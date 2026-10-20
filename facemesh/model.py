"""The FaceMesh network.

One shared encoder (PointNet) turns any set of 3D points into a fixed size code.
Two small decoders read that code:
  - expression_decoder: neutral face code + expression + time -> how each vertex moves
  - fit_decoder: noisy scan code -> clean face on the standard template
"""

import math

import torch
from torch import nn


class PointNetEncoder(nn.Module):
    """Runs the same small network on every point, then keeps the max of each feature.

    Max pooling makes the output the same no matter how the points are ordered,
    which is what we need for raw scans that have no vertex order.
    """

    def __init__(self, code_size=256):
        super().__init__()
        # Conv1d with kernel size 1 is just a Linear layer applied to every point separately.
        # No BatchNorm on purpose: this encoder sees clean neutral faces AND noisy scans, so
        # BatchNorm's saved averages would be a blur of both and hurt at test time.
        self.per_point_layers = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=1), nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=1), nn.ReLU(),
            nn.Conv1d(128, code_size, kernel_size=1),
        )

    def forward(self, points):
        # points: (batch, num_points, 3) -> features: (batch, code_size, num_points)
        per_point_features = self.per_point_layers(points.transpose(1, 2))
        # keep the strongest response for each feature across all points
        return per_point_features.max(dim=2).values


def make_decoder(input_size, hidden_size, output_size):
    # plain 3 layer MLP; last layer starts at zero so the first guess is "no change"
    decoder = nn.Sequential(
        nn.Linear(input_size, hidden_size), nn.ReLU(),
        nn.Linear(hidden_size, hidden_size), nn.ReLU(),
        nn.Linear(hidden_size, output_size),
    )
    nn.init.zeros_(decoder[-1].weight)
    nn.init.zeros_(decoder[-1].bias)
    return decoder


def time_bin_index(frame_positions, num_time_bins):
    # which of num_time_bins equal slices of the clip each time falls in (1.0 goes in the last one)
    return torch.clamp((frame_positions * num_time_bins).long(), max=num_time_bins - 1)


def time_features(frame_positions):
    # turn one number in [0, 1] into a few sine and cosine waves so the MLP can model rise and fall
    angle = math.pi * frame_positions.unsqueeze(1)
    return torch.cat([frame_positions.unsqueeze(1), torch.sin(angle), torch.cos(angle),
                      torch.sin(2 * angle), torch.cos(2 * angle)], dim=1)


class FaceMeshModel(nn.Module):
    def __init__(self, num_vertices, num_expressions, mean_face, code_size=256,
                 expression_embedding_size=32, hidden_size=512, num_time_bins=10, identity_dropout=0.0):
        super().__init__()
        self.num_vertices = num_vertices
        self.num_time_bins = num_time_bins
        self.identity_dropout = identity_dropout
        self.encoder = PointNetEncoder(code_size)
        # each expression name gets its own learned vector
        self.expression_embedding = nn.Embedding(num_expressions, expression_embedding_size)
        expression_input_size = code_size + expression_embedding_size + 5  # 5 = time feature count
        self.expression_decoder = make_decoder(expression_input_size, hidden_size, num_vertices * 3)
        self.fit_decoder = make_decoder(code_size, hidden_size, num_vertices * 3)
        # the average training face, in normalized units; saved with the model but not trained
        self.register_buffer("mean_face", torch.as_tensor(mean_face, dtype=torch.float32))
        # average training offset per (expression, time bin); all zeros unless train.py fills it in,
        # in which case the network only has to learn a correction on top of the baseline
        self.register_buffer("mean_offsets", torch.zeros(num_expressions, num_time_bins, num_vertices, 3))

    def predict_expression(self, neutral_vertices, expression_ids, frame_positions):
        # neutral_vertices: (batch, num_vertices, 3) on the template
        identity_code = self.encoder(neutral_vertices)
        if self.training and self.identity_dropout > 0:
            # sometimes hide who the person is, so the decoder also learns motion that works for anyone
            keep = (torch.rand(identity_code.shape[0], 1, device=identity_code.device) >= self.identity_dropout)
            identity_code = identity_code * keep.float()
        decoder_input = torch.cat([identity_code,
                                   self.expression_embedding(expression_ids),
                                   time_features(frame_positions)], dim=1)
        vertex_offsets = self.expression_decoder(decoder_input).view(-1, self.num_vertices, 3)
        baseline_offsets = self.mean_offsets[expression_ids, time_bin_index(frame_positions, self.num_time_bins)]
        # predict the change from neutral, not the whole face; the change is small and easier to learn
        return neutral_vertices + baseline_offsets + vertex_offsets

    def fit_scan(self, scan_points):
        # scan_points: (batch, any number of points, 3), any order
        scan_code = self.encoder(scan_points)
        vertex_offsets = self.fit_decoder(scan_code).view(-1, self.num_vertices, 3)
        return self.mean_face.unsqueeze(0) + vertex_offsets
