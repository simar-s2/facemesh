"""Loads the preprocessed face meshes and builds training samples.

Every sample gives the model two jobs:
1. Expression job: given a person's neutral face, an expression name and a
   time in the clip, predict where every vertex of the template moves.
2. Fit job: given a noisy, holey point cloud of a face, predict the clean
   face on the standard template (same vertex count and order every time).
"""

import numpy as np
import torch
from torch.utils.data import Dataset


def load_face_data(npz_path):
    # read the single .npz file made by scripts/preprocess_coma.py or make_fake_dataset.py
    raw = np.load(npz_path, allow_pickle=True)
    return {
        "vertices": raw["vertices"].astype(np.float32),            # (num_meshes, num_vertices, 3)
        "faces": raw["faces"].astype(np.int64),                    # (num_triangles, 3), same for every mesh
        "subject_ids": raw["subject_ids"].astype(np.int64),        # which person each mesh belongs to
        "expression_ids": raw["expression_ids"].astype(np.int64),  # which expression clip it came from
        "frame_positions": raw["frame_positions"].astype(np.float32),  # 0.0 = clip start, 1.0 = clip end
        "neutral_vertices": raw["neutral_vertices"].astype(np.float32),  # (num_subjects, num_vertices, 3)
        "subject_names": [str(name) for name in raw["subject_names"]],
        "expression_names": [str(name) for name in raw["expression_names"]],
    }


def split_indices(face_data, split_mode, num_validation_subjects=2):
    # pick which meshes are for training and which are held out for validation
    all_indices = np.arange(len(face_data["vertices"]))
    if split_mode == "subject":
        # hold out whole people so we test on faces the model has never seen
        num_subjects = len(face_data["subject_names"])
        validation_subjects = set(range(num_subjects - num_validation_subjects, num_subjects))
        is_validation = np.array([subject in validation_subjects for subject in face_data["subject_ids"]])
    elif split_mode == "interpolation":
        # hold out every 10th frame, like the CoMA paper's interpolation test
        is_validation = (all_indices % 10) == 0
    else:
        raise ValueError(f"unknown split_mode: {split_mode}")
    return all_indices[~is_validation], all_indices[is_validation]


def compute_normalization(face_data, train_indices):
    # one center point and one scale for the whole dataset so all inputs live near 0 with size about 1
    train_vertices = face_data["vertices"][train_indices]
    center = train_vertices.reshape(-1, 3).mean(axis=0)
    scale = float(train_vertices.reshape(-1, 3).std())
    mean_face = train_vertices.mean(axis=0)
    return {"center": center.astype(np.float32), "scale": scale, "mean_face": mean_face.astype(np.float32)}


def normalize(vertices, normalization):
    # move into the model's space
    return (vertices - normalization["center"]) / normalization["scale"]


def denormalize(vertices, normalization):
    # move back into real units (meters for CoMA)
    return vertices * normalization["scale"] + normalization["center"]


def make_noisy_scan(clean_vertices, num_points, noise_std, hole_radius, outlier_fraction, rng):
    # fake what a real 3D scanner gives you: jitter, a missing patch, and stray points
    hole_center = clean_vertices[rng.integers(len(clean_vertices))]
    distance_to_hole = np.linalg.norm(clean_vertices - hole_center, axis=1)
    kept_vertices = clean_vertices[distance_to_hole > hole_radius]
    if len(kept_vertices) < 10:
        kept_vertices = clean_vertices  # hole was too big, skip it this time

    chosen_rows = rng.integers(0, len(kept_vertices), size=num_points)
    scan_points = kept_vertices[chosen_rows] + rng.normal(0.0, noise_std, size=(num_points, 3))

    num_outliers = int(outlier_fraction * num_points)
    lowest_corner = clean_vertices.min(axis=0)
    highest_corner = clean_vertices.max(axis=0)
    scan_points[:num_outliers] = rng.uniform(lowest_corner, highest_corner, size=(num_outliers, 3))

    rng.shuffle(scan_points)  # a scan has no vertex order, so do not let the model rely on one
    return scan_points.astype(np.float32)


class FaceMeshDataset(Dataset):
    def __init__(self, face_data, indices, normalization, num_scan_points=2048,
                 noise_mm=1.5, hole_radius_mm=20.0, outlier_fraction=0.02,
                 units_to_mm=1000.0, is_training=True):
        self.face_data = face_data
        self.indices = indices
        self.normalization = normalization
        self.num_scan_points = num_scan_points
        self.noise_std = noise_mm / units_to_mm             # convert mm into data units
        self.hole_radius = hole_radius_mm / units_to_mm
        self.outlier_fraction = outlier_fraction
        self.is_training = is_training

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        mesh_index = self.indices[position]
        target_vertices = self.face_data["vertices"][mesh_index]
        subject_id = self.face_data["subject_ids"][mesh_index]
        neutral_vertices = self.face_data["neutral_vertices"][subject_id]

        # training: fresh noise every time (a new OS-seeded generator, so DataLoader workers never repeat
        # each other); validation: seeded by mesh index so every epoch is scored on identical scans
        rng = np.random.default_rng() if self.is_training else np.random.default_rng(int(mesh_index))
        noisy_scan = make_noisy_scan(target_vertices, self.num_scan_points, self.noise_std,
                                     self.hole_radius, self.outlier_fraction, rng)

        return {
            "neutral": torch.from_numpy(normalize(neutral_vertices, self.normalization)),
            "target": torch.from_numpy(normalize(target_vertices, self.normalization)),
            "noisy_scan": torch.from_numpy(normalize(noisy_scan, self.normalization)),
            "expression_id": torch.tensor(self.face_data["expression_ids"][mesh_index]),
            "frame_position": torch.tensor(self.face_data["frame_positions"][mesh_index]),
            "subject_id": torch.tensor(subject_id),
        }
