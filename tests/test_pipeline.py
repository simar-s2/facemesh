"""Quick checks that the pieces fit together. Run with: python -m pytest tests/"""

import numpy as np
import torch
from pxr import Usd, UsdGeom

from facemesh.dataset import make_noisy_scan
from facemesh.export_usd import write_mesh_usd
from facemesh.model import FaceMeshModel


def test_encoder_ignores_point_order():
    # shuffling the points must not change the code, or raw scans would give random results
    model = FaceMeshModel(num_vertices=50, num_expressions=3, mean_face=np.zeros((50, 3))).eval()
    points = torch.randn(1, 200, 3)
    shuffled = points[:, torch.randperm(200)]
    assert torch.allclose(model.encoder(points), model.encoder(shuffled), atol=1e-5)


def test_untrained_model_returns_neutral_face():
    # last layer starts at zero, so before training the prediction equals the input neutral
    model = FaceMeshModel(num_vertices=50, num_expressions=3, mean_face=np.zeros((50, 3))).eval()
    neutral = torch.randn(2, 50, 3)
    predicted = model.predict_expression(neutral, torch.tensor([0, 2]), torch.tensor([0.0, 0.5]))
    assert torch.allclose(predicted, neutral)


def test_untrained_model_with_offset_table_returns_baseline():
    # with baseline correction on, the starting prediction is exactly the mean offset baseline
    model = FaceMeshModel(num_vertices=50, num_expressions=3, mean_face=np.zeros((50, 3)), num_time_bins=10).eval()
    model.mean_offsets.copy_(torch.randn(3, 10, 50, 3))
    neutral = torch.randn(2, 50, 3)
    predicted = model.predict_expression(neutral, torch.tensor([1, 2]), torch.tensor([0.25, 1.0]))
    # time 0.25 falls in bin 2; time 1.0 goes in the last bin, 9
    assert torch.allclose(predicted[0], neutral[0] + model.mean_offsets[1, 2])
    assert torch.allclose(predicted[1], neutral[1] + model.mean_offsets[2, 9])


def test_noisy_scan_has_requested_size():
    clean = np.random.default_rng(0).normal(size=(500, 3)).astype(np.float32)
    scan = make_noisy_scan(clean, 1024, noise_std=0.01, hole_radius=0.5, outlier_fraction=0.05,
                           rng=np.random.default_rng(1))
    assert scan.shape == (1024, 3)


def test_usd_file_has_every_frame(tmp_path):
    frames = np.random.default_rng(0).normal(size=(5, 4, 3)).astype(np.float32)
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    out_path = tmp_path / "face.usda"
    write_mesh_usd(out_path, frames, faces)
    stage = Usd.Stage.Open(str(out_path))   # keep the stage in a variable or Python frees it
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/Face/Mesh"))
    assert len(mesh.GetPointsAttr().GetTimeSamples()) == 5
    assert np.allclose(np.array(mesh.GetPointsAttr().Get(3)), frames[3])
