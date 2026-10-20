"""Turn model predictions into USD files that open in Blender.

Make one animated file per expression for a person in the dataset:
    python -m facemesh.export_usd expressions --checkpoint runs/first_try/best.pt \
        --data data/coma.npz --subject FaceTalk_170908_03277_TA --out_dir exports/

Clean a raw scan (.ply or .obj) onto the template:
    python -m facemesh.export_usd fit --checkpoint runs/first_try/best.pt \
        --scan my_scan.ply --out_dir exports/
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import trimesh
from pxr import Gf, Usd, UsdGeom, Vt

from facemesh.dataset import denormalize, load_face_data, normalize
from facemesh.model import FaceMeshModel


def load_model(checkpoint_path):
    # weights_only=False because the checkpoint also stores numpy arrays (normalization, faces)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    normalization = checkpoint["normalization"]
    mean_face_normalized = normalize(normalization["mean_face"], normalization)
    model = FaceMeshModel(mean_face=mean_face_normalized, **checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def write_mesh_usd(out_path, frames, faces, frames_per_second=30):
    # frames: (num_frames, num_vertices, 3); one frame gives a still mesh, more gives an animation
    stage = Usd.Stage.CreateNew(str(out_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)   # FLAME and CoMA meshes are Y up
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)          # CoMA vertices are in meters
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(len(frames) - 1)
    stage.SetTimeCodesPerSecond(frames_per_second)

    root = UsdGeom.Xform.Define(stage, "/Face")
    stage.SetDefaultPrim(root.GetPrim())
    mesh = UsdGeom.Mesh.Define(stage, "/Face/Mesh")
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(faces)))   # every face is a triangle
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(faces.reshape(-1).tolist()))
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)            # show the mesh as is, no smoothing

    points_attribute = mesh.CreatePointsAttr()
    for frame_number, vertices in enumerate(frames):
        # a time sample per frame; Blender reads this as a mesh cache animation
        points_attribute.Set(Vt.Vec3fArray.FromNumpy(vertices.astype(np.float32)), Usd.TimeCode(frame_number))

    lowest_corner = frames.reshape(-1, 3).min(axis=0)
    highest_corner = frames.reshape(-1, 3).max(axis=0)
    mesh.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(*lowest_corner.tolist()), Gf.Vec3f(*highest_corner.tolist())]))
    stage.GetRootLayer().Save()


@torch.no_grad()
def export_expressions(model, checkpoint, neutral_vertices, out_dir, num_frames):
    # for every expression, predict the face at evenly spaced times and save it as an animation
    normalization = checkpoint["normalization"]
    neutral_normalized = torch.from_numpy(normalize(neutral_vertices, normalization)).unsqueeze(0)
    frame_positions = torch.linspace(0.0, 1.0, num_frames)
    written_paths = []
    for expression_id, expression_name in enumerate(checkpoint["expression_names"]):
        predicted = model.predict_expression(neutral_normalized.repeat(num_frames, 1, 1),
                                             torch.full((num_frames,), expression_id, dtype=torch.long),
                                             frame_positions)
        frames = denormalize(predicted.numpy(), normalization)
        out_path = Path(out_dir) / f"{expression_name}.usda"
        write_mesh_usd(out_path, frames, checkpoint["faces"])
        written_paths.append(out_path)
    return written_paths


@torch.no_grad()
def export_fitted_scan(model, checkpoint, scan_points, out_path):
    # run the noisy scan through the fit path and save the clean template mesh
    normalization = checkpoint["normalization"]
    scan_normalized = torch.from_numpy(normalize(scan_points.astype(np.float32), normalization)).unsqueeze(0)
    fitted = denormalize(model.fit_scan(scan_normalized)[0].numpy(), normalization)
    write_mesh_usd(out_path, fitted[np.newaxis], checkpoint["faces"])
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["expressions", "fit"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--data", help="dataset .npz, needed for --subject")
    parser.add_argument("--subject", help="subject name from the dataset to animate")
    parser.add_argument("--neutral_ply", help="or give a neutral face .ply already on the template")
    parser.add_argument("--scan", help="raw scan .ply/.obj for fit mode (must be roughly aligned to the template)")
    parser.add_argument("--num_frames", type=int, default=60)
    parser.add_argument("--num_scan_points", type=int, default=2048)
    args = parser.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    model, checkpoint = load_model(args.checkpoint)

    if args.mode == "expressions":
        if args.neutral_ply:
            # process=False keeps the vertex order, which must match the template
            neutral_vertices = np.asarray(trimesh.load(args.neutral_ply, process=False).vertices, dtype=np.float32)
        else:
            face_data = load_face_data(args.data)
            neutral_vertices = face_data["neutral_vertices"][face_data["subject_names"].index(args.subject)]
        for path in export_expressions(model, checkpoint, neutral_vertices, args.out_dir, args.num_frames):
            print(f"wrote {path}")
    else:
        scan = trimesh.load(args.scan, process=False)
        scan_vertices = np.asarray(scan.vertices, dtype=np.float32)
        chosen_rows = np.random.default_rng(0).integers(0, len(scan_vertices), size=args.num_scan_points)
        out_path = export_fitted_scan(model, checkpoint, scan_vertices[chosen_rows], Path(args.out_dir) / "fitted_scan.usda")
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
