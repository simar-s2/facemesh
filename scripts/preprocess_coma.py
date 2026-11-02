"""Convert the CoMA dataset folders into one .npz file the trainer can load fast.

Expected layout after you download and unzip CoMA (needs a free account at coma.is.tue.mpg.de):
    COMA_data/<subject>/<expression>/<expression>.000001.ply
    COMA_data/<subject>/<expression>/<expression>.000002.ply
    ...

Usage:
    python scripts/preprocess_coma.py --coma_root path/to/COMA_data --out data/coma.npz
    python scripts/preprocess_coma.py --coma_root path/to/COMA_data --out data/coma_small.npz --frame_step 3
"""

import argparse
from pathlib import Path

import numpy as np
import trimesh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coma_root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--frame_step", type=int, default=1,
                        help="keep every Nth frame; neighbouring frames are nearly identical, so 3 loses little")
    args = parser.parse_args()

    coma_root = Path(args.coma_root)
    subject_folders = sorted(folder for folder in coma_root.iterdir() if folder.is_dir())
    expression_names = sorted({clip.name for subject in subject_folders for clip in subject.iterdir() if clip.is_dir()})
    subject_names = [folder.name for folder in subject_folders]

    all_vertices = []
    subject_ids = []
    expression_ids = []
    frame_positions = []
    first_frames_per_subject = {subject_id: [] for subject_id in range(len(subject_names))}
    faces = None
    num_vertices = None

    for subject_id, subject_folder in enumerate(subject_folders):
        for expression_folder in sorted(subject_folder.iterdir()):
            if not expression_folder.is_dir():
                continue
            frame_files = sorted(expression_folder.glob("*.ply"))
            for frame_number, frame_file in enumerate(frame_files):
                if frame_number % args.frame_step != 0:
                    continue   # frame 0 is always kept, so the neutral face is unchanged
                # process=False is important: by default trimesh merges and reorders vertices,
                # which would break the vertex to vertex match between meshes
                mesh = trimesh.load(frame_file, process=False)
                vertices = np.asarray(mesh.vertices, dtype=np.float32)
                if faces is None:
                    faces = np.asarray(mesh.faces, dtype=np.int64)
                    num_vertices = len(vertices)
                # every mesh must share one template, or vertex number N would mean different spots
                if len(vertices) != num_vertices or not np.array_equal(mesh.faces, faces):
                    raise ValueError(f"{frame_file} does not match the template of the first mesh")
                all_vertices.append(vertices)
                subject_ids.append(subject_id)
                expression_ids.append(expression_names.index(expression_folder.name))
                frame_positions.append(frame_number / max(len(frame_files) - 1, 1))
                if frame_number == 0:
                    first_frames_per_subject[subject_id].append(vertices)  # clips start near neutral
            print(f"{subject_folder.name}/{expression_folder.name}: {len(frame_files)} frames")

    # a person's neutral face = average of the first frame of each of their clips
    neutral_vertices = np.stack([np.mean(first_frames_per_subject[subject_id], axis=0)
                                 for subject_id in range(len(subject_names))])

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out,
                        vertices=np.stack(all_vertices), faces=faces,
                        subject_ids=np.array(subject_ids), expression_ids=np.array(expression_ids),
                        frame_positions=np.array(frame_positions, dtype=np.float32),
                        neutral_vertices=neutral_vertices.astype(np.float32),
                        subject_names=np.array(subject_names), expression_names=np.array(expression_names))
    print(f"saved {len(all_vertices)} meshes, {len(subject_names)} subjects, {len(expression_names)} expressions, "
          f"{all_vertices[0].shape[0]} vertices each (FLAME template has 5023) to {args.out}")


if __name__ == "__main__":
    main()
