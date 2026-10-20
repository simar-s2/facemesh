"""Make a small fake face dataset in the same .npz format as CoMA.

Use it to test the whole pipeline on a laptop in minutes, before you have the real data.
The "faces" are bumpy ellipsoids: each fake person has their own shape, and each
fake expression pushes a region of the surface out and back over the clip.

Usage:
    python scripts/make_fake_dataset.py --out data/fake.npz
"""

import argparse
from pathlib import Path

import numpy as np
import trimesh


def smooth_bump(vertices, center, width):
    # Gaussian weight: 1.0 at the center, fading smoothly to 0 further away
    squared_distance = ((vertices - center) ** 2).sum(axis=1)
    return np.exp(-squared_distance / (2 * width ** 2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--num_subjects", type=int, default=12)
    parser.add_argument("--num_expressions", type=int, default=12)
    parser.add_argument("--frames_per_clip", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    sphere = trimesh.creation.icosphere(subdivisions=3)   # 642 vertices on a unit sphere
    # on a unit sphere the outward normal at a vertex is just its position
    normals = np.asarray(sphere.vertices, dtype=np.float32)
    head_size_meters = np.array([0.08, 0.11, 0.09], dtype=np.float32)   # half width, half height, half depth
    template = normals * head_size_meters
    faces = np.asarray(sphere.faces, dtype=np.int64)

    # each fake expression: a region of the face, how far it pushes out, and how much it twists
    expression_center_rows = rng.integers(0, len(template), size=args.num_expressions)
    expression_amplitudes = rng.uniform(0.008, 0.020, size=args.num_expressions)    # meters at full intensity
    expression_twist_angles = rng.uniform(-0.15, 0.15, size=args.num_expressions)   # radians at full intensity

    neutral_faces = []
    expression_strengths = []
    for _ in range(args.num_subjects):
        # each fake person: their own proportions plus a few bumps (think cheekbones, nose, chin)
        neutral = template * rng.uniform(0.9, 1.1, size=3)
        for _ in range(4):
            bump_center = template[rng.integers(len(template))]
            bump_height = rng.uniform(-0.006, 0.006)
            neutral = neutral + normals * (bump_height * smooth_bump(template, bump_center, 0.03))[:, None]
        neutral_faces.append(neutral.astype(np.float32))
        expression_strengths.append(rng.uniform(0.7, 1.3))   # some people make bigger faces than others

    all_vertices, subject_ids, expression_ids, frame_positions = [], [], [], []
    for subject_id, neutral in enumerate(neutral_faces):
        for expression_id in range(args.num_expressions):
            center_row = expression_center_rows[expression_id]
            region = smooth_bump(template, template[center_row], 0.035)[:, None]
            # cross product with the normal at the center gives a sideways spin around that normal
            spin_direction = np.cross(normals[center_row], template - template[center_row])
            for frame_number in range(args.frames_per_clip):
                position = frame_number / (args.frames_per_clip - 1)
                intensity = np.sin(np.pi * position) * expression_strengths[subject_id]   # 0, up to a peak, back to 0
                push_out = normals * region * expression_amplitudes[expression_id] * intensity
                # intensity squared so the motion is not just one fixed shape scaled up and down
                twist = spin_direction * region * expression_twist_angles[expression_id] * intensity ** 2
                all_vertices.append((neutral + push_out + twist).astype(np.float32))
                subject_ids.append(subject_id)
                expression_ids.append(expression_id)
                frame_positions.append(position)

    # CoMA's rule is "neutral = average of each clip's first frame"; here every first frame has
    # intensity sin(0) = 0, so that average is exactly the neutral face we built
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out,
                        vertices=np.stack(all_vertices), faces=faces,
                        subject_ids=np.array(subject_ids, dtype=np.int64),
                        expression_ids=np.array(expression_ids, dtype=np.int64),
                        frame_positions=np.array(frame_positions, dtype=np.float32),
                        neutral_vertices=np.stack(neutral_faces),
                        subject_names=np.array([f"fake_subject_{number:02d}" for number in range(args.num_subjects)]),
                        expression_names=np.array([f"fake_expression_{number:02d}" for number in range(args.num_expressions)]))
    print(f"saved {len(all_vertices)} fake meshes, {len(template)} vertices and {len(faces)} triangles each, to {args.out}")


if __name__ == "__main__":
    main()
