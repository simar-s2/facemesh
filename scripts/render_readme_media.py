"""Render the README images from a trained checkpoint.

Writes docs/images/logo.svg, hero.gif (a noisy scan cleaned onto the template, turning)
and expression.gif (predicted expression next to the real motion) for one held-out person.

Needs two extra packages that training does not:  pip install matplotlib pillow

Usage:
    python scripts/render_readme_media.py --data data/coma.npz \
        --checkpoint runs/coma_subject/best.pt --subject FaceTalk_170915_00223_TA
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # draw to images, no window
import matplotlib.pyplot as plt
import numpy as np
import torch
import trimesh
from matplotlib import colors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # so this script can import the facemesh package
from facemesh.dataset import denormalize, load_face_data, make_noisy_scan, normalize
from facemesh.export_usd import load_model

BACKGROUND = "#0b0e14"
TITLE_COLOR = "#e5e7eb"
CAPTION_COLOR = "#9ca3af"
# grey for still parts of the face, then blue, purple and pink for more and more movement
MOVEMENT_COLORS = colors.LinearSegmentedColormap.from_list("movement", ["#64748b", "#3b82f6", "#a855f7", "#f472b6"])
CAMERA_ELEVATION = 8
CAMERA_AZIMUTH = 72   # FLAME faces look along +z; this views them from the front, slightly turned


def write_logo(out_path):
    # a low-poly wireframe head on the same dark tile and blue-to-purple gradient as kbsync's logo
    sphere = trimesh.creation.icosphere(subdivisions=1)
    head = sphere.vertices * np.array([0.78, 1.0, 0.85])
    turn = trimesh.transformations.rotation_matrix(np.radians(-18), [0, 1, 0])[:3, :3]
    tilt = trimesh.transformations.rotation_matrix(np.radians(12), [1, 0, 0])[:3, :3]
    head = head @ (turn @ tilt).T
    lines = []
    for start, end in sphere.edges_unique:
        if head[start, 2] + head[end, 2] > -0.25:   # only edges on the front half
            lines.append(f'<line x1="{128 + head[start, 0] * 82:.1f}" y1="{132 - head[start, 1] * 82:.1f}" '
                         f'x2="{128 + head[end, 0] * 82:.1f}" y2="{132 - head[end, 1] * 82:.1f}"/>')
    dots = [f'<circle cx="{128 + x * 82:.1f}" cy="{132 - y * 82:.1f}" r="5"/>' for x, y, z in head if z > -0.1]
    out_path.write_text(f'''<svg xmlns="http://www.w3.org/2000/svg" width="256" height="256" viewBox="0 0 256 256">
  <defs>
    <linearGradient id="tile" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#232a3b"/>
      <stop offset="1" stop-color="#0b0e14"/>
    </linearGradient>
    <linearGradient id="mesh" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#60a5fa"/>
      <stop offset="1" stop-color="#c084fc"/>
    </linearGradient>
    <filter id="glow" x="-30%" y="-30%" width="160%" height="160%">
      <feGaussianBlur stdDeviation="6"/>
    </filter>
    <g id="wire" stroke-width="5" stroke-linecap="round">
      {"".join(lines)}
    </g>
    <g id="dots">{"".join(dots)}</g>
  </defs>
  <rect width="256" height="256" rx="56" fill="url(#tile)"/>
  <use href="#wire" stroke="url(#mesh)" filter="url(#glow)" opacity="0.6"/>
  <use href="#wire" stroke="url(#mesh)"/>
  <use href="#dots" fill="#f8fafc"/>
</svg>
''')


def set_up_axis(axis, view_center, view_half_size, azimuth, title):
    # same size box on every axis so the head is not stretched
    for set_limit, center in zip([axis.set_xlim, axis.set_ylim, axis.set_zlim], view_center):
        set_limit(center - view_half_size, center + view_half_size)
    axis.set_box_aspect((1, 1, 1))
    axis.view_init(elev=CAMERA_ELEVATION, azim=azimuth)
    axis.set_axis_off()
    axis.set_facecolor(BACKGROUND)
    axis.set_title(title, color=TITLE_COLOR, fontsize=12, pad=-4)


def draw_mesh(axis, vertices, faces, movement_per_face, largest_movement):
    # y is up in the data but z is up in matplotlib, so swap the two
    triangles = vertices[:, [0, 2, 1]][faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    light_direction = np.array([0.3, -0.8, 0.6]) / np.linalg.norm([0.3, -0.8, 0.6])
    brightness = 0.55 + 0.45 * np.clip(normals @ light_direction, 0, 1)   # simple lighting so the face looks solid
    face_colors = MOVEMENT_COLORS(np.clip(movement_per_face / largest_movement, 0, 1))
    face_colors[:, :3] *= brightness[:, None]
    axis.add_collection3d(Poly3DCollection(triangles, facecolors=face_colors, edgecolors=(1, 1, 1, 0.06), linewidths=0.3))


def figure_to_image(figure):
    figure.canvas.draw()
    image = Image.frombuffer("RGBA", figure.canvas.get_width_height(), figure.canvas.buffer_rgba()).convert("RGB")
    plt.close(figure)
    return image


def save_gif(frames, out_path, milliseconds_per_frame):
    frames = frames + frames[::-1][1:-1]   # play forward then back, so the loop has no jump
    # 128 colors per frame keeps the file a few MB instead of tens of MB
    palette_frames = [frame.quantize(colors=128, method=Image.Quantize.MEDIANCUT) for frame in frames]
    palette_frames[0].save(out_path, save_all=True, append_images=palette_frames[1:],
                           duration=milliseconds_per_frame, loop=0, optimize=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True, help="a model trained on the subject split, so --subject is unseen")
    parser.add_argument("--subject", required=True, help="a held-out person, for example FaceTalk_170915_00223_TA")
    parser.add_argument("--out_dir", default="docs/images")
    parser.add_argument("--num_frames", type=int, default=40, help="frames of the expression clip to show")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_logo(out_dir / "logo.svg")

    face_data = load_face_data(args.data)
    model, checkpoint = load_model(args.checkpoint)
    normalization = checkpoint["normalization"]
    faces = face_data["faces"]
    num_vertices = face_data["vertices"].shape[1]
    subject_id = face_data["subject_names"].index(args.subject)
    neutral = face_data["neutral_vertices"][subject_id]

    # frame the camera on this person's head, a little tight so the face fills the picture
    neutral_plotted = neutral[:, [0, 2, 1]]
    view_center = (neutral_plotted.max(axis=0) + neutral_plotted.min(axis=0)) / 2
    view_half_size = (neutral_plotted.max(axis=0) - neutral_plotted.min(axis=0)).max() / 2 * 0.8

    # show the expression where this person moves the most
    clips = {}
    for expression_id in range(len(face_data["expression_names"])):
        rows = np.where((face_data["subject_ids"] == subject_id) & (face_data["expression_ids"] == expression_id))[0]
        clips[expression_id] = rows[np.argsort(face_data["frame_positions"][rows])]
    expression_id = max(clips, key=lambda candidate: np.linalg.norm(face_data["vertices"][clips[candidate]] - neutral, axis=2).max())
    clip_rows = clips[expression_id]
    clip_rows = clip_rows[np.linspace(0, len(clip_rows) - 1, min(args.num_frames, len(clip_rows))).round().astype(int)]
    expression_name = face_data["expression_names"][expression_id]

    # hero.gif: a noisy scan of the strongest moment, and the model's clean template mesh, turning
    true_vertices = face_data["vertices"][clip_rows[len(clip_rows) // 2]]
    scan = make_noisy_scan(true_vertices, 2048, noise_std=0.0015, hole_radius=0.02, outlier_fraction=0.02,
                           rng=np.random.default_rng(3))
    with torch.no_grad():
        scan_normalized = torch.from_numpy(normalize(scan, normalization)).unsqueeze(0)
        fitted = denormalize(model.fit_scan(scan_normalized)[0].numpy(), normalization)
    scan_error_mm = np.linalg.norm(fitted - true_vertices, axis=1).mean() * 1000
    frames = []
    for azimuth in np.linspace(CAMERA_AZIMUTH - 45, CAMERA_AZIMUTH + 45, 30):
        figure = plt.figure(figsize=(8, 4.2), dpi=90, facecolor=BACKGROUND)
        axis = figure.add_subplot(1, 2, 1, projection="3d")
        scan_plotted = scan[:, [0, 2, 1]]
        axis.scatter(scan_plotted[:, 0], scan_plotted[:, 1], scan_plotted[:, 2], s=2.5, c="#60a5fa", alpha=0.85)
        set_up_axis(axis, view_center, view_half_size, azimuth, "noisy scan, 2,048 unordered points")
        axis = figure.add_subplot(1, 2, 2, projection="3d")
        draw_mesh(axis, fitted, faces, np.full(len(faces), 0.55), 1.0)
        set_up_axis(axis, view_center, view_half_size, azimuth, f"clean {num_vertices:,}-vertex template")
        figure.text(0.5, 0.04, f"person never seen in training   |   mean error {scan_error_mm:.2f} mm",
                    color=CAPTION_COLOR, ha="center", fontsize=10)
        figure.subplots_adjust(left=0, right=1, top=0.94, bottom=0.08, wspace=0)
        frames.append(figure_to_image(figure))
    save_gif(frames, out_dir / "hero.gif", milliseconds_per_frame=70)

    # expression.gif: the predicted clip next to the real one, colored by how far each triangle moved
    true_frames = face_data["vertices"][clip_rows]
    with torch.no_grad():
        neutral_normalized = torch.from_numpy(normalize(neutral, normalization)).unsqueeze(0).repeat(len(clip_rows), 1, 1)
        predicted = model.predict_expression(neutral_normalized, torch.full((len(clip_rows),), expression_id),
                                             torch.from_numpy(face_data["frame_positions"][clip_rows]))
    predicted_frames = denormalize(predicted.numpy(), normalization)
    largest_movement_mm = (np.linalg.norm(true_frames - neutral, axis=2) * 1000).max()
    frames = []
    for frame_number, row in enumerate(clip_rows):
        figure = plt.figure(figsize=(8, 4.2), dpi=90, facecolor=BACKGROUND)
        for panel, (vertices, title) in enumerate([(predicted_frames[frame_number], "model prediction"),
                                                   (true_frames[frame_number], "real motion")]):
            axis = figure.add_subplot(1, 2, panel + 1, projection="3d")
            movement_per_face = (np.linalg.norm(vertices - neutral, axis=1) * 1000)[faces].mean(axis=1)
            draw_mesh(axis, vertices, faces, movement_per_face, largest_movement_mm)
            set_up_axis(axis, view_center, view_half_size, CAMERA_AZIMUTH, title)
        frame_error_mm = np.linalg.norm(predicted_frames[frame_number] - true_frames[frame_number], axis=1).mean() * 1000
        figure.text(0.5, 0.04, f"{expression_name}, new person, t = {face_data['frame_positions'][row]:.2f}   |   "
                    f"error {frame_error_mm:.2f} mm   |   color = movement", color=CAPTION_COLOR, ha="center", fontsize=10)
        figure.subplots_adjust(left=0, right=1, top=0.94, bottom=0.08, wspace=0)
        frames.append(figure_to_image(figure))
    save_gif(frames, out_dir / "expression.gif", milliseconds_per_frame=60)

    print(f"wrote {out_dir}/logo.svg, hero.gif ({expression_name}, scan error {scan_error_mm:.2f} mm) and expression.gif")


if __name__ == "__main__":
    main()
