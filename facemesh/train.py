"""Train FaceMesh.

Example:
    python -m facemesh.train --data data/coma.npz --out_dir runs/first_try --epochs 60
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from facemesh.dataset import FaceMeshDataset, compute_normalization, load_face_data, normalize, split_indices
from facemesh.losses import edge_length_loss, edges_from_faces, mean_vertex_error_mm, vertex_loss
from facemesh.model import FaceMeshModel, time_bin_index


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="path to the .npz made by a preprocess script")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--split", default="subject", choices=["subject", "interpolation"])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--edge_weight", type=float, default=0.5, help="how much the edge length loss counts")
    parser.add_argument("--fit_weight", type=float, default=1.0, help="how much the scan fitting job counts")
    parser.add_argument("--num_scan_points", type=int, default=2048)
    parser.add_argument("--num_time_bins", type=int, default=10, help="time slices for the mean offset baseline")
    parser.add_argument("--baseline_correction", action=argparse.BooleanOptionalAction, default=False,
                        help="learn a correction on top of the mean offset baseline instead of the whole motion")
    parser.add_argument("--identity_dropout", type=float, default=0.0,
                        help="chance of hiding the identity code from the expression decoder while training")
    parser.add_argument("--augment_stretch", type=float, default=0.05,
                        help="random per-axis head stretch while training, 0.05 = up to 5 percent; 0 turns it off")
    parser.add_argument("--units_to_mm", type=float, default=1000.0, help="1000 if meshes are in meters")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true", help="use bfloat16 mixed precision on GPU")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def mean_offset_table(face_data, train_indices, num_time_bins):
    # for each (expression, time bin): the average of (mesh - that person's neutral) over training meshes
    num_expressions = len(face_data["expression_names"])
    num_vertices = face_data["vertices"].shape[1]
    table = np.zeros((num_expressions, num_time_bins, num_vertices, 3), dtype=np.float32)
    expression_ids = face_data["expression_ids"][train_indices]
    time_bins = time_bin_index(torch.from_numpy(face_data["frame_positions"][train_indices]), num_time_bins).numpy()
    for expression_id in range(num_expressions):
        for time_bin in range(num_time_bins):
            group = train_indices[(expression_ids == expression_id) & (time_bins == time_bin)]
            if len(group) == 0:
                continue   # never seen in training, so the baseline predicts "no change"
            neutral = face_data["neutral_vertices"][face_data["subject_ids"][group]]
            table[expression_id, time_bin] = (face_data["vertices"][group] - neutral).mean(axis=0)
    return table


def baseline_error_mm(face_data, validation_indices, table, units_to_mm):
    # the simple baseline: predict neutral + table entry, then measure the same mm error as the model
    num_time_bins = table.shape[1]
    time_bins = time_bin_index(torch.from_numpy(face_data["frame_positions"][validation_indices]), num_time_bins).numpy()
    errors = []
    for index, time_bin in zip(validation_indices, time_bins):
        neutral = face_data["neutral_vertices"][face_data["subject_ids"][index]]
        prediction = neutral + table[face_data["expression_ids"][index], time_bin]
        errors.append(np.linalg.norm(prediction - face_data["vertices"][index], axis=1).mean())
    return float(np.mean(errors)) * units_to_mm


def run_one_batch(model, batch, edges, edge_weight, fit_weight):
    # do both jobs on one batch and return the total loss plus the two predictions
    predicted_expression = model.predict_expression(batch["neutral"], batch["expression_id"], batch["frame_position"])
    fitted_face = model.fit_scan(batch["noisy_scan"])
    expression_loss = vertex_loss(predicted_expression, batch["target"]) + \
        edge_weight * edge_length_loss(predicted_expression, batch["target"], edges)
    fit_loss = vertex_loss(fitted_face, batch["target"]) + \
        edge_weight * edge_length_loss(fitted_face, batch["target"], edges)
    return expression_loss + fit_weight * fit_loss, predicted_expression, fitted_face


@torch.no_grad()
def evaluate(model, loader, edges, args, scale, device):
    # average millimeter error for both jobs over the whole validation set
    model.eval()
    expression_error_total = 0.0
    fit_error_total = 0.0
    sample_count = 0
    for batch in loader:
        batch = {name: value.to(device) for name, value in batch.items()}
        _, predicted_expression, fitted_face = run_one_batch(model, batch, edges, args.edge_weight, args.fit_weight)
        batch_size = batch["target"].shape[0]
        expression_error_total += mean_vertex_error_mm(predicted_expression, batch["target"], scale, args.units_to_mm) * batch_size
        fit_error_total += mean_vertex_error_mm(fitted_face, batch["target"], scale, args.units_to_mm) * batch_size
        sample_count += batch_size
    return expression_error_total / sample_count, fit_error_total / sample_count


def main():
    args = parse_arguments()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # NVIDIA GPU on AWS, Apple GPU on a Mac, otherwise CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    face_data = load_face_data(args.data)
    train_indices, validation_indices = split_indices(face_data, args.split)
    normalization = compute_normalization(face_data, train_indices)
    print(f"split={args.split}: {len(train_indices)} training meshes, {len(validation_indices)} validation meshes, "
          f"{face_data['vertices'].shape[1]} vertices each, device={device}")

    offset_table = mean_offset_table(face_data, train_indices, args.num_time_bins)
    baseline_error = baseline_error_mm(face_data, validation_indices, offset_table, args.units_to_mm)
    print(f"mean offset baseline, expression error: {baseline_error:.3f} mm   <- the number to beat")

    dataset_settings = dict(normalization=normalization, num_scan_points=args.num_scan_points,
                            units_to_mm=args.units_to_mm, augment_stretch=args.augment_stretch)
    train_loader = DataLoader(FaceMeshDataset(face_data, train_indices, is_training=True, **dataset_settings),
                              batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              drop_last=True, pin_memory=device.type == "cuda", persistent_workers=args.num_workers > 0)
    validation_loader = DataLoader(FaceMeshDataset(face_data, validation_indices, is_training=False, **dataset_settings),
                                   batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                                   persistent_workers=args.num_workers > 0)

    mean_face_normalized = normalize(normalization["mean_face"], normalization)
    model_config = dict(num_vertices=face_data["vertices"].shape[1],
                        num_expressions=len(face_data["expression_names"]),
                        num_time_bins=args.num_time_bins, identity_dropout=args.identity_dropout)
    model = FaceMeshModel(mean_face=mean_face_normalized, **model_config)
    if args.baseline_correction:
        # offsets are differences of positions, so only the scale applies, not the center
        model.mean_offsets.copy_(torch.from_numpy(offset_table / normalization["scale"]))
    model = model.to(device)
    edges = edges_from_faces(face_data["faces"]).to(device)
    print(f"model parameters: {sum(parameter.numel() for parameter in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)   # slowly lower the learning rate

    # epoch 0 = before any training: the expression job outputs the starting guess, the fit job the mean face
    expression_error, fit_error = evaluate(model, validation_loader, edges, args, normalization["scale"], device)
    print(f"epoch   0  (untrained)        expression {expression_error:.3f} mm  fit {fit_error:.3f} mm")
    history = [dict(epoch=0, train_loss=None, expression_error_mm=expression_error, fit_error_mm=fit_error, seconds=0.0)]
    untrained_fit_error = fit_error

    best_score = float("inf")
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_start = time.time()
        loss_total = 0.0
        for batch in train_loader:
            batch = {name: value.to(device, non_blocking=True) for name, value in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.amp and device.type == "cuda"):
                loss, _, _ = run_one_batch(model, batch, edges, args.edge_weight, args.fit_weight)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)   # stop rare huge updates
            optimizer.step()
            loss_total += loss.item()
        scheduler.step()

        expression_error, fit_error = evaluate(model, validation_loader, edges, args, normalization["scale"], device)
        epoch_record = dict(epoch=epoch, train_loss=loss_total / len(train_loader),
                            expression_error_mm=expression_error, fit_error_mm=fit_error,
                            seconds=time.time() - epoch_start)
        history.append(epoch_record)
        print(f"epoch {epoch:3d}  loss {epoch_record['train_loss']:.4f}  "
              f"expression {expression_error:.3f} mm  fit {fit_error:.3f} mm  ({epoch_record['seconds']:.1f}s)")

        # each job as a fraction of its simple answer (baseline, mean face), so the larger fit error
        # in mm does not decide on its own which epoch is best
        score = expression_error / baseline_error + fit_error / untrained_fit_error
        if score < best_score:
            best_score = score
            best_epoch = epoch
            torch.save({
                "model_state": model.state_dict(),
                "model_config": model_config,
                "normalization": normalization,
                "faces": face_data["faces"],
                "expression_names": face_data["expression_names"],
                "subject_names": face_data["subject_names"],
                "arguments": vars(args),
            }, out_dir / "best.pt")

    best_record = history[best_epoch]
    summary = dict(split=args.split, baseline_expression_error_mm=baseline_error, best_epoch=best_epoch,
                   best_expression_error_mm=best_record["expression_error_mm"],
                   best_fit_error_mm=best_record["fit_error_mm"], arguments=vars(args), history=history)
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(f"done. best epoch {best_epoch}: expression {best_record['expression_error_mm']:.3f} mm "
          f"(baseline {baseline_error:.3f} mm), fit {best_record['fit_error_mm']:.3f} mm")


if __name__ == "__main__":
    main()
