"""Loss functions and the main error metric."""

import numpy as np
import torch


def edges_from_faces(faces):
    # every triangle (a, b, c) gives three edges; keep each edge once
    all_edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
    all_edges = np.sort(all_edges, axis=1)
    unique_edges = np.unique(all_edges, axis=0)
    return torch.as_tensor(unique_edges, dtype=torch.long)


def vertex_loss(predicted_vertices, target_vertices):
    # L1 distance per coordinate; less hurt by a few bad vertices than L2
    return (predicted_vertices - target_vertices).abs().mean()


def edge_length_loss(predicted_vertices, target_vertices, edges):
    # compare the length of every mesh edge; punishes spiky, crumpled surfaces
    def edge_lengths(vertices):
        start_points = vertices[:, edges[:, 0]]
        end_points = vertices[:, edges[:, 1]]
        return (start_points - end_points).norm(dim=-1)
    return (edge_lengths(predicted_vertices) - edge_lengths(target_vertices)).abs().mean()


def mean_vertex_error_mm(predicted_vertices, target_vertices, scale, units_to_mm):
    # average straight line distance per vertex, converted back to millimeters
    distance = (predicted_vertices - target_vertices).norm(dim=-1).mean()
    return distance.item() * scale * units_to_mm
