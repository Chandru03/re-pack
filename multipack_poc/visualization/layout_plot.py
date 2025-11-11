"""
Plotly-based 3D visualisations for carton-in-box and box-on-pallet layouts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import plotly.graph_objects as go
import plotly.io as pio
from plotly.colors import qualitative

from multipack_poc.core.solver_box_to_pallet import BoxPalletPackingResult
from multipack_poc.core.solver_carton_to_box import CartonBoxPackingResult
from multipack_poc.core.utils_geometry import Orientation, Placement
from multipack_poc.models.box import Box
from multipack_poc.models.package import Package
from multipack_poc.models.pallet import Pallet

DEFAULT_COLOR_SEQUENCE = qualitative.Light24


def _prism_vertices(x: int, y: int, z: int, dx: int, dy: int, dz: int) -> Tuple[List[int], List[int], List[int]]:
    xs = [x, x + dx, x + dx, x, x, x + dx, x + dx, x]
    ys = [y, y, y + dy, y + dy, y, y, y + dy, y + dy]
    zs = [z, z, z, z, z + dz, z + dz, z + dz, z + dz]
    return xs, ys, zs


def _mesh_from_prism(
    x: int,
    y: int,
    z: int,
    dx: int,
    dy: int,
    dz: int,
    color: str,
    name: str,
    opacity: float = 0.92,
) -> go.Mesh3d:
    xs, ys, zs = _prism_vertices(x, y, z, dx, dy, dz)
    # Triangle indices referencing the 8 vertices.
    i = [0, 0, 0, 4, 4, 4, 0, 1, 2, 3, 5, 6]
    j = [1, 2, 3, 5, 6, 7, 4, 5, 6, 7, 6, 7]
    k = [2, 3, 1, 6, 7, 5, 1, 2, 3, 0, 7, 4]
    return go.Mesh3d(
        x=xs,
        y=ys,
        z=zs,
        i=i,
        j=j,
        k=k,
        opacity=opacity,
        color=color,
        name=name,
        showscale=False,
        flatshading=True,
        lighting=dict(ambient=0.4, diffuse=0.9, specular=0.3, fresnel=0.1, roughness=0.5),
        lightposition=dict(x=0.3, y=0.6, z=1.0),
        hoverinfo="skip",
    )


def _edge_trace(
    x: int,
    y: int,
    z: int,
    dx: int,
    dy: int,
    dz: int,
    color: str,
    name: str,
) -> go.Scatter3d:
    xs, ys, zs = _prism_vertices(x, y, z, dx, dy, dz)
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]
    x_coords: List[float] = []
    y_coords: List[float] = []
    z_coords: List[float] = []
    for start, end in edges:
        x_coords.extend([xs[start], xs[end], None])
        y_coords.extend([ys[start], ys[end], None])
        z_coords.extend([zs[start], zs[end], None])

    return go.Scatter3d(
        x=x_coords,
        y=y_coords,
        z=z_coords,
        mode="lines",
        line=dict(color=color, width=3),
        name=name,
        showlegend=False,
        hoverinfo="skip",
    )


def _color_for_index(index: int) -> str:
    return DEFAULT_COLOR_SEQUENCE[index % len(DEFAULT_COLOR_SEQUENCE)]


def _container_wireframe(
    dx: int,
    dy: int,
    dz: int,
    name: str,
    color: str = "#4a5568",
) -> go.Scatter3d:
    vertices = [
        (0, 0, 0),
        (dx, 0, 0),
        (dx, dy, 0),
        (0, dy, 0),
        (0, 0, dz),
        (dx, 0, dz),
        (dx, dy, dz),
        (0, dy, dz),
    ]
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]
    x_coords = []
    y_coords = []
    z_coords = []
    for start, end in edges:
        x_coords.extend([vertices[start][0], vertices[end][0], None])
        y_coords.extend([vertices[start][1], vertices[end][1], None])
        z_coords.extend([vertices[start][2], vertices[end][2], None])

    return go.Scatter3d(
        x=x_coords,
        y=y_coords,
        z=z_coords,
        mode="lines",
        name=name,
        line=dict(color=color, width=4),
        showlegend=True,
        hoverinfo="skip",
    )


def _pack_geometry_traces(
    placements: Sequence[Placement],
    color_offset: int = 0,
    label: str = "Item",
) -> List[go.BaseTraceType]:
    traces: List[go.BaseTraceType] = []
    for idx, placement in enumerate(placements):
        dx, dy, dz = placement.orientation
        color = _color_for_index(idx + color_offset)
        name = f"{label} {idx + 1}"
        traces.append(
            _mesh_from_prism(
                placement.x,
                placement.y,
                placement.z,
                dx,
                dy,
                dz,
                color=color,
                name=name,
            )
        )
        traces.append(
            _edge_trace(
                placement.x,
                placement.y,
                placement.z,
                dx,
                dy,
                dz,
                color="#1a202c",
                name=name,
            )
        )
    return traces


def carton_in_box_figure(
    package: Package,
    box: Box,
    result: CartonBoxPackingResult,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        _container_wireframe(box.length, box.width, box.height, name=box.name, color="#2d3748")
    )
    for trace in _pack_geometry_traces(result.placements, label=package.name):
        fig.add_trace(trace)

    fig.update_layout(
        title="Cartons inside Box",
        scene=dict(
            xaxis_title="Length (mm)",
            yaxis_title="Width (mm)",
            zaxis_title="Height (mm)",
            aspectmode="data",
            xaxis=dict(
                backgroundcolor="#f2f5fb",
                gridcolor="#cbd5e0",
                zerolinecolor="#a0aec0",
            ),
            yaxis=dict(
                backgroundcolor="#f2f5fb",
                gridcolor="#cbd5e0",
                zerolinecolor="#a0aec0",
            ),
            zaxis=dict(
                backgroundcolor="#f2f5fb",
                gridcolor="#cbd5e0",
                zerolinecolor="#a0aec0",
            ),
        ),
        paper_bgcolor="#f7f9fc",
        plot_bgcolor="#f7f9fc",
        legend=dict(
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="#cbd5e0",
            borderwidth=1,
        ),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def pallet_layout_figure(
    box: Box,
    pallet: Pallet,
    result: BoxPalletPackingResult,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        _container_wireframe(
            pallet.length,
            pallet.width,
            pallet.usable_height(),
            name=pallet.name,
            color="#2d3748",
        )
    )
    for trace in _pack_geometry_traces(result.placements, label=box.name):
        fig.add_trace(trace)

    fig.update_layout(
        title="Boxes on Pallet",
        scene=dict(
            xaxis_title="Length (mm)",
            yaxis_title="Width (mm)",
            zaxis_title="Height (mm)",
            aspectmode="data",
            xaxis=dict(
                backgroundcolor="#f2f5fb",
                gridcolor="#cbd5e0",
                zerolinecolor="#a0aec0",
            ),
            yaxis=dict(
                backgroundcolor="#f2f5fb",
                gridcolor="#cbd5e0",
                zerolinecolor="#a0aec0",
            ),
            zaxis=dict(
                backgroundcolor="#f2f5fb",
                gridcolor="#cbd5e0",
                zerolinecolor="#a0aec0",
            ),
        ),
        paper_bgcolor="#f7f9fc",
        plot_bgcolor="#f7f9fc",
        legend=dict(
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="#cbd5e0",
            borderwidth=1,
        ),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def save_figure_image(fig: go.Figure, output_path: str | Path, width: int = 900, height: int = 650) -> None:
    """
    Persist a figure to disk as a static PNG using Kaleido.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pio.write_image(fig, str(output_path), format="png", width=width, height=height, scale=2)

