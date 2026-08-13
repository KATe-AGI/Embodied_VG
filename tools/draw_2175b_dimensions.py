#!/usr/bin/env python3
"""Extract known 2175B dimensions from STEP geometry and draw a PNG sheet."""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

import cadquery as cq
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Circle


ROOT = Path(__file__).resolve().parents[1]
STEP = ROOT / "plug_model" / "2175B.stp"
OBJ = ROOT / "plug_model" / "2175B_grasp.obj"
OUTPUT = ROOT / "plug_model" / "2175B_all_known_dimensions.png"
FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
FONT = FontProperties(fname=FONT_PATH)

# Manufacturer catalogue values for the 32 A, 4-pole 2175B variant.
CATALOGUE = {"a": 186.0, "b": 94.0, "n": 145.0, "y": 22.0}


def read_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    with path.open("r", encoding="ascii", errors="ignore") as stream:
        for line in stream:
            if line.startswith("v "):
                vertices.append([float(value) * 1000.0 for value in line.split()[1:4]])
            elif line.startswith("f "):
                faces.append([int(value.split("/")[0]) - 1 for value in line.split()[1:4]])
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def projected_edges(vertices: np.ndarray, faces: np.ndarray, columns: tuple[int, int]) -> np.ndarray:
    edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), axis=0)
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    return vertices[edges][:, :, columns]


def dimension(ax, p0: tuple[float, float], p1: tuple[float, float], text: str, color: str = "#c2410c") -> None:
    ax.annotate("", xy=p1, xytext=p0, arrowprops={"arrowstyle": "<->", "lw": 1.25, "color": color})
    x = (p0[0] + p1[0]) * 0.5
    y = (p0[1] + p1[1]) * 0.5
    rotation = 90 if abs(p1[1] - p0[1]) > abs(p1[0] - p0[0]) else 0
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="bottom" if rotation == 0 else "center",
        rotation=rotation,
        fontsize=9,
        color=color,
        fontproperties=FONT,
        bbox={"fc": "white", "ec": "none", "alpha": 0.88, "pad": 1.2},
    )


def extract_geometry() -> tuple[cq.Shape, list[float], dict[float, list[tuple[float, float]]]]:
    shape = cq.importers.importStep(str(STEP)).val()
    bbox = shape.BoundingBox()
    tail_z = bbox.zmin

    plane_stations: set[float] = set()
    coaxial: dict[float, list[tuple[float, float]]] = defaultdict(list)
    for face in shape.Faces():
        surface = face._geomAdaptor()
        face_bbox = face.BoundingBox()
        if face.geomType() == "PLANE" and abs(surface.Axis().Direction().Z()) > 0.999999:
            plane_stations.add(round(face.Center().z - tail_z, 6))
        elif face.geomType() == "CYLINDER":
            direction = surface.Axis().Direction()
            location = surface.Location()
            if abs(direction.Z()) > 0.999999 and math.hypot(location.X(), location.Y()) < 1e-4:
                diameter = round(2.0 * surface.Radius(), 6)
                coaxial[diameter].append((face_bbox.zmin - tail_z, face_bbox.zmax - tail_z))
    return shape, sorted(plane_stations), dict(sorted(coaxial.items()))


def fmt(values: list[float], decimals: int = 1) -> str:
    return ", ".join(f"{value:.{decimals}f}" for value in values)


def main() -> None:
    shape, stations, coaxial = extract_geometry()
    bbox = shape.BoundingBox()
    vertices, faces = read_obj(OBJ)
    vertices[:, 0] -= vertices[:, 0].min()  # tail-referenced axial coordinate

    # Keep the model readable while avoiding a dense duplicate-edge cloud.
    side_edges = projected_edges(vertices, faces, (0, 1))
    front_edges = projected_edges(vertices, faces, (1, 2))

    plt.rcParams.update({"axes.unicode_minus": False, "font.family": "sans-serif"})
    fig = plt.figure(figsize=(24, 16), dpi=150, facecolor="#f8fafc")
    fig.text(0.045, 0.962, "MENNEKES 2175B 插头—已知尺寸汇总", fontsize=25, color="#0f172a", fontproperties=FONT)
    fig.text(
        0.047,
        0.933,
        "源模型：plug_model/2175B.stp  |  单位：mm  |  STEP AP214（2018-01-26）  |  图中几何值不含制造公差",
        fontsize=12,
        color="#475569",
        fontproperties=FONT,
    )

    ax = fig.add_axes([0.045, 0.515, 0.67, 0.37], facecolor="white")
    ax.add_collection(LineCollection(side_edges, colors="#64748b", linewidths=0.10, alpha=0.42, rasterized=True))
    ax.scatter(vertices[:, 0], vertices[:, 1], s=0.08, color="#cbd5e1", alpha=0.55, rasterized=True)
    ax.set_xlim(-8, 188)
    ax.set_ylim(-72, 65)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("距尾端的轴向距离 (mm)", fontsize=11, fontproperties=FONT)
    ax.set_ylabel("横向 (mm)", fontsize=11, fontproperties=FONT)
    ax.set_title("正交侧视图（STEP 实体）", fontsize=15, loc="left", fontproperties=FONT)
    ax.grid(True, color="#e2e8f0", linewidth=0.55)
    for spine in ax.spines.values():
        spine.set_color("#cbd5e1")

    # Exact macro stations read from axial planar faces. The chain is selected to
    # describe the major external sections, while the complete station list is below.
    macro = [0.0, 29.3, 47.3, 55.8, 105.8, 127.5, 148.5, 170.8, 180.3]
    for value in macro:
        ax.plot([value, value], [-55, 53], color="#0284c7", lw=0.6, ls=(0, (3, 3)), alpha=0.7)
    segments = list(zip(macro[:-1], macro[1:]))
    for index, (start, end) in enumerate(segments):
        y = -56.0 if index % 2 == 0 else -64.0
        dimension(ax, (start, y), (end, y), f"S{index + 1}  {end - start:.1f}", "#0369a1")
    dimension(ax, (0.0, 58.0), (180.3, 58.0), "STEP 总长 180.300", "#b91c1c")

    front = fig.add_axes([0.755, 0.555, 0.205, 0.29], facecolor="white")
    front.add_collection(LineCollection(front_edges, colors="#64748b", linewidths=0.10, alpha=0.40, rasterized=True))
    front.scatter(vertices[:, 1], vertices[:, 2], s=0.08, color="#cbd5e1", alpha=0.55, rasterized=True)
    front.set_xlim(-59, 59)
    front.set_ylim(-59, 59)
    front.set_aspect("equal", adjustable="box")
    front.set_title("端视包络", fontsize=15, fontproperties=FONT)
    front.grid(True, color="#e2e8f0", linewidth=0.55)
    dimension(front, (-47, -53), (47, -53), "94.000", "#b91c1c")
    dimension(front, (53, -47), (53, 47), "94.000", "#b91c1c")

    # These concentric analytic cylinders occur at different axial positions and
    # are superimposed in the end projection.  Leaders make that distinction more
    # useful than treating every circle as a feature on the same physical end face.
    ring_callouts = [
        (57.85, 135.0, (-54.0, 48.0), "#6d28d9"),
        (56.85, 150.0, (-54.0, 38.0), "#7c3aed"),
        (45.60, 210.0, (-54.0, -31.0), "#0369a1"),
        (41.70, 230.0, (-54.0, -42.0), "#0284c7"),
        (39.30, 300.0, (29.0, -43.0), "#0f766e"),
    ]
    for diameter_value, angle_deg, label_xy, color in ring_callouts:
        front.add_patch(
            Circle(
                (0.0, 0.0),
                diameter_value * 0.5,
                fill=False,
                edgecolor=color,
                linewidth=0.85,
                linestyle=(0, (4, 3)),
                alpha=0.72,
            )
        )
        angle = math.radians(angle_deg)
        radius = diameter_value * 0.5
        point = (radius * math.cos(angle), radius * math.sin(angle))
        front.annotate(
            f"Ø{diameter_value:.2f}",
            xy=point,
            xytext=label_xy,
            fontsize=7.5,
            color=color,
            fontproperties=FONT,
            ha="left",
            va="center",
            arrowprops={"arrowstyle": "-", "color": color, "lw": 0.8},
            bbox={"fc": "white", "ec": "none", "alpha": 0.84, "pad": 0.7},
        )

    # Bottom information cards.
    cards = [
        (0.045, 0.075, 0.29, 0.38, "STEP 实体几何值", "#0f766e"),
        (0.355, 0.075, 0.31, 0.38, "可解析旋转特征", "#7c3aed"),
        (0.685, 0.075, 0.275, 0.38, "厂商目录数据（32 A / 4 极）", "#c2410c"),
    ]
    for x, y, w, h, title, color in cards:
        panel = fig.add_axes([x, y, w, h], facecolor="white")
        panel.set_xticks([])
        panel.set_yticks([])
        for spine in panel.spines.values():
            spine.set_color("#e2e8f0")
        panel.text(0.04, 0.92, title, fontsize=15, color=color, transform=panel.transAxes, fontproperties=FONT)

    geo = fig.axes[-3]
    geo_lines = [
        f"包围尺寸：{bbox.xlen:.6f} × {bbox.ylen:.6f} × {bbox.zlen:.6f}",
        "即：最大横向 94.000042 × 94.000000",
        "轴向长度：180.300000",
        "",
        "主要轴向分段（尾→头）：",
        "S1   0.0–29.3       = 29.3",
        "S2  29.3–47.3      = 18.0",
        "S3  47.3–55.8       = 8.5",
        "S4  55.8–105.8     = 50.0",
        "S5 105.8–127.5     = 21.7",
        "S6 127.5–148.5     = 21.0",
        "S7 148.5–170.8     = 22.3",
        "S8 170.8–180.3      = 9.5",
        "",
        "全部轴向平面位置（距尾端）：",
    ]
    # Include every plane station, including fine local details, over wrapped lines.
    station_chunks = [stations[i : i + 8] for i in range(0, len(stations), 8)]
    geo_lines.extend(fmt(chunk, 1) for chunk in station_chunks)
    geo.text(0.04, 0.83, "\n".join(geo_lines), va="top", fontsize=8.8, linespacing=1.18, color="#334155", transform=geo.transAxes, fontproperties=FONT)

    analytic = fig.axes[-2]
    diameters = list(coaxial)
    analytic_lines = [
        "STEP 中与主轴同轴的解析圆柱面直径：",
        "",
        fmt(diameters[:6], 2),
        fmt(diameters[6:], 2),
        "",
        "直径集合（mm）：",
        "28.50, 39.30, 41.70, 43.00, 45.60,",
        "54.85, 55.45, 56.85, 57.25, 57.85, 60.50",
        "",
        "说明：集合同时包含外圆、内圆、槽底等面；",
        "STEP 无 PMI，因此不将它们武断命名为孔径",
        "或装配公差尺寸。",
        "",
        f"实体数：{len(shape.Solids())}  |  面：{len(shape.Faces())}  |  边：{len(shape.Edges())}",
    ]
    analytic.text(0.04, 0.83, "\n".join(analytic_lines), va="top", fontsize=10.5, linespacing=1.42, color="#334155", transform=analytic.transAxes, fontproperties=FONT)

    official = fig.axes[-1]
    official_lines = [
        "尺寸图标称：",
        f"a  总长                  {CATALOGUE['a']:.0f} mm",
        f"b  最大直径              {CATALOGUE['b']:.0f} mm",
        f"n  目录定位长度          {CATALOGUE['n']:.0f} mm",
        f"y  允许电缆外径上限      {CATALOGUE['y']:.0f} mm",
        "",
        "电缆/接线相关：",
        "导体截面积          2.5–6 mm²",
        "电缆外护套剥离长度  50 mm",
        "导体剥线长度        12–17 mm",
        "",
        "差异：目录 a=186 mm，本 STEP 实体=180.3 mm，",
        "相差 5.7 mm。选型用目录值，几何配准用 STEP 值。",
        "",
        "型号：2175B  |  400–440 V  |  3 h  |  IP67",
    ]
    official.text(0.04, 0.83, "\n".join(official_lines), va="top", fontsize=10.5, linespacing=1.42, color="#334155", transform=official.transAxes, fontproperties=FONT)

    fig.text(
        0.047,
        0.037,
        "注：“全部轴向平面位置”含局部肋、卡扣和小特征；本图不从哑实体猜测螺纹规格、配合等级和制造公差。",
        fontsize=10,
        color="#64748b",
        fontproperties=FONT,
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
