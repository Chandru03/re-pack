"""
PDF report generator using ReportLab.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from multipack_poc.core.solver_box_to_pallet import BoxPalletPackingResult
from multipack_poc.core.solver_carton_to_box import CartonBoxPackingResult
from multipack_poc.models.box import Box
from multipack_poc.models.package import Package
from multipack_poc.models.pallet import Pallet


def _build_table(data: Sequence[Sequence[str]], column_widths: Sequence[float]) -> Table:
    table = Table(data, colWidths=column_widths)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F1F1F1")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#333333")),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#DDDDDD")),
            ]
        )
    )
    return table


def _input_table(package: Package, box: Box, pallet: Pallet) -> Table:
    headers = ["Parameter", "Value"]
    rows = [
        ("Carton Name", package.name),
        ("Carton Dimensions (mm)", f"{package.length} x {package.width} x {package.height}"),
        ("Carton Thickness Axis", package.thickness_axis or "Free"),
        ("Carton Weight (g)", f"{package.weight:,}"),
        ("Box Name", box.name),
        ("Box Inner Dimensions (mm)", f"{box.length} x {box.width} x {box.height}"),
        ("Box Weight Limits (g)", f"{box.max_weight:,}"),
        ("Pallet Name", pallet.name),
        ("Pallet Footprint (mm)", f"{pallet.length} x {pallet.width}"),
        ("Pallet Height Limit (mm)", f"{pallet.usable_height()} usable / {pallet.max_height} total"),
        ("Pallet Weight Limit (g)", f"{pallet.max_weight:,}"),
    ]
    data = [headers] + [[left, right] for left, right in rows]
    return _build_table(data, column_widths=[70 * mm, 110 * mm])


def _metrics_table(carton_result: CartonBoxPackingResult, pallet_result: BoxPalletPackingResult) -> Table:
    headers = ["Metric", "Value"]
    rows = [
        ("Items per Box", carton_result.items_per_box),
        ("Box Volume Utilisation (%)", f"{carton_result.volume_utilisation_pct:.2f}"),
        ("Box Footprint Utilisation (%)", f"{carton_result.footprint_utilisation_pct:.2f}"),
        ("Boxes per Layer", pallet_result.boxes_per_layer),
        ("Layers per Pallet", pallet_result.layers_per_pallet),
        ("Total Boxes per Pallet", pallet_result.total_boxes),
        ("Total Items per Pallet", pallet_result.total_items),
        ("Pallet Volume Utilisation (%)", f"{pallet_result.volume_utilisation_pct:.2f}"),
        ("Pallet Footprint Utilisation (%)", f"{pallet_result.footprint_utilisation_pct:.2f}"),
        ("Solver Status (Box)", carton_result.solver_status),
        ("Solver Status (Pallet)", pallet_result.solver_status),
    ]
    data = [headers] + [[str(left), str(right)] for left, right in rows]
    return _build_table(data, column_widths=[80 * mm, 100 * mm])


def generate_pdf_report(
    output_path: str | Path,
    package: Package,
    box: Box,
    pallet: Pallet,
    carton_result: CartonBoxPackingResult,
    pallet_result: BoxPalletPackingResult,
    layout_images: Iterable[str | Path],
) -> Path:
    """
    Generate a palletisation PDF report and return the output path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=landscape(A4),
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        topMargin=15 * mm,
        bottomMargin=15 * mm,
        title="Multipack Optimisation Report",
    )

    styles = getSampleStyleSheet()
    title_style = styles["Title"]
    subtitle_style = ParagraphStyle(
        "Subtitle",
        parent=styles["Heading2"],
        textColor=colors.HexColor("#2D5B88"),
    )

    story: list = [
        Paragraph("Multipack Optimisation Report", title_style),
        Spacer(1, 8 * mm),
        Paragraph("Input Summary", subtitle_style),
        Spacer(1, 4 * mm),
        _input_table(package, box, pallet),
        Spacer(1, 6 * mm),
        Paragraph("Packing Performance", subtitle_style),
        Spacer(1, 4 * mm),
        _metrics_table(carton_result, pallet_result),
    ]

    for image_path in layout_images:
        image_path = Path(image_path)
        if image_path.exists():
            story.extend(
                [
                    Spacer(1, 6 * mm),
                    Paragraph(image_path.stem.replace("_", " ").title(), subtitle_style),
                    Spacer(1, 4 * mm),
                    Image(str(image_path), width=180 * mm, height=110 * mm),
                ]
            )

    doc.build(story)
    return output_path

