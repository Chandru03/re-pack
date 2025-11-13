"""
Simple CLI script to execute the Re-pack optimisation pipeline end-to-end.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from multipack_poc.core.solver_box_to_pallet import pack_boxes_on_pallet
from multipack_poc.core.solver_carton_to_box import pack_cartons_into_box
from multipack_poc.models.box import Box
from multipack_poc.models.package import Package
from multipack_poc.models.pallet import Pallet
from multipack_poc.report.pdf_generator import generate_pdf_report
from multipack_poc.visualization import layout_plot


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    config_dir = base_dir / "config"
    output_dir = base_dir / "artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)

    pallets_data = load_config(config_dir / "pallets.json")
    boxes_data = load_config(config_dir / "boxes.json")

    package = Package(
        length=200,
        width=5,
        height=300,
        weight=500,
        name="Sample Sheet",
        thickness_axis="width",
    )
    box_template = boxes_data["standard_box"]
    box = Box(
        length=box_template["length"],
        width=box_template["width"],
        height=box_template["height"],
        max_weight=box_template["max_weight"],
        tare_weight=500,
        name=box_template["name"],
    )
    pallet_template = pallets_data["euro_pallet"]
    pallet = Pallet.from_dict(pallet_template)

    carton_result = pack_cartons_into_box(package, box)
    if carton_result.items_per_box <= 0:
        raise RuntimeError("Failed to place cartons into the box with provided parameters.")

    filled_box_weight = carton_result.items_per_box * package.weight + box.tare_weight
    pallet_result = pack_boxes_on_pallet(
        box=box,
        pallet=pallet,
        items_per_box=carton_result.items_per_box,
        filled_box_weight=filled_box_weight,
    )

    carton_fig = layout_plot.carton_in_box_figure(package, box, carton_result)
    pallet_fig = layout_plot.pallet_layout_figure(box, pallet, pallet_result)

    carton_image_path = output_dir / "carton_layout.png"
    pallet_image_path = output_dir / "pallet_layout.png"
    layout_plot.save_figure_image(carton_fig, carton_image_path)
    layout_plot.save_figure_image(pallet_fig, pallet_image_path)

    pdf_path = output_dir / "pallet_scheme.pdf"
    generate_pdf_report(
        pdf_path,
        package=package,
        box=box,
        pallet=pallet,
        carton_result=carton_result,
        pallet_result=pallet_result,
        layout_images=[carton_image_path, pallet_image_path],
    )

    print("=== Re-pack Optimisation Summary ===")
    print(f"Items per Box: {carton_result.items_per_box}")
    print(f"Boxes per Layer: {pallet_result.boxes_per_layer}")
    print(f"Layers per Pallet: {pallet_result.layers_per_pallet}")
    print(f"Total Items per Pallet: {pallet_result.total_items}")
    print(f"Carton Volume Utilisation: {carton_result.volume_utilisation_pct:.2f}%")
    print(f"Pallet Volume Utilisation: {pallet_result.volume_utilisation_pct:.2f}%")
    print(f"Artifacts saved to: {output_dir}")


if __name__ == "__main__":
    main()

