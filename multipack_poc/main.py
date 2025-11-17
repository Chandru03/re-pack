"""
Streamlit entrypoint for the Re-pack PoC.
"""

from __future__ import annotations

import json
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

import streamlit as st

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from multipack_poc.core.solver_box_to_pallet import BoxPalletPackingResult, pack_boxes_on_pallet
from multipack_poc.core.solver_carton_to_box import CartonBoxPackingResult, pack_cartons_into_box
from multipack_poc.models.box import Box
from multipack_poc.models.package import Package
from multipack_poc.models.pallet import Pallet
from multipack_poc.report.pdf_generator import generate_pdf_report
from multipack_poc.visualization import layout_plot

BASE_DIR = Path(__file__).resolve().parent
CONFIG_DIR = BASE_DIR / "config"


@st.cache_data
def load_json_config(filename: str) -> Dict[str, Any]:
    with open(CONFIG_DIR / filename, "r", encoding="utf-8") as file:
        return json.load(file)


def build_package_inputs() -> Package:
    st.subheader("Carton Details")
    col1, col2, col3, col4 = st.columns(4)
    length = col1.number_input(
        "Length (mm)", min_value=0.01, value=200.0, step=0.01, format="%.2f"
    )
    width = col2.number_input(
        "Width (mm)", min_value=0.01, value=150.0, step=0.01, format="%.2f"
    )
    height = col3.number_input(
        "Height (mm)", min_value=0.01, value=100.0, step=0.01, format="%.2f"
    )
    weight = col4.number_input("Weight (g)", min_value=0.01, value=500.0, step=0.01, format="%.2f")
    name = st.text_input("Item Name", value="Carton")
    axis_label = st.selectbox(
        "Lock thickness axis",
        options=[
            "Free rotation (carton - experimental)",
            "Keep length vertical",
            "Keep width vertical (sheet thickness)",
            "Keep height vertical",
        ],
        index=2,
        help="Select how the item may be oriented inside the box.",
    )
    axis_map = {
        "Free rotation (carton)": None,
        "Keep length vertical": "length",
        "Keep width vertical (sheet thickness)": "width",
        "Keep height vertical": "height",
        "Free rotation (carton - experimental)": None,
    }
    thickness_axis = axis_map.get(axis_label, None)
    return Package(
        length=float(length),
        width=float(width),
        height=float(height),
        weight=float(weight),
        name=name,
        thickness_axis=thickness_axis,
    )


def build_box_inputs(box_templates: Dict[str, Any]) -> Box:
    st.subheader("Shipping Box")
    template_options = {value["name"]: value for value in box_templates.values()}
    template_names = list(template_options.keys())
    selected_template = st.selectbox("Template", template_names, index=0)
    template = template_options[selected_template]
    col1, col2, col3, col4, col5 = st.columns(5)
    length = col1.number_input(
        "Inner Length (mm)", min_value=0.01, value=float(template["length"]), step=0.01, format="%.2f"
    )
    width = col2.number_input(
        "Inner Width (mm)", min_value=0.01, value=float(template["width"]), step=0.01, format="%.2f"
    )
    height = col3.number_input(
        "Inner Height (mm)", min_value=0.01, value=float(template["height"]), step=0.01, format="%.2f"
    )
    max_weight = col4.number_input(
        "Max Weight (g)", min_value=0.01, value=float(template["max_weight"]), step=0.01, format="%.2f"
    )
    tare_weight = col5.number_input("Tare Weight (g)", min_value=0.0, value=500.0, step=0.01, format="%.2f")
    name = st.text_input("Box Name", value=template["name"])
    return Box(
        length=float(length),
        width=float(width),
        height=float(height),
        max_weight=float(max_weight),
        tare_weight=float(tare_weight),
        name=name,
    )


def build_pallet_inputs(pallet_templates: Dict[str, Any]) -> Pallet:
    st.subheader("Pallet")
    template_options = {value["name"]: value for value in pallet_templates.values()}
    template_names = list(template_options.keys())
    selected_template = st.selectbox("Pallet Type", template_names, index=0)
    template = template_options[selected_template]
    col1, col2, col3 = st.columns(3)
    length = col1.number_input(
        "Length (mm)", min_value=0.01, value=float(template["length"]), step=0.01, format="%.2f"
    )
    width = col2.number_input(
        "Width (mm)", min_value=0.01, value=float(template["width"]), step=0.01, format="%.2f"
    )
    height = col3.number_input(
        "Deck Height (mm)", min_value=0.01, value=float(template["height"]), step=0.01, format="%.2f"
    )

    col4, col5 = st.columns(2)
    max_height = col4.number_input(
        "Max Stack Height (mm)",
        min_value=float(height + 0.01),
        value=float(template["max_height"]),
        step=0.01,
        format="%.2f",
    )
    max_weight = col5.number_input(
        "Max Weight (g)", min_value=0.01, value=float(template["max_weight"]), step=0.01, format="%.2f"
    )
    name = st.text_input("Pallet Name", value=template["name"])

    return Pallet(
        length=float(length),
        width=float(width),
        height=float(height),
        max_height=float(max_height),
        max_weight=float(max_weight),
        name=name,
    )


def main() -> None:
    st.set_page_config(page_title="Re-pack Optimiser", layout="wide")
    st.title("Re-pack Packing Optimisation (PoC)")
    st.write("Optimise carton packing into boxes and pallet stacking using Google OR-Tools.")

    pallet_templates = load_json_config("pallets.json")
    box_templates = load_json_config("boxes.json")

    with st.form("input_form"):
        package = build_package_inputs()
        box = build_box_inputs(box_templates)
        pallet = build_pallet_inputs(pallet_templates)

        time_limit = st.slider("Solver Time Limit (seconds)", min_value=1, max_value=20, value=5)
        submitted = st.form_submit_button("Run Optimisation", type="primary")

    if submitted:
        try:
            st.session_state.pop("results", None)
            carton_result = pack_cartons_into_box(package, box, time_limit_sec=float(time_limit))
            if carton_result.items_per_box <= 0:
                st.warning("Unable to place any cartons into the box with the provided constraints.")
                st.stop()

            filled_box_weight = carton_result.items_per_box * package.weight + box.tare_weight
            pallet_result = pack_boxes_on_pallet(
                box=box,
                pallet=pallet,
                items_per_box=carton_result.items_per_box,
                filled_box_weight=filled_box_weight,
                time_limit_sec=float(time_limit),
            )
            if pallet_result.layers_per_pallet <= 0:
                st.warning(f"Unable to stack boxes on the pallet: {pallet_result.solver_status}")
                st.info(
                    f"**Pallet constraints:**\n"
                    f"- Usable height: {pallet.usable_height():.2f}mm "
                    f"(Max height: {pallet.max_height:.2f}mm - Deck height: {pallet.height:.2f}mm)\n"
                    f"- Max weight: {pallet.max_weight:,.0f}g\n\n"
                    f"**Box dimensions:**\n"
                    f"- Height: {box.height:.2f}mm\n"
                    f"- Filled box weight: {filled_box_weight:,.0f}g\n\n"
                    f"**Issue:** "
                    + (
                        f"Box height ({box.height:.2f}mm) exceeds usable height ({pallet.usable_height():.2f}mm). "
                        f"Increase Max Stack Height or decrease Box Height."
                        if "height insufficient" in pallet_result.solver_status.lower()
                        else (
                            f"Filled box weight ({filled_box_weight:,.0f}g) exceeds pallet capacity ({pallet.max_weight:,.0f}g). "
                            f"Reduce box weight or increase pallet max weight."
                            if "weight" in pallet_result.solver_status.lower()
                            else pallet_result.solver_status
                        )
                    )
                )

            carton_fig = layout_plot.carton_in_box_figure(package, box, carton_result)
            pallet_fig = layout_plot.pallet_layout_figure(box, pallet, pallet_result)

            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir_path = Path(tmpdir)
                carton_image = tmpdir_path / "carton_layout.png"
                pallet_image = tmpdir_path / "pallet_layout.png"
                layout_plot.save_figure_image(carton_fig, carton_image)
                layout_plot.save_figure_image(pallet_fig, pallet_image)
                pdf_path = tmpdir_path / "pallet_scheme.pdf"
                generate_pdf_report(
                    pdf_path,
                    package=package,
                    box=box,
                    pallet=pallet,
                    carton_result=carton_result,
                    pallet_result=pallet_result,
                    layout_images=[carton_image, pallet_image],
                )
                pdf_bytes = pdf_path.read_bytes()

            st.session_state["results"] = {
                "package": package,
                "box": box,
                "pallet": pallet,
                "carton_result": carton_result,
                "pallet_result": pallet_result,
                "carton_fig": carton_fig,
                "pallet_fig": pallet_fig,
                "pdf_bytes": pdf_bytes,
            }
            st.success("Optimisation completed.")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Optimisation failed: {exc}")

    results = st.session_state.get("results")
    if results:
        package = results["package"]
        box = results["box"]
        pallet = results["pallet"]
        carton_result: CartonBoxPackingResult = results["carton_result"]
        pallet_result: BoxPalletPackingResult = results["pallet_result"]
        carton_fig = results["carton_fig"]
        pallet_fig = results["pallet_fig"]
        pdf_bytes: bytes = results["pdf_bytes"]

        st.markdown("### Optimisation Summary")
        summary_cols = st.columns(4)
        summary_cols[0].metric("Items per Box", carton_result.items_per_box)
        summary_cols[1].metric("Boxes per Layer", pallet_result.boxes_per_layer)
        summary_cols[2].metric("Layers per Pallet", pallet_result.layers_per_pallet)
        summary_cols[3].metric("Total Items per Pallet", pallet_result.total_items)

        st.markdown("#### Box Utilisation")
        st.write(
            f"- Volume Utilisation: **{carton_result.volume_utilisation_pct:.2f}%**  \n"
            f"- Footprint Utilisation: **{carton_result.footprint_utilisation_pct:.2f}%**  \n"
            f"- Weight Utilisation: **{carton_result.weight_utilisation_pct:.2f}%**  \n"
            f"- Thickness Axis: **{package.thickness_axis or 'Free'}**"
        )

        st.markdown("#### Pallet Utilisation")
        st.write(
            f"- Volume Utilisation: **{pallet_result.volume_utilisation_pct:.2f}%**  \n"
            f"- Footprint Utilisation: **{pallet_result.footprint_utilisation_pct:.2f}%**  \n"
            f"- Weight Utilisation: **{pallet_result.weight_utilisation_pct:.2f}%**"
        )

        st.plotly_chart(carton_fig, use_container_width=True)
        st.plotly_chart(pallet_fig, use_container_width=True)

        st.download_button(
            label="Download PDF Report",
            data=pdf_bytes,
            file_name="pallet_scheme.pdf",
            mime="application/pdf",
        )


if __name__ == "__main__":
    main()

