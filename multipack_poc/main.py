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
    with st.expander("Item Details", expanded=True):
        name = st.text_input("Item Name", value="Carton", help="Name or identifier for the item")
        
        st.markdown("**Dimensions**")
        col1, col2, col3, col4 = st.columns(4)
        length = col1.number_input(
            "Length (mm)", 
            min_value=0.01, 
            value=200.0, 
            step=0.01, 
            format="%.2f",
            help="Length of the item in millimeters"
        )
        width = col2.number_input(
            "Width (mm)", 
            min_value=0.01, 
            value=150.0, 
            step=0.01, 
            format="%.2f",
            help="Width of the item in millimeters"
        )
        height = col3.number_input(
            "Height (mm)", 
            min_value=0.01, 
            value=100.0, 
            step=0.01, 
            format="%.2f",
            help="Height of the item in millimeters"
        )
        weight = col4.number_input(
            "Weight (g)", 
            min_value=0.01, 
            value=500.0, 
            step=0.01, 
            format="%.2f",
            help="Weight of a single item in grams"
        )
        
        # Calculate and display volume
        volume = length * width * height / 1_000_000  # Convert to liters
        st.caption(f"Item Volume: {volume:.3f} L | Weight: {weight:.2f} g")
        
        st.markdown("**Orientation Constraints**")
        axis_label = st.selectbox(
            "Item Orientation",
            options=[
                "Free rotation (experimental)",
                "Keep length vertical",
                "Keep width vertical (sheet thickness)",
                "Keep height vertical",
            ],
            index=2,
            help="Select how the item may be oriented inside the box. 'Free rotation' allows any orientation.",
        )
        axis_map = {
            "Free rotation (experimental)": None,
            "Keep length vertical": "length",
            "Keep width vertical (sheet thickness)": "width",
            "Keep height vertical": "height",
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
    with st.expander("Shipping Box", expanded=True):
        template_options = {value["name"]: value for value in box_templates.values()}
        template_names = list(template_options.keys())
        selected_template = st.selectbox(
            "Box Template", 
            template_names, 
            index=0,
            help="Select a predefined box template or customize dimensions below"
        )
        template = template_options[selected_template]
        
        name = st.text_input("Box Name", value=template["name"], help="Name or identifier for the box")
        
        st.markdown("**Inner Dimensions**")
        col1, col2, col3 = st.columns(3)
        length = col1.number_input(
            "Inner Length (mm)", 
            min_value=0.01, 
            value=float(template["length"]), 
            step=0.01, 
            format="%.2f",
            help="Internal length of the box"
        )
        width = col2.number_input(
            "Inner Width (mm)", 
            min_value=0.01, 
            value=float(template["width"]), 
            step=0.01, 
            format="%.2f",
            help="Internal width of the box"
        )
        height = col3.number_input(
            "Inner Height (mm)", 
            min_value=0.01, 
            value=float(template["height"]), 
            step=0.01, 
            format="%.2f",
            help="Internal height of the box"
        )
        
        # Calculate and display box volume
        box_volume = length * width * height / 1_000_000  # Convert to liters
        st.caption(f"Box Volume: {box_volume:.3f} L")
        
        st.markdown("**Weight Constraints**")
        col4, col5 = st.columns(2)
        max_weight = col4.number_input(
            "Max Weight (g)", 
            min_value=0.01, 
            value=float(template["max_weight"]), 
            step=0.01, 
            format="%.2f",
            help="Maximum total weight the box can hold (including items and box weight)"
        )
        tare_weight = col5.number_input(
            "Tare Weight (g)", 
            min_value=0.0, 
            value=500.0, 
            step=0.01, 
            format="%.2f",
            help="Weight of the empty box"
        )
    
    return Box(
        length=float(length),
        width=float(width),
        height=float(height),
        max_weight=float(max_weight),
        tare_weight=float(tare_weight),
        name=name,
    )


def build_pallet_inputs(pallet_templates: Dict[str, Any]) -> Pallet:
    with st.expander("Pallet Configuration", expanded=True):
        template_options = {value["name"]: value for value in pallet_templates.values()}
        template_names = list(template_options.keys())
        
        # Initialize session state for pallet dimensions on first run
        if "selected_pallet_type" not in st.session_state:
            st.session_state.selected_pallet_type = template_names[0]
            template = template_options[template_names[0]]
            st.session_state.pallet_length = float(template["length"])
            st.session_state.pallet_width = float(template["width"])
            st.session_state.pallet_height = float(template["height"])
            st.session_state.pallet_max_height = float(template["max_height"])
            st.session_state.pallet_max_weight = float(template["max_weight"])
            st.session_state.pallet_name = template["name"]
        
        # Get current template index
        current_index = template_names.index(st.session_state.selected_pallet_type) if st.session_state.selected_pallet_type in template_names else 0
        
        # Selectbox - update session state when type changes
        selected_template_name = st.selectbox(
            "Pallet Type", 
            template_names, 
            index=current_index,
            key="pallet_type_selectbox",
            help="Select a standard pallet size. Dimensions will auto-populate but can be customized."
        )
        
        # Get the selected template
        template = template_options[selected_template_name]
        
        # Check if template changed - if so, update session state with template values
        if selected_template_name != st.session_state.selected_pallet_type:
            st.session_state.selected_pallet_type = selected_template_name
            st.session_state.pallet_length = float(template["length"])
            st.session_state.pallet_width = float(template["width"])
            # Keep height custom - don't update from template (user can set it manually)
            st.session_state.pallet_max_height = float(template["max_height"])
            st.session_state.pallet_max_weight = float(template["max_weight"])
            st.session_state.pallet_name = template["name"]
        
        # Use session state values (which are updated when template changes)
        # Include template name in keys to force widget recreation when template changes
        key_suffix = f"_{selected_template_name}"
        
        name = st.text_input(
            "Pallet Name", 
            value=st.session_state.pallet_name, 
            key=f"pallet_name_input{key_suffix}",
            help="Name or identifier for the pallet"
        )
        
        st.markdown("**Pallet Dimensions**")
        col1, col2, col3 = st.columns(3)
        length = col1.number_input(
            "Length (mm)", 
            min_value=0.01, 
            value=st.session_state.pallet_length, 
            step=0.01, 
            format="%.2f",
            key=f"pallet_length_input{key_suffix}",
            help="Length of the pallet deck"
        )
        width = col2.number_input(
            "Width (mm)", 
            min_value=0.01, 
            value=st.session_state.pallet_width, 
            step=0.01, 
            format="%.2f",
            key=f"pallet_width_input{key_suffix}",
            help="Width of the pallet deck"
        )
        height = col3.number_input(
            "Deck Height (mm)", 
            min_value=0.01, 
            value=st.session_state.pallet_height, 
            step=0.01, 
            format="%.2f",
            key="pallet_height_input",
            help="Height of the pallet deck (customizable, not affected by template)"
        )
        
        # Calculate and display pallet footprint
        footprint = length * width / 1_000_000  # Convert to m²
        st.caption(f"Pallet Footprint: {footprint:.3f} m²")
        
        st.markdown("**Stacking Constraints**")
        col4, col5 = st.columns(2)
        max_height = col4.number_input(
            "Max Stack Height (mm)",
            min_value=float(height + 0.01),
            value=st.session_state.pallet_max_height,
            step=0.01,
            format="%.2f",
            key=f"pallet_max_height_input{key_suffix}",
            help="Maximum total height including deck (usable height = max height - deck height)"
        )
        max_weight = col5.number_input(
            "Max Weight (g)", 
            min_value=0.01, 
            value=st.session_state.pallet_max_weight, 
            step=0.01, 
            format="%.2f",
            key=f"pallet_max_weight_input{key_suffix}",
            help="Maximum total weight the pallet can hold"
        )
        
        # Display usable height
        usable_height = max_height - height
        st.caption(f"Usable Stack Height: {usable_height:.2f} mm")

        # Update session state with current values (for persistence)
        st.session_state.pallet_length = float(length)
        st.session_state.pallet_width = float(width)
        st.session_state.pallet_height = float(height)
        st.session_state.pallet_max_height = float(max_height)
        st.session_state.pallet_max_weight = float(max_weight)
        st.session_state.pallet_name = name

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
    st.title("Re-pack Packing Optimisation")
    
    st.divider()

    pallet_templates = load_json_config("pallets.json")
    box_templates = load_json_config("boxes.json")

    # Build all inputs in a form
    with st.form("input_form"):
        # Build pallet inputs first (outside form was causing issues, now inside with proper structure)
        pallet = build_pallet_inputs(pallet_templates)
        st.divider()
        package = build_package_inputs()
        st.divider()
        box = build_box_inputs(box_templates)
        st.divider()
        
        st.markdown("**Solver Settings**")
        time_limit = st.slider(
            "Solver Time Limit (seconds)", 
            min_value=1, 
            max_value=20, 
            value=5,
            help="Maximum time allowed for the optimization solver to find a solution"
        )
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            submitted = st.form_submit_button("Run Optimisation", type="primary", use_container_width=True)

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

        st.divider()
        st.markdown("## Optimisation Results")
        
        # Key metrics in cards
        summary_cols = st.columns(4)
        summary_cols[0].metric(
            "Items per Box", 
            carton_result.items_per_box,
            help="Number of items that fit in a single box"
        )
        summary_cols[1].metric(
            "Boxes per Layer", 
            pallet_result.boxes_per_layer,
            help="Number of boxes that fit on one pallet layer"
        )
        summary_cols[2].metric(
            "Layers per Pallet", 
            pallet_result.layers_per_pallet,
            help="Number of layers that can be stacked on the pallet"
        )
        summary_cols[3].metric(
            "Total Items per Pallet", 
            pallet_result.total_items,
            help="Total number of items that fit on one complete pallet"
        )
        
        # Utilisation metrics in expandable sections
        col1, col2 = st.columns(2)
        
        with col1:
            with st.expander("Box Utilisation", expanded=True):
                st.progress(carton_result.volume_utilisation_pct / 100, text=f"Volume: {carton_result.volume_utilisation_pct:.2f}%")
                st.progress(carton_result.footprint_utilisation_pct / 100, text=f"Footprint: {carton_result.footprint_utilisation_pct:.2f}%")
                st.progress(carton_result.weight_utilisation_pct / 100, text=f"Weight: {carton_result.weight_utilisation_pct:.2f}%")
                st.caption(f"**Orientation:** {package.thickness_axis or 'Free rotation'}")
        
        with col2:
            with st.expander("Pallet Utilisation", expanded=True):
                st.progress(pallet_result.volume_utilisation_pct / 100, text=f"Volume: {pallet_result.volume_utilisation_pct:.2f}%")
                st.progress(pallet_result.footprint_utilisation_pct / 100, text=f"Footprint: {pallet_result.footprint_utilisation_pct:.2f}%")
                st.progress(pallet_result.weight_utilisation_pct / 100, text=f"Weight: {pallet_result.weight_utilisation_pct:.2f}%")

        st.divider()
        st.markdown("## Visualizations")
        
        st.markdown("### Box Layout")
        st.plotly_chart(carton_fig, use_container_width=True)
        
        st.markdown("### Pallet Layout")
        st.plotly_chart(pallet_fig, use_container_width=True)

        st.divider()
        st.markdown("## Report")
        st.download_button(
            label="Download PDF Report",
            data=pdf_bytes,
            file_name="pallet_scheme.pdf",
            mime="application/pdf",
            use_container_width=True,
            help="Download a comprehensive PDF report with all optimization details and visualizations"
        )


if __name__ == "__main__":
    main()

