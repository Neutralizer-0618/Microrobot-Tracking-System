from __future__ import annotations

from dataclasses import dataclass
from hashlib import md5
from typing import Optional

import numpy as np
import streamlit as st
from PIL import Image
from streamlit.elements.image import create_layout_config
from streamlit.elements.lib.image_utils import image_to_url
from streamlit_drawable_canvas import _component_func, _data_url_to_image, _resize_img


@dataclass
class CanvasResult:
    image_data: Optional[np.ndarray] = None
    json_data: Optional[dict] = None


def st_canvas(
    fill_color: str = "#eee",
    stroke_width: int = 20,
    stroke_color: str = "black",
    background_color: str = "",
    background_image: Image.Image | None = None,
    update_streamlit: bool = True,
    height: int = 400,
    width: int = 600,
    drawing_mode: str = "freedraw",
    initial_drawing: dict | None = None,
    display_toolbar: bool = True,
    point_display_radius: int = 3,
    key=None,
) -> CanvasResult:
    background_image_url = None
    if background_image is not None:
        resized = _resize_img(background_image, height, width)
        background_image_url = image_to_url(
            resized,
            create_layout_config(width=width),
            True,
            "RGB",
            "PNG",
            f"drawable-canvas-bg-{md5(resized.tobytes()).hexdigest()}-{key}",
        )
        background_image_url = st._config.get_option("server.baseUrlPath") + background_image_url
        background_color = ""

    initial_drawing = {"version": "4.4.0"} if initial_drawing is None else initial_drawing
    initial_drawing["background"] = background_color

    component_value = _component_func(
        fillColor=fill_color,
        strokeWidth=stroke_width,
        strokeColor=stroke_color,
        backgroundColor=background_color,
        backgroundImageURL=background_image_url,
        realtimeUpdateStreamlit=update_streamlit and (drawing_mode != "polygon"),
        canvasHeight=height,
        canvasWidth=width,
        drawingMode=drawing_mode,
        initialDrawing=initial_drawing,
        displayToolbar=display_toolbar,
        displayRadius=point_display_radius,
        key=key,
        default=None,
    )
    if component_value is None:
        return CanvasResult()

    return CanvasResult(
        image_data=np.asarray(_data_url_to_image(component_value["data"])),
        json_data=component_value["raw"],
    )
