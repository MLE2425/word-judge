import os
import polars as pl

from typing import Callable


class CFG:
    project_dir: str = os.path.dirname(__file__)
    data_dir: str = os.path.join(os.path.dirname(__file__), "../data")
    source_ia: dict[
        str,
        bool | None,
    ] = {
        "argupt.*": True,
        "machine-.*": True,
        "train_drcat.*": True,
        "drcat_v3.*": None,
        ".*?_essays.*": None,
        "essay_forum_.*": False,
        "ivypanda.*": False,
        "mlm_real.*": False,
        "mlm_synthetic.*": True,
    }
    source_sep: dict[str, str] = {
        "argupt.*": ",",
        "machine-.*": ",",
        "train_drcat.*": ",",
        "drcat_v3.*": ",",
        ".*?_essays.*": ",",
        "essay_forum_.*": ",",
        "ivypanda.*": ",",
        "mlm_real.*": ",",
        "mlm_synthetic.*": ",",
    }
    add_source_to_data: bool = True
    features_map: dict[str, str] = {
        "text": "text",
        "TEXT": "text",
        "essay_text": "text",
        "discourse_text": "text",
        "Cleaned Essay": "text",
        "model": "model",
        "temperature": "temperature",
        "generated": "generated",
        "label": "generated",
    }

    extra_processing: dict[str, Callable] = {
        "iviypanda.*": lambda x: x.with_columns(
            pl.col("TEXT").map_elements(lambda s: s[:1000]).alias("TEXT")
        ),
        "mlm_synthetic.*": lambda x: x.group_by("essay_id").head(3),
    }
