import os


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
        # "ivypanda.*": False,
    }
    source_sep: dict[str, str] = {
        "argupt.*": ",",
        "machine-.*": ",",
        "train_drcat.*": ",",
        "drcat_v3.*": ",",
        ".*?_essays.*": ",",
        "essay_forum_.*": ",",
        # "ivypanda.*": ",",
    }
    add_source_to_data: bool = True
    features_map: dict[str, str] = {
        "text": "text",
        "TEXT": "text",
        "Cleaned Essay": "text",
        "model": "model",
        "temperature": "temperature",
        "generated": "generated",
        "label": "generated",
    }
