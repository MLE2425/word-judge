import os

class CFG:
    data_dir: str = os.path.join(os.path.dirname(__file__), "../data")
    source_ia: dict[list[str], bool | None,] = {
        "argupt.*" : True,
        "machine-.*": True,
        "train_drcat.*": True,
        ".*?_essays.*": None,
        "essay_forum_.*": False,
    }
    source_sep: dict[str, str] = {
        "argupt.*" : ",",
        "machine-.*": ",",
        "train_drcat.*": ",",
        ".*?_essays.*": ",",
        "essay_forum_.*": ",",
    }
    add_source_to_data: bool = True
    features_map: dict[str, str] = {
        "text": "text",
        "Cleaned Essay": "text",
        "model": "model",
        "temperature": "temperature",
        "generated": "generated",
    }
