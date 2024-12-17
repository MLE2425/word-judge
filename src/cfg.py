import logging

class CFG:
    data_dir: str = 'data/'
    source_ia: dict[bool | None, list[str]] = {
        True: ["argupt.*", "machine-.*", "train_drcat.*"],
        False: ["essay_forum_.*"],
        None: [".*?_essays.*"],
    }
    source_sep: dict[str, str] = {
        "argupt.*" : ",",
        "machine-.*": ",",
        "train_drcat.*": ",",
        ".*?_essays.*": ",",
        "essay_forum_.*": ",",
    }
    source_features: dict[str, list[str]] = {
        "argupt.*" : ["id","prompt_id","prompt","text","model","temperature","exam_type","score","score_level"],
        "machine-.*": ["id","prompt_id","prompt","text","model","temperature","exam_type","score","score_level"],
        "train_drcat.*": ["text", "label", "source", "fold"],
        ".*?_essays.*": ["id","prompt_id","text","generated"],
        "essay_forum_.*": ["Cleaned Essay", "Correct Grammar"]
    }
    add_source_to_data: bool = True
    features_map: dict[str, str] = {
        "text": "text",
        "Cleaned Essay": "text",
        "model": "model",
        "temperature": "temperature",
        "generated": "generated",
    }
