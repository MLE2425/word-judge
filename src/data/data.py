import os
import re
import logging;
logger = logging.getLogger(__name__)

import polars as pl
from cfg import CFG

def load_sources() -> pl.DataFrame:
    sources: list[pl.DataFrame] = []
    for csv in os.listdir(CFG.data_dir):
        key: str | None = next(iter([k for k in CFG.source_sep if re.match(rf"{k}", csv)]), None)
        if key is None:
            continue

        sep: str = CFG.source_sep[key]
        source: pl.DataFrame
        try:
            source = pl.read_csv(f"{CFG.data_dir}/{csv}", separator=sep)
        except:
            logger.debug(f"Error reading {csv}")
            continue

        # Mapping features
        source = source.select([
            f for f in CFG.source_features[key] if f in source.columns
        ])
        source = source.rename(
            *[
                {
                    f: CFG.features_map[f]
                } for f in CFG.features_map if f in source.columns
            ]
        )

        sources.append(source)
