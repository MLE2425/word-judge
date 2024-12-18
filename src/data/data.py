import logging
import os
import re
from typing import List

import polars as pl
from cfg import CFG


logger = logging.getLogger(__name__)


def load_and_merge_sources() -> pl.DataFrame:
    sources: List[pl.DataFrame] = []
    for csv in os.listdir(CFG.data_dir):
        key: str | None = next(
            iter([k for k in CFG.source_sep if re.match(rf"{k}", csv)]), None
        )
        if key is None:
            continue

        sep: str = CFG.source_sep[key]
        source: pl.DataFrame
        try:
            source = pl.read_csv(f"{CFG.data_dir}/{csv}", separator=sep)
        except Exception:
            logger.debug(f"Error reading {csv}")
            continue

        # Mapping features
        source = source.rename(
            {f: CFG.features_map[f] for f in CFG.features_map if f in source.columns}
        )

        if CFG.source_ia[key] is not None:
            source = source.with_columns(
                generated=pl.lit(1 if CFG.source_ia[key] else 0)
            )

        source = source.select(["text", "generated"])

        source = source.with_columns(generated=source["generated"].cast(pl.Int8))

        if CFG.add_source_to_data:
            source = source.with_columns(
                source=pl.lit(csv),
            )

        sources.append(source)
    combined_sources = pl.concat(sources) if sources else pl.DataFrame()
    return combined_sources
