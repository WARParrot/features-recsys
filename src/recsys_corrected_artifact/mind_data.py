"""Load and normalize the public Microsoft MIND dataset.

The adapter keeps the logged impression as ``query_id`` and expands each
``item-label`` token into one candidate row.  It deliberately keeps ``history``
as metadata instead of turning it into a model feature: the first comparison
must test point-in-time aggregate feature search, not a special history model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


_BEHAVIORS_COLUMNS = ["query_id", "user", "time_text", "history", "impressions"]
_NEWS_COLUMNS = [
    "item",
    "category",
    "subcategory",
    "title",
    "abstract",
    "url",
    "title_entities",
    "abstract_entities",
]


def _resolve_split_dir(root: Path, split: str) -> Path:
    """Resolve common extracted MIND directory layouts."""
    candidates = [
        root / split,
        root / split.lower(),
        root / f"MIND_{split}",
        root / f"MIND{split}",
        root / f"MINDsmall_{split}",
        root / f"MINDlarge_{split}",
    ]
    for candidate in candidates:
        if (candidate / "behaviors.tsv").is_file() and (candidate / "news.tsv").is_file():
            return candidate
    if (root / "behaviors.tsv").is_file() and (root / "news.tsv").is_file():
        return root
    names = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"could not find MIND split {split!r}; checked: {names}")


def _resolve_requested_splits(root: Path, splits) -> list[tuple[str, Path]]:
    """Resolve splits without reading a root-level single split more than once."""
    requested = list(splits)
    if not requested:
        raise ValueError("at least one MIND split is required")
    root_is_split = (root / "behaviors.tsv").is_file() and (root / "news.tsv").is_file()
    if root_is_split:
        inferred = next(
            (split for split in requested if split.lower() in root.name.lower()),
            requested[0],
        )
        return [(inferred, root)]
    return [(split, _resolve_split_dir(root, split)) for split in requested]


def _read_tsv(path: Path, names: list[str]) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=names,
        dtype=str,
        keep_default_na=False,
        quoting=3,
    )


def _parse_impression(token: str) -> tuple[str, int]:
    item, separator, label = token.rpartition("-")
    if not separator or not item or label not in {"0", "1"}:
        raise ValueError(f"invalid MIND impression token: {token!r}")
    return item, int(label)


def _category_codes(news: pd.DataFrame) -> dict[str, int]:
    categories = (
        news["category"].fillna("").astype(str).str.strip()
        + "::"
        + news["subcategory"].fillna("").astype(str).str.strip()
    )
    return {value: code for code, value in enumerate(sorted(categories.unique()))}


def _canonicalize_inputs(behaviors: pd.DataFrame, news: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    behavior_aliases = {
        "impression_id": "query_id",
        "user_id": "user",
        "time": "time_text",
    }
    news_aliases = {"news_id": "item"}
    behaviors = behaviors.rename(columns=behavior_aliases).copy()
    news = news.rename(columns=news_aliases).copy()
    required_behaviors = set(_BEHAVIORS_COLUMNS)
    required_news = {"item", "category", "subcategory"}
    if not required_behaviors.issubset(behaviors.columns):
        raise KeyError(f"behaviors is missing {sorted(required_behaviors - set(behaviors.columns))}")
    if not required_news.issubset(news.columns):
        raise KeyError(f"news is missing {sorted(required_news - set(news.columns))}")
    return behaviors, news


def explode_mind_impressions(
    behaviors: pd.DataFrame,
    news: pd.DataFrame,
    query_prefix: str | None = None,
) -> pd.DataFrame:
    """Expand MIND behavior rows into one causally rankable candidate row."""
    behaviors, news = _canonicalize_inputs(behaviors, news)
    news_meta = news.drop_duplicates("item").set_index("item")
    category_codes = _category_codes(news)
    rows: list[dict] = []
    event_id = 0
    for behavior in behaviors.itertuples(index=False):
        # Pin parsing to UTC so the same extracted dataset is reproducible
        # across hosts with different local timezone settings.
        timestamp = pd.Timestamp(behavior.time_text, tz="UTC")
        timestamp_seconds = int(timestamp.timestamp())
        for token in str(behavior.impressions).split():
            item, label = _parse_impression(token)
            if item in news_meta.index:
                category = f"{news_meta.at[item, 'category']}::{news_meta.at[item, 'subcategory']}"
                item_cat = category_codes[category]
            else:
                item_cat = -1
            rows.append(
                {
                    "event_id": event_id,
                    "query_id": (
                        f"{query_prefix}:{behavior.query_id}"
                        if query_prefix
                        else str(behavior.query_id)
                    ),
                    "source_split": getattr(behavior, "source_split", "unknown"),
                    "user": str(behavior.user),
                    "item": item,
                    "timestamp": timestamp_seconds,
                    "label": label,
                    "is_click": float(label),
                    "is_purchase": 0.0,
                    "binary_label": label,
                    "item_cat": int(item_cat),
                    "history": str(behavior.history),
                }
            )
            event_id += 1
    if not rows:
        raise ValueError("MIND behaviors contain no impression candidates")
    return (
        pd.DataFrame(rows)
        .sort_values(["timestamp", "event_id"], kind="mergesort")
        .reset_index(drop=True)
    )


def summarize_mind_metadata(frame: pd.DataFrame, splits=()):
    """Summarize metadata for the exact MIND frame used by an experiment."""
    unknown = frame["item_cat"] == -1
    return {
        "splits": list(splits),
        "impression_rows": int(len(frame)),
        "unknown_item_rows": int(unknown.sum()),
        "unknown_item_fraction": float(unknown.mean()) if len(frame) else 0.0,
        "unknown_item_ids": int(frame.loc[unknown, "item"].nunique()),
    }


def load_mind(root: str | Path, splits=("train", "dev"), strict=False) -> pd.DataFrame:
    """Load extracted MIND split directories into the artifact schema.

    ``root`` may contain ``train/`` and ``dev/`` directories, or be one split
    directory itself.  All requested behavior rows are concatenated before
    category encoding so item categories remain stable across splits.
    """
    root = Path(root)
    resolved_splits = _resolve_requested_splits(root, splits)
    behavior_frames = []
    news_frames = []
    for split, split_dir in resolved_splits:

        behaviors = _read_tsv(split_dir / "behaviors.tsv", _BEHAVIORS_COLUMNS)
        news = _read_tsv(split_dir / "news.tsv", _NEWS_COLUMNS)
        behavior_frames.append((split, behaviors))
        news_frames.append(news[["item", "category", "subcategory"]])

    prefixed_frames = [
        frame.assign(
            query_id=frame["query_id"].map(
                lambda value, split=split: f"{split}:{value}"
            ),
            source_split=split,
        )
        for split, frame in behavior_frames
    ]
    behaviors = pd.concat(prefixed_frames, ignore_index=True)
    news = pd.concat(news_frames, ignore_index=True).drop_duplicates("item")
    frame = explode_mind_impressions(behaviors, news)
    frame["event_id"] = np.arange(len(frame), dtype=np.int64)
    frame = frame.sort_values(["timestamp", "event_id"], kind="mergesort").reset_index(drop=True)
    frame.attrs["mind_metadata"] = summarize_mind_metadata(
        frame, splits=[split for split, _ in resolved_splits]
    )
    if strict and frame.attrs["mind_metadata"]["unknown_item_rows"]:
        raise ValueError(
            "MIND contains impression items absent from news.tsv: "
            f"{frame.attrs['mind_metadata']['unknown_item_rows']} rows"
        )
    return frame
