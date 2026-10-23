"""KION dataset adapter for causal temporal ranking experiments."""

from __future__ import annotations

from pathlib import Path
import zipfile

import numpy as np
import pandas as pd


def _read_interactions(path: Path) -> pd.DataFrame:
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as archive:
            names = [name for name in archive.namelist() if name.endswith("interactions.csv")]
            if len(names) != 1:
                raise ValueError("KION archive must contain exactly one interactions.csv")
            with archive.open(names[0]) as stream:
                return pd.read_csv(stream)
    return pd.read_csv(path)


def load_kion(
    interactions: str | Path,
    users: str | Path | None = None,
    items: str | Path | None = None,
    *,
    max_users: int | None = None,
    seed: int = 42,
    watch_threshold: float = 50.0,
) -> pd.DataFrame:
    """Load KION into the artifact schema, preserving complete selected histories.

    ``max_users`` samples users, not rows: every interaction of a selected user is
    retained.  The target is a high-watch event (watched_pct >= threshold).
    Metadata is reduced to stable categorical codes to keep the first experiment
    focused on temporal aggregate feature construction rather than text models.
    """
    frame = _read_interactions(Path(interactions))
    if frame.empty or frame.iloc[-1].isna().any():
        raise ValueError("KION interactions contain an incomplete final row")
    required = {"user_id", "item_id", "last_watch_dt", "watched_pct"}
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"KION interactions missing columns: {sorted(missing)}")
    frame = frame.rename(columns={"user_id": "user", "item_id": "item"}).copy()
    frame["user"] = pd.to_numeric(frame["user"], errors="raise").astype("int64")
    frame["item"] = pd.to_numeric(frame["item"], errors="raise").astype("int64")
    if max_users is not None:
        if max_users <= 0:
            raise ValueError("max_users must be positive")
        all_users = np.sort(frame["user"].unique())
        rng = np.random.default_rng(seed)
        chosen = set(rng.choice(all_users, size=min(max_users, len(all_users)), replace=False))
        frame = frame[frame["user"].isin(chosen)].copy()
    parsed_dates = pd.to_datetime(frame["last_watch_dt"], utc=True, errors="coerce")
    if parsed_dates.isna().any():
        raise ValueError("KION interactions contain invalid timestamps")
    frame["timestamp"] = parsed_dates.astype("int64") // 10**9
    frame["watched_pct"] = pd.to_numeric(frame["watched_pct"], errors="coerce").fillna(0.0)
    frame["binary_label"] = (frame["watched_pct"] >= watch_threshold).astype("int8")
    frame["label"] = frame["binary_label"]
    # KION has observed watch interactions, not impressions/clicks. Keep
    # feature names semantically tied to the high-watch outcome.
    frame["is_positive_watch"] = frame["binary_label"].astype("float32")
    frame["is_click"] = frame["is_positive_watch"]  # compatibility alias; not CTR
    frame["is_purchase"] = 0.0

    item_cat = pd.Series("unknown", index=frame.index, dtype="object")
    user_segment = pd.Series("unknown", index=frame.index, dtype="object")
    if items is not None:
        meta = pd.read_csv(items)
        if "item_id" not in meta:
            raise KeyError("KION items metadata must contain item_id")
        fields = [c for c in ("content_type", "genres") if c in meta]
        if fields:
            values = meta[fields].fillna("").astype(str).agg("::".join, axis=1)
            item_cat = frame["item"].map(pd.Series(values.to_numpy(), index=meta["item_id"]).to_dict()).fillna("unknown")
    if users is not None:
        meta = pd.read_csv(users)
        if "user_id" not in meta:
            raise KeyError("KION users metadata must contain user_id")
        fields = [c for c in ("age", "income", "sex", "kids_flg") if c in meta]
        if fields:
            values = meta[fields].fillna("").astype(str).agg("::".join, axis=1)
            user_segment = frame["user"].map(pd.Series(values.to_numpy(), index=meta["user_id"]).to_dict()).fillna("unknown")
    frame["item_cat"] = pd.Categorical(item_cat).codes.astype("int64")
    frame["user_segment"] = pd.Categorical(user_segment).codes.astype("int64")
    frame["event_id"] = np.arange(len(frame), dtype="int64")
    frame = frame.sort_values(["timestamp", "event_id"], kind="mergesort").reset_index(drop=True)
    frame.attrs["kion_metadata"] = {
        "users": int(frame["user"].nunique()),
        "items": int(frame["item"].nunique()),
        "events": int(len(frame)),
        "max_users": max_users,
        "seed": seed,
        "watch_threshold": watch_threshold,
        "target": "watched_pct >= threshold",
        "feedback_semantics": "observed_watch_interaction; is_click is a compatibility alias, not an impression CTR",
        "source": "https://github.com/irsafilo/KION_DATASET",
    }
    return frame
