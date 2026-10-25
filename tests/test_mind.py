import pandas as pd
import pytest

from recsys_corrected_artifact.mind_data import explode_mind_impressions, load_mind, summarize_mind_metadata


def test_explode_mind_impressions_preserves_impression_groups_and_labels():
    behaviors = pd.DataFrame(
        {
            "impression_id": ["10", "11"],
            "user_id": ["U1", "U1"],
            "time": ["11/15/2019 10:22:32 AM", "11/15/2019 10:22:32 AM"],
            "history": ["N1 N2", "N2"],
            "impressions": ["N10-0 N11-1", "N12-0 N13-0"],
        }
    )
    news = pd.DataFrame(
        {
            "news_id": ["N10", "N11", "N12", "N13"],
            "category": ["sports", "sports", "business", "business"],
            "subcategory": ["golf", "golf", "markets", "markets"],
        }
    )

    frame = explode_mind_impressions(behaviors, news)

    assert list(frame["query_id"]) == ["10", "10", "11", "11"]
    assert list(frame["label"]) == [0, 1, 0, 0]
    assert frame.groupby("query_id").size().to_dict() == {"10": 2, "11": 2}
    assert frame["timestamp"].nunique() == 1
    assert set(frame["item_cat"]) == {0, 1}
    assert set(frame["history"]) == {"N1 N2", "N2"}




def test_mind_metadata_reports_unknown_items_for_loaded_frame():
    behaviors = pd.DataFrame(
        {
            "impression_id": ["10"],
            "user_id": ["U1"],
            "time": ["11/15/2019 10:22:32 AM"],
            "history": [""],
            "impressions": ["N1-1 N_missing-0"],
        }
    )
    news = pd.DataFrame(
        {
            "news_id": ["N1"],
            "category": ["sports"],
            "subcategory": ["golf"],
        }
    )
    frame = explode_mind_impressions(behaviors, news)
    metadata = summarize_mind_metadata(frame, splits=("train",))

    assert metadata["impression_rows"] == 2
    assert metadata["unknown_item_rows"] == 1
    assert metadata["unknown_item_fraction"] == 0.5
    assert metadata["unknown_item_ids"] == 1


def test_load_mind_strict_mode_rejects_unknown_items(tmp_path):
    split = tmp_path / "train"
    split.mkdir()
    (split / "news.tsv").write_text(
        "N1\tsports\tgolf\tT1\tA1\tU1\t{}\t{}\n",
        encoding="utf-8",
    )
    (split / "behaviors.tsv").write_text(
        "1\tU1\t11/15/2019 10:22:32 AM\t\tN_missing-0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="absent from news.tsv"):
        load_mind(tmp_path, splits=("train",), strict=True)


def test_load_mind_root_level_single_split_is_not_duplicated(tmp_path):
    (tmp_path / "news.tsv").write_text(
        "N1\tsports\tgolf\tT1\tA1\tU1\t{}\t{}\n",
        encoding="utf-8",
    )
    (tmp_path / "behaviors.tsv").write_text(
        "1\tU1\t11/15/2019 10:22:32 AM\t\tN1-1\n",
        encoding="utf-8",
    )

    frame = load_mind(tmp_path)

    assert len(frame) == 1
    assert frame["query_id"].tolist() == ["train:1"]
    assert frame["source_split"].tolist() == ["train"]
    assert frame.attrs["mind_metadata"]["splits"] == ["train"]


def test_load_mind_reads_train_and_dev_without_treating_history_as_candidates(tmp_path):
    train = tmp_path / "train"
    dev = tmp_path / "dev"
    train.mkdir()
    dev.mkdir()
    news_text = "N1\tsports\tgolf\tT1\tA1\tU1\t{}\t{}\nN2\tbusiness\tmarkets\tT2\tA2\tU2\t{}\t{}\n"
    (train / "news.tsv").write_text(news_text, encoding="utf-8")
    (dev / "news.tsv").write_text(news_text, encoding="utf-8")
    (train / "behaviors.tsv").write_text(
        "1\tU1\t11/15/2019 10:22:32 AM\tN1\tN1-1 N2-0\n",
        encoding="utf-8",
    )
    (dev / "behaviors.tsv").write_text(
        "1\tU1\t11/16/2019 10:22:32 AM\tN1\tN2-1 N1-0\n",
        encoding="utf-8",
    )

    frame = load_mind(tmp_path)

    assert len(frame) == 4
    assert frame["timestamp"].is_monotonic_increasing
    assert frame["query_id"].nunique() == 2
    assert "history" in frame.columns
    assert list(frame.loc[frame["query_id"].str.contains("dev:"), "item"]) == ["N2", "N1"]
    assert set(frame["binary_label"]) == {0, 1}
    assert "source_split" in frame.columns
    assert set(frame["source_split"]) == {"train", "dev"}
    assert frame.attrs["mind_metadata"]["splits"] == ["train", "dev"]
    assert frame.attrs["mind_metadata"]["impression_rows"] == 4
