import pytest

from recsys_corrected_artifact.experiment_contracts import validate_result, validate_run_collection


def test_result_contract_requires_split_metadata():
    with pytest.raises(ValueError, match="missing keys"):
        validate_result({"seed": 1, "selected": "manual"})


def test_result_collection_accepts_rolling_fold_results():
    result = {
        "seed": 1,
        "folds": [
            {"seed": 1, "selected": "manual", "split_metadata": {"test": {"rows": 1}}}
        ],
    }
    validate_run_collection([result])
