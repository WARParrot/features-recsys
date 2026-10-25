import pandas as pd
from lightgbm import LGBMClassifier, early_stopping
from sklearn.metrics import roc_auc_score


def matrix(frame, columns, category_columns=()):
    missing = set(columns) - set(frame.columns)
    if missing:
        raise KeyError(f"frame is missing feature columns: {sorted(missing)}")
    result = frame.loc[:, columns].copy()
    for column in category_columns:
        if column in result.columns:
            result[column] = result[column].astype("category")
    return result


def _params(config=None, rounds=1000, random_state=17):
    params = {
        "n_estimators": rounds,
        "learning_rate": 0.04,
        "num_leaves": 31,
        "min_child_samples": 40,
        "colsample_bytree": 0.9,
        "subsample": 0.9,
        "subsample_freq": 1,
        "reg_lambda": 1.0,
        "random_state": random_state,
        "verbosity": -1,
    }
    if config:
        params.update(config)
    return params


def _validate_features(frame, columns, split_name):
    missing = set(columns) - set(frame.columns)
    if missing:
        raise KeyError(f"{split_name} frame is missing features: {sorted(missing)}")


def _validate_labels(frame, split_name):
    if "binary_label" not in frame.columns:
        raise KeyError(f"{split_name} frame is missing binary_label")
    if frame.empty:
        raise ValueError(f"{split_name} frame is empty")
    if frame["binary_label"].nunique() < 2:
        raise ValueError(f"{split_name} frame must contain both binary classes")


def fit_lgbm(train, valid, columns, category_columns=(), config=None, rounds=1000, random_state=17):
    _validate_labels(train, "train")
    _validate_labels(valid, "valid")
    _validate_features(train, columns, "train")
    _validate_features(valid, columns, "valid")
    categories = [column for column in category_columns if column in columns]
    params = _params(config, rounds, random_state)
    # The validation metric is the primary AUC below, regardless of model config.
    params["metric"] = "None"
    model = LGBMClassifier(**params)
    train_matrix = matrix(train, columns, categories)
    valid_matrix = matrix(valid, columns, categories)
    model.fit(
        train_matrix,
        train["binary_label"],
        eval_set=[(valid_matrix, valid["binary_label"])],
        eval_metric="auc",
        callbacks=[early_stopping(50, first_metric_only=True, verbose=False)],
        categorical_feature=categories or "auto",
    )
    return model


def fit_lgbm_fixed(frame, columns, category_columns=(), config=None, rounds=350, random_state=17):
    _validate_labels(frame, "train")
    _validate_features(frame, columns, "train")
    categories = [column for column in category_columns if column in columns]
    model = LGBMClassifier(**_params(config, rounds, random_state))
    model.fit(
        matrix(frame, columns, categories),
        frame["binary_label"],
        categorical_feature=categories or "auto",
    )
    return model


def predict_proba(model, frame, columns, category_columns=()):
    _validate_features(frame, columns, "prediction")
    return model.predict_proba(matrix(frame, columns, category_columns))[:, 1]


def auc(model, frame, columns, category_columns=()):
    _validate_labels(frame, "evaluation")
    _validate_features(frame, columns, "evaluation")
    return float(
        roc_auc_score(
            frame["binary_label"],
            predict_proba(model, frame, columns, category_columns),
        )
    )
