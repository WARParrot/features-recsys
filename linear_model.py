from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import roc_auc_score


def _pipeline(frame, columns, category_columns, c, random_state=17):
    categories = [column for column in category_columns if column in columns]
    numeric = [column for column in columns if column not in categories]
    transforms = []
    if numeric:
        transforms.append(("numeric", StandardScaler(), numeric))
    if categories:
        transforms.append(("categorical", OneHotEncoder(handle_unknown="ignore"), categories))
    return make_pipeline(
        ColumnTransformer(transforms, remainder="drop"),
        LogisticRegression(C=c, max_iter=300, solver="liblinear", random_state=random_state),
    )


def _validate_features(frame, columns, split_name):
    missing = set(columns) - set(frame.columns)
    if missing:
        raise KeyError(f"{split_name} frame is missing features: {sorted(missing)}")


def _validate_labels(frame, split_name):
    if "binary_label" not in frame.columns:
        raise KeyError(f"{split_name} frame is missing binary_label")
    if frame.empty or frame["binary_label"].nunique() < 2:
        raise ValueError(f"{split_name} frame must contain both binary classes")


def tune_logistic(train, select, columns, category_columns=(), c_values=(0.03, 0.1, 0.3), random_state=17):
    _validate_labels(train, "train")
    _validate_labels(select, "select")
    _validate_features(train, columns, "train")
    _validate_features(select, columns, "select")
    trials = []
    for c in c_values:
        model = _pipeline(train, columns, category_columns, c, random_state=random_state)
        model.fit(train[columns], train["binary_label"])
        score = roc_auc_score(select["binary_label"], model.predict_proba(select[columns])[:, 1])
        trials.append((float(score), c))
    return max(trials, key=lambda trial: trial[0])


def fit_logistic(train, columns, category_columns=(), c=0.1, random_state=17):
    _validate_labels(train, "train")
    _validate_features(train, columns, "train")
    model = _pipeline(train, columns, category_columns, c, random_state=random_state)
    model.fit(train[columns], train["binary_label"])
    return model


def logistic_auc(model, frame, columns):
    _validate_labels(frame, "evaluation")
    _validate_features(frame, columns, "evaluation")
    return float(roc_auc_score(frame["binary_label"], model.predict_proba(frame[columns])[:, 1]))
