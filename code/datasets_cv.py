"""Dataset registry and resampling design for the larger study.

Design (approved): every dataset is capped at CAP = 500 points (stratified subsample with a fixed seed, so the dataset
definition does not change between repetitions), then evaluated with repeated stratified 10-fold cross-validation,
2 repetitions = 20 resamples.  Preprocessing (standardise, PCA to at most 4 components, min-max to [0, pi]) is fitted
on the training part of each resample only.

Datasets
  offline (scikit-learn): moons, circles, breast_cancer, iris, wine, digits38, digits17, digits49
  download (OpenML by numeric id, with a shape check; a failed download or a shape mismatch skips the dataset):
    ionosphere, sonar, banknote, pima, blood, haberman, ilpd, spambase
  The shape check protects against a wrong id.  Downloads are cached in ./data_cache.  If OpenML is not reachable,
  put <name>.csv (numeric feature columns, last column = class label, with header) into a folder and pass --csv-dir.
  (OpenML 'wdbc' is not used: it is the same data as scikit-learn's breast cancer.)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

BASE_SEED = 2026
CAP = 500
OFFLINE = ["moons", "circles", "breast_cancer", "iris", "wine", "digits38", "digits17", "digits49"]
# name: (OpenML data id, expected n_samples, expected n_features)
OPENML = {
    "ionosphere": (59, 351, 34),
    "sonar": (40, 208, 60),
    "banknote": (1462, 1372, 4),
    "pima": (37, 768, 8),
    "blood": (1464, 748, 4),
    "haberman": (43, 306, 3),
    "ilpd": (1480, 583, 10),
    "spambase": (44, 4601, 57),
}
ALL_DATASETS = OFFLINE + list(OPENML)


class DatasetSkip(Exception):
    pass


def _lift(X2):
    x1, x2 = X2[:, 0], X2[:, 1]
    return np.column_stack([x1, x2, x1 * x2, x1 ** 2 - x2 ** 2])


def _digits_pair(a, b):
    X, y = datasets.load_digits(return_X_y=True)
    m = (y == a) | (y == b)
    return X[m], (y[m] == b).astype(int)


def _openml(name, cache, csv_dir):
    if csv_dir and (Path(csv_dir) / f"{name}.csv").exists():

        df = pd.read_csv(Path(csv_dir) / f"{name}.csv")
        return df.iloc[:, :-1].select_dtypes("number").to_numpy(float), LabelEncoder().fit_transform(df.iloc[:, -1])
    did, n_exp, d_exp = OPENML[name]
    try:
        d = datasets.fetch_openml(data_id=did, as_frame=True, data_home=cache, parser="auto")
    except Exception as e:  # network, id, parser
        raise DatasetSkip(f"download failed ({type(e).__name__}: {str(e)[:80]})")
    X = d.data.copy()
    for c in X.columns:  # nominal feature columns (e.g. gender, year coded as category) -> integer codes
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].astype("category").cat.codes.astype(float)
    if X.shape != (n_exp, d_exp):
        raise DatasetSkip(f"shape {X.shape} does not match expected {(n_exp, d_exp)} - wrong id? skipped")
    if X.isna().any().any():
        raise DatasetSkip("missing values - skipped")
    return X.to_numpy(float), LabelEncoder().fit_transform(d.target.astype(str))


def load_full(name, cap=CAP, cache="data_cache", csv_dir=None):
    """Return X (raw features), y (int labels), info dict.  Raises DatasetSkip."""
    if name == "moons":
        X, y = datasets.make_moons(n_samples=cap, noise=0.15, random_state=BASE_SEED)
        X = _lift(X)
    elif name == "circles":
        X, y = datasets.make_circles(n_samples=cap, noise=0.1, factor=0.5, random_state=BASE_SEED)
        X = _lift(X)
    elif name == "breast_cancer":
        X, y = datasets.load_breast_cancer(return_X_y=True)
    elif name == "iris":
        X, y = datasets.load_iris(return_X_y=True)
    elif name == "wine":
        X, y = datasets.load_wine(return_X_y=True)
    elif name == "digits38":
        X, y = _digits_pair(3, 8)
    elif name == "digits17":
        X, y = _digits_pair(1, 7)
    elif name == "digits49":
        X, y = _digits_pair(4, 9)
    elif name in OPENML:
        X, y = _openml(name, cache, csv_dir)
    else:
        raise DatasetSkip("unknown dataset name")
    n_full = len(y)
    if len(y) > cap:
        X, _, y, _ = train_test_split(X, y, train_size=cap, stratify=y, random_state=BASE_SEED)
    info = dict(dataset=name, n_full=n_full, n_used=len(y), n_features=X.shape[1], n_classes=int(len(np.unique(y))),
                majority_share=float(np.bincount(y).max() / len(y)))
    return np.asarray(X, float), np.asarray(y, int), info


def resamples(y, folds=10, repeats=2, seed=BASE_SEED):
    """List of (train_idx, test_idx): repeated stratified k-fold."""
    rskf = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=seed)
    return list(rskf.split(np.zeros(len(y)), y))


def preprocess(Xtr, Xte, n_max=4, seed=BASE_SEED, upper=np.pi):
    """Standardise, PCA to at most n_max components (only if more features), min-max to [0, upper]; train-fit only."""
    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    if Xtr.shape[1] > n_max:
        p = PCA(n_components=n_max, random_state=seed).fit(Xtr)
        Xtr, Xte = p.transform(Xtr), p.transform(Xte)
    mm = MinMaxScaler(feature_range=(0.0, upper)).fit(Xtr)
    return mm.transform(Xtr), mm.transform(Xte)
