"""
train_test_split.py

Road-aware train/test split and model training / evaluation.

  make_train_test_split()    split the labeled pixels so that every road goes
                             wholly to train OR test (never both); handles
                             class balancing (see `balance`)
  train_random_forest()      split + Random Forest + feature ranking
  train_best_model()         random search over several models, grouped CV,
                             best model saved to disk
  train_svm_top_features()   SVM-only random search (not used by app.py)

All models use the EXACT SAME split created by make_train_test_split().

`balance`:
  "weights"          keep all pixels, use class weights
  "downsample_train" split first, then downsample TRAIN only (test keeps its
                     natural class mix)
  "downsample_all"   downsample everything before the split (test is
                     balanced too)
  "none"             keep all pixels, no balancing
"""

from typing import Optional, Sequence

import joblib
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


# ===========================================================================
# 1. Road-aware train/test split
# ===========================================================================
def make_train_test_split(
    df: pd.DataFrame,
    class_col: str = "class_label",
    road_col: str = "road_id",
    test_size: float = 0.3,
    random_state: int = 42,
    block_size: int = 100,
    max_per_class: Optional[int] = None,
    balance: str = "weights",
    road_only: bool = True,
) -> dict:
    """
    Returns a dict with:
        train_df, test_df   both contain a "split_group" column
        class_counts        class counts of the input table
        n_groups            number of split groups
    """

    # ------------------------------------------------------------
    # Class handling (controlled by `balance`)
    # ------------------------------------------------------------
    valid_modes = ("weights", "downsample_train", "downsample_all", "none")
    if balance not in valid_modes:
        raise ValueError(f"balance must be one of {valid_modes}, got '{balance}'")

    class_counts = df[class_col].value_counts()

    print(f"Class counts before balancing (mode = {balance}):")
    print(class_counts)

    if balance == "downsample_all":
        n_per_class = int(class_counts.min())
        balanced = pd.concat(
            [
                grp.sample(n=n_per_class, random_state=random_state)
                for _, grp in df.groupby(class_col)
            ]
        ).reset_index(drop=True)
        print(f"Downsampled all classes to n={n_per_class} pixels each.")
    elif max_per_class is not None:
        balanced = pd.concat(
            [
                grp.sample(
                    n=min(len(grp), max_per_class),
                    random_state=random_state,
                )
                for _, grp in df.groupby(class_col)
            ]
        ).reset_index(drop=True)
        print(f"Capped every class at {max_per_class} pixels.")
    else:
        balanced = df.reset_index(drop=True)

    # ------------------------------------------------------------
    # Spatial blocks
    # ------------------------------------------------------------
    balanced = balanced.copy()

    balanced["block_x"] = np.floor(balanced["x"] / block_size).astype(int)
    balanced["block_y"] = np.floor(balanced["y"] / block_size).astype(int)

    balanced["spatial_block"] = (
        balanced["block_x"].astype(str) + "_" + balanced["block_y"].astype(str)
    )

    print(f"\nNumber of spatial blocks: {balanced['spatial_block'].nunique()}")
    print(f"Number of roads: {balanced[road_col].nunique()}")

    # ------------------------------------------------------------
    # Build split groups: roads (and blocks linked through roads)
    # ------------------------------------------------------------
    if road_only:
        balanced["split_group"] = balanced[road_col].astype(str)
    else:
        parent = {}

        def find(a):
            parent.setdefault(a, a)
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        pairs = balanced[[road_col, "spatial_block"]].drop_duplicates()
        for r, b in pairs.itertuples(index=False):
            union(("road", r), ("block", b))

        road_to_group = {
            r: str(find(("road", r))) for r in balanced[road_col].unique()
        }
        balanced["split_group"] = balanced[road_col].map(road_to_group)

    group_sizes = balanced["split_group"].value_counts()

    print(f"\nNumber of split groups: {len(group_sizes)}")
    print(f"Largest group holds {group_sizes.iloc[0] / len(balanced):.1%} of pixels")

    # ------------------------------------------------------------
    # Group train/test split (fill test until ~test_size of pixels)
    # ------------------------------------------------------------
    rng = np.random.RandomState(random_state)
    shuffled = rng.permutation(group_sizes.index.to_numpy())

    target = test_size * len(balanced)
    test_groups, running = set(), 0
    for g in shuffled:
        if running >= target:
            break
        test_groups.add(g)
        running += group_sizes[g]

    test_mask = balanced["split_group"].isin(test_groups)
    train_mask = ~test_mask

    train_df = balanced.loc[train_mask].copy()
    test_df = balanced.loc[test_mask].copy()

    if balance == "downsample_train":
        n_min = int(train_df[class_col].value_counts().min())
        train_df = (
            train_df.groupby(class_col)
            .sample(n=n_min, random_state=random_state)
            .copy()
        )
        print(f"\nDownsampled TRAIN only to n={n_min} pixels per class.")

    # Sanity check: no road on both sides
    overlap = set(train_df[road_col]) & set(test_df[road_col])
    assert not overlap, f"Roads leaked across split: {sorted(overlap)[:5]}"

    # ------------------------------------------------------------
    # Check all classes
    # ------------------------------------------------------------
    all_classes = set(balanced[class_col].unique())
    missing_train = all_classes - set(train_df[class_col].unique())
    missing_test = all_classes - set(test_df[class_col].unique())

    if missing_train:
        raise ValueError(
            f"Some classes are missing from training set: {sorted(missing_train)}"
        )

    if missing_test:
        raise ValueError(
            f"Some classes are missing from test set: {sorted(missing_test)}"
        )

    print(
        f"\nSpatial + road-aware split:"
        f"\n  Train pixels: {len(train_df)}"
        f"\n  Test pixels:  {len(test_df)}"
        f"\n  Train roads:  {train_df[road_col].nunique()}"
        f"\n  Test roads:   {test_df[road_col].nunique()}"
        f"\n  Train blocks: {train_df['spatial_block'].nunique()}"
        f"\n  Test blocks:  {test_df['spatial_block'].nunique()}"
        f"\n  Train groups: {train_df['split_group'].nunique()}"
        f"\n  Test groups:  {test_df['split_group'].nunique()}"
    )

    print("\nTrain class distribution:")
    print(train_df[class_col].value_counts())

    print("\nTest class distribution:")
    print(test_df[class_col].value_counts())

    return {
        "train_df": train_df,
        "test_df": test_df,
        "class_counts": class_counts,
        "n_groups": len(group_sizes),
    }


# ===========================================================================
# 2. Random Forest
# ===========================================================================
def create_train_test_dict(
    df: pd.DataFrame,
    class_col: str = "class_label",
    road_col: str = "road_id",
    feature_cols: Optional[Sequence[str]] = None,
    band_prefix: str = "B",
    test_size: float = 0.3,
    n_estimators: int = 300,
    random_state: int = 42,
    block_size: int = 100,
    max_per_class: Optional[int] = None,  # optional cap for huge classes (None = use all)
    balance: str = "weights",  # "weights" | "downsample_train" | "downsample_all" | "none"
    road_only: bool = True,  # True = group by road only, ignore blocks
) -> dict:

    # ------------------------------------------------------------
    # Detect features
    # ------------------------------------------------------------
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c.startswith(band_prefix)]

    if not feature_cols:
        raise ValueError("No feature columns found.")

    # ------------------------------------------------------------
    # Check required columns
    # ------------------------------------------------------------
    required_cols = ["x", "y", class_col, road_col]

    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # ------------------------------------------------------------
    # Split
    # ------------------------------------------------------------
    split = make_train_test_split(
        df,
        class_col=class_col,
        road_col=road_col,
        test_size=test_size,
        random_state=random_state,
        block_size=block_size,
        max_per_class=max_per_class,
        balance=balance,
        road_only=road_only,
    )

    train_df = split["train_df"]
    test_df = split["test_df"]
    class_counts = split["class_counts"]

    return {
        "balance": balance,
        "classes": sorted(class_counts.index.tolist()),
        "train_df": train_df,
        "test_df": test_df,
        "n_train": len(train_df),
        "n_test": len(test_df),
        "n_train_blocks": train_df["spatial_block"].nunique(),
        "n_test_blocks": test_df["spatial_block"].nunique(),
        "n_train_roads": train_df[road_col].nunique(),
        "n_test_roads": test_df[road_col].nunique(),
        "n_groups": split["n_groups"],
        "road_col": road_col,
        "road_only": road_only,
        "block_size": block_size,
    }


# ===========================================================================
# 3. Compare several models with random search, keep the best
# ===========================================================================
class LabelledModel:
    """Wraps a fitted model so predict() returns the original class names."""

    def __init__(self, model, encoder):
        self.model = model
        self.encoder = encoder
        self.classes_ = encoder.classes_

    def predict(self, X):
        return self.encoder.inverse_transform(self.model.predict(X))

    def predict_proba(self, X):
        return self.model.predict_proba(X)


def _model_specs(random_state: int, n_features: int) -> dict:
    """Each entry: (estimator, search space, needs_scaling)."""
    # For standardized data, sklearn's default gamma is 1 / n_features.
    # Search around that value instead of a fixed range, otherwise many
    # draws are far too large and the SVM predicts a single class.
    g0 = 1.0 / n_features

    specs = {
        "svm": (
            SVC(probability=False, random_state=random_state),
            {
                "clf__C": loguniform(1e-2, 1e3),
                "clf__gamma": loguniform(g0 / 30, g0 * 30),
                "clf__kernel": ["rbf"],
            },
            True,
        ),
        "rf": (
            RandomForestClassifier(random_state=random_state, n_jobs=1),
            {
                "clf__n_estimators": randint(100, 500),
                "clf__max_depth": [None, 10, 20, 30],
                "clf__min_samples_leaf": randint(1, 10),
                "clf__max_features": ["sqrt", "log2", 0.5],
            },
            False,
        ),
        "extra_trees": (
            ExtraTreesClassifier(random_state=random_state, n_jobs=1),
            {
                "clf__n_estimators": randint(100, 500),
                "clf__max_depth": [None, 10, 20, 30],
                "clf__min_samples_leaf": randint(1, 10),
                "clf__max_features": ["sqrt", "log2", 0.5],
            },
            False,
        ),
        "hist_gb": (
            HistGradientBoostingClassifier(random_state=random_state),
            {
                "clf__learning_rate": loguniform(0.01, 0.3),
                "clf__max_depth": [None, 3, 5, 8],
                "clf__max_leaf_nodes": randint(8, 64),
                "clf__l2_regularization": loguniform(1e-4, 10),
            },
            False,
        ),
        "logreg": (
            LogisticRegression(max_iter=3000, random_state=random_state),
            {"clf__C": loguniform(1e-3, 1e2)},
            True,
        ),
        "knn": (
            KNeighborsClassifier(n_jobs=1),
            {
                "clf__n_neighbors": randint(3, 31),
                "clf__weights": ["uniform", "distance"],
            },
            True,
        ),
    }

    if HAS_XGB:
        specs["xgboost"] = (
            XGBClassifier(
                eval_metric="mlogloss",
                random_state=random_state,
                n_jobs=1,
                tree_method="hist",
            ),
            {
                "clf__n_estimators": randint(100, 600),
                "clf__max_depth": randint(3, 10),
                "clf__learning_rate": loguniform(0.01, 0.3),
                "clf__subsample": uniform(0.6, 0.4),
                "clf__colsample_bytree": uniform(0.6, 0.4),
                "clf__min_child_weight": randint(1, 10),
            },
            False,
        )

    return specs


def train_best_model(
    train_test_splitted_data: dict,
    class_col: str = "class_label",
    group_col: str = "split_group",  # road-aware groups made by the split
    models: Sequence[str] = (
        "svm", "rf", "extra_trees", "xgboost", "hist_gb", "logreg", "knn"
    ),
    n_iter: int = 30,
    cv_folds: int = 5,
    scoring: str = "f1_macro",
    random_state: int = 42,
    n_jobs: int = -1,
    save_path: Optional[str] = "best_model.pkl",  # None = do not save
) -> dict:

    # Select the band columns
    selected_features = [
        col for col in train_test_splitted_data["train_df"].columns
        if col.startswith("B") and col[1:].isdigit()
    ]

    print(f"The following columns will be used for training {selected_features}")

    # ------------------------------------------------------------
    # SAME RF spatial + road split
    # ------------------------------------------------------------
    train_df = train_test_splitted_data["train_df"]
    test_df = train_test_splitted_data["test_df"]

    if group_col not in train_df.columns:
        raise ValueError(
            f"'{group_col}' not found in train_df. "
            "Run train_random_forest first."
        )

    X_train = train_df[selected_features].to_numpy(dtype="float64")
    X_test = test_df[selected_features].to_numpy(dtype="float64")

    y_train_str = train_df[class_col].to_numpy(dtype=str)
    y_test_str = test_df[class_col].to_numpy(dtype=str)

    # Labels as integers (XGBoost needs this; harmless for the others)
    encoder = LabelEncoder().fit(y_train_str)
    y_train = encoder.transform(y_train_str)

    # Balanced weights only when the RF step was run in "weights" mode
    balance = train_test_splitted_data.get("balance", "weights")
    use_weights = balance == "weights"
    sample_weight = (
        compute_sample_weight("balanced", y_train) if use_weights else None
    )

    groups_train = train_df[group_col].to_numpy()

    n_groups = len(np.unique(groups_train))
    if n_groups < cv_folds:
        raise ValueError(
            f"Only {n_groups} groups in the training set, "
            f"but cv_folds={cv_folds}. Lower cv_folds."
        )

    # ------------------------------------------------------------
    # Random search for every model (same grouped folds for all)
    # ------------------------------------------------------------
    specs = _model_specs(random_state, len(selected_features))

    unknown = [m for m in models if m not in specs]
    if unknown:
        msg = f"Unknown or unavailable models: {unknown}. Available: {list(specs)}"
        if "xgboost" in unknown and not HAS_XGB:
            msg += "  (xgboost is not installed: pip install xgboost)"
        raise ValueError(msg)

    cv = GroupKFold(n_splits=cv_folds)

    searches = {}
    rows = []

    for name in models:
        estimator, space, needs_scaling = specs[name]

        steps = []
        if needs_scaling:
            steps.append(("scaler", StandardScaler()))
        steps.append(("clf", estimator))
        pipe = Pipeline(steps)

        print(f"\n--- Searching {name} ({n_iter} tries x {cv_folds} folds) ---")

        search = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=space,
            n_iter=n_iter,
            scoring=scoring,
            cv=cv,
            random_state=random_state,
            n_jobs=n_jobs,
            refit=False,
            verbose=0,
        )

        # KNN cannot use sample weights; all other models can
        fit_params = (
            {}
            if (name == "knn" or not use_weights)
            else {"clf__sample_weight": sample_weight}
        )

        search.fit(X_train, y_train, groups=groups_train, **fit_params)

        best_std = search.cv_results_["std_test_score"][search.best_index_]
        print(f"Best CV {scoring}: {search.best_score_:.4f} (+/- {best_std:.4f})")

        searches[name] = (search, pipe)
        rows.append(
            {
                "model": name,
                "cv_score": search.best_score_,
                "cv_std": best_std,
                "best_params": {
                    k.replace("clf__", ""): v
                    for k, v in search.best_params_.items()
                },
            }
        )

    comparison = (
        pd.DataFrame(rows)
        .sort_values("cv_score", ascending=False)
        .reset_index(drop=True)
    )

    best_name = comparison.loc[0, "model"]
    best_params = comparison.loc[0, "best_params"]

    print("\n==========================================")
    print("MODEL COMPARISON (grouped CV on training set)")
    print("==========================================")
    print(
        comparison[["model", "cv_score", "cv_std"]].to_string(
            index=False, float_format=lambda v: f"{v:.4f}"
        )
    )
    print(f"\nBest model: {best_name}")
    print(f"Best params: {best_params}")

    # ------------------------------------------------------------
    # Refit the winner on all training data
    # ------------------------------------------------------------
    _, best_pipe = searches[best_name]
    final_pipe = clone(best_pipe).set_params(
        **{f"clf__{k}": v for k, v in best_params.items()}
    )

    if best_name == "svm":
        final_pipe.set_params(clf__probability=True)  # for later inference

    final_fit_params = (
        {}
        if (best_name == "knn" or not use_weights)
        else {"clf__sample_weight": sample_weight}
    )
    final_pipe.fit(X_train, y_train, **final_fit_params)

    model = LabelledModel(final_pipe, encoder)

    # ------------------------------------------------------------
    # Test set: used ONCE, only for the winning model
    # ------------------------------------------------------------
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test_str, y_pred)

    classes = sorted(np.unique(y_test_str))

    f1_per_class = f1_score(
        y_test_str,
        y_pred,
        labels=classes,
        average=None,
        zero_division=0,
    )
    f1_by_class = dict(zip(classes, f1_per_class))

    f1_macro = f1_score(y_test_str, y_pred, average="macro", zero_division=0)

    report = classification_report(y_test_str, y_pred, zero_division=0)

    cm = confusion_matrix(y_test_str, y_pred, labels=classes)

    print("\n==========================================")
    print(f"TEST RESULTS — {best_name.upper()}")
          # f" — TOP {n_top_features} FEATURES")
    print("==========================================")

    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Macro F1: {f1_macro:.4f}")

    print("\nF1 by class:")
    for class_name, f1 in f1_by_class.items():
        print(f"  {class_name}: {f1:.4f}")

    print("\nClassification report:")
    print(report)

    print("Confusion matrix:")
    print(cm)

    # ------------------------------------------------------------
    # Save the best model to disk
    #
    # Stored as plain sklearn objects (no custom classes), so it can
    # be loaded anywhere with joblib.load (see inference.load_saved_model).
    #   bundle["pipeline"].predict(X) returns integer codes;
    #   bundle["classes"][code] gives the class name.
    # ------------------------------------------------------------
    if save_path is not None:
        bundle = {
            "pipeline": final_pipe,
            "classes": [str(c) for c in encoder.classes_],
            "features": selected_features,  # column order used for training
            "model_name": best_name,
            "best_params": best_params,
            "best_cv_score": float(comparison.loc[0, "cv_score"]),
            "test_accuracy": float(accuracy),
            "test_f1_macro": float(f1_macro),
        }
        joblib.dump(bundle, save_path)
        print(f"\nBest model saved to: {save_path}")

    return {
        "saved_to": save_path,
        "model": model,  # predict() returns class names
        "best_model_name": best_name,
        "best_params": best_params,
        "best_cv_score": comparison.loc[0, "cv_score"],
        "comparison": comparison,  # leaderboard of all models
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_by_class": f1_by_class,
        "classification_report": report,
        "confusion_matrix": cm,
        "selected_features": selected_features,
        "classes": classes,
    }


# ===========================================================================
# 4. SVM-only random search (kept from the original script, not used by app.py)
# ===========================================================================
def train_svm_top_features(
    rf_results: dict,
    n_top_features: int = 10,
    class_col: str = "class_label",
    group_col: str = "split_group",  # road-aware groups made by the split
    kernels: Sequence[str] = ("rbf",),
    C_range: tuple = (1e-2, 1e3),
    gamma_range: tuple = (1e-4, 1e1),
    n_iter: int = 30,
    cv_folds: int = 5,
    scoring: str = "f1_macro",
    random_state: int = 42,
    n_jobs: int = -1,
) -> dict:

    ranked_features = rf_results["feature_importances"].sort_values(
        ascending=False
    )

    if n_top_features < 1:
        raise ValueError("n_top_features must be at least 1.")

    if n_top_features > len(ranked_features):
        raise ValueError(
            f"Requested {n_top_features} features, "
            f"but only {len(ranked_features)} are available."
        )

    selected_features = ranked_features.head(n_top_features).index.tolist()

    train_df = rf_results["train_df"]
    test_df = rf_results["test_df"]

    if group_col not in train_df.columns:
        raise ValueError(
            f"'{group_col}' not found in train_df. "
            "Run train_random_forest first."
        )

    X_train = train_df[selected_features].to_numpy(dtype="float64")
    X_test = test_df[selected_features].to_numpy(dtype="float64")

    y_train = train_df[class_col].to_numpy(dtype=str)
    y_test = test_df[class_col].to_numpy(dtype=str)

    groups_train = train_df[group_col].to_numpy()

    n_groups = len(np.unique(groups_train))
    if n_groups < cv_folds:
        raise ValueError(
            f"Only {n_groups} groups in the training set, "
            f"but cv_folds={cv_folds}. Lower cv_folds."
        )

    # probability=False during the search (much faster); the final model
    # is refit with probability=True below.
    search_pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svm", SVC(probability=False)),
        ]
    )

    param_distributions = {
        "svm__C": loguniform(*C_range),
        "svm__gamma": loguniform(*gamma_range),
        "svm__kernel": list(kernels),
    }

    search = RandomizedSearchCV(
        estimator=search_pipe,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=scoring,
        cv=GroupKFold(n_splits=cv_folds),
        random_state=random_state,
        n_jobs=n_jobs,
        refit=False,
        verbose=1,
    )

    search.fit(X_train, y_train, groups=groups_train)

    best_params = {
        k.replace("svm__", ""): v for k, v in search.best_params_.items()
    }

    print(f"\nBest CV {scoring}: {search.best_score_:.4f}")
    print(f"Best params: {best_params}")

    svm_model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svm", SVC(probability=True, random_state=random_state, **best_params)),
        ]
    )

    svm_model.fit(X_train, y_train)

    y_pred = svm_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    classes = sorted(np.unique(y_test))

    f1_per_class = f1_score(
        y_test,
        y_pred,
        labels=classes,
        average=None,
        zero_division=0,
    )
    f1_by_class = dict(zip(classes, f1_per_class))

    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)

    report = classification_report(y_test, y_pred, zero_division=0)

    cm = confusion_matrix(y_test, y_pred, labels=classes)

    print("\n==========================================")
    print(f"SVM RESULTS — TOP {n_top_features} FEATURES")
    print("==========================================")

    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Macro F1: {f1_macro:.4f}")

    print("\nF1 by class:")
    for class_name, f1 in f1_by_class.items():
        print(f"  {class_name}: {f1:.4f}")

    print("\nClassification report:")
    print(report)

    print("Confusion matrix:")
    print(cm)

    print("\nSelected features:")
    for rank, feature in enumerate(selected_features, start=1):
        print(f"{rank:>3}. {feature}  RF importance = {ranked_features[feature]:.6f}")

    return {
        "model": svm_model,
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_by_class": f1_by_class,
        "classification_report": report,
        "confusion_matrix": cm,
        "selected_features": selected_features,
        "n_top_features": n_top_features,
        "classes": classes,
        "best_params": best_params,
        "best_cv_score": search.best_score_,
        "cv_results": search.cv_results_,
        "C": best_params.get("C"),
        "kernel": best_params.get("kernel"),
        "gamma": best_params.get("gamma"),
    }
