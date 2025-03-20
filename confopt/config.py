from typing import List
from confopt.data_classes import IntRange, FloatRange, CategoricalRange

# Reference names of search estimator architectures:
QGBM_NAME: str = "qgbm"
QRF_NAME: str = "qrf"
KR_NAME: str = "kr"
GBM_NAME: str = "gbm"
LGBM_NAME: str = "lgbm"
KNN_NAME: str = "knn"
RF_NAME: str = "rf"
QKNN_NAME: str = "qknn"
QL_NAME: str = "ql"
QLGBM_NAME: str = "qlgbm"
SFQENS_NAME: str = "sfqens"  # New quantile ensemble model
MFENS_NAME: str = "mfqens"  # New ensemble model name for QLGBM + QL combination
PENS_NAME: str = "pens"  # New point ensemble model for GBM + KNN combination

# Reference names of quantile regression estimators:
MULTI_FIT_QUANTILE_ESTIMATOR_ARCHITECTURES: List[str] = [
    QGBM_NAME,
    QLGBM_NAME,
    QL_NAME,
    MFENS_NAME,
]

SINGLE_FIT_QUANTILE_ESTIMATOR_ARCHITECTURES: List[str] = [
    QRF_NAME,
    QKNN_NAME,
    SFQENS_NAME,
]

POINT_ESTIMATOR_ARCHITECTURES: List[str] = [
    KR_NAME,
    GBM_NAME,
    LGBM_NAME,
    KNN_NAME,
    RF_NAME,
    PENS_NAME,
]

# Define tuning spaces using the data classes based on original values

# Random Forest tuning space
RF_TUNING_SPACE = {
    "n_estimators": IntRange(min_value=10, max_value=75),
    "max_features": CategoricalRange(choices=[0.3, 0.5, 0.7, "sqrt"]),
    "min_samples_split": IntRange(min_value=2, max_value=7),
    "min_samples_leaf": IntRange(min_value=1, max_value=6),
    "bootstrap": CategoricalRange(choices=[True, False]),
}

# KNN tuning space
KNN_TUNING_SPACE = {
    "n_neighbors": IntRange(min_value=3, max_value=9),
    "weights": CategoricalRange(choices=["uniform", "distance"]),
    "p": CategoricalRange(choices=[1, 2]),
}

# LGBM tuning space
LGBM_TUNING_SPACE = {
    "learning_rate": FloatRange(min_value=0.05, max_value=0.2),
    "n_estimators": IntRange(min_value=10, max_value=30),
    "max_depth": IntRange(min_value=2, max_value=4),
    "min_child_samples": IntRange(min_value=3, max_value=7),
    "subsample": FloatRange(min_value=0.7, max_value=0.9),
    "colsample_bytree": FloatRange(min_value=0.6, max_value=0.8),
    "reg_alpha": FloatRange(min_value=0.1, max_value=1.0),
    "reg_lambda": FloatRange(min_value=0.1, max_value=1.0),
}

# GBM tuning space
GBM_TUNING_SPACE = {
    "learning_rate": FloatRange(min_value=0.05, max_value=0.3),
    "n_estimators": IntRange(min_value=10, max_value=50),
    "min_samples_split": IntRange(min_value=2, max_value=7),
    "min_samples_leaf": IntRange(min_value=2, max_value=5),
    "max_depth": IntRange(min_value=2, max_value=4),
    "subsample": FloatRange(min_value=0.8, max_value=1.0),
}

# GP tuning space
GP_TUNING_SPACE = {
    "kernel": CategoricalRange(choices=["RBF", "RationalQuadratic"]),
    "alpha": FloatRange(min_value=1e-10, max_value=1e-6, log_scale=True),
    "normalize_y": CategoricalRange(choices=[True, False]),
}

# KR tuning space
KR_TUNING_SPACE = {
    "alpha": FloatRange(min_value=0.1, max_value=10.0, log_scale=True),
    "kernel": CategoricalRange(choices=["linear", "rbf", "poly"]),
}

# QRF tuning space
QRF_TUNING_SPACE = {
    "n_estimators": IntRange(min_value=10, max_value=50),
    "max_depth": IntRange(min_value=3, max_value=5),
    "max_features": FloatRange(min_value=0.6, max_value=0.8),
    "min_samples_split": IntRange(min_value=2, max_value=3),
    "bootstrap": CategoricalRange(choices=[True, False]),
}

# QKNN tuning space
QKNN_TUNING_SPACE = {
    "n_neighbors": IntRange(min_value=3, max_value=10),
}

# QL tuning space
QL_TUNING_SPACE = {
    "alpha": FloatRange(min_value=0.01, max_value=0.3, log_scale=True),
    "max_iter": IntRange(min_value=100, max_value=500),
    "p_tol": FloatRange(min_value=1e-5, max_value=1e-3, log_scale=True),
}

# QGBM tuning space
QGBM_TUNING_SPACE = {
    "learning_rate": FloatRange(min_value=0.1, max_value=0.3),
    "n_estimators": IntRange(min_value=20, max_value=50),
    "min_samples_split": IntRange(min_value=5, max_value=10),
    "min_samples_leaf": IntRange(min_value=3, max_value=5),
    "max_depth": IntRange(min_value=3, max_value=7),
    "subsample": FloatRange(min_value=0.8, max_value=0.9),
    "max_features": FloatRange(min_value=0.8, max_value=1.0),
}

# QLGBM tuning space
QLGBM_TUNING_SPACE = {
    "learning_rate": FloatRange(min_value=0.05, max_value=0.2),
    "n_estimators": IntRange(min_value=10, max_value=30),
    "max_depth": IntRange(min_value=2, max_value=3),
    "min_child_samples": IntRange(min_value=3, max_value=7),
    "subsample": FloatRange(min_value=0.7, max_value=0.9),
    "colsample_bytree": FloatRange(min_value=0.6, max_value=0.8),
    "reg_alpha": FloatRange(min_value=0.1, max_value=1.0),
    "reg_lambda": FloatRange(min_value=0.1, max_value=1.0),
}

# SFQENS tuning space
SFQENS_TUNING_SPACE = {
    "cv": IntRange(min_value=2, max_value=3),
    "weighting_strategy": CategoricalRange(
        choices=["inverse_error", "rank", "uniform", "meta_learner"]
    ),
    "qrf_n_estimators": IntRange(min_value=10, max_value=50),
    "qrf_max_depth": IntRange(min_value=3, max_value=5),
    "qrf_max_features": FloatRange(min_value=0.6, max_value=0.8),
    "qrf_min_samples_split": IntRange(min_value=2, max_value=3),
    "qrf_bootstrap": CategoricalRange(choices=[True, False]),
    "qknn_n_neighbors": IntRange(min_value=3, max_value=10),
}

# MFENS tuning space
MFENS_TUNING_SPACE = {
    "cv": IntRange(min_value=2, max_value=3),
    "weighting_strategy": CategoricalRange(
        choices=["inverse_error", "rank", "uniform", "meta_learner"]
    ),
    "qlgbm_learning_rate": FloatRange(min_value=0.05, max_value=0.2),
    "qlgbm_n_estimators": IntRange(min_value=10, max_value=30),
    "qlgbm_max_depth": IntRange(min_value=2, max_value=3),
    "qlgbm_min_child_samples": IntRange(min_value=3, max_value=7),
    "qlgbm_subsample": FloatRange(min_value=0.7, max_value=0.9),
    "qlgbm_colsample_bytree": FloatRange(min_value=0.6, max_value=0.8),
    "qlgbm_reg_alpha": FloatRange(min_value=0.1, max_value=0.5),
    "qlgbm_reg_lambda": FloatRange(min_value=0.1, max_value=0.5),
    "ql_alpha": FloatRange(min_value=0.01, max_value=0.1, log_scale=True),
    "ql_max_iter": IntRange(min_value=100, max_value=500),
    "ql_p_tol": FloatRange(min_value=1e-4, max_value=1e-3, log_scale=True),
}

# PENS tuning space
PENS_TUNING_SPACE = {
    "cv": IntRange(min_value=2, max_value=3),
    "weighting_strategy": CategoricalRange(
        choices=["inverse_error", "rank", "uniform", "meta_learner"]
    ),
    "gbm_learning_rate": FloatRange(min_value=0.05, max_value=0.3),
    "gbm_n_estimators": IntRange(min_value=10, max_value=50),
    "gbm_min_samples_split": IntRange(min_value=2, max_value=7),
    "gbm_min_samples_leaf": IntRange(min_value=2, max_value=5),
    "gbm_max_depth": IntRange(min_value=2, max_value=4),
    "gbm_subsample": FloatRange(min_value=0.8, max_value=1.0),
    "knn_n_neighbors": IntRange(min_value=3, max_value=9),
    "knn_weights": CategoricalRange(choices=["uniform", "distance"]),
    "knn_p": CategoricalRange(choices=[1, 2]),
}

# Default configurations from the original file
SEARCH_MODEL_DEFAULT_CONFIGURATIONS = {
    RF_NAME: {
        "n_estimators": 25,
        "max_features": "sqrt",
        "min_samples_split": 3,
        "min_samples_leaf": 2,
        "bootstrap": True,
    },
    KNN_NAME: {
        "n_neighbors": 5,
        "weights": "distance",
    },
    GBM_NAME: {
        "learning_rate": 0.1,
        "n_estimators": 25,
        "min_samples_split": 3,
        "min_samples_leaf": 3,
        "max_depth": 2,
        "subsample": 0.9,
    },
    LGBM_NAME: {
        "learning_rate": 0.1,
        "n_estimators": 20,
        "max_depth": 2,
        "min_child_samples": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "min_child_weight": 3,
    },
    KR_NAME: {
        "alpha": 1.0,
        "kernel": "rbf",
    },
    QRF_NAME: {
        "n_estimators": 25,
        "max_depth": 5,
        "max_features": 0.8,
        "min_samples_split": 2,
        "bootstrap": True,
    },
    QKNN_NAME: {
        "n_neighbors": 5,
    },
    QL_NAME: {
        "alpha": 0.05,
        "max_iter": 200,
        "p_tol": 1e-4,
    },
    QGBM_NAME: {
        "learning_rate": 0.2,
        "n_estimators": 25,
        "min_samples_split": 5,
        "min_samples_leaf": 3,
        "max_depth": 5,
        "subsample": 0.8,
        "max_features": 0.8,
    },
    QLGBM_NAME: {
        "learning_rate": 0.1,
        "n_estimators": 20,
        "max_depth": 2,
        "min_child_samples": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "min_child_weight": 3,
    },
    SFQENS_NAME: {
        "cv": 3,
        "weighting_strategy": "inverse_error",
        "qrf_n_estimators": 25,
        "qrf_max_depth": 5,
        "qrf_max_features": 0.8,
        "qrf_min_samples_split": 2,
        "qrf_bootstrap": True,
        "qknn_n_neighbors": 5,
    },
    MFENS_NAME: {
        "cv": 3,
        "weighting_strategy": "inverse_error",
        "qlgbm_learning_rate": 0.1,
        "qlgbm_n_estimators": 20,
        "qlgbm_max_depth": 2,
        "qlgbm_min_child_samples": 5,
        "qlgbm_subsample": 0.8,
        "qlgbm_colsample_bytree": 0.7,
        "qlgbm_reg_alpha": 0.1,
        "qlgbm_reg_lambda": 0.1,
        "ql_alpha": 0.05,
        "ql_max_iter": 200,
        "ql_p_tol": 1e-4,
    },
    PENS_NAME: {
        "cv": 3,
        "weighting_strategy": "inverse_error",
        "gbm_learning_rate": 0.1,
        "gbm_n_estimators": 25,
        "gbm_min_samples_split": 3,
        "gbm_min_samples_leaf": 3,
        "gbm_max_depth": 2,
        "gbm_subsample": 0.9,
        "knn_n_neighbors": 5,
        "knn_weights": "distance",
        "knn_p": 2,
    },
}

# Mapping of tuning spaces using constants as keys
SEARCH_MODEL_TUNING_SPACE = {
    RF_NAME: RF_TUNING_SPACE,
    KNN_NAME: KNN_TUNING_SPACE,
    LGBM_NAME: LGBM_TUNING_SPACE,
    GBM_NAME: GBM_TUNING_SPACE,
    KR_NAME: KR_TUNING_SPACE,
    QRF_NAME: QRF_TUNING_SPACE,
    QKNN_NAME: QKNN_TUNING_SPACE,
    QL_NAME: QL_TUNING_SPACE,
    QGBM_NAME: QGBM_TUNING_SPACE,
    QLGBM_NAME: QLGBM_TUNING_SPACE,
    SFQENS_NAME: SFQENS_TUNING_SPACE,
    MFENS_NAME: MFENS_TUNING_SPACE,
    PENS_NAME: PENS_TUNING_SPACE,
}
