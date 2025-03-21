from enum import Enum
from typing import Dict, Any, Type, List, Optional
from pydantic import BaseModel

from confopt.data_classes import IntRange, FloatRange, CategoricalRange

# Import estimator classes
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsRegressor
from lightgbm import LGBMRegressor
from confopt.quantile_wrappers import (
    QuantileGBM,
    QuantileLightGBM,
    QuantileForest,
    QuantileKNN,
    QuantileLasso,
)
from confopt.ensembling import (
    SingleFitQuantileEnsembleEstimator,
    MultiFitQuantileEnsembleEstimator,
    PointEnsembleEstimator,
)


class EstimatorType(str, Enum):
    POINT = "point"
    SINGLE_FIT_QUANTILE = "single_fit_quantile"
    MULTI_FIT_QUANTILE = "multi_fit_quantile"
    ENSEMBLE_POINT = "ensemble_point"
    ENSEMBLE_QUANTILE_SINGLE_FIT = "ensemble_quantile_single_fit"
    ENSEMBLE_QUANTILE_MULTI_FIT = "ensemble_quantile_multi_fit"


# Pydantic model for estimator configuration
class EstimatorConfig(BaseModel):
    name: str
    estimator_class: Type
    estimator_type: EstimatorType
    default_config: Dict[str, Any]
    tuning_space: Dict[str, Any]
    component_estimators: Optional[List[str]] = None  # For ensemble models

    class Config:
        arbitrary_types_allowed = True


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
RF_DEFAULT_CONFIG = {
    "n_estimators": 25,
    "max_features": "sqrt",
    "min_samples_split": 3,
    "min_samples_leaf": 2,
    "bootstrap": True,
}

KNN_DEFAULT_CONFIG = {
    "n_neighbors": 5,
    "weights": "distance",
}

GBM_DEFAULT_CONFIG = {
    "learning_rate": 0.1,
    "n_estimators": 25,
    "min_samples_split": 3,
    "min_samples_leaf": 3,
    "max_depth": 2,
    "subsample": 0.9,
}

LGBM_DEFAULT_CONFIG = {
    "learning_rate": 0.1,
    "n_estimators": 20,
    "max_depth": 2,
    "min_child_samples": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "min_child_weight": 3,
}

KR_DEFAULT_CONFIG = {
    "alpha": 1.0,
    "kernel": "rbf",
}

QRF_DEFAULT_CONFIG = {
    "n_estimators": 25,
    "max_depth": 5,
    "max_features": 0.8,
    "min_samples_split": 2,
    "bootstrap": True,
}

QKNN_DEFAULT_CONFIG = {
    "n_neighbors": 5,
}

QL_DEFAULT_CONFIG = {
    "alpha": 0.05,
    "max_iter": 200,
    "p_tol": 1e-4,
}

QGBM_DEFAULT_CONFIG = {
    "learning_rate": 0.2,
    "n_estimators": 25,
    "min_samples_split": 5,
    "min_samples_leaf": 3,
    "max_depth": 5,
    "subsample": 0.8,
    "max_features": 0.8,
}

QLGBM_DEFAULT_CONFIG = {
    "learning_rate": 0.1,
    "n_estimators": 20,
    "max_depth": 2,
    "min_child_samples": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "min_child_weight": 3,
}

SFQENS_DEFAULT_CONFIG = {
    "weighting_strategy": "inverse_error",
    "qrf_n_estimators": 25,
    "qrf_max_depth": 5,
    "qrf_max_features": 0.8,
    "qrf_min_samples_split": 2,
    "qrf_bootstrap": True,
    "qknn_n_neighbors": 5,
}

MFENS_DEFAULT_CONFIG = {
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
}

PENS_DEFAULT_CONFIG = {
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
}


def create_ensemble_config(
    name: str,
    estimator_class: Type,
    estimator_type: EstimatorType,
    component_names: List[str],
) -> EstimatorConfig:
    """
    Create a simplified EstimatorConfig for an ensemble model.
    """
    # Ensemble-specific parameters only include weighting strategy
    tuning_space = {
        "weighting_strategy": CategoricalRange(
            choices=["inverse_error", "rank", "uniform", "meta_learner"]
        )
    }

    default_config = {
        "weighting_strategy": "inverse_error",
        "cv": 3,  # Fixed parameter, not tuned
        "component_estimators": component_names,  # Store component names for initialization
    }

    return EstimatorConfig(
        name=name,
        estimator_class=estimator_class,
        estimator_type=estimator_type,
        default_config=default_config,
        tuning_space=tuning_space,
        component_estimators=component_names,
    )


# Consolidated estimator configurations
ESTIMATOR_REGISTRY = {
    # Point estimators
    RF_NAME: EstimatorConfig(
        name=RF_NAME,
        estimator_class=RandomForestRegressor,
        estimator_type=EstimatorType.POINT,
        default_config=RF_DEFAULT_CONFIG,
        tuning_space=RF_TUNING_SPACE,
    ),
    KNN_NAME: EstimatorConfig(
        name=KNN_NAME,
        estimator_class=KNeighborsRegressor,
        estimator_type=EstimatorType.POINT,
        default_config=KNN_DEFAULT_CONFIG,
        tuning_space=KNN_TUNING_SPACE,
    ),
    GBM_NAME: EstimatorConfig(
        name=GBM_NAME,
        estimator_class=GradientBoostingRegressor,
        estimator_type=EstimatorType.POINT,
        default_config=GBM_DEFAULT_CONFIG,
        tuning_space=GBM_TUNING_SPACE,
    ),
    LGBM_NAME: EstimatorConfig(
        name=LGBM_NAME,
        estimator_class=LGBMRegressor,
        estimator_type=EstimatorType.POINT,
        default_config=LGBM_DEFAULT_CONFIG,
        tuning_space=LGBM_TUNING_SPACE,
    ),
    KR_NAME: EstimatorConfig(
        name=KR_NAME,
        estimator_class=KernelRidge,
        estimator_type=EstimatorType.POINT,
        default_config=KR_DEFAULT_CONFIG,
        tuning_space=KR_TUNING_SPACE,
    ),
    # Single-fit quantile estimators
    QRF_NAME: EstimatorConfig(
        name=QRF_NAME,
        estimator_class=QuantileForest,
        estimator_type=EstimatorType.SINGLE_FIT_QUANTILE,
        default_config=QRF_DEFAULT_CONFIG,
        tuning_space=QRF_TUNING_SPACE,
    ),
    QKNN_NAME: EstimatorConfig(
        name=QKNN_NAME,
        estimator_class=QuantileKNN,
        estimator_type=EstimatorType.SINGLE_FIT_QUANTILE,
        default_config=QKNN_DEFAULT_CONFIG,
        tuning_space=QKNN_TUNING_SPACE,
    ),
    # Multi-fit quantile estimators
    QGBM_NAME: EstimatorConfig(
        name=QGBM_NAME,
        estimator_class=QuantileGBM,
        estimator_type=EstimatorType.MULTI_FIT_QUANTILE,
        default_config=QGBM_DEFAULT_CONFIG,
        tuning_space=QGBM_TUNING_SPACE,
    ),
    QLGBM_NAME: EstimatorConfig(
        name=QLGBM_NAME,
        estimator_class=QuantileLightGBM,
        estimator_type=EstimatorType.MULTI_FIT_QUANTILE,
        default_config=QLGBM_DEFAULT_CONFIG,
        tuning_space=QLGBM_TUNING_SPACE,
    ),
    QL_NAME: EstimatorConfig(
        name=QL_NAME,
        estimator_class=QuantileLasso,
        estimator_type=EstimatorType.MULTI_FIT_QUANTILE,
        default_config=QL_DEFAULT_CONFIG,
        tuning_space=QL_TUNING_SPACE,
    ),
}

# Add ensemble estimators with simplified configs
ESTIMATOR_REGISTRY[PENS_NAME] = create_ensemble_config(
    name=PENS_NAME,
    estimator_class=PointEnsembleEstimator,
    estimator_type=EstimatorType.ENSEMBLE_POINT,
    component_names=[GBM_NAME, KNN_NAME],
)

ESTIMATOR_REGISTRY[SFQENS_NAME] = create_ensemble_config(
    name=SFQENS_NAME,
    estimator_class=SingleFitQuantileEnsembleEstimator,
    estimator_type=EstimatorType.ENSEMBLE_QUANTILE_SINGLE_FIT,
    component_names=[QRF_NAME, QKNN_NAME],
)

ESTIMATOR_REGISTRY[MFENS_NAME] = create_ensemble_config(
    name=MFENS_NAME,
    estimator_class=MultiFitQuantileEnsembleEstimator,
    estimator_type=EstimatorType.ENSEMBLE_QUANTILE_MULTI_FIT,
    component_names=[QLGBM_NAME, QL_NAME],
)

# Helper lists for backwards compatibility
MULTI_FIT_QUANTILE_ESTIMATOR_ARCHITECTURES = [
    name
    for name, config in ESTIMATOR_REGISTRY.items()
    if config.estimator_type
    in [EstimatorType.MULTI_FIT_QUANTILE, EstimatorType.ENSEMBLE_QUANTILE_MULTI_FIT]
]

SINGLE_FIT_QUANTILE_ESTIMATOR_ARCHITECTURES = [
    name
    for name, config in ESTIMATOR_REGISTRY.items()
    if config.estimator_type
    in [EstimatorType.SINGLE_FIT_QUANTILE, EstimatorType.ENSEMBLE_QUANTILE_SINGLE_FIT]
]

POINT_ESTIMATOR_ARCHITECTURES = [
    name
    for name, config in ESTIMATOR_REGISTRY.items()
    if config.estimator_type in [EstimatorType.POINT, EstimatorType.ENSEMBLE_POINT]
]
