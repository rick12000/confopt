from typing import Dict, Any, Type, Optional, List
from pydantic import BaseModel

from confopt.wrapping import IntRange, FloatRange, CategoricalRange

# Import estimator classes
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsRegressor
from confopt.selection.estimators.quantile_estimation import (
    BaseSingleFitQuantileEstimator,
    BaseMultiFitQuantileEstimator,
    QuantileGBM,
    QuantileForest,
    QuantileKNN,
    QuantileLasso,
    QuantileGP,
    QuantileLeaf,  # Added QuantileLeaf to imports
)
from confopt.wrapping import ParameterRange
from confopt.selection.estimators.ensembling import (
    BaseEnsembleEstimator,
    QuantileEnsembleEstimator,
)


class EstimatorConfig(BaseModel):
    estimator_name: str
    estimator_class: Type
    default_params: Dict[str, Any]
    estimator_parameter_space: Dict[str, ParameterRange]
    ensemble_components: Optional[
        List[Dict[str, Any]]
    ] = None  # New field for ensemble components

    class Config:
        arbitrary_types_allowed = True

    def is_ensemble_estimator(self) -> bool:
        return issubclass(self.estimator_class, BaseEnsembleEstimator)

    def is_quantile_estimator(self) -> bool:
        return issubclass(
            self.estimator_class,
            (
                BaseSingleFitQuantileEstimator,
                BaseMultiFitQuantileEstimator,
                QuantileEnsembleEstimator,
            ),
        )


# Reference names of search estimator architectures:
QGBM_NAME: str = "qgbm"
QRF_NAME: str = "qrf"
KR_NAME: str = "kr"
GBM_NAME: str = "gbm"
KNN_NAME: str = "knn"
RF_NAME: str = "rf"
QKNN_NAME: str = "qknn"
QL_NAME: str = "ql"
QGP_NAME: str = "qgp"  # Gaussian Process Quantile Estimator
QLEAF_NAME: str = "qleaf"  # New quantile estimator

# New ensemble estimator names
QENS1_NAME: str = "qens1"  # Ensemble of QL + QKNN + QRF
QENS2_NAME: str = "qens2"  # Ensemble of QL + QKNN + QGBM
QENS3_NAME: str = "qens3"  # Ensemble of QRF + QL
QENS4_NAME: str = "qens4"  # Ensemble of QRF + QGP
QENS5_NAME: str = "qens5"  # Ensemble of QGP + QRF + QKNN

QUANTILE_TO_POINT_ESTIMATOR_MAPPING = {
    QRF_NAME: RF_NAME,
    QKNN_NAME: KNN_NAME,
    QLEAF_NAME: RF_NAME,
    QGBM_NAME: GBM_NAME,
}

# Consolidated estimator configurations
ESTIMATOR_REGISTRY = {
    # Point estimators
    RF_NAME: EstimatorConfig(
        estimator_name=RF_NAME,
        estimator_class=RandomForestRegressor,
        default_params={
            "n_estimators": 50,
            "max_features": "sqrt",
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_depth": 3,
            "bootstrap": True,
            "random_state": None,  # added to allow seeding
        },
        estimator_parameter_space={
            "n_estimators": IntRange(min_value=25, max_value=200),
            "max_features": CategoricalRange(choices=[0.5, 0.7, "sqrt"]),
            "min_samples_split": IntRange(min_value=2, max_value=6),
            "min_samples_leaf": IntRange(min_value=1, max_value=4),
            "max_depth": IntRange(min_value=2, max_value=6),
            "bootstrap": CategoricalRange(choices=[True, False]),
        },
    ),
    KNN_NAME: EstimatorConfig(
        estimator_name=KNN_NAME,
        estimator_class=KNeighborsRegressor,
        default_params={
            "n_neighbors": 10,
            "weights": "distance",
        },
        estimator_parameter_space={
            "n_neighbors": IntRange(min_value=5, max_value=20),
            "weights": CategoricalRange(choices=["uniform", "distance"]),
            "p": CategoricalRange(choices=[1, 2]),
        },
    ),
    GBM_NAME: EstimatorConfig(
        estimator_name=GBM_NAME,
        estimator_class=GradientBoostingRegressor,
        default_params={
            "learning_rate": 0.05,
            "n_estimators": 100,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_depth": 3,
            "subsample": 0.8,
            "random_state": None,  # added
        },
        estimator_parameter_space={
            "learning_rate": FloatRange(min_value=0.02, max_value=0.15),
            "n_estimators": IntRange(min_value=10, max_value=200),
            "min_samples_split": IntRange(min_value=4, max_value=10),
            "min_samples_leaf": IntRange(min_value=3, max_value=7),
            "max_depth": IntRange(min_value=2, max_value=4),
            "subsample": FloatRange(min_value=0.7, max_value=0.9),
        },
    ),
    KR_NAME: EstimatorConfig(
        estimator_name=KR_NAME,
        estimator_class=KernelRidge,
        default_params={
            "alpha": 5.0,
            "kernel": "rbf",
        },
        estimator_parameter_space={
            "alpha": FloatRange(min_value=1.0, max_value=20.0, log_scale=True),
            "kernel": CategoricalRange(choices=["linear", "rbf", "poly"]),
        },
    ),
    # Single-fit quantile estimators
    QRF_NAME: EstimatorConfig(
        estimator_name=QRF_NAME,
        estimator_class=QuantileForest,
        default_params={
            "n_estimators": 50,
            "max_depth": 4,
            "max_features": 0.7,
            "min_samples_split": 4,
            "bootstrap": True,
            "random_state": None,  # added
        },
        estimator_parameter_space={
            "n_estimators": IntRange(min_value=25, max_value=100),
            "max_depth": IntRange(min_value=2, max_value=6),
            "max_features": FloatRange(min_value=0.6, max_value=0.8),
            "min_samples_split": IntRange(min_value=2, max_value=6),
            "bootstrap": CategoricalRange(choices=[True, False]),
        },
    ),
    QKNN_NAME: EstimatorConfig(
        estimator_name=QKNN_NAME,
        estimator_class=QuantileKNN,
        default_params={
            "n_neighbors": 6,
        },
        estimator_parameter_space={
            "n_neighbors": IntRange(min_value=5, max_value=20),
        },
    ),
    QLEAF_NAME: EstimatorConfig(
        estimator_name=QLEAF_NAME,
        estimator_class=QuantileLeaf,
        default_params={
            "n_estimators": 50,
            "max_depth": 3,
            "max_features": 0.8,
            "min_samples_split": 2,
            "bootstrap": True,
            "random_state": None,
        },
        estimator_parameter_space={
            "n_estimators": IntRange(min_value=25, max_value=200),
            "max_depth": IntRange(min_value=2, max_value=6),
            "max_features": FloatRange(min_value=0.7, max_value=1.0),
            "min_samples_split": IntRange(min_value=1, max_value=8),
            "bootstrap": CategoricalRange(choices=[True, False]),
        },
    ),
    # Multi-fit quantile estimators
    QGBM_NAME: EstimatorConfig(
        estimator_name=QGBM_NAME,
        estimator_class=QuantileGBM,
        default_params={
            "learning_rate": 0.1,
            "n_estimators": 100,
            "min_samples_split": 6,
            "min_samples_leaf": 1,
            "max_depth": 2,
            "subsample": 0.7,
            "max_features": 0.7,
            "random_state": None,  # added
        },
        estimator_parameter_space={
            "learning_rate": FloatRange(min_value=0.05, max_value=0.2),
            "n_estimators": IntRange(min_value=25, max_value=200),
            "min_samples_split": IntRange(min_value=2, max_value=8),
            "min_samples_leaf": IntRange(min_value=1, max_value=3),
            "max_depth": IntRange(min_value=2, max_value=6),
            "subsample": FloatRange(min_value=0.6, max_value=0.8),
            "max_features": FloatRange(min_value=0.6, max_value=0.8),
        },
    ),
    QL_NAME: EstimatorConfig(
        estimator_name=QL_NAME,
        estimator_class=QuantileLasso,
        default_params={
            "max_iter": 300,
            "p_tol": 1e-4,
            "random_state": None,  # added
        },
        estimator_parameter_space={
            "max_iter": IntRange(min_value=200, max_value=800),
            "p_tol": FloatRange(min_value=1e-5, max_value=1e-3, log_scale=True),
        },
    ),
    # Ensemble estimators
    QENS1_NAME: EstimatorConfig(
        estimator_name=QENS1_NAME,
        estimator_class=QuantileEnsembleEstimator,
        default_params={
            "weighting_strategy": "linear_stack",
            "cv": 5,
            "alpha": 0.001,
        },
        estimator_parameter_space={
            "weighting_strategy": CategoricalRange(choices=["uniform", "linear_stack"]),
            "alpha": FloatRange(min_value=0.001, max_value=0.1, log_scale=True),
        },
        ensemble_components=[
            {
                "class": QuantileLasso,
                "params": {
                    "max_iter": 300,
                    "p_tol": 1e-4,
                },
            },
            {
                "class": QuantileKNN,
                "params": {
                    "n_neighbors": 6,
                },
            },
            {
                "class": QuantileGBM,
                "params": {
                    "learning_rate": 0.1,
                    "n_estimators": 100,
                    "min_samples_split": 6,
                    "min_samples_leaf": 1,
                    "max_depth": 2,
                    "subsample": 0.7,
                    "max_features": 0.7,
                    "random_state": None,
                },
            },
        ],
    ),
    QENS2_NAME: EstimatorConfig(
        estimator_name=QENS2_NAME,
        estimator_class=QuantileEnsembleEstimator,
        default_params={
            "weighting_strategy": "linear_stack",
            "cv": 5,
            "alpha": 0.001,
        },
        estimator_parameter_space={
            "weighting_strategy": CategoricalRange(choices=["uniform", "linear_stack"]),
            "alpha": FloatRange(min_value=0.001, max_value=0.1, log_scale=True),
        },
        ensemble_components=[
            {
                "class": QuantileGBM,
                "params": {
                    "learning_rate": 0.1,
                    "n_estimators": 100,
                    "min_samples_split": 6,
                    "min_samples_leaf": 1,
                    "max_depth": 2,
                    "subsample": 0.7,
                    "max_features": 0.7,
                    "random_state": None,
                },
            },
            {
                "class": QuantileForest,
                "params": {
                    "n_estimators": 50,
                    "max_depth": 4,
                    "max_features": 0.7,
                    "min_samples_split": 4,
                    "bootstrap": True,
                    "random_state": None,
                },
            },
        ],
    ),
    QENS3_NAME: EstimatorConfig(
        estimator_name=QENS3_NAME,
        estimator_class=QuantileEnsembleEstimator,
        default_params={
            "weighting_strategy": "linear_stack",
            "cv": 5,
            "alpha": 0.001,
        },
        estimator_parameter_space={
            "weighting_strategy": CategoricalRange(choices=["uniform", "linear_stack"]),
            "alpha": FloatRange(min_value=0.001, max_value=0.1, log_scale=True),
        },
        ensemble_components=[
            {
                "class": QuantileGBM,
                "params": {
                    "learning_rate": 0.1,
                    "n_estimators": 100,
                    "min_samples_split": 6,
                    "min_samples_leaf": 1,
                    "max_depth": 2,
                    "subsample": 0.7,
                    "max_features": 0.7,
                    "random_state": None,
                },
            },
            {
                "class": QuantileLasso,
                "params": {
                    "max_iter": 300,
                    "p_tol": 1e-4,
                },
            },
        ],
    ),
    QENS4_NAME: EstimatorConfig(
        estimator_name=QENS4_NAME,
        estimator_class=QuantileEnsembleEstimator,
        default_params={
            "weighting_strategy": "linear_stack",
            "cv": 5,
            "alpha": 0.001,
        },
        estimator_parameter_space={
            "weighting_strategy": CategoricalRange(choices=["uniform", "linear_stack"]),
            "alpha": FloatRange(min_value=0.001, max_value=0.1, log_scale=True),
        },
        ensemble_components=[
            {
                "class": QuantileGBM,
                "params": {
                    "learning_rate": 0.1,
                    "n_estimators": 100,
                    "min_samples_split": 6,
                    "min_samples_leaf": 1,
                    "max_depth": 2,
                    "subsample": 0.7,
                    "max_features": 0.7,
                    "random_state": None,
                },
            },
            {
                "class": QuantileGP,
                "params": {
                    "kernel": "matern",
                    "alpha": 1e-8,
                },
            },
        ],
    ),
    QENS5_NAME: EstimatorConfig(
        estimator_name=QENS5_NAME,
        estimator_class=QuantileEnsembleEstimator,
        default_params={
            "weighting_strategy": "linear_stack",
            "cv": 5,
            "alpha": 0.001,
        },
        estimator_parameter_space={
            "weighting_strategy": CategoricalRange(choices=["uniform", "linear_stack"]),
            "alpha": FloatRange(min_value=0.001, max_value=0.1, log_scale=True),
        },
        ensemble_components=[
            {
                "class": QuantileLasso,
                "params": {
                    "max_iter": 300,
                    "p_tol": 1e-4,
                },
            },
            {
                "class": QuantileGP,
                "params": {
                    "kernel": "matern",
                    "alpha": 1e-8,
                    "random_state": None,
                },
            },
            {
                "class": QuantileGBM,
                "params": {
                    "learning_rate": 0.1,
                    "n_estimators": 100,
                    "min_samples_split": 6,
                    "min_samples_leaf": 1,
                    "max_depth": 2,
                    "subsample": 0.7,
                    "max_features": 0.7,
                    "random_state": None,
                },
            },
        ],
    ),
    # Add new quantile estimators
    QGP_NAME: EstimatorConfig(
        estimator_name=QGP_NAME,
        estimator_class=QuantileGP,
        default_params={
            "kernel": "matern",
            "alpha": 1e-8,
            "random_state": None,
        },
        estimator_parameter_space={
            "kernel": CategoricalRange(choices=["rbf", "matern", "rational_quadratic"]),
            "alpha": FloatRange(min_value=1e-10, max_value=1e-6, log_scale=True),
        },
    ),
}
