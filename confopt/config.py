from enum import Enum
from typing import Dict, Any, Type
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

DUMMY_QUANTILES = [0.2, 0.8]


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
    default_estimator: Any
    tuning_space: Dict[str, Any]

    class Config:
        arbitrary_types_allowed = True

    def is_ensemble(self) -> bool:
        """Determine if this estimator is an ensemble model"""
        return self.estimator_type in [
            EstimatorType.ENSEMBLE_POINT,
            EstimatorType.ENSEMBLE_QUANTILE_SINGLE_FIT,
            EstimatorType.ENSEMBLE_QUANTILE_MULTI_FIT,
        ]

    def is_quantile_estimator(self) -> bool:
        """Determine if this estimator produces quantile predictions"""
        return self.estimator_type in [
            EstimatorType.SINGLE_FIT_QUANTILE,
            EstimatorType.MULTI_FIT_QUANTILE,
            EstimatorType.ENSEMBLE_QUANTILE_SINGLE_FIT,
            EstimatorType.ENSEMBLE_QUANTILE_MULTI_FIT,
        ]

    def needs_multiple_fits(self) -> bool:
        """Determine if this estimator requires multiple fits for different quantiles"""
        return self.estimator_type in [
            EstimatorType.MULTI_FIT_QUANTILE,
            EstimatorType.ENSEMBLE_QUANTILE_MULTI_FIT,
        ]

    def is_single_fit_quantile(self) -> bool:
        """Determine if this estimator is a single-fit quantile estimator"""
        return self.estimator_type in [
            EstimatorType.SINGLE_FIT_QUANTILE,
            EstimatorType.ENSEMBLE_QUANTILE_SINGLE_FIT,
        ]

    def is_point_estimator(self) -> bool:
        """Determine if this estimator is a point estimator"""
        return self.estimator_type in [
            EstimatorType.POINT,
            EstimatorType.ENSEMBLE_POINT,
        ]


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
SFQENS_NAME: str = "sfqens"  # Quantile ensemble model
MFENS_NAME: str = "mfqens"  # Ensemble model name for QLGBM + QL combination
PENS_NAME: str = "pens"  # Point ensemble model for GBM + KNN combination

# Consolidated estimator configurations
ESTIMATOR_REGISTRY = {
    # Point estimators
    RF_NAME: EstimatorConfig(
        name=RF_NAME,
        estimator_class=RandomForestRegressor,
        estimator_type=EstimatorType.POINT,
        default_estimator=RandomForestRegressor(
            n_estimators=25,
            max_features="sqrt",
            min_samples_split=3,
            min_samples_leaf=2,
            bootstrap=True,
        ),
        tuning_space={
            "n_estimators": IntRange(min_value=10, max_value=75),
            "max_features": CategoricalRange(choices=[0.3, 0.5, 0.7, "sqrt"]),
            "min_samples_split": IntRange(min_value=2, max_value=7),
            "min_samples_leaf": IntRange(min_value=1, max_value=6),
            "bootstrap": CategoricalRange(choices=[True, False]),
        },
    ),
    KNN_NAME: EstimatorConfig(
        name=KNN_NAME,
        estimator_class=KNeighborsRegressor,
        estimator_type=EstimatorType.POINT,
        default_estimator=KNeighborsRegressor(
            n_neighbors=5,
            weights="distance",
        ),
        tuning_space={
            "n_neighbors": IntRange(min_value=3, max_value=9),
            "weights": CategoricalRange(choices=["uniform", "distance"]),
            "p": CategoricalRange(choices=[1, 2]),
        },
    ),
    GBM_NAME: EstimatorConfig(
        name=GBM_NAME,
        estimator_class=GradientBoostingRegressor,
        estimator_type=EstimatorType.POINT,
        default_estimator=GradientBoostingRegressor(
            learning_rate=0.1,
            n_estimators=25,
            min_samples_split=3,
            min_samples_leaf=3,
            max_depth=2,
            subsample=0.9,
        ),
        tuning_space={
            "learning_rate": FloatRange(min_value=0.05, max_value=0.3),
            "n_estimators": IntRange(min_value=10, max_value=50),
            "min_samples_split": IntRange(min_value=2, max_value=7),
            "min_samples_leaf": IntRange(min_value=2, max_value=5),
            "max_depth": IntRange(min_value=2, max_value=4),
            "subsample": FloatRange(min_value=0.8, max_value=1.0),
        },
    ),
    LGBM_NAME: EstimatorConfig(
        name=LGBM_NAME,
        estimator_class=LGBMRegressor,
        estimator_type=EstimatorType.POINT,
        default_estimator=LGBMRegressor(
            learning_rate=0.1,
            n_estimators=20,
            max_depth=2,
            min_child_samples=5,
            subsample=0.8,
            colsample_bytree=0.7,
            reg_alpha=0.1,
            reg_lambda=0.1,
            min_child_weight=3,
        ),
        tuning_space={
            "learning_rate": FloatRange(min_value=0.05, max_value=0.2),
            "n_estimators": IntRange(min_value=10, max_value=30),
            "max_depth": IntRange(min_value=2, max_value=4),
            "min_child_samples": IntRange(min_value=3, max_value=7),
            "subsample": FloatRange(min_value=0.7, max_value=0.9),
            "colsample_bytree": FloatRange(min_value=0.6, max_value=0.8),
            "reg_alpha": FloatRange(min_value=0.1, max_value=1.0),
            "reg_lambda": FloatRange(min_value=0.1, max_value=1.0),
        },
    ),
    KR_NAME: EstimatorConfig(
        name=KR_NAME,
        estimator_class=KernelRidge,
        estimator_type=EstimatorType.POINT,
        default_estimator=KernelRidge(
            alpha=1.0,
            kernel="rbf",
        ),
        tuning_space={
            "alpha": FloatRange(min_value=0.1, max_value=10.0, log_scale=True),
            "kernel": CategoricalRange(choices=["linear", "rbf", "poly"]),
        },
    ),
    # Single-fit quantile estimators
    QRF_NAME: EstimatorConfig(
        name=QRF_NAME,
        estimator_class=QuantileForest,
        estimator_type=EstimatorType.SINGLE_FIT_QUANTILE,
        default_estimator=QuantileForest(
            n_estimators=25,
            max_depth=5,
            max_features=0.8,
            min_samples_split=2,
            bootstrap=True,
        ),
        tuning_space={
            "n_estimators": IntRange(min_value=10, max_value=50),
            "max_depth": IntRange(min_value=3, max_value=5),
            "max_features": FloatRange(min_value=0.6, max_value=0.8),
            "min_samples_split": IntRange(min_value=2, max_value=3),
            "bootstrap": CategoricalRange(choices=[True, False]),
        },
    ),
    QKNN_NAME: EstimatorConfig(
        name=QKNN_NAME,
        estimator_class=QuantileKNN,
        estimator_type=EstimatorType.SINGLE_FIT_QUANTILE,
        default_estimator=QuantileKNN(
            n_neighbors=5,
        ),
        tuning_space={
            "n_neighbors": IntRange(min_value=3, max_value=10),
        },
    ),
    # Multi-fit quantile estimators
    QGBM_NAME: EstimatorConfig(
        name=QGBM_NAME,
        estimator_class=QuantileGBM,
        estimator_type=EstimatorType.MULTI_FIT_QUANTILE,
        default_estimator=QuantileGBM(
            quantiles=DUMMY_QUANTILES,
            learning_rate=0.2,
            n_estimators=25,
            min_samples_split=5,
            min_samples_leaf=3,
            max_depth=5,
            subsample=0.8,
            max_features=0.8,
        ),
        tuning_space={
            "learning_rate": FloatRange(min_value=0.1, max_value=0.3),
            "n_estimators": IntRange(min_value=20, max_value=50),
            "min_samples_split": IntRange(min_value=5, max_value=10),
            "min_samples_leaf": IntRange(min_value=3, max_value=5),
            "max_depth": IntRange(min_value=3, max_value=7),
            "subsample": FloatRange(min_value=0.8, max_value=0.9),
            "max_features": FloatRange(min_value=0.8, max_value=1.0),
        },
    ),
    QLGBM_NAME: EstimatorConfig(
        name=QLGBM_NAME,
        estimator_class=QuantileLightGBM,
        estimator_type=EstimatorType.MULTI_FIT_QUANTILE,
        default_estimator=QuantileLightGBM(
            quantiles=DUMMY_QUANTILES,
            learning_rate=0.1,
            n_estimators=20,
            max_depth=2,
            min_child_samples=5,
            subsample=0.8,
            colsample_bytree=0.7,
            reg_alpha=0.1,
            reg_lambda=0.1,
            min_child_weight=3,
        ),
        tuning_space={
            "learning_rate": FloatRange(min_value=0.05, max_value=0.2),
            "n_estimators": IntRange(min_value=10, max_value=30),
            "max_depth": IntRange(min_value=2, max_value=3),
            "min_child_samples": IntRange(min_value=3, max_value=7),
            "subsample": FloatRange(min_value=0.7, max_value=0.9),
            "colsample_bytree": FloatRange(min_value=0.6, max_value=0.8),
            "reg_alpha": FloatRange(min_value=0.1, max_value=1.0),
            "reg_lambda": FloatRange(min_value=0.1, max_value=1.0),
        },
    ),
    QL_NAME: EstimatorConfig(
        name=QL_NAME,
        estimator_class=QuantileLasso,
        estimator_type=EstimatorType.MULTI_FIT_QUANTILE,
        default_estimator=QuantileLasso(
            quantiles=DUMMY_QUANTILES,
            alpha=0.05,
            max_iter=200,
            p_tol=1e-4,
        ),
        tuning_space={
            "alpha": FloatRange(min_value=0.01, max_value=0.3, log_scale=True),
            "max_iter": IntRange(min_value=100, max_value=500),
            "p_tol": FloatRange(min_value=1e-5, max_value=1e-3, log_scale=True),
        },
    ),
}

# Create point ensemble estimator with GBM and KNN components
point_ensemble = PointEnsembleEstimator(weighting_strategy="inverse_error", cv=3)
point_ensemble.add_estimator(ESTIMATOR_REGISTRY[GBM_NAME].default_estimator)
point_ensemble.add_estimator(ESTIMATOR_REGISTRY[KNN_NAME].default_estimator)

# Create single-fit quantile ensemble with QRF and QKNN components
sfq_ensemble = SingleFitQuantileEnsembleEstimator(
    weighting_strategy="inverse_error", cv=3
)
sfq_ensemble.add_estimator(ESTIMATOR_REGISTRY[QRF_NAME].default_estimator)
sfq_ensemble.add_estimator(ESTIMATOR_REGISTRY[QKNN_NAME].default_estimator)

# Create multi-fit quantile ensemble with QLGBM and QL components
mfq_ensemble = MultiFitQuantileEnsembleEstimator(
    weighting_strategy="inverse_error", cv=3
)
mfq_ensemble.add_estimator(ESTIMATOR_REGISTRY[QLGBM_NAME].default_estimator)
mfq_ensemble.add_estimator(ESTIMATOR_REGISTRY[QL_NAME].default_estimator)

# Add ensemble estimators to registry
ESTIMATOR_REGISTRY[PENS_NAME] = EstimatorConfig(
    name=PENS_NAME,
    estimator_class=PointEnsembleEstimator,
    estimator_type=EstimatorType.ENSEMBLE_POINT,
    default_estimator=point_ensemble,
    tuning_space={
        "weighting_strategy": CategoricalRange(
            choices=["inverse_error", "rank", "uniform", "meta_learner"]
        ),
        "component_0.learning_rate": FloatRange(min_value=0.05, max_value=0.3),
        "component_0.n_estimators": IntRange(min_value=10, max_value=50),
        "component_1.n_neighbors": IntRange(min_value=3, max_value=9),
    },
)

ESTIMATOR_REGISTRY[SFQENS_NAME] = EstimatorConfig(
    name=SFQENS_NAME,
    estimator_class=SingleFitQuantileEnsembleEstimator,
    estimator_type=EstimatorType.ENSEMBLE_QUANTILE_SINGLE_FIT,
    default_estimator=sfq_ensemble,
    tuning_space={
        "weighting_strategy": CategoricalRange(
            choices=["inverse_error", "rank", "uniform", "meta_learner"]
        ),
        "component_0.n_estimators": IntRange(min_value=10, max_value=50),
        "component_0.max_depth": IntRange(min_value=3, max_value=5),
        "component_1.n_neighbors": IntRange(min_value=3, max_value=10),
    },
)

ESTIMATOR_REGISTRY[MFENS_NAME] = EstimatorConfig(
    name=MFENS_NAME,
    estimator_class=MultiFitQuantileEnsembleEstimator,
    estimator_type=EstimatorType.ENSEMBLE_QUANTILE_MULTI_FIT,
    default_estimator=mfq_ensemble,
    tuning_space={
        "weighting_strategy": CategoricalRange(
            choices=["inverse_error", "rank", "uniform", "meta_learner"]
        ),
        "component_0.learning_rate": FloatRange(min_value=0.05, max_value=0.2),
        "component_0.n_estimators": IntRange(min_value=10, max_value=30),
        "component_1.alpha": FloatRange(min_value=0.01, max_value=0.3, log_scale=True),
    },
)
