from typing import Dict, Any
from pydantic import BaseModel

from confopt.data_classes import IntRange, FloatRange, CategoricalRange

# Import estimator classes
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsRegressor
from lightgbm import LGBMRegressor
from confopt.selection.quantile_estimation import (
    BaseSingleFitQuantileEstimator,
    BaseMultiFitQuantileEstimator,
    QuantileGBM,
    QuantileLightGBM,
    QuantileForest,
    QuantileKNN,
    QuantileLasso,
)
from confopt.data_classes import ParameterRange
from confopt.selection.ensembling import (
    BaseEnsembleEstimator,
    QuantileEnsembleEstimator,
    PointEnsembleEstimator,
)
from copy import deepcopy


class EstimatorConfig(BaseModel):
    estimator_name: str
    estimator_instance: Any
    estimator_parameter_space: Dict[str, ParameterRange]

    class Config:
        arbitrary_types_allowed = True

    def is_ensemble_estimator(self) -> bool:
        return isinstance(self.estimator_instance, BaseEnsembleEstimator)

    def is_quantile_estimator(self) -> bool:
        return isinstance(
            self.estimator_instance,
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
        estimator_name=RF_NAME,
        estimator_instance=RandomForestRegressor(
            n_estimators=25,
            max_features="sqrt",
            min_samples_split=3,
            min_samples_leaf=2,
            bootstrap=True,
        ),
        estimator_parameter_space={
            "n_estimators": IntRange(min_value=10, max_value=75),
            "max_features": CategoricalRange(choices=[0.3, 0.5, 0.7, "sqrt"]),
            "min_samples_split": IntRange(min_value=2, max_value=7),
            "min_samples_leaf": IntRange(min_value=1, max_value=6),
            "bootstrap": CategoricalRange(choices=[True, False]),
        },
    ),
    KNN_NAME: EstimatorConfig(
        estimator_name=KNN_NAME,
        estimator_instance=KNeighborsRegressor(
            n_neighbors=5,
            weights="distance",
        ),
        estimator_parameter_space={
            "n_neighbors": IntRange(min_value=3, max_value=9),
            "weights": CategoricalRange(choices=["uniform", "distance"]),
            "p": CategoricalRange(choices=[1, 2]),
        },
    ),
    GBM_NAME: EstimatorConfig(
        estimator_name=GBM_NAME,
        estimator_instance=GradientBoostingRegressor(
            learning_rate=0.1,
            n_estimators=25,
            min_samples_split=3,
            min_samples_leaf=3,
            max_depth=2,
            subsample=0.9,
        ),
        estimator_parameter_space={
            "learning_rate": FloatRange(min_value=0.05, max_value=0.3),
            "n_estimators": IntRange(min_value=10, max_value=50),
            "min_samples_split": IntRange(min_value=2, max_value=7),
            "min_samples_leaf": IntRange(min_value=2, max_value=5),
            "max_depth": IntRange(min_value=2, max_value=4),
            "subsample": FloatRange(min_value=0.8, max_value=1.0),
        },
    ),
    LGBM_NAME: EstimatorConfig(
        estimator_name=LGBM_NAME,
        estimator_instance=LGBMRegressor(
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
        estimator_parameter_space={
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
        estimator_name=KR_NAME,
        estimator_instance=KernelRidge(
            alpha=1.0,
            kernel="rbf",
        ),
        estimator_parameter_space={
            "alpha": FloatRange(min_value=0.1, max_value=10.0, log_scale=True),
            "kernel": CategoricalRange(choices=["linear", "rbf", "poly"]),
        },
    ),
    # Single-fit quantile estimators
    QRF_NAME: EstimatorConfig(
        estimator_name=QRF_NAME,
        estimator_instance=QuantileForest(
            n_estimators=25,
            max_depth=5,
            max_features=0.8,
            min_samples_split=2,
            bootstrap=True,
        ),
        estimator_parameter_space={
            "n_estimators": IntRange(min_value=10, max_value=50),
            "max_depth": IntRange(min_value=3, max_value=5),
            "max_features": FloatRange(min_value=0.6, max_value=0.8),
            "min_samples_split": IntRange(min_value=2, max_value=3),
            "bootstrap": CategoricalRange(choices=[True, False]),
        },
    ),
    QKNN_NAME: EstimatorConfig(
        estimator_name=QKNN_NAME,
        estimator_instance=QuantileKNN(
            n_neighbors=5,
        ),
        estimator_parameter_space={
            "n_neighbors": IntRange(min_value=3, max_value=10),
        },
    ),
    # Multi-fit quantile estimators
    QGBM_NAME: EstimatorConfig(
        estimator_name=QGBM_NAME,
        estimator_instance=QuantileGBM(
            learning_rate=0.2,
            n_estimators=25,
            min_samples_split=5,
            min_samples_leaf=3,
            max_depth=5,
            subsample=0.8,
            max_features=0.8,
        ),
        estimator_parameter_space={
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
        estimator_name=QLGBM_NAME,
        estimator_instance=QuantileLightGBM(
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
        estimator_parameter_space={
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
        estimator_name=QL_NAME,
        estimator_instance=QuantileLasso(
            alpha=0.05,
            max_iter=200,
            p_tol=1e-4,
        ),
        estimator_parameter_space={
            "alpha": FloatRange(min_value=0.01, max_value=0.3, log_scale=True),
            "max_iter": IntRange(min_value=100, max_value=500),
            "p_tol": FloatRange(min_value=1e-5, max_value=1e-3, log_scale=True),
        },
    ),
    # Ensemble estimators - added directly to the registry
    PENS_NAME: EstimatorConfig(
        estimator_name=PENS_NAME,
        estimator_instance=PointEnsembleEstimator(
            estimators=[
                deepcopy(
                    GradientBoostingRegressor(
                        learning_rate=0.1,
                        n_estimators=25,
                        min_samples_split=3,
                        min_samples_leaf=3,
                        max_depth=2,
                        subsample=0.9,
                    )
                ),
                deepcopy(
                    KNeighborsRegressor(
                        n_neighbors=5,
                        weights="distance",
                    )
                ),
            ],
            weighting_strategy="linear_stack",
            cv=3,
        ),
        estimator_parameter_space={
            "weighting_strategy": CategoricalRange(choices=["uniform", "linear_stack"]),
        },
    ),
    SFQENS_NAME: EstimatorConfig(
        estimator_name=SFQENS_NAME,
        estimator_instance=QuantileEnsembleEstimator(
            estimators=[
                deepcopy(
                    QuantileForest(
                        n_estimators=25,
                        max_depth=5,
                        max_features=0.8,
                        min_samples_split=2,
                        bootstrap=True,
                    )
                ),
                deepcopy(
                    QuantileKNN(
                        n_neighbors=5,
                    )
                ),
            ],
            weighting_strategy="linear_stack",
            cv=3,
        ),
        estimator_parameter_space={
            "weighting_strategy": CategoricalRange(choices=["uniform", "linear_stack"]),
        },
    ),
    MFENS_NAME: EstimatorConfig(
        estimator_name=MFENS_NAME,
        estimator_instance=QuantileEnsembleEstimator(
            estimators=[
                deepcopy(
                    QuantileLightGBM(
                        learning_rate=0.1,
                        n_estimators=20,
                        max_depth=2,
                        min_child_samples=5,
                        subsample=0.8,
                        colsample_bytree=0.7,
                        reg_alpha=0.1,
                        reg_lambda=0.1,
                        min_child_weight=3,
                    )
                ),
                deepcopy(
                    QuantileLasso(
                        alpha=0.05,
                        max_iter=200,
                        p_tol=1e-4,
                    )
                ),
            ],
            weighting_strategy="linear_stack",
            cv=3,
        ),
        estimator_parameter_space={
            "weighting_strategy": CategoricalRange(choices=["uniform", "linear_stack"]),
        },
    ),
}
