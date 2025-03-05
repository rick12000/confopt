from typing import List, Dict

# Reference names of search estimator architectures:
QGBM_NAME: str = "qgbm"
QRF_NAME: str = "qrf"
KR_NAME: str = "kr"
GP_NAME: str = "gp"
GBM_NAME: str = "gbm"
LGBM_NAME: str = "lgbm"
KNN_NAME: str = "knn"
RF_NAME: str = "rf"
DNN_NAME: str = "dnn"
QKNN_NAME: str = "qknn"
QL_NAME: str = "ql"
QLGBM_NAME: str = "qlgbm"

# Reference names of quantile regression estimators:
QUANTILE_ESTIMATOR_ARCHITECTURES: List[str] = [
    QGBM_NAME,
    QLGBM_NAME,
]

POINT_ESTIMATOR_ARCHITECTURES: List[str] = [
    KR_NAME,
    GP_NAME,
    GBM_NAME,
    LGBM_NAME,
    KNN_NAME,
    RF_NAME,
]

# Reference names of estimators that don't need their input data normalized:
NON_NORMALIZING_ARCHITECTURES: List[str] = [
    RF_NAME,
    GBM_NAME,
    QRF_NAME,
    QGBM_NAME,
    QLGBM_NAME,
    LGBM_NAME,
]

# Lookup of metrics to their direction of optimization (direct
# for performance metrics, inverse for loss or error metrics)
METRIC_PROPORTIONALITY_LOOKUP: Dict[str, str] = {
    "accuracy_score": "direct",
    "log_loss": "inverse",
    "mean_squared_error": "inverse",
}
