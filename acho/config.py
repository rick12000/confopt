from typing import List, Dict

# Model reference names:
QGBM_NAME: str = "qgbm"
QRF_NAME: str = "qrf"
KR_NAME: str = "kr"
GP_NAME: str = "gp"
GBM_NAME: str = "gbm"
KNN_NAME: str = "knn"
RF_NAME: str = "rf"
DNN_NAME: str = "dnn"

# Reference names of quantile regression models:
QUANTILE_MODEL_TYPES: List[str] = [QGBM_NAME, QRF_NAME]

# Reference names of models that don't need their input data normalized:
NON_NORMALIZING_MODEL_TYPES: List[str] = [RF_NAME, GBM_NAME, QRF_NAME, QGBM_NAME]

# Lookup of metrics to their direction of optimization (direct
# for performance metrics, inverse for loss or error metrics)
METRIC_PROPORTIONALITY_LOOKUP: Dict[str, str] = {
    "accuracy_score": "direct",
    "log_loss": "inverse",
    "mean_squared_error": "inverse",
}
