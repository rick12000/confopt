from typing import List

# Reference names of search estimator architectures:
QGBM_NAME: str = "qgbm"
QRF_NAME: str = "qrf"
KR_NAME: str = "kr"
GP_NAME: str = "gp"
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
    # GP_NAME,
    GBM_NAME,
    LGBM_NAME,
    KNN_NAME,
    RF_NAME,
    PENS_NAME,
]
