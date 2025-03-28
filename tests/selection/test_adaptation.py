import numpy as np
import pytest
from sklearn.linear_model import LinearRegression
from confopt.selection.adaptation import DtACI

COVERAGE_TOLERANCE: float = 0.03


def check_breach(alpha_level, y_pred, y_test, cal_res):
    quantile = np.quantile(cal_res, 1 - alpha_level)
    lower = y_pred - quantile
    upper = y_pred + quantile
    return int(not (lower <= y_test <= upper))


@pytest.mark.parametrize("target_alpha", [0.1, 0.2, 0.5, 0.8, 0.9])
def test_regression_conformal_adaptation(linear_data_drift, target_alpha):
    dtaci = DtACI(alpha=target_alpha, gamma_values=[0.01, 0.05])

    initial_window = 30
    no_adapt_breaches = []
    dtaci_breaches = []

    X, y = linear_data_drift

    for i in range(initial_window, len(X) - 1):
        X_past = X[: i - 1]
        y_past = y[: i - 1]
        X_test = X[i].reshape(1, -1)
        y_test = y[i]

        n_cal = max(int(len(X_past) * 0.3), 5)
        X_train, X_cal = X_past[:-n_cal], X_past[-n_cal:]
        y_train, y_cal = y_past[:-n_cal], y_past[-n_cal:]

        model = LinearRegression()
        model.fit(X_train, y_train)
        y_cal_pred = model.predict(X_cal)
        cal_residuals = np.abs(y_cal - y_cal_pred)
        y_test_pred = model.predict(X_test)[0]

        residual = np.abs(y_test - y_test_pred)
        beta_t = np.mean(cal_residuals >= residual)
        dtaci.update(beta=beta_t)

        no_adapt_breaches.append(
            check_breach(target_alpha, y_test_pred, y_test, cal_residuals)
        )
        dtaci_breaches.append(
            check_breach(dtaci.alpha_t, y_test_pred, y_test, cal_residuals)
        )

    no_adapt_coverage = 1 - np.mean(no_adapt_breaches)
    dtaci_coverage = 1 - np.mean(dtaci_breaches)
    target_coverage = 1 - target_alpha

    no_adapt_error = abs(no_adapt_coverage - target_coverage)
    dtaci_error = abs(dtaci_coverage - target_coverage)

    assert abs(dtaci_coverage - target_coverage) < COVERAGE_TOLERANCE

    assert dtaci_error <= no_adapt_error
