from confopt.adaptation import ACI  # , DtACI
import pytest


@pytest.mark.parametrize("breach", [True, False])
@pytest.mark.parametrize("alpha", [0.2, 0.8])
@pytest.mark.parametrize("gamma", [0.01, 0.1])
def test_update_adaptive_interval(breach, alpha, gamma):
    aci = ACI(alpha=alpha, gamma=gamma)
    stored_alpha = aci.alpha
    updated_alpha = aci.update(breach_indicator=breach)

    assert 0 < updated_alpha < 1
    if breach:
        assert updated_alpha <= alpha
    else:
        assert updated_alpha >= alpha

    assert stored_alpha == aci.alpha
