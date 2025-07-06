Bound-Based Acquisition Strategies
==================================

...existing overview and features sections...

...existing architecture section...

Mathematical Foundation and Derivation
-------------------------------------

Bound-based acquisition strategies utilize specific bounds from prediction intervals to make conservative or exploration-enhanced optimization decisions.

**Lower Confidence Bound Framework**

The Lower Confidence Bound (LCB) approach adapts Upper Confidence Bound (UCB) strategies for minimization problems:

.. math::
   \text{LCB}(x) = \mu(x) - \beta_t \sigma(x)

where :math:`\mu(x)$ is the point estimate, :math:`\sigma(x)$ quantifies uncertainty, and :math:`\beta_t$ controls exploration.

**Conformal Prediction Adaptation**

In conformal settings, we approximate this using:

1. **Point Estimate**: Use conformal predictor's point prediction :math:`\hat{y}(x)$
2. **Uncertainty Quantification**: Use interval width as uncertainty measure:

   .. math::
      w(x) = U_\alpha(x) - L_\alpha(x)

   where :math:`[L_\alpha(x), U_\alpha(x)]$ is the :math:`(1-\alpha)$-confidence interval.

3. **LCB Formulation**:

   .. math::
      \text{LCB}(x) = \hat{y}(x) - \beta_t w(x)

**Exploration Parameter Decay**

Theoretical guarantees require time-dependent exploration:

**Logarithmic Decay**:

.. math::
   \beta_t = \sqrt{\frac{c \log t}{t}}

This provides :math:`O(\sqrt{t \log t})$ regret bounds under appropriate conditions.

**Inverse Square Root Decay**:

.. math::
   \beta_t = \sqrt{\frac{c}{t}}

This offers more aggressive exploration decay with :math:`O(\sqrt{t})$ regret.

**Pessimistic Lower Bound**

The conservative approach uses only lower bounds:

.. math::
   \text{PLB}(x) = L_\alpha(x)

This provides risk-averse acquisition by assuming pessimistic scenarios within the confidence intervals.

**Interval Width Adaptation**

The confidence level :math:`\alpha$ can be adapted based on empirical coverage:

.. math::
   \alpha_{t+1} = \text{adapter}(\alpha_t, \beta_t)

where :math:`\beta_t$ is the observed coverage rate and the adapter maintains target coverage while optimizing interval efficiency.

**Decision Rule**

Select the candidate minimizing the acquisition function:

.. math::
   x^* = \arg\min_{x \in \mathcal{X}} \text{LCB}(x)

**Theoretical Properties**

Under regularity conditions, LCB achieves:

1. **Convergence**: :math:`\lim_{t \to \infty} \text{LCB}(x_t) = f(x^*)$
2. **Regret Bounds**: :math:`R_T = O(\sqrt{T \log T})$ for logarithmic decay
3. **Exploration-Exploitation Balance**: :math:`\beta_t \to 0$ ensures convergence while maintaining exploration

**Multi-Scale Intervals**

When multiple confidence levels are available, combine bounds:

.. math::
   \text{LCB}_{\text{multi}}(x) = \sum_{j=1}^k w_j L_{\alpha_j}(x)

where :math:`w_j$ are weights reflecting confidence in each interval level.

...existing content continues from "Bound-based methodology" section...
