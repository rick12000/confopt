<div align="center">
  <img src="assets/logo.png" alt="ConfOpt Logo" width="450"/>
</div>

<br>

<div align="center">

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![PyPI version](https://badge.fury.io/py/confopt.svg)](https://badge.fury.io/py/confopt)
[![PyPI downloads](https://img.shields.io/pypi/dm/confopt.svg)](https://pypi.org/project/confopt/)
[![Python versions](https://img.shields.io/pypi/pyversions/confopt.svg)](https://pypi.org/project/confopt/)
[![Build Status](https://github.com/rick12000/confopt/workflows/CI/badge.svg)](https://github.com/rick12000/confopt/actions)
[![arXiv](https://img.shields.io/badge/arXiv-ACHO-cyan)](https://doi.org/10.48550/arXiv.2207.03017)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://confopt.readthedocs.io/)

</div>

---

Built for machine learning practitioners who need both architecture flexibility and statistical rigor, **ConfOpt** delivers superior optimization performance through conformal uncertainty quantification and a wide selection of surrogate models.

## 📦 Installation

Install ConfOpt from PyPI using pip:

```bash
pip install confopt
```

For the latest development version:

```bash
git clone https://github.com/rick12000/confopt.git
cd confopt
pip install -e .
```

## 🎯 Getting Started

The example below shows how to optimize hyperparameters for a RandomForest classifier.
Just define your objective function, specify the search space, and let ConfOpt handle the rest:

```python
from confopt.tuning import ConformalTuner
from confopt.wrapping import IntRange, FloatRange, CategoricalRange
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Define your objective function
def objective_function(configuration):
    X, y = load_wine(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=configuration['n_estimators'],
        max_features=configuration['max_features'],
        criterion=configuration['criterion'],
        random_state=42
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    return accuracy_score(y_test, predictions)

# Define search space
search_space = {
    'n_estimators': IntRange(50, 200),
    'max_features': FloatRange(0.1, 1.0),
    'criterion': CategoricalRange(['gini', 'entropy', 'log_loss'])
}

# Create and run tuner
tuner = ConformalTuner(
    objective_function=objective_function,
    search_space=search_space,
    metric_optimization='maximize'
)
tuner.tune(max_searches=50, n_random_searches=10)

# Get results
best_params = tuner.get_best_params()
best_score = tuner.get_best_value()

print(f"Best accuracy: {best_score:.4f}")
print(f"Best parameters: {best_params}")
```

For detailed examples and explanations see the [documentation](https://confopt.readthedocs.io/).

## 📚 Documentation

The documentation provides everything you need to get started with ConfOpt or contribute to its codebase:

### **[Getting Started Guide](https://confopt.readthedocs.io/en/latest/getting_started.html)**
Complete tutorials for classification and regression tasks with step-by-step explanations.

### **[Examples](https://confopt.readthedocs.io/en/latest/basic_usage.html)**
- **[Classification Example](https://confopt.readthedocs.io/en/latest/basic_usage/classification_example.html)**: RandomForest hyperparameter tuning on a classification task.
- **[Regression Example](https://confopt.readthedocs.io/en/latest/basic_usage/regression_example.html)**: RandomForest hyperparameter tuning on a regression task.

### **[Advanced Usage](https://confopt.readthedocs.io/en/latest/advanced_usage.html)**
Custom acquisition functions, samplers, and warm starting.

### **[API Reference](https://confopt.readthedocs.io/en/latest/api_reference.html)**
Complete reference for main classes, methods, and parameters.

### **[Developer Resources](https://confopt.readthedocs.io/en/latest/architecture.html)**
- **[Architecture Overview](https://confopt.readthedocs.io/en/latest/architecture.html)**: System design and component interactions.
- **[Components Guide](https://confopt.readthedocs.io/en/latest/components.html)**: Deep dive into modules and mechanics.

## 🤝 Contributing

TBI

## 🔬 Theory

ConfOpt implements surrogate models and acquisition functions from the following papers:

> **Adaptive Conformal Hyperparameter Optimization**
> [arXiv, 2022](https://doi.org/10.48550/arXiv.2207.03017)

> **[Optimizing Hyperparameters with Conformal Quantile Regression]**
> [PMLR, 2023](https://proceedings.mlr.press/v202/salinas23a/salinas23a.pdf)

## 📈 Benchmarks

TBI

## 📄 License

[Apache License 2.0](https://github.com/rick12000/confopt/blob/main/LICENSE).

---

<div align="center">
  <strong>Ready to take your hyperparameter optimization to the next level?</strong><br>
  <a href="https://confopt.readthedocs.io/en/latest/getting_started.html">Get Started</a> |
  <a href="https://confopt.readthedocs.io/en/latest/basic_usage.html">Examples</a> |
  <a href="https://confopt.readthedocs.io/en/latest/api_reference.html">API Docs</a> |
</div>
