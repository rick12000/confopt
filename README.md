<div align="center">

  <img src="https://raw.githubusercontent.com/rick12000/confopt/add-objective-search/assets/logo.png" alt="ConfOpt Logo" width="300"/>
</div>

<br>

<div align="center">

[![Downloads](https://pepy.tech/badge/confopt)](https://pepy.tech/project/confopt)
[![Downloads](https://pepy.tech/badge/confopt/month)](https://pepy.tech/project/confopt)
[![PyPI version](https://badge.fury.io/py/confopt.svg)](https://badge.fury.io/py/confopt)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://confopt.readthedocs.io/)
[![Python versions](https://img.shields.io/pypi/pyversions/confopt.svg?color=brightgreen)](https://pypi.org/project/confopt/)
<!-- [![License](https://img.shields.io/badge/License-Apache_2.0-orange.svg)](https://opensource.org/licenses/Apache-2.0) -->

</div>

---

Built for machine learning practitioners requiring flexible and robust hyperparameter tuning, **ConfOpt** delivers superior optimization performance through conformal uncertainty quantification and a wide selection of surrogate models.

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

### Step 1: Import Required Libraries

```python
from confopt.tuning import ConformalTuner
from confopt.wrapping import IntRange, FloatRange, CategoricalRange
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
```
We import the necessary libraries for tuning and model evaluation. The `load_wine` function is used to load the wine dataset, which serves as our example data for optimizing the hyperparameters of the RandomForest classifier.

### Step 2: Define the Objective Function

```python
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
```
This function defines the objective we want to optimize. It loads the wine dataset, splits it into training and testing sets, and trains a RandomForest model using the provided configuration. The function returns the accuracy score, which serves as the optimization metric.

### Step 3: Define the Search Space

```python
search_space = {
    'n_estimators': IntRange(50, 200),
    'max_features': FloatRange(0.1, 1.0),
    'criterion': CategoricalRange(['gini', 'entropy', 'log_loss'])
}
```
Here, we specify the search space for hyperparameters. This includes defining the range for the number of estimators, the proportion of features to consider when looking for the best split, and the criterion for measuring the quality of a split.

### Step 4: Create and Run the Tuner

```python
tuner = ConformalTuner(
    objective_function=objective_function,
    search_space=search_space,
    minimize=False
)
tuner.tune(max_searches=50, n_random_searches=10)
```
We initialize the `ConformalTuner` with the objective function and search space. The tuner is then run to find the best hyperparameters by maximizing the accuracy score.

### Step 5: Retrieve and Display Results

```python
best_params = tuner.get_best_params()
best_score = tuner.get_best_value()

print(f"Best accuracy: {best_score:.4f}")
print(f"Best parameters: {best_params}")
```
Finally, we retrieve the best parameters and score from the tuning process and print them to the console for review.

For detailed examples and explanations see the [documentation](https://confopt.readthedocs.io/).

## 📚 Documentation

### **User Guide**
- **[Classification Example](https://confopt.readthedocs.io/en/latest/basic_usage/classification_example.html)**: RandomForest hyperparameter tuning on a classification task.
- **[Regression Example](https://confopt.readthedocs.io/en/latest/basic_usage/regression_example.html)**: RandomForest hyperparameter tuning on a regression task.

### **Developer Resources**
- **[Architecture Overview](https://confopt.readthedocs.io/en/latest/architecture.html)**: System design and module interactions.
- **[API Reference](https://confopt.readthedocs.io/en/latest/api_reference.html)**:
Complete reference for main classes, methods, and parameters.

## 🤝 Contributing

TBI

## 🔬 Theory

ConfOpt implements surrogate models and acquisition functions from the following papers:

> **Adaptive Conformal Hyperparameter Optimization**
> [arXiv, 2022](https://doi.org/10.48550/arXiv.2207.03017)

> **Optimizing Hyperparameters with Conformal Quantile Regression**
> [PMLR, 2023](https://proceedings.mlr.press/v202/salinas23a/salinas23a.pdf)

## 📈 Benchmarks

TBI

## 📄 License

[Apache License 2.0](https://github.com/rick12000/confopt/blob/main/LICENSE)

---

<div align="center">
  <strong>Ready to take your hyperparameter optimization to the next level?</strong><br>
  <a href="https://confopt.readthedocs.io/en/latest/getting_started.html">Get Started</a> |
  <a href="https://confopt.readthedocs.io/en/latest/api_reference.html">API Docs</a> |
</div>
