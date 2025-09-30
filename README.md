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

The example below shows how to optimize hyperparameters for a RandomForest classifier. You can find more examples in the [documentation](https://confopt.readthedocs.io/).

### Step 1: Import Required Libraries

```python
from confopt.tuning import ConformalTuner
from confopt.wrapping import IntRange, FloatRange, CategoricalRange
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
```
We import the necessary libraries for tuning and model evaluation. The `load_wine` function is used to load the wine dataset, which serves as our example data for optimizing the hyperparameters of the RandomForest classifier (the dataset is trivial and we can easily reach 100% accuracy, this is for example purposes only).

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
This function defines the objective we want to optimize. It loads the wine dataset, splits it into training and testing sets, and trains a RandomForest model using the provided configuration. The function returns test accuracy, which will be the objective value ConfOpt will optimize for.

### Step 3: Define the Search Space

```python
search_space = {
    'n_estimators': IntRange(min_value=50, max_value=200),
    'max_features': FloatRange(min_value=0.1, max_value=1.0),
    'criterion': CategoricalRange(choices=['gini', 'entropy', 'log_loss'])
}
```
Here, we specify the search space for hyperparameters. In this Random Forest example, this includes defining the range for the number of estimators, the proportion of features to consider when looking for the best split, and the criterion for measuring the quality of a split.

### Step 4: Create and Run the Tuner

```python
tuner = ConformalTuner(
    objective_function=objective_function,
    search_space=search_space,
    minimize=False
)
tuner.tune(max_searches=50, n_random_searches=10)
```
We initialize the `ConformalTuner` with the objective function and search space. The `tune` method then kickstarts hyperparameter search and finds the hyperparameters that maximize test accuracy.

### Step 5: Retrieve and Display Results

```python
best_params = tuner.get_best_params()
best_score = tuner.get_best_value()

print(f"Best accuracy: {best_score:.4f}")
print(f"Best parameters: {best_params}")
```
Finally, we retrieve the optimization's best parameters and test accuracy score and print them to the console for review.

For detailed examples and explanations see the [documentation](https://confopt.readthedocs.io/).

## 📚 Documentation

### **User Guide**
- **[Classification Example](https://confopt.readthedocs.io/en/latest/basic_usage/classification_example.html)**: RandomForest hyperparameter tuning on a classification task.
- **[Regression Example](https://confopt.readthedocs.io/en/latest/basic_usage/regression_example.html)**: RandomForest hyperparameter tuning on a regression task.

### **Developer Resources**
- **[Architecture Overview](https://confopt.readthedocs.io/en/latest/architecture.html)**: System design and module interactions.
- **[API Reference](https://confopt.readthedocs.io/en/latest/api_reference.html)**:
Complete reference for main classes, methods, and parameters.

## 📈 Benchmarks

<div align="center">
  <img src="https://raw.githubusercontent.com/rick12000/confopt/add-objective-search/assets/benchmark_results.png" alt="ConfOpt Logo" width="450"/>
</div>

**ConfOpt** is significantly better than plain old random search, but it also beats established tools like **Optuna** or traditional **Gaussian Processes**!

The above benchmark considers neural architecture search on complex image recognition datasets (JAHS-201) and neural network tuning on tabular classification datasets (LCBench-L).

For a fuller analysis of caveats and benchmarking results, refer to the latest methodological paper.

## 🔬 Theory

ConfOpt implements surrogate models and acquisition functions from the following papers:

> **Adaptive Conformal Hyperparameter Optimization**
> [arXiv, 2022](https://doi.org/10.48550/arXiv.2207.03017)

> **Optimizing Hyperparameters with Conformal Quantile Regression**
> [PMLR, 2023](https://proceedings.mlr.press/v202/salinas23a/salinas23a.pdf)

> **Enhancing Performance and Calibration in Quantile Hyperparameter Optimization**
> [arXiv, 2025](https://www.arxiv.org/abs/2509.17051)

## 🤝 Contributing

If you'd like to contribute, please email [r.doyle.edu@gmail.com](mailto:r.doyle.edu@gmail.com) with a quick summary of the feature you'd like to add and we can discuss it before setting up a PR!

If you want to contribute a fix relating to a new bug, first raise an [issue](https://github.com/rick12000/confopt/issues) on GitHub, then email [r.doyle.edu@gmail.com](mailto:r.doyle.edu@gmail.com) referencing the issue. Issues will be regularly monitored, only send an email if you want to contribute a fix.

## 📄 License

[MIT License](https://github.com/rick12000/confopt/blob/main/LICENSE)

---

<div align="center">
  <strong>Ready to take your hyperparameter optimization to the next level?</strong><br>
  <a href="https://confopt.readthedocs.io/en/latest/getting_started.html">Get Started</a> |
  <a href="https://confopt.readthedocs.io/en/latest/api_reference.html">API Docs</a>
</div>
