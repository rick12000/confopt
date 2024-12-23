import logging
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def get_perceptron_layers(
    n_layers_grid: List[int],
    layer_size_grid: List[int],
    random_seed: Optional[int] = None,
) -> List[Tuple]:
    """
    Construct list of randomly sampled multilayer perceptron
    configuration tuples.

    Each tuple is randomly constructed given a grid of layer
    counts and a grid of layer sizes. A single tuple is just
    a sequence of layer sizes, eg. (10, 20, 60, 20, 10), for
    some diamond shaped perceptron.

    Parameters
    ----------
    n_layers_grid :
        List of potential layer counts determining how many
        perceptron layers there can be in a configuration tuple.
    layer_size_grid :
        List of potential perceptron layer sizes from which
        to construct a configuration tuple.
    random_seed :
        Random seed.

    Returns
    -------
    layer_tuples :
        Collection of tuples, each of which contains the layer sizes
        determining the architecture of a multilayer perceptron.
    """
    random.seed(random_seed)
    np.random.seed(random_seed)

    layer_tuples = []
    # Hard coded:
    discretization = 1000
    for _ in range(discretization):
        tuple_len = random.choice(n_layers_grid)
        layer_tuple = ()
        for _ in range(tuple_len):
            layer_tuple = layer_tuple + (random.choice(layer_size_grid),)
        layer_tuples.append(layer_tuple)

    return layer_tuples


def get_tuning_configurations(
    parameter_grid: Dict, n_configurations: int, random_state: Optional[int] = None
) -> List[Dict]:
    """
    Randomly sample list of unique hyperparameter configurations.

    Each configuration is constructed from a broader parameter grid of
    possible parameter values.

    Parameters
    ----------
    parameter_grid :
        Dictionary of parameter names to possible ranged parameter values.
    n_configurations :
        Number of desired configurations to randomly construct from the
        raw parameter grid.
    random_state :
        Random seed.

    Returns
    -------
    configurations :
        Unique randomly constructed hyperparameter configurations.
    """
    random.seed(random_state)

    configurations = []
    for _ in range(n_configurations):
        configuration = {}
        for parameter_name in parameter_grid:
            parameter_value = random.choice(parameter_grid[parameter_name])
            configuration[parameter_name] = parameter_value
        if configuration not in configurations:
            configurations.append(configuration)

    return configurations


def tabularize_configurations(configurations: List[Dict]) -> pd.DataFrame:
    """
    Transform list of configuration dictionaries into tabular training data.

    Configurations are type transformed, one hot encoded and wrapped in a
    pandas dataframe to enable regression tasks.

    Parameters
    ----------
    configurations :
        List of hyperparameter configurations to tabularize.

    Returns
    -------
    tabularized_configurations :
        Tabularized hyperparameter configurations (hyperparameter names
        as columns and hyperparameter values as rows).
    """
    logger.debug(f"Received {len(configurations)} configurations to tabularize.")

    # Get maximum length of any list or tuple parameter in configuration (this is
    # important for configuration inputs where lists and tuples can be of variable
    # length depending on the parameter values passed):
    max_tuple_or_list_lens_per_parameter = {}
    for configuration in configurations:
        for parameter_name, parameter in configuration.items():
            if isinstance(parameter, (tuple, list)):
                if parameter_name not in max_tuple_or_list_lens_per_parameter:
                    max_tuple_or_list_lens_per_parameter[parameter_name] = len(
                        parameter
                    )
                elif (
                    len(parameter)
                    > max_tuple_or_list_lens_per_parameter[parameter_name]
                ):
                    max_tuple_or_list_lens_per_parameter[parameter_name] = len(
                        parameter
                    )

    # Create new configurations with flattened list/tuple parameter inputs:
    expanded_configurations = []
    for configuration in configurations:
        expanded_record = {}
        for parameter_name, parameter in configuration.items():
            if isinstance(parameter, (tuple, list)):
                for i in range(max_tuple_or_list_lens_per_parameter[parameter_name]):
                    if i < len(parameter):
                        expanded_record[f"{parameter_name}_{i}"] = parameter[i]
                    else:
                        # Below assumes that missing dimensions are equivalent to 0 entries
                        # (This works for eg. for the tuple layer sizes of an MLPRegressor)
                        expanded_record[f"{parameter_name}_{i}"] = 0
            else:
                expanded_record[parameter_name] = parameter

        expanded_configurations.append(expanded_record)

    logger.debug(
        f"Expanded configuration list's first element: {expanded_configurations[0]}"
    )

    # NOTE: None values are converted to np.nan during pandas ingestion.
    tabularized_configurations = pd.DataFrame(expanded_configurations).replace(
        {np.nan: None}
    )

    categorical_columns = []
    column_types = list(tabularized_configurations.dtypes)
    # Loop through each column type in the tabular data and wherever an
    # object column is present (due to None parameter values being mixed
    # in with other types) check whether the column is a None + str mix
    # or a None + float/int mix.
    # For inference purposes, the None values in an otherwise str filled
    # column should be considered another category, and are thus set to
    # "None", while in the None + numericals case they are assumed to mean
    # zero (this last conversion is not accurate for all parameters,
    # eg. the maximum number of leaves in a random forest algorithm,
    # TODO: consider turning the None + numerical columns to categoricals).
    for original_column_idx, column_type in enumerate(column_types):
        if str(column_type) == "object":
            types = []
            column_name = tabularized_configurations.columns[original_column_idx]
            for element in list(tabularized_configurations[column_name]):
                if type(element) not in types:
                    types.append(type(element))
            if str in types:
                tabularized_configurations[column_name] = (
                    tabularized_configurations[column_name]
                    .infer_objects(copy=False)
                    .fillna("None")
                )
                categorical_columns.append(column_name)
            elif float in types or int in types:
                tabularized_configurations[column_name] = (
                    tabularized_configurations[column_name]
                    .infer_objects(copy=False)
                    .fillna(0)
                )
            else:
                raise ValueError(
                    "Type other than 'str', 'int', 'float' was detected in 'None' handling."
                )

    # One hot encode categorical columns (parameters) in tabularized dataset:
    for column_name in categorical_columns:
        tabularized_configurations = pd.concat(
            [
                tabularized_configurations,
                pd.get_dummies(tabularized_configurations[column_name]),
            ],
            axis=1,
        )
        tabularized_configurations = tabularized_configurations.drop(
            [column_name], axis=1
        )

    logger.debug(
        f"Tabularized configuration dataframe shape: {tabularized_configurations.shape}"
    )

    return tabularized_configurations
