import logging
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


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

    configurations_set = set()
    configurations = []

    for _ in range(n_configurations):
        configuration = {}
        for parameter_name in parameter_grid:
            parameter_value = random.choice(parameter_grid[parameter_name])
            configuration[parameter_name] = parameter_value

        # Convert the configuration dictionary to a tuple of sorted items
        configuration_tuple = tuple(sorted(configuration.items()))

        if configuration_tuple not in configurations_set:
            configurations_set.add(configuration_tuple)
            configurations.append(configuration)

    return configurations


def tabularize_configurations(
    searchable_configurations: List[Dict], searched_configurations: List[Dict]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Transform list of configuration dictionaries into tabular training data.

    Configurations are type transformed, one hot encoded and wrapped in a
    pandas dataframe to enable regression tasks.

    Parameters
    ----------
    searchable_configurations :
        List of hyperparameter configurations to tabularize.
    searched_configurations :
        List of hyperparameter configurations to tabularize.

    Returns
    -------
    tabularized_configurations :
        Tabularized hyperparameter configurations (hyperparameter names
        as columns and hyperparameter values as rows).
    """
    configurations = searchable_configurations + searched_configurations

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
    # NOTE: Order of list of dicts must be preserved during pandas ingestion, if
    # this ever changes in future versions, return to this:
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
                    # .infer_objects(copy=False)
                    .fillna("None")
                )
                categorical_columns.append(column_name)
            elif float in types or int in types:
                tabularized_configurations[column_name] = (
                    tabularized_configurations[column_name]
                    # .infer_objects(copy=False)
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

    tabularized_searchable_configurations = tabularized_configurations.iloc[
        : len(searchable_configurations), :
    ]
    tabularized_searched_configurations = tabularized_configurations.iloc[
        len(searchable_configurations) :, :
    ]

    return tabularized_searchable_configurations, tabularized_searched_configurations
