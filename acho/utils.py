import numpy as np
import random
from typing import Dict, List, Optional, Any
import pandas as pd
import logging
import math

logger = logging.getLogger(__name__)


def get_perceptron_layers(
    n_layers_grid: List[int],
    layer_size_grid: List[int],
    random_seed: Optional[int] = None,
) -> List:
    random.seed(random_seed)
    np.random.seed(random_seed)

    layer_tuples = []
    discretization = 1000
    for _ in range(discretization):
        tuple_len = random.choice(n_layers_grid)
        layer_tuple = ()
        for _ in range(0, tuple_len):
            layer_tuple = layer_tuple + (random.choice(layer_size_grid),)
        layer_tuples.append(layer_tuple)

    return layer_tuples


def get_tuning_configurations(
    parameter_grid: Dict, n_configurations: int, random_state: Optional[int] = None
) -> List[Dict]:
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
    logger.debug(f"Received {len(configurations)} configurations to tabularize.")

    # Get maximum length of any list or tuple parameter in configuration:
    max_tuple_or_list_lens_per_parameter = {}
    for record in configurations:
        for parameter_name, parameter in record.items():
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

    # Create new list of dicts with expanded list or tuple parameter inputs:
    configuration_list = []
    for record in configurations:
        expanded_record = {}
        for parameter_name, parameter in record.items():
            if isinstance(parameter, (str, int, float)):
                expanded_record[parameter_name] = parameter
            elif isinstance(parameter, (tuple, list)):
                for i in range(max_tuple_or_list_lens_per_parameter[parameter_name]):
                    if i < len(parameter):
                        expanded_record[f"{parameter_name}_{i}"] = parameter[i]
                    else:
                        # NOTE: Below this is a custom default that works for things like layers of a neural
                        # network (this tuple decomposition is built primarily for sklearn MLP models), where
                        # a missing value of a variable tuple means that element is not present (or 0).
                        expanded_record[f"{parameter_name}_{i}"] = 0

        configuration_list.append(expanded_record)

    logger.debug(
        f"Expanded configuration list's first element: {configuration_list[0]}"
    )

    # TODO: This doesn't seem to work as well for nonetypes and nans in practice, revisit
    types_dict = {}
    mixed_keys = []
    for configuration in configuration_list:
        for k, v in configuration.items():
            if k not in set(mixed_keys):
                if isinstance(v, str):
                    types_dict[k] = "string"
                elif isinstance(v, int):
                    if math.isnan(v):
                        types_dict[k] = "object"
                        mixed_keys.append(k)
                    else:
                        types_dict[k] = "int32"
                elif isinstance(v, float):
                    if math.isnan(v):
                        types_dict[k] = "object"
                        mixed_keys.append(k)
                    else:
                        types_dict[k] = "float64"
                elif v is None:
                    types_dict[k] = "object"
                    mixed_keys.append(k)
                else:
                    types_dict[k] = "object"
                    mixed_keys.append(k)

    tabularized_configurations = (
        pd.DataFrame(configuration_list).fillna(0).astype(types_dict)
    )

    # One hot encode string variables:
    for column_name in tabularized_configurations.columns:
        if tabularized_configurations[column_name].dtype == "string":
            logger.debug(
                f"Detected string typed {column_name} column; one hot encoding..."
            )
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
