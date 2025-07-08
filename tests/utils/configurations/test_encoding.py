from confopt.utils.configurations.encoding import ConfigurationEncoder
from confopt.wrapping import IntRange, FloatRange, CategoricalRange


def test_configuration_encoder():
    """Test that ConfigurationEncoder properly encodes configurations"""
    # Create configurations with mixed parameter types
    configs = [
        {"numeric1": 1.0, "numeric2": 5, "cat1": "a", "cat2": True},
        {"numeric1": 2.0, "numeric2": 10, "cat1": "b", "cat2": False},
        {"numeric1": 3.0, "numeric2": 15, "cat1": "a", "cat2": True},
    ]

    # Define search space with categorical parameters
    search_space = {
        "numeric1": FloatRange(min_value=0.0, max_value=10.0),
        "numeric2": IntRange(min_value=0, max_value=20),
        "cat1": CategoricalRange(choices=["a", "b", "c"]),
        "cat2": CategoricalRange(choices=[True, False]),
    }

    # Test initialization
    encoder = ConfigurationEncoder(search_space)

    # Verify categorical mappings are created correctly
    assert "cat1" in encoder.categorical_mappings
    assert "cat2" in encoder.categorical_mappings

    # Test transformation
    df = encoder.transform(configs)

    # Check shape - should have columns for numeric1, numeric2, cat1_a, cat1_b, cat1_c, cat2_False, cat2_True
    assert df.shape[0] == 3  # 3 rows

    # Verify numeric columns are preserved
    assert "numeric1" in df.columns
    assert "numeric2" in df.columns

    # Check one-hot encoding worked correctly for string categorical values
    cat1_cols = [col for col in df.columns if col.startswith("cat1_")]
    assert (
        len(cat1_cols) == 3
    )  # "a", "b", and "c" (all possible values from search space)

    cat1_a_col = next(col for col in cat1_cols if "a" in col)
    cat1_b_col = next(col for col in cat1_cols if "b" in col)

    # First row has cat1="a", so a=1, b=0
    assert df.loc[0, cat1_a_col] == 1
    assert df.loc[0, cat1_b_col] == 0

    # Second row has cat1="b", so a=0, b=1
    assert df.loc[1, cat1_a_col] == 0
    assert df.loc[1, cat1_b_col] == 1

    # Check boolean categorical values
    cat2_cols = [col for col in df.columns if col.startswith("cat2_")]
    assert len(cat2_cols) == 2  # False and True mapped to 0 and 1

    # Boolean values get sorted as str representations: False -> 'False', True -> 'True'
    # When sorted: ['False', 'True'] -> cat2_0 for False, cat2_1 for True
    cat2_false_col = "cat2_0"
    cat2_true_col = "cat2_1"

    # First row has cat2=True, so False=0, True=1
    assert df.loc[0, cat2_true_col] == 1
    assert df.loc[0, cat2_false_col] == 0
