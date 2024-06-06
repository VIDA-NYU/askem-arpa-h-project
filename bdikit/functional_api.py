from enum import Enum
from os.path import join, dirname
from typing import Union, Type
import pandas as pd
from bdikit.mapping_algorithms.column_mapping.algorithms import (
    BaseColumnMappingAlgorithm,
    SimFloodAlgorithm,
    ComaAlgorithm,
    CupidAlgorithm,
    DistributionBasedAlgorithm,
    JaccardDistanceAlgorithm,
    GPTAlgorithm,
)

GDC_DATA_PATH = join(dirname(__file__), "./resource/gdc_table.csv")


class ColumnMappingMethod(Enum):
    SIMFLOOD = ("similarity_flooding", SimFloodAlgorithm)
    COMA = ("coma", ComaAlgorithm)
    CUPID = ("cupid", CupidAlgorithm)
    DISTRIBUTION_BASED = ("distribution_based", DistributionBasedAlgorithm)
    JACCARD_DISTANCE = ("jaccard_distance", JaccardDistanceAlgorithm)
    GPT = ("gpt", GPTAlgorithm)

    def __init__(
        self, method_name: str, method_class: Type[BaseColumnMappingAlgorithm]
    ):
        self.method_name = method_name
        self.method_class = method_class

    @staticmethod
    def get_class(method_name: str):
        methods = {
            method.method_name: method.method_class for method in ColumnMappingMethod
        }
        try:
            return methods[method_name]
        except KeyError:
            names = ", ".join(list(methods.keys()))
            raise ValueError(
                f"The {method_name} algorithm is not supported. "
                f"Supported algorithms are: {names}"
            )


def match_columns(
    source: pd.DataFrame,
    target: Union[str, pd.DataFrame] = "gdc",
    method: str = ColumnMappingMethod.SIMFLOOD.name,
) -> pd.DataFrame:
    """
    Performs schema mapping between the source table and the given target. The target
    either is a DataFrame or a string representing a standard data vocabulary.
    """
    if isinstance(target, str):
        target_table = _load_table_for_standard(target)
    else:
        target_table = target

    matcher_instance = ColumnMappingMethod.get_class(method)(source, target_table)
    matches = matcher_instance.map()

    return pd.DataFrame(matches.items(), columns=["source", "target"])


def _load_table_for_standard(name: str) -> pd.DataFrame:
    """
    Load the table for the given standard data vocabulary. Currently, only the
    GDC standard is supported.
    """
    if name == "gdc":
        return pd.read_csv(GDC_DATA_PATH)
    else:
        raise ValueError(f"The {name} standard is not supported")
