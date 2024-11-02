import pandas as pd
import json

from mleco.constants import DIR2DATA


def load_dataset(dataset_name: str, description: str = False) -> pd.DataFrame:
    """
    Load a dataset from the data directory.

    Parameters
    ----------
    dataset_name : str
        The name of the dataset to load.

    Returns
    -------
    pd.DataFrame
        The dataset.
    """
    if description:
        with open(DIR2DATA / f"{dataset_name}.json", "r", encoding="utf-8") as f:
            dataset_info = json.load(f)
        print(dataset_info)
    return dataset_info, pd.read_csv(DIR2DATA / f"{dataset_name}.csv")
