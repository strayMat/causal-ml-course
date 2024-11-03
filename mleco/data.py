import pandas as pd
import json

from mleco.constants import DIR2DATA


def list_datasets() -> None:
    """
    List the available datasets.
    """
    datasets = [f.stem for f in DIR2DATA.glob("*.csv")]
    print("\n".join(datasets))


def load_dataset(dataset_name: str, description: str = False) -> pd.DataFrame:
    """
    Load a dataset.
    Args:
        dataset_name (str): The name of the dataset to load.
        description (bool, optional): If True, prints the description of the dataset. Defaults to False.

    Returns:
        pd.DataFrame: The dataset.
    """
    if description:
        print_description(dataset_name)
    return pd.read_csv(DIR2DATA / f"{dataset_name}.csv")


def print_description(dataset_name: str) -> None:
    """
    Print the description of a dataset.

     Args:
         dataset_name (str): The name of the dataset.
    """
    with open(DIR2DATA / f"{dataset_name}.json", "r", encoding="utf-8") as f:
        dataset_info = json.load(f)
        breakpoint()
        dataset_schema = pd.DataFrame.from_dict(
            dataset_info.pop("schema"), orient="index"
        ).reset_index()
        dataset_schema.columns = ["Variable", "Description"]
        print(f"Description of the {dataset_name} dataset:")
        print(dataset_schema.to_markdown(index=False))
        print(json.dumps(dataset_info, indent=4))
