import warnings
from typing import Dict, Optional
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import joblib

from pathlib import Path
from script_utils import SPLIT_SEEDS, DATA_IDS


def load_california_housing(data_id: str, data_path: Path, random_state: int, frac: float = 0.8) -> dict:
    raw_dir = data_path / "raw"

    data_fname = raw_dir / "california-housing.csv"

    if not data_fname.exists():
        data_fname.parent.mkdir(parents=True, exist_ok=True)

        _data: pd.DataFrame = fetch_california_housing(as_frame=True).frame
        _data.to_csv(data_fname, index=False)
    else:
        _data = pd.read_csv(data_fname)

    target_col = "MedHouseVal"
    target_pos_val = None

    # Place the target col as the last column
    _data = pd.concat([_data.drop(target_col, axis=1), _data[target_col]], axis=1)
    train_data, test_data = train_test_split(_data, train_size=frac, random_state=random_state)

    return dict(
        data_id=data_id,
        data=_data,
        frac=frac,
        seed=random_state,
        train=train_data,
        test=test_data,
        target_col=target_col,
        target_pos_val=target_pos_val
    )


def load_heloc(data_id: str, data_path: Path, random_state: int, frac: float = 0.8) -> dict:
    raw_dir = data_path / "raw"
    _data = pd.read_csv(raw_dir / "heloc_dataset_v1.csv")

    target_col = "RiskPerformance"
    target_pos_val = "Good"

    # Place the target col as the last column
    _data = pd.concat([_data.drop(target_col, axis=1), _data[target_col]], axis=1)

    train_data, test_data = train_test_split(_data, train_size=frac, random_state=random_state, stratify=_data[target_col])

    return dict(
        data_id=data_id,
        data=_data,
        frac=frac,
        seed=random_state,
        train=train_data,
        test=test_data,
        target_col=target_col,
        target_pos_val=target_pos_val
    )


def load_adult_income(data_id: str, data_path: Path, random_state: int, frac: float = 0.8) -> dict:
    raw_dir = data_path / "raw"

    # Extract column headers
    _text = (raw_dir / "adult.names").read_text()
    col_names = [l.split(":")[0] for l in _text.split("\n\n")[-1].split("\n") if l.strip()]
    col_names.append("income")

    # Load the data
    train_data = pd.read_csv(raw_dir / "adult.data", header=None)
    train_data.columns = col_names

    # Normalize the target variable
    train_data["income"] = train_data["income"].str.replace(".", "", regex=False).str.strip()

    _data = train_data

    target_col = "income"
    target_pos_val = ">50K"

    # Place the target col as the last column
    _data = pd.concat([_data.drop(target_col, axis=1), _data[target_col]], axis=1)

    train_data, test_data = train_test_split(_data, train_size=frac, random_state=random_state, stratify=_data[target_col])

    # test_data = pd.read_csv(RAW_DATA_DIR / data_id / "adult.test", header=None, skiprows=1)
    # test_data.columns = col_names
    # test_data["income"] = test_data["income"].str.replace(".", "", regex=False).str.strip()

    # _data = pd.concat([train_data, test_data])

    return dict(
        data_id=data_id,
        data=_data,
        frac=frac,
        seed=random_state,
        train=train_data,
        test=test_data,
        target_col=target_col,
        target_pos_val=target_pos_val
    )


def load_travel_customers(data_id: str, data_path: Path, random_state: int, frac: float = 0.8) -> dict:
    raw_dir = data_path / "raw"

    target_col = "Target"
    target_pos_val = 1

    _data = pd.read_csv(raw_dir / "Customertravel.csv")

    # Place the target col as the last column
    _data = pd.concat([_data.drop(target_col, axis=1), _data[target_col]], axis=1)

    train_data, test_data = train_test_split(_data, train_size=frac, random_state=random_state, stratify=_data[target_col])

    return dict(
        data_id=data_id,
        data=_data,
        frac=frac,
        seed=random_state,
        train=train_data,
        test=test_data,
        target_col=target_col,
        target_pos_val=target_pos_val
    )


def load_split_save_data(data_id: str, data_path: Path, random_state: int, frac: float = 0.8, return_payload: bool = False) -> Optional[dict]:
    assert data_id in DATA_IDS

    part_path = data_path / f"split_{random_state}" / f"{data_id}_seed-{random_state}.pkl"
    part_path.parent.mkdir(parents=True, exist_ok=True)

    if part_path.exists():
        print(f"Data split exists: {part_path}", flush=True)
        return

    if data_id == "california-housing":
        payload = load_california_housing(data_id, data_path, random_state, frac)
    elif data_id == "heloc":
        payload = load_heloc(data_id, data_path, random_state, frac)
    elif data_id == "adult-income":
        payload = load_adult_income(data_id, data_path, random_state, frac)
    elif data_id == "travel-customers":
        payload = load_travel_customers(data_id, data_path, random_state, frac)
    else:
        warnings.warn(f"The data_id ({data_id}) has no data loader implementation yet. Skipping...")
        return

    assert not any(payload["data"].index.duplicated()), "The data has non-unique index values. Please reindex the data."

    # Perform test here.
    # Check if the keys in the payload are complete.
    expected_keys = ["data_id", "data", "frac", "seed", "train", "test", "target_col", "target_pos_val"]
    assert len(payload) == len(expected_keys)
    for k in expected_keys:
        assert k in payload

    joblib.dump(payload, part_path)

    if return_payload:
        return payload


# Fraction of the training data
DATA_FRAC = 0.8

# Set the input dir in data directory as the DATA_DIR
DATA_DIR = Path(__file__).parent.parent / "data" / "input"
assert DATA_DIR.exists() and DATA_DIR.is_dir()

data_id_path: Dict[str, Path] = {}

for p in DATA_DIR.glob("*"):
    data_id_path[p.name] = p


for data_id in data_id_path:
    if data_id not in DATA_IDS:
        warnings.warn(f"The data_id {data_id} is not registered. Skipping...")
        continue

    for seed in SPLIT_SEEDS:
        print(f"Generating data splits for {data_id} using seed {seed}...", flush=True)
        load_split_save_data(
            data_id=data_id,
            data_path=data_id_path[data_id],
            random_state=seed,
            frac=DATA_FRAC,
            return_payload=False
        )
