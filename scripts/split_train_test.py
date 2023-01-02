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
    assert _data.isnull().sum().sum() == 0, "We expect that the HELOC data has no NA values."

    # Remove observations that are likely to have all variables missing (imputed with -9)
    _data = _data.replace(-9, None).dropna(thresh=23).fillna(-9)
    assert _data.isnull().sum().sum() == 0, "Again, we expect that the HELOC data has no NA values."

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


def load_customer_personality(data_id: str, data_path: Path, random_state: int, frac: float = 0.8) -> dict:
    assert data_id == "customer-personality"

    raw_dir = data_path / "raw"

    target_col = "Response"
    target_pos_val = 1

    _data = pd.read_csv(raw_dir / "marketing_campaign.csv", sep="\t")
    _data.drop(["ID", "Z_CostContact", "Z_Revenue"], axis=1, inplace=True)
    _data.dropna(subset=["Income"], inplace=True)

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


def load_mobile_price(data_id: str, data_path: Path, random_state: int, frac: float = 0.8) -> dict:
    assert data_id == "mobile-price"

    raw_dir = data_path / "raw"

    target_col = "price_range"
    target_pos_val = [0, 1, 2, 3]

    _data = pd.read_csv(raw_dir / "train.csv")

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


def load_oil_spill(data_id: str, data_path: Path, random_state: int, frac: float = 0.8) -> dict:
    assert data_id == "oil-spill"

    raw_dir = data_path / "raw"

    target_col = "target"
    target_pos_val = 1

    _data = pd.read_csv(raw_dir / "oil_spill.csv")
    _data.drop(["f_23"], axis=1, inplace=True)

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


def load_predict_diabetes(data_id: str, data_path: Path, random_state: int, frac: float = 0.8) -> dict:
    assert data_id == "predict-diabetes"

    raw_dir = data_path / "raw"

    target_col = "Outcome"
    target_pos_val = 1

    _data = pd.read_csv(raw_dir / "diabetes.csv")

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

    if data_id == "california-housing": load_func = load_california_housing
    elif data_id == "heloc": load_func = load_heloc
    elif data_id == "adult-income": load_func = load_adult_income
    elif data_id == "travel-customers": load_func = load_travel_customers
    elif data_id == "customer-personality": load_func = load_customer_personality
    elif data_id == "mobile-price": load_func = load_mobile_price
    elif data_id == "oil-spill": load_func = load_oil_spill
    elif data_id == "predict-diabetes": load_func = load_predict_diabetes
    else:
        warnings.warn(f"The data_id ({data_id}) has no data loader implementation yet. Skipping...")
        return

    payload = load_func(data_id, data_path, random_state, frac)

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
