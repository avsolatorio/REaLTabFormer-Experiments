# Based from: https://github.com/Yura52/tabular-dl-num-embeddings/blob/88ffa9fe0f6bb0446464896937cef91fe944296d/bin/datasets.py

import argparse
import enum
import json
import math
import random
import shutil
import sys
import zipfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional, cast, List, Tuple, Dict, Union
from urllib.request import urlretrieve
from urllib.error import URLError

import catboost.datasets
import joblib
import numpy as np
import pandas as pd
import sklearn.datasets
import sklearn.utils
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

ArrayDict = Dict[str, np.ndarray]
ListDict = Dict[str, Union[list, str]]
Info = Dict[str, Any]

DATA_DIR = Path(__file__).parent.parent / 'data' / 'repo'
SEED = 0
CAT_MISSING_VALUE = '__nan__'

EXPECTED_FILES = {
    'abalone': ['dataset_187_abalone.arff', 'abalone_idx.json'],
    # 'adult': [],
    # 'buddy': [],
    'california': [],
    'cardio': ['cardio_train.csv', 'cardio_idx.json'],
    # Source: https://www.kaggle.com/shrutimechlearn/churn-modelling
    'churn2': ['Churn_Modelling.csv'],
    # 'default': [],
    'diabetes': ['dataset_37_diabetes.arff', 'diabetes_idx.json'],
    # Source: https://archive.ics.uci.edu/ml/machine-learning-databases/00363/Dataset.zip
    'fb-comments': ['Dataset.zip'],
    'gesture': [],
    'higgs-small': [],
    'house': [],
    'insurance': ['insurance.csv', 'insurance_idx.json'],
    # 'king': [],
    # 'miniboone': [],
    # 'wilt': [],
}
# EXPECTED_FILES['fb-c'] = EXPECTED_FILES['wd-fb-comments'] = EXPECTED_FILES[
#     'fb-comments'
# ]


class TaskType(enum.Enum):
    REGRESSION = 'regression'
    BINCLASS = 'binclass'
    MULTICLASS = 'multiclass'


# %%
def _set_random_seeds():
    random.seed(SEED)
    np.random.seed(SEED)


def _unzip(path: Path, members: Optional[List[str]] = None) -> None:
    with zipfile.ZipFile(path) as f:
        f.extractall(path.parent, members)


def _start(dirname: str) -> Tuple[Path, List[Path]]:
    print(f'>>> {dirname}')
    _set_random_seeds()
    dataset_dir = DATA_DIR / dirname
    expected_files = EXPECTED_FILES[dirname]
    if expected_files:
        assert dataset_dir.exists()
        assert set(expected_files) == set(x.name for x in dataset_dir.iterdir())
    else:
        assert not dataset_dir.exists()
        dataset_dir.mkdir()
    return dataset_dir, [dataset_dir / x for x in expected_files]


def _fetch_openml(data_id: int) -> sklearn.utils.Bunch:
    bunch = cast(
        sklearn.utils.Bunch,
        sklearn.datasets.fetch_openml(data_id=data_id, as_frame=True),
    )
    assert not bunch['categories']
    return bunch


def _get_sklearn_dataset(name: str) -> Tuple[np.ndarray, np.ndarray]:
    get_data = getattr(sklearn.datasets, f'load_{name}', None)
    if get_data is None:
        get_data = getattr(sklearn.datasets, f'fetch_{name}', None)
    assert get_data is not None, f'No such dataset in scikit-learn: {name}'
    return get_data(return_X_y=True)


def _encode_classification_target(y: np.ndarray) -> np.ndarray:
    assert not str(y.dtype).startswith('float')
    if str(y.dtype) not in ['int32', 'int64', 'uint32', 'uint64']:
        y = LabelEncoder().fit_transform(y)
    else:
        labels = set(map(int, y))
        if sorted(labels) != list(range(len(labels))):
            y = LabelEncoder().fit_transform(y)
    return y.astype(np.int64)


def _make_split(size: int, stratify: Optional[np.ndarray], n_parts: int) -> ArrayDict:
    # n_parts == 3:      all -> train & val & test
    # n_parts == 2: trainval -> train & val
    assert n_parts in (2, 3)
    all_idx = np.arange(size, dtype=np.int64)
    a_idx, b_idx = train_test_split(
        all_idx,
        test_size=0.2,
        stratify=stratify,
        random_state=SEED + (1 if n_parts == 2 else 0),
    )
    if n_parts == 2:
        return cast(ArrayDict, {'train': a_idx, 'val': b_idx})
    a_stratify = None if stratify is None else stratify[a_idx]
    a1_idx, a2_idx = train_test_split(
        a_idx, test_size=0.2, stratify=a_stratify, random_state=SEED + 1
    )
    return cast(ArrayDict, {'train': a1_idx, 'val': a2_idx, 'test': b_idx})


def _apply_split(data: ArrayDict, split: ArrayDict) -> Dict[str, ArrayDict]:
    return {k: {part: v[idx] for part, idx in split.items()} for k, v in data.items()}


def _save(
    dataset_dir: Path,
    name: str,
    task_type: TaskType,
    *,
    X_num: Optional[ArrayDict],
    X_cat: Optional[ArrayDict],
    y: ArrayDict,
    idx: Optional[ArrayDict],
    id_: Optional[str] = None,
    id_suffix: str = '--default',
    float_type: type = np.float32,
    cols: Optional[ListDict] = None,
) -> None:
    if id_ is not None:
        assert id_suffix == '--default'
    assert (
        X_num is not None or X_cat is not None
    ), 'At least one type of features must be presented.'
    if X_num is not None:
        X_num = {k: v.astype(float_type) for k, v in X_num.items()}
    if X_cat is not None:
        X_cat = {k: v.astype(str) for k, v in X_cat.items()}
    if idx is not None:
        idx = {k: v.astype(np.int64) for k, v in idx.items()}
    y = {
        k: v.astype(float_type if task_type == TaskType.REGRESSION else np.int64)
        for k, v in y.items()
    }
    if task_type != TaskType.REGRESSION:
        y_unique = {k: set(v.tolist()) for k, v in y.items()}
        assert y_unique['train'] == set(range(max(y_unique['train']) + 1))
        for x in ['val', 'test']:
            assert y_unique[x] <= y_unique['train']
        del x

    info = {
        'name': name,
        'id': (dataset_dir.name + id_suffix) if id_ is None else id_,
        'task_type': task_type.value,
        'n_num_features': (0 if X_num is None else next(iter(X_num.values())).shape[1]),
        'n_cat_features': (0 if X_cat is None else next(iter(X_cat.values())).shape[1]),
        'cols': cols
    } | {f'{k}_size': len(v) for k, v in y.items()}
    if task_type == TaskType.MULTICLASS:
        info['n_classes'] = len(set(y['train']))
    (dataset_dir / 'info.json').write_text(json.dumps(info, indent=4))

    for data_name in ['X_num', 'X_cat', 'y', 'idx']:
        data = locals()[data_name]
        if data is not None:
            for k, v in data.items():
                np.save(dataset_dir / f'{data_name}_{k}.npy', v)

    if cols:
        for dpart in ["train", "val", "test"]:
            target = locals()["y"]

            if dpart not in target:
                continue

            cat = locals()["X_cat"]
            num = locals()["X_num"]
            part_idx = locals()["idx"]

            data = []

            # Here we put the categorical variables first.
            # In the ddpm data, the numerical variables come first.
            if cat is not None and dpart in cat:
                cat = cat[dpart]
                data.append(pd.DataFrame(cat, columns=cols["cat"]))

            if num is not None and dpart in num:
                num = num[dpart]
                data.append(pd.DataFrame(num, columns=cols["num"]))

            if len(data) > 1:
                data = pd.concat(data, axis=1)
            else:
                data = data[0]

            data[cols["target"]] = target[dpart]

            if part_idx:
                data.index = part_idx[dpart]

            joblib.dump(data, dataset_dir / f'full_{dpart}.df.pkl')

    (dataset_dir / 'READY').touch()
    print('Done\n')


def _load_idx(data_id: str, idx_file: Path) -> ArrayDict:
    # Load idx data derived by reverse-engineering the
    # train-val-test split in the data dump below.
    # # conda activate tddpm
    # # cd $PROJECT_DIR
    # # wget "https://www.dropbox.com/s/rpckvcs3vx7j605/data.tar?dl=0" -O data.tar
    # # tar -xvf data.tar

    assert data_id in ["abalone", "cardio", "diabetes", "insurance"]
    idx = json.loads(Path(idx_file).read_text())
    idx = {k: np.array(v) for k, v in idx.items()}

    return cast(ArrayDict, idx)


# %%
def abalone():
    # Get the file here: https://www.kaggle.com/shrutimechlearn/churn-modelling
    dataset_dir, files = _start('abalone')
    target_col = "Class_number_of_rings"

    try:
        bunch = _fetch_openml(183)
        df = bunch["data"]
        y_all = bunch["target"].values.astype(np.int64)
    except URLError:
        # In case SSL is blocked by the firewall, fallback to local
        # copy of the file.
        data = arff.loadarff(files[0])
        df = pd.DataFrame(data[0])
        y_all = df.pop(target_col).values.astype(np.int64)

    # Make dataset consistent with the data used in https://github.com/rotot0/tab-ddpm.
    # We move the variables ['Gender', 'HasCrCard', 'IsActiveMember'] to the cat_columns
    # since the nunique == 2 for these variables.
    num_columns = [
        'Length',
        'Diameter',
        'Height',
        'Whole_weight',
        'Shucked_weight',
        'Viscera_weight',
        'Shell_weight',
    ]
    cat_columns = ['Sex']
    assert set(num_columns) | set(cat_columns) == set(df.columns.tolist())
    X_num_all = df[num_columns].astype(np.float64).values
    X_cat_all = df[cat_columns].astype(str).values

    cols = {
        "num": num_columns,
        "cat": cat_columns,
        "target": target_col
    }

    idx = _load_idx('abalone', files[1])

    _save(
        dataset_dir,
        'Abalone',
        TaskType.REGRESSION,
        **_apply_split(
            {'X_num': X_num_all, 'X_cat': X_cat_all, 'y': y_all},
            idx,
        ),
        idx=idx,
        float_type=np.float64,
        cols=cols
    )


def adult():
    dataset_dir, _ = _start('adult')
    target_col = "income"

    df_trainval, df_test = catboost.datasets.adult()
    df_trainval = cast(pd.DataFrame, df_trainval)
    df_test = cast(pd.DataFrame, df_test)
    assert (df_trainval.dtypes == df_test.dtypes).all()
    assert (df_trainval.columns == df_test.columns).all()
    categorical_mask = cast(pd.Series, df_trainval.dtypes != np.float64)

    num_columns = categorical_mask[~categorical_mask].index.tolist()
    cat_columns = [c for c in df_trainval.columns if c not in num_columns]
    if target_col in cat_columns:
        cat_columns.pop(target_col)

    def get_Xy(df: pd.DataFrame):
        y = (df.pop(target_col) == '>50K').values.astype('int64')
        return {
            'X_num': df.loc[:, ~categorical_mask].values,
            'X_cat': df.loc[:, categorical_mask].values,
            'y': y,
        }

    data = {k: {'test': v} for k, v in get_Xy(df_test).items()} | {
        'idx': {
            'test': np.arange(
                len(df_trainval), len(df_trainval) + len(df_test), dtype=np.int64
            )
        }
    }
    trainval_data = get_Xy(df_trainval)
    train_val_idx = _make_split(len(df_trainval), trainval_data['y'], 2)
    data['idx'].update(train_val_idx)
    for x in data['X_cat'].values():
        x[x == 'nan'] = CAT_MISSING_VALUE
    for k, v in _apply_split(trainval_data, train_val_idx).items():
        data[k].update(v)

    cols = {
        "num": num_columns,
        "cat": cat_columns,
        "target": target_col
    }

    _save(dataset_dir, 'Adult', TaskType.BINCLASS, **data, cols=cols)


def buddy():
    pass


def california_housing():
    dataset_dir, _ = _start('california')

    X_num_all, y_all = _get_sklearn_dataset('california_housing')
    idx = _make_split(len(X_num_all), None, 3)

    _save(
        dataset_dir,
        'California Housing',
        TaskType.REGRESSION,
        **_apply_split({'X_num': X_num_all, 'y': y_all}, idx),
        X_cat=None,
        idx=idx,
    )


def cardio():
    data_id = "cardio"
    dataset_dir, files = _start(data_id)
    target_col = "cardio"

    df = pd.read_csv(files[0], sep=";")
    y_all = df.pop(target_col).values.astype(np.int64)

    num_columns = [
        "age",
        "height",
        "weight",
        "ap_hi",
        "ap_lo"
    ]

    cat_columns = [c for c in df.columns if c not in num_columns]
    assert set(num_columns) | set(cat_columns) == set(df.columns.tolist())
    X_num_all = df[num_columns].astype(np.float64).values
    X_cat_all = df[cat_columns].astype(str).values

    cols = {
        "num": num_columns,
        "cat": cat_columns,
        "target": target_col
    }

    idx = _load_idx(data_id, files[1])

    _save(
        dataset_dir,
        'Cardio',
        TaskType.BINCLASS,
        **_apply_split(
            {'X_num': X_num_all, 'X_cat': X_cat_all, 'y': y_all},
            idx,
        ),
        idx=idx,
        float_type=np.float64,
        cols=cols,
    )


def churn2_modelling():
    # Get the file here: https://www.kaggle.com/shrutimechlearn/churn-modelling
    dataset_dir, files = _start('churn2')
    df = pd.read_csv(files[0])
    target_col = "Exited"

    df = df.drop(columns=['RowNumber', 'CustomerId', 'Surname'])
    df['Gender'] = df['Gender'].astype('category').cat.codes.values.astype(np.int64)
    y_all = df.pop(target_col).values.astype(np.int64)

    # Make dataset consistent with the data used in https://github.com/rotot0/tab-ddpm.
    # We move the variables ['Gender', 'HasCrCard', 'IsActiveMember'] to the cat_columns
    # since the nunique == 2 for these variables.
    num_columns = [
        'CreditScore',
        # 'Gender',
        'Age',
        'Tenure',
        'Balance',
        'NumOfProducts',
        'EstimatedSalary',
        # 'HasCrCard',
        # 'IsActiveMember',
        'EstimatedSalary',
    ]
    cat_columns = ['Geography', 'Gender', 'HasCrCard', 'IsActiveMember']
    assert set(num_columns) | set(cat_columns) == set(df.columns.tolist())
    X_num_all = df[num_columns].astype(np.float32).values
    X_cat_all = df[cat_columns].astype(str).values
    idx = _make_split(len(df), y_all, 3)

    cols = {
        "num": num_columns,
        "cat": cat_columns,
        "target": target_col
    }

    _save(
        dataset_dir,
        'Churn Modelling',
        TaskType.BINCLASS,
        **_apply_split(
            {'X_num': X_num_all, 'X_cat': X_cat_all, 'y': y_all},
            idx,
        ),
        idx=idx,
        cols=cols
    )


def default():
    pass


def diabetes():
    # Get the file here: https://www.kaggle.com/shrutimechlearn/churn-modelling
    data_id = "diabetes"
    dataset_dir, files = _start(data_id)
    target_col = "Outcome"

    try:
        bunch = _fetch_openml(42608)
        df = bunch["data"]
    except URLError:
        # In case SSL is blocked by the firewall, fallback to local
        # copy of the file.
        data = arff.loadarff(files[0])
        df = pd.DataFrame(data[0])

    y_all = df.pop(target_col).values.astype(np.int64)

    num_columns = [
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age"
    ]

    cat_columns = [c for c in df.columns if c not in num_columns]
    assert set(num_columns) | set(cat_columns) == set(df.columns.tolist())
    X_num_all = df[num_columns].astype(np.float64).values
    X_cat_all = df[cat_columns].astype(str).values

    cols = {
        "num": num_columns,
        "cat": cat_columns,
        "target": target_col
    }

    idx = _load_idx(data_id, files[1])

    _save(
        dataset_dir,
        'Diabetes',
        TaskType.BINCLASS,
        **_apply_split(
            {'X_num': X_num_all, 'X_cat': X_cat_all, 'y': y_all},
            idx,
        ),
        idx=idx,
        float_type=np.float64,
        cols=cols,
    )


def facebook_comments_volume(keep_derived: bool):
    # This is our preprocessing. The difference with Wide & Deep:
    # - (let columns be: [c0, c1, ..., c51, c52, target])
    # - c3 is the only categorical feature (as described at the UCI page)
    # - c14 is removed (it contains three unique values with the following distribution: [157631, 4, 3])
    # - c37 is removed (it contains one unique value)
    # - if keep_derived is False, then [c4, c5, ..., c28] are removed
    dataset_dir, files = _start('fb-comments' if keep_derived else 'fb-c')
    csv_path = 'Dataset/Training/Features_Variant_5.csv'
    _unzip(files[0], [csv_path])

    target_col = "target"

    df = pd.read_csv(
        dataset_dir / csv_path, names=[f'c{i}' for i in range(53)] + [target_col]
    )
    extra_columns = {'c14', 'c37'}
    if not keep_derived:
        extra_columns.update(f'c{i}' for i in range(4, 29))
    df.drop(columns=sorted(extra_columns), inplace=True)

    seed = 2
    dfs = {}
    dfs['train'], dfs['test'] = train_test_split(df, random_state=seed, test_size=0.2)
    dfs['val'], dfs['test'] = train_test_split(
        dfs['test'], random_state=seed, test_size=0.5
    )
    max_target_value = dfs['train'][target_col].quantile(0.99)  # type: ignore[code]
    dfs = {k: v[v[target_col] <= max_target_value] for k, v in dfs.items()}

    cat_columns = ['c3']
    # Using difference here is fine because the column names are alphabetical.
    num_columns = dfs['train'].columns.difference(cat_columns + [target_col])

    # Make dataset consistent with the data used in https://github.com/rotot0/tab-ddpm.
    nunique = dfs['train'][num_columns].nunique()
    num_columns = nunique[nunique > 2].index
    cat_columns.extend([c for c in nunique.index if c not in num_columns])

    cols = {
        "num": num_columns.tolist(),
        "cat": cat_columns,
        "target": target_col
    }

    # Store idx
    idx = {
        "train": np.array(dfs["train"].index.tolist()),
        "val": np.array(dfs["val"].index.tolist()),
        "test": np.array(dfs["test"].index.tolist()),
    }

    _save(
        dataset_dir,
        'Facebook Comments Volume'
        + ('' if keep_derived else ' (without derived features)'),
        TaskType.REGRESSION,
        X_num={k: v[num_columns].astype(np.float32).values for k, v in dfs.items()},
        X_cat={k: v[cat_columns].astype(str).values for k, v in dfs.items()},
        y={k: v[target_col].astype(np.float32).values for k, v in dfs.items()},
        idx=idx,
        id_='fb-comments--'
        + ('default' if keep_derived else 'without-derived-features'),
        cols=cols,
    )


def gesture_phase():
    dataset_dir, _ = _start('gesture')
    bunch = _fetch_openml(4538)
    target_col = "target"

    num_columns = bunch['data'].columns.tolist()
    cat_columns = []

    cols = {
        "num": num_columns,
        "cat": cat_columns,
        "target": target_col
    }

    X_num_all = bunch['data'].values.astype(np.float32)
    y_all = _encode_classification_target(bunch['target'].cat.codes.values)
    idx = _make_split(len(X_num_all), y_all, 3)

    _save(
        dataset_dir,
        'Gesture Phase',
        TaskType.MULTICLASS,
        **_apply_split({'X_num': X_num_all, 'y': y_all}, idx),
        X_cat=None,
        idx=idx,
        cols=cols
    )


def higgs_small():
    dataset_dir, _ = _start('higgs-small')
    bunch = _fetch_openml(23512)

    target_col = "target"

    num_columns = bunch['data'].columns.tolist()
    cat_columns = []

    cols = {
        "num": num_columns,
        "cat": cat_columns,
        "target": target_col
    }

    X_num_all = bunch['data'].values.astype(np.float32)
    y_all = _encode_classification_target(bunch['target'].cat.codes.values)
    nan_mask = np.isnan(X_num_all)
    valid_objects_mask = ~(nan_mask.any(1))
    # There is just one object with nine(!) missing values; let's drop it
    assert valid_objects_mask.sum() + 1 == len(X_num_all) and nan_mask.sum() == 9
    X_num_all = X_num_all[valid_objects_mask]
    y_all = y_all[valid_objects_mask]
    idx = _make_split(len(X_num_all), y_all, 3)

    _save(
        dataset_dir,
        'Higgs Small',
        TaskType.BINCLASS,
        **_apply_split({'X_num': X_num_all, 'y': y_all}, idx),
        X_cat=None,
        idx=idx,
        cols=cols,
    )


def house_16h():
    dataset_dir, _ = _start('house')
    bunch = _fetch_openml(574)

    target_col = "target"

    num_columns = bunch['data'].columns.tolist()
    cat_columns = []

    cols = {
        "num": num_columns,
        "cat": cat_columns,
        "target": target_col
    }

    X_num_all = bunch['data'].values.astype(np.float32)
    y_all = bunch['target'].values.astype(np.float32)
    idx = _make_split(len(X_num_all), None, 3)

    _save(
        dataset_dir,
        'House 16H',
        TaskType.REGRESSION,
        **_apply_split({'X_num': X_num_all, 'y': y_all}, idx),
        X_cat=None,
        idx=idx,
        cols=cols,
    )


def insurance():
    data_id = "insurance"
    dataset_dir, files = _start(data_id)
    target_col = "charges"

    df = pd.read_csv(files[0])
    y_all = df.pop(target_col).values.astype(np.float64)

    num_columns = [
        "age",
        "bmi",
        "children",
    ]

    cat_columns = [c for c in df.columns if c not in num_columns]
    assert set(num_columns) | set(cat_columns) == set(df.columns.tolist())
    X_num_all = df[num_columns].astype(np.float64).values
    X_cat_all = df[cat_columns].astype(str).values

    cols = {
        "num": num_columns,
        "cat": cat_columns,
        "target": target_col
    }

    idx = _load_idx(data_id, files[1])

    _save(
        dataset_dir,
        'Insurance',
        TaskType.REGRESSION,
        **_apply_split(
            {'X_num': X_num_all, 'X_cat': X_cat_all, 'y': y_all},
            idx,
        ),
        idx=idx,
        float_type=np.float64,
        cols=cols,
    )

def king():
    pass


def miniboone():
    pass


def wilt():
    pass


# %%
def main(argv):
    assert DATA_DIR.exists()
    _set_random_seeds()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--clear',
        action='store_true',
        help='Remove everything except for the expected files.',
    )
    args = parser.parse_args(argv[1:])

    if args.clear:
        for dirname, filenames in EXPECTED_FILES.items():
            dataset_dir = DATA_DIR / dirname
            for x in dataset_dir.iterdir():
                if x.name not in filenames:
                    if x.is_dir():
                        shutil.rmtree(x)
                    else:
                        x.unlink()
                    print(f'Removed: {x}')
            if not list(dataset_dir.iterdir()):
                dataset_dir.rmdir()
                print(f'Removed: {dataset_dir}')
        return

    # Below, datasets are grouped by file sources that we use, not by original sources.

    # # Selection are based from
    # https://github.com/rotot0/tab-ddpm/issues/3#issuecomment-1287205628
    # (adult, california, fb-comments, gesture, higgs-small, house)

    # OpenML
    abalone()
    # adult()  # CatBoost *
    # buddy()
    california_housing()  # Scikit-Learn *
    cardio()
    churn2_modelling()
    # default()
    diabetes()
    facebook_comments_volume(True)
    gesture_phase()
    higgs_small()
    house_16h()
    insurance()
    # king()
    # miniboone()
    # wilt()

    print('-----')
    print('Done!')


if __name__ == '__main__':
    sys.exit(main(sys.argv))