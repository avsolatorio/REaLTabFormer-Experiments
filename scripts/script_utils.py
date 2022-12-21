model_types = ["ctgan", "tvae", "copulagan", "gaussiancopula"]
from pathlib import Path


BASE_DIR = Path(__file__).parent.parent / "data"
assert BASE_DIR.exists(), f"Make sure that the DATA_DIR ({BASE_DIR}) is correct..."


def get_dirs(model_type, data_id):
    assert model_type in model_types

    print(model_type, data_id)

    EXP_DIR: Path = BASE_DIR / "models" / model_type / data_id
    DATA_DIR: Path = BASE_DIR / "input" / data_id

    save_dir = (EXP_DIR / "trained_model")
    save_dir.mkdir(parents=True, exist_ok=True)

    samples_save_dir = (EXP_DIR / "samples")
    samples_save_dir.mkdir(parents=True, exist_ok=True)

    save_dir = save_dir.as_posix()
    samples_save_dir = samples_save_dir.as_posix()

    return EXP_DIR, DATA_DIR, save_dir, samples_save_dir


def get_fnames(model_type: str, data_id: str, seed: int, verbose: bool = True):
    data_fname = f"{data_id}_seed-{seed}.pkl"
    name = f"{model_type}_model-{data_id}_seed-{seed}"

    if verbose: print(name)


    model_fname = f"{name}.pkl"
    samples_fname = f"{name}.csv"

    return data_fname, model_fname, samples_fname
