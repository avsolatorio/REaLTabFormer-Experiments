from pathlib import Path
import torch


MODEL_TYPES = ["ctgan", "tvae", "copulagan", "gaussiancopula", "distillgreat", "great", "smallrealtabformer", "realtabformer", "bigrealtabformer"]
RANDOM_SEED = 1029

BASE_DIR = Path(__file__).parent.parent / "data"
assert BASE_DIR.exists(), f"Make sure that the DATA_DIR ({BASE_DIR}) is correct..."


def get_data_target_batch_size(data_id: str) -> int:
    target_batch_size = 512

    if data_id == "travel-customers":
        target_batch_size = 32

    return target_batch_size


def get_batch_size(data_id: str, model_type: str):
    # # Batch sizes are set to be approximately
    # # similar across models.
    # SDV_BATCH_SIZE = 510
    # RTF_BATCH_SIZE = 512  # Adjust gradient_accumulation_steps
    # GREAT_BATCH_SIZE = 512  # Adjust gradient_accumulation_steps and distributed

    target_batch_size = get_data_target_batch_size(data_id)

    if model_type in ["ctgan", "tvae", "copulagan", "gaussiancopula"]:
        # The batch_size for SDV models should be multiple of 10.
        batch_size = (target_batch_size // 10) * 10

    if model_type in ["distillgreat", "great", "smallrealtabformer", "realtabformer", "bigrealtabformer"]:
        cuda_count = torch.cuda.device_count()
        batch_size = target_batch_size // cuda_count

    return batch_size


def get_epochs(data_id: str, model_type: str):
    epochs = 200

    return epochs


def get_dirs(data_id: str, model_type: str):
    assert model_type in MODEL_TYPES

    print(data_id, model_type)

    EXP_DIR: Path = BASE_DIR / "models" / model_type / data_id
    DATA_DIR: Path = BASE_DIR / "input" / data_id

    save_dir = (EXP_DIR / "trained_model")
    save_dir.mkdir(parents=True, exist_ok=True)

    samples_save_dir = (EXP_DIR / "samples")
    samples_save_dir.mkdir(parents=True, exist_ok=True)

    save_dir = save_dir.as_posix()
    samples_save_dir = samples_save_dir.as_posix()

    return EXP_DIR, DATA_DIR, save_dir, samples_save_dir


def get_fnames(data_id: str, model_type: str, seed: int, verbose: bool = True):
    data_fname = f"{data_id}_seed-{seed}.pkl"
    name = f"{model_type}_model-{data_id}_seed-{seed}"

    if verbose: print(name)

    model_fname = f"{name}.pkl"
    samples_fname = f"{name}.csv"

    return data_fname, model_fname, samples_fname
