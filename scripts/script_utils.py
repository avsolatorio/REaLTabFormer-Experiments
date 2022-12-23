from pathlib import Path

DATA_IDS = ["california-housing", "travel-customers", "adult-income", "heloc"]
SPLIT_SEEDS = [610, 1029, 1004, 2019, 2009]
RANDOM_SEED = 1029

SDV_MODEL_TYPES = ["ctgan", "tvae", "copulagan", "gaussiancopula"]
GREAT_MODEL_TYPES = ["distillgreat", "great"]
REALTABFORMER_MODEL_TYPES = ["realtabformer", "smallrealtabformer", "bigrealtabformer"]
MODEL_TYPES = SDV_MODEL_TYPES + GREAT_MODEL_TYPES + REALTABFORMER_MODEL_TYPES

BASE_DIR = Path(__file__).parent.parent / "data"
assert BASE_DIR.exists(), f"Make sure that the DATA_DIR ({BASE_DIR}) is correct..."
GRADIENT_ACCUMULATION_STEPS = 4
MIN_BATCH_SIZE = 4


def get_data_target_batch_size(data_id: str) -> int:
    target_batch_size = 512

    if data_id == "travel-customers":
        target_batch_size = 32

    return target_batch_size


def get_batch_size(data_id: str, model_type: str, return_accumulation: bool = False) -> int:
    import torch

    # # Batch sizes are set to be approximately
    # # similar across models.
    # SDV_BATCH_SIZE = 510
    # RTF_BATCH_SIZE = 512  # Adjust gradient_accumulation_steps
    # GREAT_BATCH_SIZE = 512  # Adjust gradient_accumulation_steps and distributed

    target_batch_size = get_data_target_batch_size(data_id)
    gradient_accumulation_steps = GRADIENT_ACCUMULATION_STEPS

    if model_type in ["ctgan", "tvae", "copulagan", "gaussiancopula"]:
        # The batch_size for SDV models should be multiple of 10.
        batch_size = max(10, (target_batch_size // 10) * 10)

    elif model_type == "distillgreat":
        cuda_count = torch.cuda.device_count()
        batch_size = max(MIN_BATCH_SIZE, target_batch_size // cuda_count // GRADIENT_ACCUMULATION_STEPS)

        if data_id == "california-housing":
            batch_size = 64  # batch size for a single cuda device
            gradient_accumulation_steps = target_batch_size // cuda_count // batch_size

        elif data_id == "heloc":
            batch_size = 32  # batch size for a single cuda device
            gradient_accumulation_steps = target_batch_size // cuda_count // batch_size

        elif data_id == "adult-income":
            batch_size = 64  # batch size for a single cuda device
            gradient_accumulation_steps = target_batch_size // cuda_count // batch_size

    elif model_type == "great":
        cuda_count = torch.cuda.device_count()
        batch_size = max(MIN_BATCH_SIZE, target_batch_size // cuda_count // GRADIENT_ACCUMULATION_STEPS)

        if data_id == "california-housing":
            batch_size = 32  # batch size for a single cuda device
            gradient_accumulation_steps = target_batch_size // cuda_count // batch_size

        elif data_id == "heloc":
            batch_size = 16  # batch size for a single cuda device
            gradient_accumulation_steps = target_batch_size // cuda_count // batch_size

        elif data_id == "adult-income":
            batch_size = 32  # batch size for a single cuda device
            gradient_accumulation_steps = target_batch_size // cuda_count // batch_size

    elif model_type in ["smallrealtabformer", "realtabformer"]:
        cuda_count = torch.cuda.device_count()
        batch_size = max(MIN_BATCH_SIZE, target_batch_size // cuda_count // GRADIENT_ACCUMULATION_STEPS)

    elif model_type == "bigrealtabformer":
        cuda_count = torch.cuda.device_count()
        batch_size = max(MIN_BATCH_SIZE, target_batch_size // cuda_count // GRADIENT_ACCUMULATION_STEPS)

        if data_id == "california-housing":
            batch_size = 32  # batch size for a single cuda device
            gradient_accumulation_steps = target_batch_size // cuda_count // batch_size

        elif data_id == "heloc":
            batch_size = 16  # batch size for a single cuda device
            gradient_accumulation_steps = target_batch_size // cuda_count // batch_size

        elif data_id == "adult-income":
            batch_size = 32  # batch size for a single cuda device
            gradient_accumulation_steps = target_batch_size // cuda_count // batch_size


    if return_accumulation:
        return batch_size, gradient_accumulation_steps
    else:
        return batch_size


def get_epochs(data_id: str, model_type: str) -> int:
    assert model_type in MODEL_TYPES
    # Default number of epochs
    epochs = 200

    if model_type == "distillgreat":
        # We fine-tune the Distill-GReaT model for each data set
        # for 200 epochs, except for the California housing data
        # set, for it, we fine-tune it for 100 epochs.
        epochs = 200
        if data_id == "california-housing":
            epochs = 100
    elif model_type == "great":
        # The GReaT baseline is fine-tuned for 110, 310, 400, 255 epochs
        # for California, Adult, Travel, and HELOC data sets, respectively.
        if data_id == "california-housing":
            epochs = 110
        elif data_id == "adult-income":
            epochs = 310
        elif data_id == "travel-customers":
            epochs = 400
        elif data_id == "heloc":
            epochs = 255
    elif model_type == "gaussiancopula":
        # No epochs parameter
        epochs = 0

    return epochs


def get_dirs(data_id: str, model_type: str, return_checkpoint: bool = False):
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

    if return_checkpoint:
        checkpoints_dir = (EXP_DIR / "checkpoints")
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        return EXP_DIR, DATA_DIR, save_dir, samples_save_dir, checkpoints_dir
    else:
        return EXP_DIR, DATA_DIR, save_dir, samples_save_dir

def get_fnames(data_id: str, model_type: str, seed: int, epochs: int = None, verbose: bool = True):
    base_name = f"{data_id}_seed-{seed}"
    data_fname = f"{base_name}.pkl"

    if epochs is not None:
        model_base_name = f"{base_name}_epochs-{epochs}"

    name = f"{model_type}_model-{model_base_name}"

    if verbose: print(name)

    model_fname = f"{name}.pkl"
    samples_fname = f"{name}.csv"

    return data_fname, model_fname, samples_fname


def rename_samples(epochs, parent):
    import subprocess as sub

    samples_fnames = list((parent / "samples").glob("*.csv"))
    models_fnames = list((parent / "trained_model").glob("*.pkl"))

    for f in samples_fnames:
        src = f.as_posix()

        if src.endswith(f"_epochs-{epochs}.csv"):
            print(f"Skipping {src}...")
            continue

        dest = (f.parent / (f.stem + f"_epochs-{epochs}.csv")).as_posix()
        sub.call(["mv", src, dest])

        print(f"Moved {src} to {dest}...")

    for f in models_fnames:
        src = f.as_posix()
        if src.endswith(f"_epochs-{epochs}.pkl"):
            print(f"Skipping {src}...")
            continue

        dest = (f.parent / (f.stem + f"_epochs-{epochs}.pkl")).as_posix()
        sub.call(["mv", src, dest])

        print(f"Moved {src} to {dest}...")
