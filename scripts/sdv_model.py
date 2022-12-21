import random
import numpy as np
import torch
import joblib
from pathlib import Path
from sdv.tabular import GaussianCopula, CopulaGAN, CTGAN, TVAE
from script_utils import BASE_DIR, model_types, get_dirs, get_fnames


RANDOM_SEED = 1029
BASELINE_EPOCHS = 200
SDV_BATCH_SIZE = 510


def get_sdv_model(model_type):
    if model_type == "ctgan":
        model = CTGAN(epochs=BASELINE_EPOCHS, verbose=True, batch_size=SDV_BATCH_SIZE)
    elif model_type == "tvae":
        model = TVAE(epochs=BASELINE_EPOCHS, batch_size=SDV_BATCH_SIZE)
    elif model_type == "copulagan":
        model = CopulaGAN(epochs=BASELINE_EPOCHS, verbose=True, batch_size=SDV_BATCH_SIZE)
    elif model_type == "gaussiancopula":
        model = GaussianCopula()

    return model


def train_sample(model_type, data_id, sample_multiple: int = 10, verbose: bool = True):
    _, DATA_DIR, save_dir, samples_save_dir = get_dirs(model_type, data_id)

    model = get_sdv_model(model_type)

    for path in DATA_DIR.glob("split_*"):
        split = path.name
        seed = int(split.split("_")[-1])

        data_fname, model_fname, samples_fname = get_fnames(model_type, data_id, seed, verbose=verbose)
        data_fname = path / data_fname
        model_fname = Path(save_dir) / model_fname
        samples_fname = Path(samples_save_dir) / samples_fname

        if not data_fname.exists():
            print(f"Data ({data_fname}) doesn't exist... Skipping...")
            continue

        if samples_fname.exists():
            continue

        payload = joblib.load(data_fname)

        # Set random seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if not model_fname.exists():
            model.fit(payload["train"])
            # Save the trained model
            model.save(model_fname)
        else:
            model = model.load(model_fname)

        # Generate samples
        samples = model.sample(num_rows=sample_multiple * len(payload["data"]))
        samples.to_csv(samples_fname)


if __name__ == "__main__":
    for model_type in model_types:
        for data_path in (BASE_DIR / "input").glob("*"):
            data_id = data_path.name
            train_sample(model_type, data_id)
