import random
import numpy as np
import torch
import joblib
from pathlib import Path
from sdv.tabular import GaussianCopula, CopulaGAN, CTGAN, TVAE
from script_utils import BASE_DIR, SDV_MODEL_TYPES, SPLIT_SEEDS, DATA_IDS, get_batch_size, get_epochs, get_dirs, get_fnames


def get_sdv_model(data_id: str, model_type: str, epochs: int = None):
    batch_size = get_batch_size(data_id, model_type)

    if epochs is None:
        epochs = get_epochs(data_id, model_type)

    if model_type == "ctgan":
        model = CTGAN(epochs=epochs, verbose=True, batch_size=batch_size)
    elif model_type == "tvae":
        model = TVAE(epochs=epochs, batch_size=batch_size)
    elif model_type == "copulagan":
        model = CopulaGAN(epochs=epochs, verbose=True, batch_size=batch_size)
    elif model_type == "gaussiancopula":
        model = GaussianCopula()

    return model


def train_sample(data_id: str, model_type: str, seed: int, sample_multiple: int = 10, verbose: bool = True):
    _, DATA_DIR, save_dir, samples_save_dir = get_dirs(data_id, model_type)

    epochs = get_epochs(data_id, model_type)
    model = get_sdv_model(data_id, model_type, epochs=epochs)
    path = DATA_DIR / f"split_{seed}"

    data_fname, model_fname, samples_fname = get_fnames(data_id, model_type, seed, epochs=epochs, verbose=verbose)
    data_fname = path / data_fname
    model_fname = Path(save_dir) / model_fname
    samples_fname = Path(samples_save_dir) / samples_fname

    if not data_fname.exists():
        print(f"Data ({data_fname}) doesn't exist... Skipping...")
        return

    if samples_fname.exists():
        return

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
    samples.to_csv(samples_fname, index=None)


if __name__ == "__main__":
    for model_type in SDV_MODEL_TYPES:
        for seed in SPLIT_SEEDS:
            for data_id in DATA_IDS:
                if data_id == "adult-income" and model_type == "tvae":
                    continue
                train_sample(data_id, model_type, seed=seed)
