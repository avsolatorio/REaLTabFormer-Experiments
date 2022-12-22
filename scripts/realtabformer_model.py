import random
import numpy as np
import pandas as pd
import torch
import joblib
from pathlib import Path
from realtabformer import REaLTabFormer

from transformers import GPT2Config
from script_utils import BASE_DIR, REALTABFORMER_MODEL_TYPES, GRADIENT_ACCUMULATION_STEPS, DATA_IDS, SPLIT_SEEDS, get_batch_size, get_epochs, get_dirs, get_fnames



def get_realtabformer_model(data_id: str, model_type: str, epochs: int = None, seed: int = 1029):
    torch.cuda.empty_cache()

    batch_size, gradient_accumulation_steps = get_batch_size(data_id, model_type, return_accumulation=True)

    n_critic = 5
    training_args_kwargs = dict(
        logging_steps=100,
        save_steps=100,
        eval_steps=100,
        save_total_limit=1,
    )

    if data_id == "heloc":
        n_critic = 5
        training_args_kwargs.update(
            dict(
                logging_steps=50,
                save_steps=50,
                eval_steps=50,
            )
        )

    if data_id == "travel-customers":
        n_critic = 5
        training_args_kwargs.update(
            dict(
                logging_steps=5,
                save_steps=5,
                eval_steps=5,
            )
        )

    training_args_kwargs["gradient_accumulation_steps"] = gradient_accumulation_steps

    if epochs is None:
        epochs = get_epochs(data_id, model_type)

    if model_type == "smallrealtabformer":
        gpt_config = GPT2Config(n_layer=4, n_head=8, n_embd=512)
    elif model_type == "realtabformer":
        gpt_config = GPT2Config(n_layer=6, n_head=12, n_embd=768)
    elif model_type == "bigrealtabformer":
        gpt_config = GPT2Config(n_layer=24, n_head=16, n_embd=1024)
    else:
        raise ValueError(f"Unknown model_type ({model_type}) for REaLTabFormer...")

    model = REaLTabFormer(
        model_type="tabular",
        tabular_config=gpt_config,
        epochs=epochs, batch_size=batch_size,
        train_size=1,
        random_state=seed,
        early_stopping_patience=0,  # Don't need to have so many checkpoints.
        **training_args_kwargs)

    return model, n_critic


def train_sample(data_id: str, model_type: str, seed: int, sample_multiple: int = 10, verbose: bool = True):
    _, DATA_DIR, save_dir, samples_save_dir, checkpoints_dir = get_dirs(data_id, model_type, return_checkpoint=True)

    epochs = get_epochs(data_id, model_type)
    model, n_critic = get_realtabformer_model(data_id, model_type, epochs=epochs, seed=seed)

    path = DATA_DIR / f"split_{seed}"

    data_fname, model_fname, samples_fname = get_fnames(data_id, model_type, seed, epochs=epochs, verbose=verbose)
    data_fname = path / data_fname
    model_fname = Path(save_dir) / model_fname
    samples_fname = Path(samples_save_dir) / samples_fname

    # The REaLTabFormer model requires a directory for saving the model,
    # so we remove the .pkl extension.
    samples_dir = samples_fname.with_suffix("")
    model_fname = model_fname.with_suffix("")
    experiment_dir = checkpoints_dir / model_fname.name

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
        # Set checkpoint directory
        model.checkpoints_dir = experiment_dir
        model.training_args_kwargs["output_dir"] = model.checkpoints_dir.as_posix()
        model.samples_save_dir = samples_dir

        model.fit(
            payload["train"],
            device="cuda" if torch.cuda.is_available() else "cpu",
            qt_max=0.1, qt_interval=100, qt_interval_unique=100, n_critic=n_critic,
            num_bootstrap=500,
            sensitivity_max_col_nums=5,
            use_ks=False,
        )
        # Save the trained model
        model.save(path=model_fname)
    else:
        model = model.load_from_dir(model_fname.as_posix())

    # Generate samples
    samples = model.sample(
        n_samples=sample_multiple * len(payload["data"]),
        save_samples=True,
        gen_batch=128,
        device="cuda",
    )
    samples.to_csv(samples_fname, index=None)


if __name__ == "__main__":
    for model_type in REALTABFORMER_MODEL_TYPES:
        for seed in SPLIT_SEEDS:
            for data_id in DATA_IDS:
                train_sample(data_id, model_type, seed=seed)
