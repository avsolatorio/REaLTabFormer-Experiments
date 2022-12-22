import random
import numpy as np
import pandas as pd
import torch
import joblib
from pathlib import Path
from be_great import GReaT

from script_utils import BASE_DIR, GREAT_MODEL_TYPES, GRADIENT_ACCUMULATION_STEPS, get_batch_size, get_epochs, get_dirs, get_fnames


def get_great_model(data_id: str, model_type: str, epochs: int = None):
    batch_size = get_batch_size(data_id, model_type)

    if epochs is None:
        epochs = get_epochs(data_id, model_type)

    if model_type == "distillgreat":
        model = GReaT(llm='distilgpt2', batch_size=batch_size, epochs=epochs, gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS)

    elif model_type == "great":
        model = GReaT(llm='gpt2', batch_size=batch_size, epochs=epochs, gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS)
    else:
        raise ValueError(f"Unknown model_type ({model_type}) for GReaT...")

    return model


def sample_great(model, target_samples: int, sampling_batch: int = 128, sampling_rate: int = 10, device: str = "cuda", random_state: int = 1029):
    synthetic_data = pd.DataFrame()
    # split_samples = (target_samples // split_sampling) + 1
    split_samples = sampling_batch * sampling_rate
    continuous_error_limit = 20

    while (synthetic_data.shape[0] < target_samples) and continuous_error_limit:
        try:
            synthetic_split = model.sample(n_samples=split_samples, k=sampling_batch, device=device)
            if synthetic_data.empty:
                synthetic_data = synthetic_split
            else:
                synthetic_data = pd.concat([synthetic_data, synthetic_split])

            continuous_error_limit = 20

        except IndexError:
            # IndexError                                Traceback (most recent call last)
            # Cell In [9], line 1
            # ----> 1 synthetic_data = model.sample(n_samples=data.shape[0])

            # File ~/.local/share/virtualenvs/be_great-GAXskHpf/lib/python3.9/site-packages/be_great/great.py:162, in GReaT.sample(self, n_samples, start_col, start_col_dist, temperature, k, max_length, device)
            #     160 # Convert tokens back to tabular data
            #     161 text_data = _convert_tokens_to_text(tokens, self.tokenizer)
            # --> 162 df_gen = _convert_text_to_tabular_data(text_data, df_gen)
            #     164 # Remove rows with flawed numerical values
            #     165 for i_num_cols in self.num_cols:

            # File ~/.local/share/virtualenvs/be_great-GAXskHpf/lib/python3.9/site-packages/be_great/great_utils.py:91, in _convert_text_to_tabular_data(text, df_gen)
            #      89         values = f.strip().split(" is ")
            #      90         if values[0] in columns and not td[values[0]]:
            # ---> 91             td[values[0]] = [values[1]]
            #      93     df_gen = pd.concat([df_gen, pd.DataFrame(td)], ignore_index=True, axis=0)
            #      94 return df_gen

            # IndexError: list index out of range

            continuous_error_limit -= 1

            print("IndexError encounterd, retrying...")

    if (continuous_error_limit == 0) and synthetic_data.shape[0] < target_samples:
        raise ValueError("Sampling failed...")

    synthetic_data = synthetic_data.sample(n=target_samples, replace=False, random_state=random_state)
    synthetic_data = synthetic_data.reset_index(drop="index")

    return synthetic_data


def train_sample(data_id: str, model_type: str, sample_multiple: int = 10, verbose: bool = True):
    _, DATA_DIR, save_dir, samples_save_dir, checkpoints_dir = get_dirs(data_id, model_type, return_checkpoint=True)

    epochs = get_epochs(data_id, model_type)
    model = get_great_model(data_id, model_type, epochs=epochs)

    for path in DATA_DIR.glob("split_*"):
        split = path.name
        seed = int(split.split("_")[-1])

        data_fname, model_fname, samples_fname = get_fnames(data_id, model_type, seed, epochs=epochs, verbose=verbose)
        data_fname = path / data_fname
        model_fname = Path(save_dir) / model_fname
        samples_fname = Path(samples_save_dir) / samples_fname

        # The GReaT model requires a directory for saving the model,
        # so we remove the .pkl extension.
        model_fname = model_fname.with_suffix("")
        experiment_dir = checkpoints_dir / model_fname.name

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
            # Set checkpoint directory
            model.experiment_dir = experiment_dir.as_posix()

            model.fit(payload["train"])
            # Save the trained model
            model.save(model_fname.as_posix())
        else:
            model = model.load_from_dir(model_fname.as_posix())

        # Generate samples
        samples = sample_great(model, target_samples=sample_multiple * len(payload["data"]), random_state=seed)
        samples.to_csv(samples_fname, index=None)


if __name__ == "__main__":
    for model_type in GREAT_MODEL_TYPES:
        for data_path in (BASE_DIR / "input").glob("*"):
            data_id = data_path.name
            train_sample(data_id, model_type)
