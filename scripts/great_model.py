import warnings
import random
import numpy as np
import pandas as pd
import torch
import joblib
from pathlib import Path
from be_great import GReaT

from script_utils import (
    GReaTModelType,
    BASE_DIR, GREAT_MODEL_TYPES, GRADIENT_ACCUMULATION_STEPS, DATA_IDS,
    SPLIT_SEEDS, get_batch_size, get_epochs, get_dirs, get_fnames
)


def get_great_model(data_id: str, model_type: str, epochs: int = None):
    batch_size, gradient_accumulation_steps = get_batch_size(data_id, model_type, return_accumulation=True)

    if epochs is None:
        epochs = get_epochs(data_id, model_type)

    if model_type == GReaTModelType.distillgreat:
        model = GReaT(llm='distilgpt2', batch_size=batch_size, epochs=epochs, gradient_accumulation_steps=gradient_accumulation_steps)

    elif model_type == GReaTModelType.great:
        model = GReaT(llm='gpt2-medium', batch_size=batch_size, epochs=epochs, gradient_accumulation_steps=gradient_accumulation_steps)
    else:
        raise ValueError(f"Unknown model_type ({model_type}) for GReaT...")

    return model


def sample_great(model, target_samples: int, sampling_batch: int = 128, sampling_rate: int = 20, device: str = "cuda", random_state: int = 1029):
    synthetic_data = pd.DataFrame()
    # split_samples = (target_samples // split_sampling) + 1
    split_samples = sampling_batch * sampling_rate
    continuous_error_limit = 20

    while (synthetic_data.shape[0] < target_samples) and continuous_error_limit:
        print("Generated samples:", synthetic_data.shape[0], target_samples)
        try:
            synthetic_split = model.sample(n_samples=split_samples, k=sampling_batch, max_length=2048, device=device)
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
        except RuntimeError:
            # Generated samples: 15360 98710
            # 85%|██████████████████████████████████████████████████████████████████████████████████████████▎               | 2182/2560 [00:52<00:07, 51.01it/s]../aten/src/ATen/native/cuda/Indexing.cu:1141: indexSelectLargeIndex: block: [18,0,0], thread: [0,0,0] Assertion `srcIndex < srcSelectDimSize` failed.
            # ../aten/src/ATen/native/cuda/Indexing.cu:1141: indexSelectLargeIndex: block: [18,0,0], thread: [1,0,0] Assertion `srcIndex < srcSelectDimSize` failed.
            # ../aten/src/ATen/native/cuda/Indexing.cu:1141: indexSelectLargeIndex: block: [18,0,0], thread: [2,0,0] Assertion `srcIndex < srcSelectDimSize` failed.
            # ../aten/src/ATen/native/cuda/Indexing.cu:1141: indexSelectLargeIndex: block: [18,0,0], thread: [3,0,0] Assertion `srcIndex < srcSelectDimSize` failed.
            # ../aten/src/ATen/native/cuda/Indexing.cu:1141: indexSelectLargeIndex: block: [18,0,0], thread: [4,0,0] Assertion `srcIndex < srcSelectDimSize` failed.
            # ../aten/src/ATen/native/cuda/Indexing.cu:1141: indexSelectLargeIndex: block: [18,0,0], thread: [5,0,0] Assertion `srcIndex < srcSelectDimSize` failed.
            # ../aten/src/ATen/native/cuda/Indexing.cu:1141: indexSelectLargeIndex: block: [18,0,0], thread: [6,0,0] Assertion `srcIndex < srcSelectDimSize` failed.
            # ../aten/src/ATen/native/cuda/Indexing.cu:1141: indexSelectLargeIndex: block: [18,0,0], thread: [7,0,0] Assertion `srcIndex < srcSelectDimSize` failed.
            # ../aten/src/ATen/native/cuda/Indexing.cu:1141: indexSelectLargeIndex: block: [18,0,0], thread: [8,0,0] Assertion `srcIndex < srcSelectDimSize` failed.
            # ../aten/src/ATen/native/cuda/Indexing.cu:1141: indexSelectLargeIndex: block: [18,0,0], thread: [9,0,0] Assertion `srcIndex < srcSelectDimSize` failed.
            # ../aten/src/ATen/native/cuda/Indexing.cu:1141: indexSelectLargeIndex: block: [18,0,0], thread: [10,0,0] Assertion `srcIndex < srcSelectDimSize` failed.
            # ../aten/src/ATen/native/cuda/Indexing.cu:1141: indexSelectLargeIndex: block: [18,0,0], thread: [11,0,0] Assertion `srcIndex < srcSelectDimSize` failed.
            # ../aten/src/ATen/native/cuda/Indexing.cu:1141: indexSelectLargeIndex: block: [18,0,0], thread: [12,0,0] Assertion `srcIndex < srcSelectDimSize` failed.
            # ../aten/src/ATen/native/cuda/Indexing.cu:1141: indexSelectLargeIndex: block: [18,0,0], thread: [13,0,0] Assertion `srcIndex < srcSelectDimSize` failed.
            # ../aten/src/ATen/native/cuda/Indexing.cu:1141: indexSelectLargeIndex: block: [18,0,0], thread: [14,0,0] Assertion `srcIndex < srcSelectDimSize` failed.
            # ../aten/src/ATen/native/cuda/Indexing.cu:1141: indexSelectLargeIndex: block: [18,0,0], thread: [15,0,0] Assertion `srcIndex < srcSelectDimSize` failed.
            # ../aten/src/ATen/native/cuda/Indexing.cu:1141: indexSelectLargeIndex: block: [18,0,0], thread: [16,0,0] Assertion `srcIndex < srcSelectDimSize` failed.
            # ../aten/src/ATen/native/cuda/Indexing.cu:1141: indexSelectLargeIndex: block: [18,0,0], thread: [17,0,0] Assertion `srcIndex < srcSelectDimSize` failed.
            # ../aten/src/ATen/native/cuda/Indexing.cu:1141: indexSelectLargeIndex: block: [18,0,0], thread: [18,0,0] Assertion `srcIndex < srcSelectDimSize` failed.
            # ../aten/src/ATen/native/cuda/Indexing.cu:1141: indexSelectLargeIndex: block: [18,0,0], thread: [19,0,0] Assertion `srcIndex < srcSelectDimSize` failed.
            # ../aten/src/ATen/native/cuda/Indexing.cu:1141: indexSelectLargeIndex: block: [18,0,0], thread: [20,0,0] Assertion `srcIndex < srcSelectDimSize` failed.
            # ../aten/src/ATen/native/cuda/Indexing.cu:1141: indexSelectLargeIndex: block: [18,0,0], thread: [21,0,0] Assertion `srcIndex < srcSelectDimSize` failed.
            # ../aten/src/ATen/native/cuda/Indexing.cu:1141: indexSelectLargeIndex: block: [18,0,0], thread: [22,0,0] Assertion `srcIndex < srcSelectDimSize` failed.
            # ../aten/src/ATen/native/cuda/Indexing.cu:1141: indexSelectLargeIndex: block: [18,0,0], thread: [23,0,0] Assertion `srcIndex < srcSelectDimSize` failed.
            # ../aten/src/ATen/native/cuda/Indexing.cu:1141: indexSelectLargeIndex: block: [18,0,0], thread: [24,0,0] Assertion `srcIndex < srcSelectDimSize` failed.
            # ../aten/src/ATen/native/cuda/Indexing.cu:1141: indexSelectLargeIndex: block: [18,0,0], thread: [25,0,0] Assertion `srcIndex < srcSelectDimSize` failed.
            # ../aten/src/ATen/native/cuda/Indexing.cu:1141: indexSelectLargeIndex: block: [18,0,0], thread: [26,0,0] Assertion `srcIndex < srcSelectDimSize` failed.
            # ../aten/src/ATen/native/cuda/Indexing.cu:1141: indexSelectLargeIndex: block: [18,0,0], thread: [27,0,0] Assertion `srcIndex < srcSelectDimSize` failed.
            # ../aten/src/ATen/native/cuda/Indexing.cu:1141: indexSelectLargeIndex: block: [18,0,0], thread: [28,0,0] Assertion `srcIndex < srcSelectDimSize` failed.
            # ../aten/src/ATen/native/cuda/Indexing.cu:1141: indexSelectLargeIndex: block: [18,0,0], thread: [29,0,0] Assertion `srcIndex < srcSelectDimSize` failed.
            # ../aten/src/ATen/native/cuda/Indexing.cu:1141: indexSelectLargeIndex: block: [18,0,0], thread: [30,0,0] Assertion `srcIndex < srcSelectDimSize` failed.
            # ../aten/src/ATen/native/cuda/Indexing.cu:1141: indexSelectLargeIndex: block: [18,0,0], thread: [31,0,0] Assertion `srcIndex < srcSelectDimSize` failed.
            # 85%|██████████████████████████████████████████████████████████████████████████████████████████▎               | 2182/2560 [00:58<00:10, 37.45it/s]
            # Traceback (most recent call last):
            # File "/home/jupyter-wb536061/REaLTabFormer-Experiments/be-great-env/../scripts/great_model.py", line 148, in <module>
            #     train_sample(data_id, model_type, seed=seed)
            # File "/home/jupyter-wb536061/REaLTabFormer-Experiments/be-great-env/../scripts/great_model.py", line 125, in train_sample
            #     samples = sample_great(model, target_samples=sample_multiple * len(payload["data"]), random_state=seed)
            # File "/home/jupyter-wb536061/REaLTabFormer-Experiments/be-great-env/../scripts/great_model.py", line 39, in sample_great
            #     synthetic_split = model.sample(n_samples=split_samples, k=sampling_batch, max_length=2048, device=device)
            # File "/home/jupyter-wb536061/.local/share/virtualenvs/be-great-env-BAUbSh5F/src/be-great/be_great/great.py", line 157, in sample
            #     tokens = self.model.generate(input_ids=start_tokens, max_length=max_length,
            # File "/home/jupyter-wb536061/.local/share/virtualenvs/be-great-env-BAUbSh5F/lib/python3.9/site-packages/torch/autograd/grad_mode.py", line 27, in decorate_context
            #     return func(*args, **kwargs)
            # File "/home/jupyter-wb536061/.local/share/virtualenvs/be-great-env-BAUbSh5F/lib/python3.9/site-packages/transformers/generation/utils.py", line 1571, in generate
            #     return self.sample(
            # File "/home/jupyter-wb536061/.local/share/virtualenvs/be-great-env-BAUbSh5F/lib/python3.9/site-packages/transformers/generation/utils.py", line 2534, in sample
            #     outputs = self(
            # File "/home/jupyter-wb536061/.local/share/virtualenvs/be-great-env-BAUbSh5F/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1194, in _call_impl
            #     return forward_call(*input, **kwargs)
            # File "/home/jupyter-wb536061/.local/share/virtualenvs/be-great-env-BAUbSh5F/lib/python3.9/site-packages/transformers/models/gpt2/modeling_gpt2.py", line 1046, in forward
            #     transformer_outputs = self.transformer(
            # File "/home/jupyter-wb536061/.local/share/virtualenvs/be-great-env-BAUbSh5F/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1194, in _call_impl
            #     return forward_call(*input, **kwargs)
            # File "/home/jupyter-wb536061/.local/share/virtualenvs/be-great-env-BAUbSh5F/lib/python3.9/site-packages/transformers/models/gpt2/modeling_gpt2.py", line 889, in forward
            #     outputs = block(
            # File "/home/jupyter-wb536061/.local/share/virtualenvs/be-great-env-BAUbSh5F/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1194, in _call_impl
            #     return forward_call(*input, **kwargs)
            # File "/home/jupyter-wb536061/.local/share/virtualenvs/be-great-env-BAUbSh5F/lib/python3.9/site-packages/transformers/models/gpt2/modeling_gpt2.py", line 389, in forward
            #     attn_outputs = self.attn(
            # File "/home/jupyter-wb536061/.local/share/virtualenvs/be-great-env-BAUbSh5F/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1194, in _call_impl
            #     return forward_call(*input, **kwargs)
            # File "/home/jupyter-wb536061/.local/share/virtualenvs/be-great-env-BAUbSh5F/lib/python3.9/site-packages/transformers/models/gpt2/modeling_gpt2.py", line 330, in forward
            #     attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)
            # File "/home/jupyter-wb536061/.local/share/virtualenvs/be-great-env-BAUbSh5F/lib/python3.9/site-packages/transformers/models/gpt2/modeling_gpt2.py", line 201, in _attn
            #     attn_weights = torch.where(causal_mask, attn_weights, mask_value)
            # RuntimeError: The size of tensor a (1024) must match the size of tensor b (1025) at non-singleton dimension 3

            continuous_error_limit -= 1
            print("RuntimeError encounterd, retrying...")

    if (continuous_error_limit == 0) and synthetic_data.shape[0] < target_samples:
        raise ValueError("Sampling failed...")

    synthetic_data = synthetic_data.sample(n=target_samples, replace=False, random_state=random_state)
    synthetic_data = synthetic_data.reset_index(drop="index")

    return synthetic_data


def train_sample(data_id: str, model_type: str, seed: int, sample_multiple: int = 10, verbose: bool = True):
    _, DATA_DIR, save_dir, samples_save_dir, checkpoints_dir = get_dirs(data_id, model_type, return_checkpoint=True)

    epochs = get_epochs(data_id, model_type)
    model = get_great_model(data_id, model_type, epochs=epochs)

    path = DATA_DIR / f"split_{seed}"

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
        model.experiment_dir = experiment_dir.as_posix()

        model.fit(payload["train"])
        # Save the trained model
        model.save(model_fname.as_posix())
    else:
        model = model.load_from_dir(model_fname.as_posix())

    # Generate samples
    try:
        samples = sample_great(model, target_samples=sample_multiple * len(payload["data"]), random_state=seed)
        samples.to_csv(samples_fname, index=None)
    except ValueError:
        warnings.warn("No samples were generated!")
        return


if __name__ == "__main__":
    import sys
    filter_data_ids = None
    if len(sys.argv) > 1:
        filter_data_ids = []
        for arg in sys.argv[1:]:
            if arg in DATA_IDS:
                filter_data_ids.append(arg)

    for model_type in GREAT_MODEL_TYPES:
        for seed in SPLIT_SEEDS:
            for data_id in DATA_IDS:
                if filter_data_ids and data_id not in filter_data_ids:
                    print(f"Skipping {data_id}...")
                    continue

                train_sample(data_id, model_type, seed=seed)
