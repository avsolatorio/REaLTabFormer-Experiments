import json
import random
import joblib
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import be_great
from be_great import GReaT


def wrap_great_sampling(model, target_samples: int, sampling_batch: int = 128, sampling_rate: int = 20, max_length: int = 8192, device: str = "cuda", random_state: int = 1029):
    """Implement this wrapper around the GReaT sampler because there's a lot of random issues currently
    that can intermittently be raised during sampling.
    """
    synthetic_data = pd.DataFrame()
    # split_samples = (target_samples // split_sampling) + 1
    split_samples = sampling_batch * sampling_rate
    continuous_error_limit = 20

    while (synthetic_data.shape[0] < target_samples) and continuous_error_limit:
        print("Generated samples:", synthetic_data.shape[0], target_samples)
        try:
            synthetic_split = model.sample(n_samples=split_samples, k=sampling_batch, max_length=max_length, device=device)
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



def train_great(
    parent_dir,
    real_data_path,
    model_params,
):
    real_data_path = Path(real_data_path)
    parent_dir = Path(parent_dir)
    save_model_path = parent_dir / "trained_model"

    # # Make sure that the rtf model specified in the config
    # # is similar to the one installed in the environment.
    # model_params["version"] == be_great.__version__

    if (save_model_path / "model.pt").exists():
        print(f"Model for this experiment is already available...")
        return

    cuda_count = max(1, torch.cuda.device_count())
    train_params = model_params["train"]

    # Calculate the gradient_accumulation_steps
    # based on the target_batch_size, available cuda count,
    # and batch size
    training_args_kwargs = train_params.pop("training_args_kwargs")
    training_args_kwargs["gradient_accumulation_steps"] = (
        model_params["meta"]["target_batch_size"] //
        cuda_count //
        train_params["batch_size"])

    great_model = GReaT(
        llm=train_params["llm"],
        batch_size=train_params["batch_size"],
        epochs=train_params["epochs"],
        **training_args_kwargs)

    train_data = joblib.load(real_data_path / "full_train.df.pkl")

    if model_params["meta"].get("use_val", False):
        val_data = joblib.load(real_data_path / "full_val.df.pkl")
        train_data = pd.concat([train_data, val_data])

    drop_cols = model_params["meta"]["drop_cols"]
    if drop_cols:
        drop_cols = [col for col in drop_cols if col in train_data.columns]
        train_data = train_data.drop(drop_cols, axis=1)

    great_model.fit(train_data)
    great_model.save(save_model_path.as_posix())

    return great_model


def sample_great(
    parent_dir,
    real_data_path,
    model_params,
    n_datasets: int = 1,
):
    real_data_path = Path(real_data_path)
    parent_dir = Path(parent_dir)

    data_id = real_data_path.name

    data_info = json.loads((real_data_path / "info.json").read_text())
    sample_gen_size = data_info.get("train_size", 0) + data_info.get("val_size", 0) + data_info.get("test_size", 0)

    save_model_path = parent_dir / "trained_model"
    exp_samples_dir = parent_dir / "great_samples"

    assert (save_model_path / "model.pt").exists()

    great_model = GReaT.load_from_dir(save_model_path.as_posix())

    sample_params = model_params["sample"]

    for seed in range(n_datasets):
        sample_save_path = exp_samples_dir / f"great_sample-seed_{seed}.pkl"
        sample_save_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"Sampling for {data_id}, seed::{seed} / {n_datasets}")

        if sample_save_path.exists():
            print(f"Sample already available for {sample_save_path}, skipping...")
            continue

        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

        sampling_rate = sample_params["sampling_rate"]

        if (sample_params["sampling_batch"] * sampling_rate) > sample_gen_size:
            # No need to oversample if a single set is sufficient.
            # Reduce sampling rate.
            sampling_rate = (sample_gen_size // sample_params["sampling_batch"]) + 1

        sample = wrap_great_sampling(
            great_model,
            target_samples=sample_gen_size,
            sampling_batch=sample_params["sampling_batch"],
            sampling_rate=sampling_rate,
            max_length=sample_params["max_length"],
            random_state=seed)

        joblib.dump(sample, sample_save_path)
