import joblib
import torch
from pathlib import Path
from realtabformer import REaLTabFormer, data_utils
from transformers import GPT2Config


def train_realtabformer(
    parent_dir,
    real_data_path,
    model_params,
    device = "cpu"
):
    real_data_path = Path(real_data_path)
    parent_dir = Path(parent_dir)

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

    train_params[f"{train_params['model_type']}_config"] = GPT2Config(
        **train_params.pop("gpt_config"))

    # Set up the model
    rtf_model = REaLTabFormer(**train_params, **training_args_kwargs)

    train_data = joblib.load(real_data_path / "full_train.df.pkl")
    fit_params = model_params["fit"]

    fit_params["frac"] = min(
        model_params["meta"]["frac_max_data"] / len(train_data),
        fit_params["frac"])

    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")

    if device.split(":")[0] in devices:
        fit_params["device"] = device
    else:
        fit_params["device"] = devices[0]

    rtf_model.fit(
        train_data,
        **fit_params
    )

    rtf_model.save(path=parent_dir / "trained_model")

    return rtf_model


def sample_realtabformer():
    pass