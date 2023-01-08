import json
import shutil
import joblib
import torch
from pathlib import Path
import realtabformer
from realtabformer import REaLTabFormer
from transformers import GPT2Config


def train_realtabformer(
    parent_dir,
    real_data_path,
    model_params,
    device = "cpu",
    config_file = None,
):
    # Make sure that the rtf model specified in the config
    # is similar to the one installed in the environment.
    assert model_params["version"] == realtabformer.__version__

    real_data_path = Path(real_data_path)
    parent_dir = Path(parent_dir)

    data_info = json.loads((real_data_path / "info.json").read_text())
    sample_gen_size = data_info.get("train_size", 0) + data_info.get("val_size", 0) + data_info.get("test_size", 0)

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

    dir_params = {
        "checkpoints_dir": parent_dir / model_params["meta"]["checkpoints_dir"],
        "samples_save_dir": parent_dir / model_params["meta"]["samples_save_dir"],
    }

    # Set up the model
    rtf_model = REaLTabFormer(**train_params, **dir_params, **training_args_kwargs)

    if Path("/content").exists():
        print("In colab....")
        # We are in colab???
        rtf_model.training_args_kwargs["output_dir"] = (Path("/content") / f"{dir_params['checkpoints_dir'].name}-{real_data_path.name}").as_posix()
    else:
        rtf_model.training_args_kwargs["output_dir"] = dir_params["checkpoints_dir"].as_posix()

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

    save_model_path = parent_dir / "trained_model"
    rtf_model.save(path=save_model_path)

    rtf_model.sample(
        n_samples=sample_gen_size, gen_batch=128, device=fit_params["device"],
        save_samples=True
    )

    # Save other artefacts
    experiment_save_path = save_model_path / rtf_model.experiment_id
    if config_file:
        assert experiment_save_path.exists()
        model_config_file = experiment_save_path / "config.toml"
        shutil.copy2(
            config_file,
            model_config_file
        )

    if (rtf_model.checkpoints_dir / "best-disc-model").exists():
        shutil.copytree(rtf_model.checkpoints_dir / "best-disc-model", experiment_save_path)

    if (rtf_model.checkpoints_dir / "mean-best-disc-model").exists():
        shutil.copytree(rtf_model.checkpoints_dir / "mean-best-disc-model", experiment_save_path)
    if (rtf_model.checkpoints_dir / "last-epoch-model").exists():
        shutil.copytree(rtf_model.checkpoints_dir / "last-epoch-model", experiment_save_path)

    return rtf_model


def sample_realtabformer():
    pass