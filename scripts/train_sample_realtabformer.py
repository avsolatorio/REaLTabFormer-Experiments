import json
import random
import shutil
import joblib
import numpy as np
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

    in_colab = Path("/content").exists()
    if in_colab:
        print("In colab....")
        # We are in colab???
        rtf_model.training_args_kwargs["output_dir"] = (Path("/content") / f"{dir_params['checkpoints_dir'].name}-{real_data_path.name}").as_posix()
    else:
        rtf_model.training_args_kwargs["output_dir"] = dir_params["checkpoints_dir"].as_posix()

    train_data = joblib.load(real_data_path / "full_train.df.pkl")
    fit_params = model_params["fit"]

    if data_info["id"].startswith("cardio"):
        # Drop the "id" column if exists.
        if "id" in train_data.columns:
            train_data = train_data.drop("id", axis=1)

    if model_params["meta"].get("use_target_col", False):
        fit_params["target_col"] = data_info["cols"]["target"]

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

    experiment_save_checkpoints_path = experiment_save_path / "rtf_checkpoints"

    for artefact in ["best-disc-model", "mean-best-disc-model", "last-epoch-model"]:
        print("Copying artefacts from:", artefact)
        if (rtf_model.checkpoints_dir / artefact).exists():
            shutil.copytree(
                rtf_model.checkpoints_dir / artefact,
                experiment_save_checkpoints_path / artefact,
                dirs_exist_ok=True)

    if in_colab:
        # Clean up checkpoints when in colab
        shutil.rmtree(
            rtf_model.training_args_kwargs["output_dir"],
            ignore_errors=True)

    return rtf_model


def sample_realtabformer(
    parent_dir,
    real_data_path,
    experiment_id: str = None,
    n_datasets: int = 1,
    device = "cpu",
    gen_batch: int = 128,
):
    real_data_path = Path(real_data_path)
    parent_dir = Path(parent_dir)

    data_id = real_data_path.name

    data_info = json.loads((real_data_path / "info.json").read_text())
    sample_gen_size = data_info.get("train_size", 0) + data_info.get("val_size", 0) + data_info.get("test_size", 0)

    save_model_path = parent_dir / "trained_model"

    if experiment_id is None:
        experiment_paths = [path.parent for path in save_model_path.glob("*/rtf_model.pt")]
    else:
        experiment_paths = [save_model_path / experiment_id]

    for exp_path in experiment_paths:
        rtf_model = REaLTabFormer.load_from_dir(exp_path)

        experiment_id = rtf_model.experiment_id
        exp_samples_dir: Path = rtf_model.samples_save_dir / experiment_id
        exp_samples_dir.mkdir(parents=True, exist_ok=True)

        for saved_type in ["best-disc-model", "mean-best-disc-model"]:
            saved_path = (exp_path / "rtf_checkpoints" / saved_type)
            if not saved_path.exists():
                print(f"Skipping {saved_path}, not exists...")

            rtf_model.model = rtf_model.model.from_pretrained(saved_path.as_posix())

            saved_type = saved_type.replace("-", "_")
            for seed in range(n_datasets):
                sample_save_path = exp_samples_dir / saved_type / f"rtf_sample-{experiment_id}-{saved_type}-seed_{seed}.pkl"
                sample_save_path.parent.mkdir(parents=True, exist_ok=True)

                print(f"Sampling for {data_id}, exp::{experiment_id}, sv::{saved_type}, seed::{seed} / {n_datasets}")

                if sample_save_path.exists():
                    print(f"Sample already available for {sample_save_path}, skipping...")
                    continue

                np.random.seed(seed)
                random.seed(seed)
                torch.manual_seed(seed)

                sample = rtf_model.sample(
                    n_samples=sample_gen_size,
                    gen_batch=gen_batch,
                    device=device,
                    save_samples=False
                )

                joblib.dump(sample, sample_save_path)
