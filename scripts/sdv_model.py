import joblib
from pathlib import Path
from sdv.tabular import GaussianCopula, CopulaGAN, CTGAN, TVAE


RANDOM_SEED = 1029
BASELINE_EPOCHS = 200
SDV_BATCH_SIZE = 510
BASE_DIR = Path(__file__).parent.parent / "data"
assert BASE_DIR.exists(), f"Make sure that the DATA_DIR ({BASE_DIR}) is correct..."

model_types = ["ctgan", "tvae", "copulagan", "gaussiancopula"]


def get_dirs(model_type, data_id):
    assert model_type in model_types

    print(model_type, data_id)

    EXP_DIR = BASE_DIR / "models" / model_type / data_id
    DATA_DIR = BASE_DIR / "input" / data_id

    save_dir = (EXP_DIR / "trained_model")
    save_dir.mkdir(parents=True, exist_ok=True)

    samples_save_dir = (EXP_DIR / "samples")
    samples_save_dir.mkdir(parents=True, exist_ok=True)

    save_dir = save_dir.as_posix()
    samples_save_dir = samples_save_dir.as_posix()

    return EXP_DIR, DATA_DIR, save_dir, samples_save_dir


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


def train_sample(model_type, data_id, sample_multiple: int = 10):

    model_type = "ctgan"
    _, DATA_DIR, save_dir, samples_save_dir = get_dirs(model_type, data_id)

    model = get_sdv_model(model_type)

    for path in DATA_DIR.glob("split_*"):
        split = path.name
        seed = split.split("_")[-1]
        data_fname = path / f"{data_id}_seed-{seed}.pkl"

        name = f"{model_type}_model-{data_id}_seed-{seed}"
        print(name)

        payload = joblib.load(data_fname)
        model.fit(payload["train"])

        # Save the trained model
        model.save(Path(save_dir) / f"{name}.pkl")

        # Generate samples
        samples = model.sample(num_rows=sample_multiple * len(payload["data"]))
        samples.to_csv(Path(samples_save_dir) / f"{name}.pkl")