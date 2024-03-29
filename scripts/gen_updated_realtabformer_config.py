"""
This module manages the configuration for the realtabformer experiments.

We define a base config file in `exp/base_rtf_config.toml`.
This config file contains all configuration values that are common
for all data experiments.

Then, for each data, we create a exp/{data_id}/realtabformer/base_config.toml.
This is were any config values that are specific to the data should be set. For
example, the miniboone data generally require smaller batch size than the other
datasets, so we set the batch_size value that works here.

Then, each variation of value(s) set in the `base_rtf_config.toml` file should
correspond to an updated `conf_version`. A `conf_version` is the experimental unit
where the full training and sampling of the realtabformer model for the data is
required.

This setup will allow us to track all experiment settings, artefacts, and results.
"""

import argparse
import copy
from pathlib import Path
import toml
import shutil
from config_lib import load_config
import subprocess as sub


EXP_DIR = Path(__file__).parent.parent / "exp"
BASE_CONF_PATH = EXP_DIR / "base_rtf_config.toml"
print(EXP_DIR)


def merge_conf(base, other):
    base = copy.deepcopy(base)
    if not isinstance(base, dict) or not isinstance(other, dict):
        return other
    for k in other:
        if k in base:
            base[k] = merge_conf(base[k], other[k])
        else:
            base[k] = other[k]
    return base


def copy_base_config():
    src_config = BASE_CONF_PATH

    conf = src_config.read_text()

    for data_path in EXP_DIR.glob("*"):
        if data_path.is_file():
            continue

        rtf_dir = (data_path / "realtabformer")

        rtf_dir.mkdir(parents=True, exist_ok=True)

        # if (rtf_dir / "config.toml").exists():
        #     continue
        # else:

        cconf = str(conf)
        cconf = cconf.replace("/DATA_ID/", f"/{data_path.name}/")
        (rtf_dir / "config.toml").write_text(cconf)

        # shutil.copy2(src_config, rtf_dir / "config.toml")
        assert (rtf_dir / "config.toml").exists()


def gen_base_configs():
    # We only need to do this once to migrate from
    # `copy_base_config`.
    def _copy_base(data_id: str):
        # https://unix.stackexchange.com/a/428422
        base_conf = BASE_CONF_PATH
        data_conf = (EXP_DIR / data_id / "realtabformer" / "config.toml")

        if not (base_conf.exists() and data_conf.exists()):
            return

        with open((EXP_DIR / data_id / "realtabformer" / "base_config.toml"), "w") as fout:
            sub.check_call([
                "comm", "-13",
                base_conf.as_posix(),
                data_conf.as_posix()],
                stdout=fout)

    for data_path in EXP_DIR.glob("*"):
        if data_path.is_file():
            continue

        _copy_base(data_path.name)


def gen_exp_config(from_exp_version: str = None):
    base_conf = toml.loads(BASE_CONF_PATH.read_text())

    for data_path in EXP_DIR.glob("*"):
        if data_path.is_file():
            continue
        data_id = data_path.name

        data_conf = (EXP_DIR / data_id / "realtabformer" / "base_config.toml")
        base_data_conf = base_conf

        if data_conf.exists():
            base_data_conf = merge_conf(
                base_data_conf, toml.loads(data_conf.read_text()))

        parent_dir = Path(base_data_conf["parent_dir"])
        assert parent_dir.exists(), f"Dir {parent_dir.as_posix()} should exist... Check full path..."

        parent_dir: Path = parent_dir / base_data_conf["conf_version"]

        # Change the parent dir to the experiment version dir.
        base_data_conf["parent_dir"] = f"{base_data_conf['parent_dir']}/{base_data_conf['conf_version']}"

        if from_exp_version and from_exp_version > base_data_conf['conf_version']:
            print(f"Skipping {base_data_conf['conf_version']}")
            continue

        if parent_dir.exists():
            current_conf = toml.loads((parent_dir / "config.toml").read_text())
            assert current_conf == base_data_conf, f"Version dir and config exists, but the config content are different for {data_id}..."
            print(f"Experiment version directory and config exists for {data_id}... Skipping...")
            continue

        parent_dir.mkdir()
        (parent_dir / "config.toml").write_text(toml.dumps(base_data_conf))


def update_rtf_version(from_rtf_version: str, to_rtf_version: str, from_exp_version: str = None):

    def _update_conf_version(conf_file):
        conf = conf_file.read_text()

        if f'version = "{from_rtf_version}"' in conf:

            conf = conf.replace(f'version = "{from_rtf_version}"', f'version = "{to_rtf_version}"')
            conf_file.write_text(conf)

            print(f"Updated conf version from {from_rtf_version} to {to_rtf_version}: {conf_file}")

    _update_conf_version(EXP_DIR / "base_rtf_config.toml")

    for data_path in EXP_DIR.glob("*"):
        if data_path.is_file():
            continue

        for conf_v in (data_path / "realtabformer").glob("0.*"):
            if not conf_v.is_dir():
                continue

            if from_exp_version is not None:
                if conf_v.name < from_exp_version:
                    continue

            conf_file = conf_v / "config.toml"
            if not conf_file.exists():
                continue

            _update_conf_version(conf_file)


def move_old_trained_models(from_exp_version: str = None):
    for data_path in EXP_DIR.glob("*"):
        if data_path.is_file():
            continue

        DATA_ID = data_path.name

        for conf_v in (data_path / "realtabformer").glob("0.*"):
            if not conf_v.is_dir():
                continue

            EXP_VERSION = conf_v.name
            if from_exp_version and EXP_VERSION < from_exp_version:
                continue

            for pname in ["trained_model", "rtf_samples", "rtf_checkpoints"]:
                src_path = conf_v / pname
                if src_path.exists():
                    dup = 1
                    tgt_path = src_path.with_name(f"old-{src_path.name}")
                    while tgt_path.exists():
                        tgt_path = tgt_path.with_name(f"old{dup:02}-{src_path.name}")
                        dup += 1

                    shutil.move(src_path, tgt_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--copy_base_config", action="store_true", default=False)
    parser.add_argument("--gen_base_configs", action="store_true", default=False)
    parser.add_argument("--gen_exp_config", action="store_true", default=False)
    parser.add_argument("--update_rtf_version", action="store_true", default=False)
    parser.add_argument("--move_old_trained_models", action="store_true", default=False)



    parser.add_argument("--from_rtf_version", default=None, type=str)
    parser.add_argument("--to_rtf_version", default=None, type=str)
    parser.add_argument("--from_exp_version", default=None, type=str)



    args = parser.parse_args()

    if args.copy_base_config:
        copy_base_config()

    if args.gen_base_configs:
        gen_base_configs()

    if args.gen_exp_config:
        gen_exp_config(from_exp_version=args.from_exp_version)

    if args.move_old_trained_models:
        move_old_trained_models(from_exp_version=args.from_exp_version)

    if args.update_rtf_version:
        assert (args.from_rtf_version and args.to_rtf_version)
        update_rtf_version(
            from_rtf_version=args.from_rtf_version,
            to_rtf_version=args.to_rtf_version,
            from_exp_version=args.from_exp_version,
        )


if __name__ == "__main__":
    main()
