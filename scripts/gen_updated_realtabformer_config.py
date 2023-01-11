import argparse
from pathlib import Path
import shutil
from config_lib import load_config
import subprocess as sub


EXP_DIR = Path(__file__).parent.parent / "exp"
print(EXP_DIR)


def copy_base_config():
    src_config = EXP_DIR / "base_rtf_config.toml"

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
    def _copy_base(data_id):
        # https://unix.stackexchange.com/a/428422
        with open((EXP_DIR / data_id / "realtabformer" / "base_config.toml"), "w") as fout:
            sub.check_call([
                "comm", "-13",
                (EXP_DIR / "base_rtf_config.toml").as_posix(),
                (EXP_DIR / data_id / "realtabformer" / "config.toml").as_posix()],
                stdout=fout)

    for data_path in EXP_DIR.glob("*"):
        if data_path.is_file():
            continue

        _copy_base(data_path.name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--copy_base_config", action="store_true", default=False)
    parser.add_argument("--gen_base_configs", action="store_true", default=False)

    args = parser.parse_args()

    if args.copy_base_config:
        copy_base_config()

    if args.gen_base_configs:
        gen_base_configs()


if __name__ == "__main__":
    main()
