import argparse
import sys
import subprocess as sub
from pathlib import Path


EXP_DIR = Path(__file__).parent.parent / "exp"


def run_training_sampling(data_ids, cuda_device: int = None, from_exp_version: str = None, to_before_exp_version: str = None):
    assert "realtabformer-env" in sys.executable
    PROJ_DIR = EXP_DIR.parent.as_posix()

    if cuda_device == "":
        cuda_device = None

    for data_id in data_ids:
        VERSIONS = sorted([rl.name for rl in (EXP_DIR / data_id / "realtabformer").glob("0.*") if rl.is_dir()])

        for version in VERSIONS:
            print(version, data_id)

            if from_exp_version and (version < from_exp_version):
                print(f"Skipping lower version than specified {from_exp_version}...")
                continue

            if to_before_exp_version and (version >= to_before_exp_version):
                print(f"Skipping this ({to_before_exp_version}) or higher version...")
                continue

            train_command = f"{sys.executable} scripts/pipeline_realtabformer.py --config exp/{data_id}/realtabformer/{version}/config.toml --train"
            sample_command = f"{sys.executable} scripts/pipeline_realtabformer.py --config exp/{data_id}/realtabformer/{version}/config.toml --sample --gen_batch=512"

            if cuda_device is not None:
                train_command = f"CUDA_VISIBLE_DEVICES={cuda_device} {train_command}"
                sample_command = f"CUDA_VISIBLE_DEVICES={cuda_device} {sample_command}"

            train_command = f"cd {PROJ_DIR} && {train_command}"
            sample_command = f"cd {PROJ_DIR} && {sample_command}"

            print(train_command)
            print(sample_command)

            sub.call(train_command, shell=True)
            sub.call(sample_command, shell=True)


def run_server_cuda0(from_exp_version: str = None, to_before_exp_version: str = None):
    run_training_sampling(
        data_ids=["cardio", "gesture", "miniboone"],
        # data_ids=["cardio", "gesture"],
        # data_ids=["cardio", "miniboone"],
        cuda_device=0,
        from_exp_version=from_exp_version,
        to_before_exp_version=to_before_exp_version,
    )


def run_server_cuda1(from_exp_version: str = None, to_before_exp_version: str = None):
    run_training_sampling(
        data_ids=["fb-comments", "house", "higgs-small"],
        cuda_device=1,
        from_exp_version=from_exp_version,
        to_before_exp_version=to_before_exp_version,
    )


def run_other(from_exp_version: str = None, to_before_exp_version: str = None):
    run_other_small(from_exp_version=from_exp_version, to_before_exp_version=to_before_exp_version)
    run_other_big(from_exp_version=from_exp_version, to_before_exp_version=to_before_exp_version)


def run_other_small(from_exp_version: str = None, to_before_exp_version: str = None):
    run_training_sampling(
        data_ids=["churn2", "diabetes", "insurance", "abalone", "wilt"],
        cuda_device=None,
        from_exp_version=from_exp_version,
        to_before_exp_version=to_before_exp_version,
    )


def run_other_big(from_exp_version: str = None, to_before_exp_version: str = None):
    run_training_sampling(
        data_ids=["buddy", "california", "adult"],
        cuda_device=None,
        from_exp_version=from_exp_version,
        to_before_exp_version=to_before_exp_version,
    )


def run_data_id(data_id, cuda_device: int = None, from_exp_version: str = None, to_before_exp_version: str = None):
    run_training_sampling(
        data_ids=[data_id],
        cuda_device=cuda_device,
        from_exp_version=from_exp_version,
        to_before_exp_version=to_before_exp_version,
    )


def run_icml_ablation(cuda_device: int = None, from_exp_version: str = None, to_before_exp_version: str = None):
    small = ["abalone", "diabetes"]
    mid = ["buddy", "california", "adult"]
    large = ["fb-comments"]

    data_ids = small + mid + large

    run_training_sampling(
        data_ids=data_ids,
        cuda_device=cuda_device,
        from_exp_version=from_exp_version,
        to_before_exp_version=to_before_exp_version,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--run_server_cuda0', action='store_true',  default=False)
    parser.add_argument('--run_server_cuda1', action='store_true',  default=False)
    parser.add_argument('--run_other', action='store_true',  default=False)
    parser.add_argument('--run_other_small', action='store_true',  default=False)
    parser.add_argument('--run_other_big', action='store_true',  default=False)
    parser.add_argument('--run_data_id', action='store_true',  default=False)
    parser.add_argument('--run_icml_ablation', action='store_true',  default=False)

    parser.add_argument('--data_id', type=str,  default=None)
    parser.add_argument('--cuda_device', type=str,  default=None)
    parser.add_argument('--from_exp_version', type=str,  default=None)
    parser.add_argument('--to_before_exp_version', type=str,  default=None)


    args = parser.parse_args()

    cuda_device = None
    if args.cuda_device:
        cuda_device = int(args.cuda_device)

    if args.run_server_cuda0:
        run_server_cuda0(args.from_exp_version, args.to_before_exp_version)

    if args.run_server_cuda1:
        run_server_cuda1(args.from_exp_version, args.to_before_exp_version)

    if args.run_other:
        run_other(args.from_exp_version, args.to_before_exp_version)

    if args.run_other_small:
        run_other_small(args.from_exp_version, args.to_before_exp_version)

    if args.run_other_big:
        run_other_big(args.from_exp_version, args.to_before_exp_version)

    if args.run_data_id:
        assert args.data_id is not None
        run_data_id(args.data_id, cuda_device, args.from_exp_version, args.to_before_exp_version)

    if args.run_icml_ablation:
        run_icml_ablation(cuda_device, args.from_exp_version, args.to_before_exp_version)


if __name__ == '__main__':
    main()
