import argparse
import sys
import subprocess as sub
from pathlib import Path


EXP_DIR = Path(__file__).parent.parent / "exp"
VERSIONS = sorted([rl.name for rl in (EXP_DIR / "abalone" / "realtabformer").glob("0.*") if rl.is_dir()])


def run_training_sampling(data_ids, cuda_device: int = None):
    assert "realtabformer-env" in sys.executable
    PROJ_DIR = EXP_DIR.parent.as_posix()

    for version in VERSIONS:
        for data_id in data_ids:
            print(version, data_id)

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


def run_server_cuda0():
    run_training_sampling(
        data_ids=["cardio", "gesture", "miniboone"],
        # data_ids=["cardio", "miniboone"],
        cuda_device=0
    )


def run_server_cuda1():
    run_training_sampling(
        data_ids=["fb-comments", "house", "higgs-small"],
        cuda_device=1
    )


def run_other():
    run_other_small()
    run_other_big()


def run_other_small():
    run_training_sampling(
        data_ids=["churn2", "diabetes", "insurance", "abalone", "wilt"],
        cuda_device=None
    )


def run_other_big():
    run_training_sampling(
        data_ids=["buddy", "california", "adult"],
        cuda_device=None
    )


def run_data_id(data_id, cuda_device: int = None):
    run_training_sampling(
        data_ids=[data_id],
        cuda_device=cuda_device
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--run_server_cuda0', action='store_true',  default=False)
    parser.add_argument('--run_server_cuda1', action='store_true',  default=False)
    parser.add_argument('--run_other', action='store_true',  default=False)
    parser.add_argument('--run_other_small', action='store_true',  default=False)
    parser.add_argument('--run_other_big', action='store_true',  default=False)
    parser.add_argument('--run_data_id', action='store_true',  default=False)

    parser.add_argument('--data_id', type=str,  default=None)
    parser.add_argument('--cuda_device', type=int,  default=None)


    args = parser.parse_args()

    if args.run_server_cuda0:
        run_server_cuda0()
    elif args.run_server_cuda1:
        run_server_cuda1()
    elif args.run_other:
        run_other()
    elif args.run_other_small:
        run_other_small()
    elif args.run_other_big:
        run_other_big()
    elif args.run_data_id:
        assert args.data_id is not None
        run_data_id(args.data_id, args.cuda_device)

if __name__ == '__main__':
    main()
