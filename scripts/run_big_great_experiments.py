import argparse
import sys
import subprocess as sub
from pathlib import Path


EXP_DIR = Path(__file__).parent.parent / "exp"


def run_training_sampling(data_ids, cuda_device: int = None):
    assert "be-great-env" in sys.executable
    PROJ_DIR = EXP_DIR.parent.as_posix()

    if cuda_device == "":
        cuda_device = None

    for data_id in data_ids:
        train_sample_command = f"{sys.executable} scripts/pipeline_great.py --config exp/{data_id}/big_great/config.toml --train --sample"

        if cuda_device is not None:
            train_sample_command = f"CUDA_VISIBLE_DEVICES={cuda_device} {train_sample_command}"

        train_sample_command = f"cd {PROJ_DIR} && {train_sample_command}"

        print(train_sample_command)

        sub.call(train_sample_command, shell=True)


def run_server_cuda0():
    run_training_sampling(
        data_ids=["cardio", "gesture", "miniboone"],
        # data_ids=["cardio", "gesture"],
        # data_ids=["cardio", "miniboone"],
        cuda_device=0,

    )


def run_server_cuda1():
    run_training_sampling(
        data_ids=["fb-comments", "house", "higgs-small"],
        cuda_device=1,

    )


def run_other():
    run_other_small()
    run_other_big()


def run_other_small():
    run_training_sampling(
        data_ids=["churn2", "diabetes", "insurance", "abalone", "wilt"],
        cuda_device=None,

    )


def run_other_big():
    run_training_sampling(
        data_ids=["buddy", "california", "adult"],
        cuda_device=None,

    )


def run_data_id(data_id, cuda_device: int = None, ):
    run_training_sampling(
        data_ids=[data_id],
        cuda_device=cuda_device,

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
    parser.add_argument('--cuda_device', type=str,  default=None)


    args = parser.parse_args()

    cuda_device = None
    if args.cuda_device:
        cuda_device = int(args.cuda_device)

    if args.run_server_cuda0:
        run_server_cuda0()

    if args.run_server_cuda1:
        run_server_cuda1()

    if args.run_other:
        run_other()

    if args.run_other_small:
        run_other_small()

    if args.run_other_big:
        run_other_big()

    if args.run_data_id:
        assert args.data_id is not None
        run_data_id(args.data_id, cuda_device)

if __name__ == '__main__':
    main()
