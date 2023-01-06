from typing import Dict, cast, Any, Union
from pathlib import Path
import tomli
import shutil
import os
import argparse
from train_sample_realtabformer import train_realtabformer, sample_realtabformer
import zero

RawConfig = Dict[str, Any]

_CONFIG_NONE = '__none__'

def _replace(data, condition, value):
    def do(x):
        if isinstance(x, dict):
            return {k: do(v) for k, v in x.items()}
        elif isinstance(x, list):
            return [do(y) for y in x]
        else:
            return value if condition(x) else x

    return do(data)


def unpack_config(config: RawConfig) -> RawConfig:
    config = cast(RawConfig, _replace(config, lambda x: x == _CONFIG_NONE, None))
    return config


def load_config(path: Union[Path, str]) -> Any:
    with open(path, 'rb') as f:
        return unpack_config(tomli.load(f))


def save_file(parent_dir, config_path):
    try:
        dst = os.path.join(parent_dir)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(os.path.abspath(config_path), dst)
    except shutil.SameFileError:
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', metavar='FILE')
    parser.add_argument('--train', action='store_true',  default=False)
    parser.add_argument('--sample', action='store_true',  default=False)
    parser.add_argument('--eval', action='store_true',  default=False)
    parser.add_argument('--change_val', action='store_true',  default=False)

    args = parser.parse_args()
    raw_config = load_config(args.config)
    timer = zero.Timer()
    timer.run()
    save_file(os.path.join(raw_config['parent_dir'], 'config.toml'), args.config)
    rtf_model = None
    if args.train:
        rtf_model = train_realtabformer(
            parent_dir=raw_config['parent_dir'],
            real_data_path=raw_config['real_data_path'],
            model_params=raw_config['model_params'],
            device=raw_config['device']
        )
    # if args.sample:
    #     sample_realtabformer(
    #         synthesizer=ctabgan,
    #         parent_dir=raw_config['parent_dir'],
    #         real_data_path=raw_config['real_data_path'],
    #         num_samples=raw_config['sample']['num_samples'],
    #         model_params=raw_config['train_params'],
    #         change_val=args.change_val,
    #         seed=raw_config['sample']['seed'],
    #         device=raw_config['device']
    #     )

    save_file(os.path.join(raw_config['parent_dir'], 'info.json'), os.path.join(raw_config['real_data_path'], 'info.json'))

    print(f'Elapsed time: {str(timer)}')

if __name__ == '__main__':
    main()