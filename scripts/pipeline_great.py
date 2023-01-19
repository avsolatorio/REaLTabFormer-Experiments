import os
import argparse
from train_sample_great import train_great, sample_great
import zero

from config_lib import load_config, save_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', metavar='FILE')
    parser.add_argument('--train', action='store_true',  default=False)
    parser.add_argument('--sample', action='store_true',  default=False)

    parser.add_argument('--n_datasets', type=int,  default=5)

    args = parser.parse_args()
    raw_config = load_config(args.config)
    timer = zero.Timer()
    timer.run()
    save_file(os.path.join(raw_config['parent_dir'], 'config.toml'), args.config)
    great_model = None
    if args.train:
        great_model = train_great(
            parent_dir=raw_config['parent_dir'],
            real_data_path=raw_config['real_data_path'],
            model_params=raw_config['model_params'],
        )
    if args.sample:
        sample_great(
            parent_dir=raw_config['parent_dir'],
            real_data_path=raw_config['real_data_path'],
            model_params=raw_config['model_params'],
            n_datasets=args.n_datasets,
        )

    save_file(os.path.join(raw_config['parent_dir'], 'info.json'), os.path.join(raw_config['real_data_path'], 'info.json'))

    print(f'Elapsed time: {str(timer)}')

if __name__ == '__main__':
    main()