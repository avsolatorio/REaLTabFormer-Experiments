import os
import argparse
from train_sample_realtabformer import train_realtabformer, sample_realtabformer
import zero

from config_lib import load_config, save_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', metavar='FILE')
    parser.add_argument('--train', action='store_true',  default=False)
    parser.add_argument('--sample', action='store_true',  default=False)
    parser.add_argument('--eval', action='store_true',  default=False)
    parser.add_argument('--change_val', action='store_true',  default=False)

    parser.add_argument('--experiment_id', type=str,  default=None)
    parser.add_argument('--n_datasets', type=int,  default=10)
    parser.add_argument('--gen_batch', type=int,  default=128)

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
            device=raw_config['device'],
            config_file=args.config
        )
    if args.sample:
        sample_realtabformer(
            parent_dir=raw_config['parent_dir'],
            real_data_path=raw_config['real_data_path'],
            experiment_id=args.experiment_id,
            n_datasets=args.n_datasets,
            device=raw_config['device'],
            gen_batch=args.gen_batch,
        )

    save_file(os.path.join(raw_config['parent_dir'], 'info.json'), os.path.join(raw_config['real_data_path'], 'info.json'))

    print(f'Elapsed time: {str(timer)}')

if __name__ == '__main__':
    main()