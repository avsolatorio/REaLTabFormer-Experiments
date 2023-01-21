# bash icml_realtabformer_exp.sh /home/jupyter-wb536061/.local/share/virtualenvs/be-great-env-gTWcYI-F/bin/python

PYTHON_GREAT_ENV=$1
CUDA_DEVICE=$2

${PYTHON_GREAT_ENV} scripts/run_great_experiments.py --run_data_id --data_id=diabetes --cuda_device=${CUDA_DEVICE}

${PYTHON_GREAT_ENV} scripts/run_great_experiments.py --run_data_id --data_id=california --cuda_device=${CUDA_DEVICE}

${PYTHON_GREAT_ENV} scripts/run_great_experiments.py --run_data_id --data_id=adult --cuda_device=${CUDA_DEVICE}

${PYTHON_GREAT_ENV} scripts/run_great_experiments.py --run_data_id --data_id=abalone --cuda_device=${CUDA_DEVICE}

${PYTHON_GREAT_ENV} scripts/run_great_experiments.py --run_data_id --data_id=fb-comments --cuda_device=${CUDA_DEVICE}

${PYTHON_GREAT_ENV} scripts/run_great_experiments.py --run_data_id --data_id=cardio --cuda_device=${CUDA_DEVICE}

${PYTHON_GREAT_ENV} scripts/run_great_experiments.py --run_data_id --data_id=buddy --cuda_device=${CUDA_DEVICE}
