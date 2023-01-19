# bash icml_realtabformer_exp.sh /home/jupyter-wb536061/.local/share/virtualenvs/realtabformer-env-gTWcYI-F/bin/python 0.0.6

PYTHON_RTF_ENV=$1
FROM_EXP_VERSION=$2
CUDA_DEVICE=$3

${PYTHON_RTF_ENV} scripts/run_experiments.py --run_data_id --data_id=diabetes --from_exp_version=${FROM_EXP_VERSION} --cuda_device=${CUDA_DEVICE}

${PYTHON_RTF_ENV} scripts/run_experiments.py --run_data_id --data_id=california --from_exp_version=${FROM_EXP_VERSION} --cuda_device=${CUDA_DEVICE}

${PYTHON_RTF_ENV} scripts/run_experiments.py --run_data_id --data_id=adult --from_exp_version=${FROM_EXP_VERSION} --cuda_device=${CUDA_DEVICE}

${PYTHON_RTF_ENV} scripts/run_experiments.py --run_data_id --data_id=fb-comments --from_exp_version=${FROM_EXP_VERSION} --cuda_device=${CUDA_DEVICE}


${PYTHON_RTF_ENV} scripts/run_experiments.py --run_data_id --data_id=abalone --from_exp_version=${FROM_EXP_VERSION} --cuda_device=${CUDA_DEVICE}

${PYTHON_RTF_ENV} scripts/run_experiments.py --run_data_id --data_id=cardio --from_exp_version=${FROM_EXP_VERSION} --cuda_device=${CUDA_DEVICE}

${PYTHON_RTF_ENV} scripts/run_experiments.py --run_data_id --data_id=buddy --from_exp_version=${FROM_EXP_VERSION} --cuda_device=${CUDA_DEVICE}
