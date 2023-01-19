# bash icml_realtabformer_exp.sh /home/jupyter-wb536061/.local/share/virtualenvs/realtabformer-env-gTWcYI-F/bin/python 0.0.6

PYTHON_RTF_ENV=$1
FROM_EXP_VERSION=$2
CUDA_DEVICE=$3


${PYTHON_RTF_ENV} scripts/run_experiments.py --run_data_id --data_id=diabetes --cuda_device=${CUDA_DEVICE} --from_exp_version=${FROM_EXP_VERSION}

${PYTHON_RTF_ENV} scripts/run_experiments.py --run_data_id --data_id=california --cuda_device=${CUDA_DEVICE} --from_exp_version=${FROM_EXP_VERSION}

${PYTHON_RTF_ENV} scripts/run_experiments.py --run_data_id --data_id=adult --cuda_device=${CUDA_DEVICE} --from_exp_version=${FROM_EXP_VERSION}

${PYTHON_RTF_ENV} scripts/run_experiments.py --run_data_id --data_id=fb-comments --cuda_device=${CUDA_DEVICE} --from_exp_version=${FROM_EXP_VERSION}


${PYTHON_RTF_ENV} scripts/run_experiments.py --run_data_id --data_id=abalone --cuda_device=${CUDA_DEVICE} --from_exp_version=${FROM_EXP_VERSION}

${PYTHON_RTF_ENV} scripts/run_experiments.py --run_data_id --data_id=cardio --cuda_device=${CUDA_DEVICE} --from_exp_version=${FROM_EXP_VERSION}

${PYTHON_RTF_ENV} scripts/run_experiments.py --run_data_id --data_id=buddy --cuda_device=${CUDA_DEVICE} --from_exp_version=${FROM_EXP_VERSION}
