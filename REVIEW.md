# Review Experiments

We document here the planned experiments that we will undertake for the ICML review comments.

## Additional experiments

First, we will need to train the models using the following variations:
1. Set the target mask rate to:
   1. (0.0.6.1) mask_rate=0
   2. (0.0.6.2) mask_rate=0.2
   3. (0.0.6) mask_rate=0.1 (done)
   4. (0.0.6.5) mask_rate=0.05
2. (0.0.6.3) Remove the use of overfitting auto-detection.
3. (0.0.6.4) Remove the use of overfitting auto-detection and set the target mask rate to 0.

We generate sub-versions of experiment 0.0.6 for all of these configurations.


Process:

Update `exp/base_rtf_config.toml` with the conf_version and the changes.

Then activate the REaLTabFormer-Experiments shell (pipenv shell).

Run `python scripts/gen_updated_realtabformer_config.py --gen_exp_config` to generate the conf files for all data.

Run the experiments:

```
cd realtabformer-env && pipenv shell && cd ../
python scripts/run_experiments.py --run_icml_ablation --from_exp_version=0.0.6.1 --cuda_device=1
```

## Data copying measure

Report on the number of data points that may have been generated exactly from the training data if we did not implement the target masking and the overfitting condition.

