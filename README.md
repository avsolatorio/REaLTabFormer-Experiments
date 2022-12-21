# REaLTabFormer-Experiments
This repository contains the materials of the experiments conducted for the REaLTabFormer paper.


# Experiment design

First we identify the datasets that we test in the experiments. For each dataset, we generate `N` random train-test splits of the data. We structure the data directory as folllows:

```
- data
    |- input
        |- data_id
            |- split-<seed_1>
            |- split-<seed_2>
            |- split-<seed_3>
            |- split-<seed_...>
            |- split-<seed_N>
    |- models
        |- model_id
            |- data_id
                |- trained_model
                    |- split-<seed_1>
                    |- split-<seed_2>
                    |- split-<seed_3>
                    |- split-<seed_...>
                    |- split-<seed_N>
                |- samples
                    |- split-<seed_1>
                    |- split-<seed_2>
                    |- split-<seed_3>
                    |- split-<seed_...>
                    |- split-<seed_N>
```

# Data sources

The following are the sources of the datasets used in the experiments:

- **Adult Income Dataset**: https://archive.ics.uci.edu/ml/datasets/Adult/
- **HELOC Dataset**: https://www.kaggle.com/datasets/averkiyoliabev/home-equity-line-of-creditheloc
- **Travel Customers Dataset**: https://www.kaggle.com/datasets/tejashvi14/tour-travels-customer-churn-prediction
- **California Housing Dataset**: https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset
- **Beijing PM2.5 Dataset**: https://archive.ics.uci.edu/ml/datasets/Beijing+PM2.5+Data
- **Online News Popularity Dataset**: https://archive.ics.uci.edu/ml/datasets/online+news+popularity


# Generating dataset splits

```
pipenv run python scripts/split_train_test.py
```
