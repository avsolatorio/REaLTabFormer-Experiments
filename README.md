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
                |- split-<seed_1>
                |- split-<seed_2>
                |- split-<seed_3>
                |- split-<seed_...>
                |- split-<seed_N>
    |- samples
        |- model_id
            |- data_id
                |- split-<seed_1>
                |- split-<seed_2>
                |- split-<seed_3>
                |- split-<seed_...>
                |- split-<seed_N>
```

# Data sources

The following are the sources of the datasets used in the experiments:

- **Adult Income Dataset**:
- **HELOC Dataset**:
- **Travel Customers Dataset**:
- **California Housing Dataset**:
