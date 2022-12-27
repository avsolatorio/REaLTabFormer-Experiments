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
                |- checkpoints
                    |- split-<seed_1>
                    |- split-<seed_2>
                    |- split-<seed_3>
                    |- split-<seed_...>
                    |- split-<seed_N>
```

# Special installation notes

We use the github install since the pip package is not yet updated with the fix that handles the random `IndexError` in `great_utils.py:_convert_text_to_tabular_data:td[values[0]] = [values[1]]`.
```
pipenv install -e git+https://github.com/kathrinse/be_great@main#egg=be_great
```

## Environment

Miniconda can be installed, then simply create a python environment.

```
conda create --name py39 python=3.9
conda activate py39
pip install pipenv
```


# Data sources

The following are the sources of the datasets used in the experiments:

- **Adult Income Dataset**: https://archive.ics.uci.edu/ml/datasets/Adult/
- **HELOC Dataset**: https://www.kaggle.com/datasets/averkiyoliabev/home-equity-line-of-creditheloc
- **Travel Customers Dataset**: https://www.kaggle.com/datasets/tejashvi14/tour-travels-customer-churn-prediction
- **California Housing Dataset**: https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset
- **Beijing PM2.5 Dataset**: https://archive.ics.uci.edu/ml/datasets/Beijing+PM2.5+Data
- **Online News Popularity Dataset**: https://archive.ics.uci.edu/ml/datasets/online+news+popularity
- **Predict Diabetes Dataset**: https://www.kaggle.com/datasets/whenamancodes/predict-diabities

# Data summary

## Adult Income Dataset

- Number of observations: 32,561
- Numeric columns: 6
- Categorical columns: 9
- Target variable: "income"
- Positive target value: ">50K"
- Target classes: ["<=50K", ">50K"]
- Missing values: assumed to be encoded by "?"
- Variable types:
  <pre>
    - age                int64
    - workclass         object
    - fnlwgt             int64
    - education         object
    - education-num      int64
    - marital-status    object
    - occupation        object
    - relationship      object
    - race              object
    - sex               object
    - capital-gain       int64
    - capital-loss       int64
    - hours-per-week     int64
    - native-country    object
    - income            object</pre>

## HELOC Dataset

- Number of observations: 9,871
- Numeric columns: 23
- Categorical columns: 1
- Target variable: "RiskPerformance"
- Positive target value: "Good"
- Target classes: ["Bad", "Good"]
- Missing values: assumed to be encoded by -9
- Variable types:
  <pre>
    - RiskPerformance                       object
    - ExternalRiskEstimate                   int64
    - MSinceOldestTradeOpen                  int64
    - MSinceMostRecentTradeOpen              int64
    - AverageMInFile                         int64
    - NumSatisfactoryTrades                  int64
    - NumTrades60Ever2DerogPubRec            int64
    - NumTrades90Ever2DerogPubRec            int64
    - PercentTradesNeverDelq                 int64
    - MSinceMostRecentDelq                   int64
    - MaxDelq2PublicRecLast12M               int64
    - MaxDelqEver                            int64
    - NumTotalTrades                         int64
    - NumTradesOpeninLast12M                 int64
    - PercentInstallTrades                   int64
    - MSinceMostRecentInqexcl7days           int64
    - NumInqLast6M                           int64
    - NumInqLast6Mexcl7days                  int64
    - NetFractionRevolvingBurden             int64
    - NetFractionInstallBurden               int64
    - NumRevolvingTradesWBalance             int64
    - NumInstallTradesWBalance               int64
    - NumBank2NatlTradesWHighUtilization     int64
    - PercentTradesWBalance                  int64</pre>
- Notes:
  - Removed 588 out of the original 10,459 observations. These observations have all missing values across the variables.

## Travel Customers Dataset

- Number of observations: 954
- Numeric columns: 3
- Categorical columns: 4
- Target variable: "Target"
- Positive target value: 1
- Target classes: [0, 1]
- Missing values: None
- Variable types:
  <pre>
    - Age                            int64
    - FrequentFlyer                 object
    - AnnualIncomeClass             object
    - ServicesOpted                  int64
    - AccountSyncedToSocialMedia    object
    - BookedHotelOrNot              object
    - Target                         int64</pre>

## Predict Diabetes Dataset

- Number of observations: 768
- Numeric columns: 9
- Categorical columns: 0
- Target variable: "Outcome"
- Positive target value: 1
- Target classes: [0, 1]
- Missing values: None
- Variable types:
  <pre>
    - Pregnancies                   int64
    - Glucose                       int64
    - BloodPressure                 int64
    - SkinThickness                 int64
    - Insulin                       int64
    - BMI                         float64
    - DiabetesPedigreeFunction    float64
    - Age                           int64
    - Outcome                       int64</pre>


# Generating dataset splits

```
pipenv run python scripts/split_train_test.py
```
