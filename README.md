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
- **Mobile Price Dataset**: https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification
- **Oil Spill Dataset**: https://www.kaggle.com/datasets/sudhanshu2198/oil-spill-detection
- **Customer Personality Dataset**: https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis

# Data summary

Variables annotated with `^` implies categorical data.

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
- Numeric columns: 2
- Categorical columns: 5
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
    - Target                         int64^</pre>

## Predict Diabetes Dataset

- Number of observations: 768
- Numeric columns: 8
- Categorical columns: 1
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
    - Outcome                       int64^</pre>

## Mobile Price Dataset

- Number of observations: 2000
- Numeric columns: 14
- Categorical columns: 7
- Target variable: "price_range"
- Positive target value: 3
- Target classes: [0, 1, 2, 3]
- Missing values: None
- Variable types:
  <pre>
    - battery_power      int64
    - blue               int64^
    - clock_speed      float64
    - dual_sim           int64^
    - fc                 int64
    - four_g             int64^
    - int_memory         int64
    - m_dep            float64
    - mobile_wt          int64
    - n_cores            int64
    - pc                 int64
    - px_height          int64
    - px_width           int64
    - ram                int64
    - sc_h               int64
    - sc_w               int64
    - talk_time          int64
    - three_g            int64^
    - touch_screen       int64^
    - wifi               int64^
    - price_range        int64^</pre>

## Oil Spill Dataset

- Number of observations: 937
- Numeric columns: 49
- Categorical columns: 1
- Target variable: "target"
- Positive target value: 1
- Target classes: [0, 1]
- Missing values: None
- Variable types:
  <pre>
    - f_1         int64
    - f_2         int64
    - f_3       float64
    - f_4       float64
    - f_5         int64
    - f_6         int64
    - f_7       float64
    - f_8       float64
    - f_9       float64
    - f_10      float64
    - f_11      float64
    - f_12      float64
    - f_13      float64
    - f_14      float64
    - f_15      float64
    - f_16      float64
    - f_17      float64
    - f_18      float64
    - f_19      float64
    - f_20      float64
    - f_21      float64
    - f_22      float64
    - f_23        int64
    - f_24      float64
    - f_25      float64
    - f_26      float64
    - f_27      float64
    - f_28      float64
    - f_29      float64
    - f_30      float64
    - f_31      float64
    - f_32      float64
    - f_33      float64
    - f_34      float64
    - f_35        int64
    - f_36        int64
    - f_37      float64
    - f_38      float64
    - f_39        int64
    - f_40        int64
    - f_41      float64
    - f_42      float64
    - f_43      float64
    - f_44      float64
    - f_45      float64
    - f_46        int64
    - f_47      float64
    - f_48      float64
    - f_49      float64
    - target      int64^</pre>
- Notes:
  - Dropped variable `f_23` in the data since there is no variability in it. All values is zero across observations.

## Customer Personality Dataset

- Number of observations: 2240
- Numeric columns: 29 - 7 - 3
- Categorical columns: 7
- Target variable: "Response"
- Positive target value: 1
- Target classes: [0, 1]
- Missing values: None
- Variable types:
  <pre>
    - ID                       int64@
    - Year_Birth               int64
    - Education               object
    - Marital_Status          object
    - Income                 float64
    - Kidhome                  int64
    - Teenhome                 int64
    - Dt_Customer             object
    - Recency                  int64
    - MntWines                 int64
    - MntFruits                int64
    - MntMeatProducts          int64
    - MntFishProducts          int64
    - MntSweetProducts         int64
    - MntGoldProds             int64
    - NumDealsPurchases        int64
    - NumWebPurchases          int64
    - NumCatalogPurchases      int64
    - NumStorePurchases        int64
    - NumWebVisitsMonth        int64
    - AcceptedCmp3             int64^
    - AcceptedCmp4             int64^
    - AcceptedCmp5             int64^
    - AcceptedCmp1             int64^
    - AcceptedCmp2             int64^
    - Complain                 int64^
    - Z_CostContact            int64@
    - Z_Revenue                int64@
    - Response                 int64^</pre>
- Notes:
  - We drop 24 observations with missing value for the `Income` variable. We also drop the `ID` variable. We also drop th `Z_CostContact` and `Z_Revenue` variables due to no variability. The `Z_CostContact` variable all has `3` as its value while the `Z_Revenue` variable all has `11` for its value.

# Generating dataset splits

```
pipenv run python scripts/split_train_test.py
```


# Generating the datasets

We benchmark our model for standard tabular data on datasets used in the https://arxiv.org/abs/2209.15421 paper.

The [GitHub repo](https://github.com/rotot0/tab-ddpm) of the paper specifies how to download the dataset they used in the paper.

```
conda activate tddpm
cd $PROJECT_DIR
wget "https://www.dropbox.com/s/rpckvcs3vx7j605/data.tar?dl=0" -O data.tar
tar -xvf data.tar
```

We can also train their models by creating a conda environment as specified below (taken from their repo).

```
export REPO_DIR=/path/to/the/code
cd $REPO_DIR

conda create -n tddpm python=3.9.7
conda activate tddpm

# if the following commands do not succeed, update conda
conda env config vars set PYTHONPATH=${PYTHONPATH}:${REPO_DIR}
conda env config vars set PROJECT_DIR=${REPO_DIR}

pip install torch==1.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt

conda deactivate
conda activate tddpm
```

For reproducibility, we store a copy of the data dump. Note that a copy of this (our project) repo is synced in Google Drive. This allows us to link the data and use it on Google Colab.

Under the project directory, we clone the tab-ddpm repo. We then download the data as specified. Since the datasets included in the dump come in numpy stored object, we implemented a method to recreate the datasets and match, as much as possible, the train-val-test data in the data dump.

For this data generation, the relevant scripts used are:
- First, identify the actual index ids for the datasets.
  - Refer to colab: [Tab-DDPM REaLTabFormer Data Recreation](https://colab.research.google.com/drive/1zFQSqJFhtXl8n8TOmNfYrwzR5rKNBER5)
  - The output of this is already versioned in the repo.

- Then generate the datasets with pickled DataFrames as output and other metadata. Use `scripts/prep_datasets.py`.
  - Run as `pipenv run python scripts/prep_datasets.py`
  - If datasets exist, it will raise an error. Clear the data first, then rerun as usual:
    - `pipenv run python scripts/prep_datasets.py --clear`
    - `pipenv run python scripts/prep_datasets.py`

> :warning: **pandas and joblib are used to pickle and store the DataFrames**: to be sure, use the same pandas version (`pandas==1.5.2`) used to pickle the data when loading. Otherwise, an error may be raised.


# Batch size

We set the target batch size for datasets with training size:
- below 1,000 to 32.
- below 10,000 to 64.
- else 512.

Smaller batch size on small datasets could help in training.


# Training the models

```
# ## Train and Gen samples
# # Bizon Server GPU0
export EXP_VERSION=0.0.1

CUDA_VISIBLE_DEVICES=0 python scripts/pipeline_realtabformer.py --config exp/cardio/realtabformer/${EXP_VERSION}/config.toml --train && \
CUDA_VISIBLE_DEVICES=0 python scripts/pipeline_realtabformer.py --config exp/cardio/realtabformer/${EXP_VERSION}/config.toml --sample --gen_batch=512 && \

CUDA_VISIBLE_DEVICES=0 python scripts/pipeline_realtabformer.py --config exp/gesture/realtabformer/${EXP_VERSION}/config.toml --train && \
CUDA_VISIBLE_DEVICES=0 python scripts/pipeline_realtabformer.py --config exp/gesture/realtabformer/${EXP_VERSION}/config.toml --sample --gen_batch=512 && \

CUDA_VISIBLE_DEVICES=0 python scripts/pipeline_realtabformer.py --config exp/miniboone/realtabformer/${EXP_VERSION}/config.toml --train && \
CUDA_VISIBLE_DEVICES=0 python scripts/pipeline_realtabformer.py --config exp/miniboone/realtabformer/${EXP_VERSION}/config.toml --sample --gen_batch=512


# # Bizon Server GPU1
export EXP_VERSION=0.0.1

CUDA_VISIBLE_DEVICES=1 python scripts/pipeline_realtabformer.py --config exp/fb-comments/realtabformer/${EXP_VERSION}/config.toml --train && \
CUDA_VISIBLE_DEVICES=1 python scripts/pipeline_realtabformer.py --config exp/fb-comments/realtabformer/${EXP_VERSION}/config.toml --sample --gen_batch=512 && \

CUDA_VISIBLE_DEVICES=1 python scripts/pipeline_realtabformer.py --config exp/house/realtabformer/${EXP_VERSION}/config.toml --train && \
CUDA_VISIBLE_DEVICES=1 python scripts/pipeline_realtabformer.py --config exp/house/realtabformer/${EXP_VERSION}/config.toml --sample --gen_batch=512 && \

CUDA_VISIBLE_DEVICES=1 python scripts/pipeline_realtabformer.py --config exp/higgs-small/realtabformer/${EXP_VERSION}/config.toml --train && \
CUDA_VISIBLE_DEVICES=1 python scripts/pipeline_realtabformer.py --config exp/higgs-small/realtabformer/${EXP_VERSION}/config.toml --sample --gen_batch=512



# # Other: https://colab.research.google.com/drive/1bkspGMSimJntE1zBGZsKv3t7RyjSlL28
# if [ `basename "$PWD"` = "REaLTabFormer-Experiments" ]; then echo "hello"; fi
export EXP_VERSION=0.0.1

python scripts/pipeline_realtabformer.py --config exp/churn2/realtabformer/${EXP_VERSION}/config.toml --train && \
python scripts/pipeline_realtabformer.py --config exp/churn2/realtabformer/${EXP_VERSION}/config.toml --sample && \

python scripts/pipeline_realtabformer.py --config exp/diabetes/realtabformer/${EXP_VERSION}/config.toml --train && \
python scripts/pipeline_realtabformer.py --config exp/diabetes/realtabformer/${EXP_VERSION}/config.toml --sample && \

python scripts/pipeline_realtabformer.py --config exp/insurance/realtabformer/${EXP_VERSION}/config.toml --train && \
python scripts/pipeline_realtabformer.py --config exp/insurance/realtabformer/${EXP_VERSION}/config.toml --sample && \

python scripts/pipeline_realtabformer.py --config exp/abalone/realtabformer/${EXP_VERSION}/config.toml --train && \
python scripts/pipeline_realtabformer.py --config exp/abalone/realtabformer/${EXP_VERSION}/config.toml --sample && \

python scripts/pipeline_realtabformer.py --config exp/wilt/realtabformer/${EXP_VERSION}/config.toml --train && \
python scripts/pipeline_realtabformer.py --config exp/wilt/realtabformer/${EXP_VERSION}/config.toml --sample && \

python scripts/pipeline_realtabformer.py --config exp/buddy/realtabformer/${EXP_VERSION}/config.toml --train && \
python scripts/pipeline_realtabformer.py --config exp/buddy/realtabformer/${EXP_VERSION}/config.toml --sample && \

python scripts/pipeline_realtabformer.py --config exp/california/realtabformer/${EXP_VERSION}/config.toml --train && \
python scripts/pipeline_realtabformer.py --config exp/california/realtabformer/${EXP_VERSION}/config.toml --sample && \

python scripts/pipeline_realtabformer.py --config exp/adult/realtabformer/${EXP_VERSION}/config.toml --train && \
python scripts/pipeline_realtabformer.py --config exp/adult/realtabformer/${EXP_VERSION}/config.toml --sample
```