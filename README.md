# ETA Watch - Late Delivery Prediction

**Flag late deliveries at the moment an order is placed, not after it ships.**

## The problem

Today the company only finds out an order is late after it has shipped, when nothing can be done about it. Late orders lead to complaints, refunds and fewer repeat purchases.

ETA Watch predicts, when an order is placed, whether it will arrive late. It uses only what is known at that moment, so the team gets a warning while there is still time to act.

## How it was built

| # | Notebook | What it does |
|---|---|---|
| 1 | `01_data_understanding` | Data quality, distributions, the target, late rate by group, correlations |
| 2 | `02_data_preprocessing` | Clean names and dates, remove orders that never shipped, drop leaky and duplicate columns |
| 3 | `03_feature_engineering` | One row per order, split by date, build and test new features |
| 4 | `04_model_training` | Compare models, tune, pick a cutoff, score the test set once |

## How the results are kept honest

- **Orders that never shipped were removed.** 7,754 rows (canceled or suspected fraud) look late by their days columns but are labeled on time. With them removed, the label follows the delivery rule on 100% of rows instead of 97.55%.
- **Columns that give away the answer were dropped.** `delivery_status` and the actual shipping days are only known after delivery.
- **One row per order.** An order with three items has one outcome, so the 172,765 item rows became 62,897 orders. Otherwise the same order could land in both train and test.
- **Split by date, not at random.** The model is trained on the past and tested on the future, as it would be used in real life.

| Split | Orders | Period |
|---|---|---|
| Train | 44,028 | Jan 2015 to Mar 2017 |
| Validation | 9,434 | Mar 2017 to Sep 2017 |
| Test | 9,435 | Sep 2017 to Jan 2018 |

- **Nothing is learned from validation or test.** The noon cutoff for Same Day orders, the location encodings and the thresholds come from the training period only. Test labels were not loaded until the final comparison.
- **Seeds are fixed** (random state 42), so reruns give the same results.

## Features

26 features were built and 7 kept:

`shipping_mode`, `same_day_after_cutoff`, `order_hour`, `order_day_of_week`, `order_month`, `is_weekend`, `order_is_holiday_season`

These groups were tested and did not improve validation results enough to keep:

- location of the order (country, state, city)
- customer segment and payment type
- market and region
- order size and order value
- shipping mode combined with hour, weekday, region and size

Order size was also excluded on purpose. In the last months of the data, orders have about 1 item each instead of about 3, so a model trained on earlier orders would be looking at a different kind of order.

## Results

All models were trained on the same 7 features and scored on the test period, using the normal 0.50 cutoff. Always guessing "late" would score 57.5% accuracy on this test set.

| Model | Test accuracy | Test AUC | Precision | Recall |
|---|---|---|---|---|
| Logistic regression | 0.724 | 0.775 | 0.897 | 0.587 |
| Random forest | 0.723 | 0.788 | 0.882 | 0.598 |
| **LightGBM (chosen)** | **0.724** | **0.800** | **0.883** | **0.601** |
| XGBoost | 0.725 | 0.789 | 0.887 | 0.598 |
| CatBoost | 0.724 | 0.782 | 0.895 | 0.591 |
| LightGBM, tuned | 0.724 | 0.777 | 0.897 | 0.587 |
| XGBoost, tuned | 0.724 | 0.775 | 0.897 | 0.587 |

**What the columns mean**
- **Accuracy:** how often the model is right overall.
- **AUC:** the area under the ROC curve. It measures how well the model ranks late orders above on-time ones. 0.5 is guessing and 1.0 is perfect.
- **Precision:** of the orders flagged as late, how many really were late.
- **Recall:** of the orders that really were late, how many were flagged.

All models finish within 0.002 accuracy of each other. LightGBM was chosen because it has the highest AUC. Tuning with Optuna (25 trials each for LightGBM and XGBoost) did not help, so the untuned models were the ones kept. Even plain logistic regression matches the accuracy.

### Choosing a cutoff

The model gives every order a probability of being late. A cutoff turns that into a yes or no flag. Lowering it catches more late orders but raises false alarms. Results for the chosen model on the test set:

| Cutoff | Late orders caught | Late orders missed | False alarms | Accuracy | Orders flagged |
|---|---|---|---|---|---|
| 0.50 (normal) | 3,261 (60%) | 2,168 | 434 | 72.4% | 39% |
| 0.40 (catch more) | 4,359 (80%) | 1,070 | 1,821 | 69.4% | 66% |

I picked 0.40 as the highest cutoff that still catches at least 75% of late orders on validation data. That 75% is my assumption, because I have no real cost figures. In practice the cutoff should be set by what a missed late order costs compared with a false alarm.

## Limits and next steps

- Standard Class is 60% of test orders and only 40% of them are late. The model is right on 60% of them, no better than always guessing "on time".
- The data has no distance, carrier, dispatch load or weather, which are the things that would explain late Standard Class orders.
- The test set is one time period. A rolling test over several windows would be a stronger check.
- Customer history features (for example a customer's past late rate) were not built. They would need care to avoid using future information.
- Dollar impact is not estimated, because there are no real cost figures.

## Project structure

```
.
├── data/
│   ├── raw/          # original CSV (not included)
│   ├── interim/      # cleaned data (notebook 02)
│   └── processed/    # train, validation and test tables (notebook 03)
├── notebooks/        # 01 to 04, run in order
├── report/
│   ├── report.md         # reasoning behind each decision
│   └── data_dictionary.md
├── .gitignore
└── README.md
```

## Run it yourself

1. Download the [DataCo Supply Chain dataset](https://www.kaggle.com/datasets/saicharankomati/dataco-supply-chain-dataset) and put `DataCoSupplyChainDataset.csv` in `data/raw/`.
2. Install the libraries (built with Python 3.13):
   ```
   pip install pandas numpy matplotlib seaborn scikit-learn lightgbm xgboost catboost optuna
   ```
3. Run the notebooks in order, `01` to `04`. Each saves what the next one needs.

Data files are not stored in the repository. Notebooks 02 and 03 recreate them.

## More detail

- [`report/report.md`](report/report.md): the reasoning behind each decision
- [`report/data_dictionary.md`](report/data_dictionary.md): what each original column means
