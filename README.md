# ETA Watch: predicting late deliveries

When an order is placed, this model estimates whether it will arrive late. It only uses things known at that moment (shipping mode, time of order), so the team gets a warning while there is still time to act.

I built it on the DataCo Supply Chain dataset as a four-notebook project, from raw data to a tested model.

## The data

The raw file has one row per item ordered. I turned it into one row per order, which gives 62,897 orders. The target is `late_delivery_risk` (1 = late). About 57% of orders are late.

Cleaning, in short:
- Removed 7,754 rows for orders that were canceled or flagged as suspected fraud and never shipped. The model is therefore only meant for orders that ship.
- Dropped columns that are only known after delivery (delivery status and the actual shipping days), since a model can't use them at order time.

I split the data by date, so the model trains on the past and is tested on the future:

| Split | Orders | Period | Late share |
|---|---|---|---|
| Train | 44,028 | Jan 2015 to Mar 2017 | 57.3% |
| Validation | 9,434 | Mar 2017 to Sep 2017 | 57.0% |
| Test | 9,435 | Sep 2017 to Jan 2018 | 57.5% |

The same-day cutoff (orders placed from noon onward) was worked out from the training period only. No order appears in more than one split (checked in notebook 04).

## The notebooks

| # | Notebook | What it does |
|---|---|---|
| 1 | `01_data_understanding` | Data quality, distributions, the target, late rate by group |
| 2 | `02_data_preprocessing` | Cleaning, removing orders that never shipped, dropping columns that give away the answer |
| 3 | `03_feature_engineering` | One row per order, the date split, building and testing features |
| 4 | `04_model_training` | Compare models, tune, choose a cutoff, score the test set |

## The model

**Features.** I started with 23 candidate columns (some raw, some built) and kept 7:

`shipping_mode`, `same_day_after_cutoff`, `order_hour`, `order_day_of_week`, `order_month`, `is_weekend`, `order_is_holiday_season`

I tested adding each of the other groups back one at a time, using the same model settings: order location (country, state, city), customer segment and payment type, market and region, and shipping mode combined with hour, weekday, region or the same-day period. Every one of them gave lower PR-AUC and lower recall on validation, and made the model fit the training data more tightly, so I left them out.

Order size also changes over time. Orders in the training and validation periods have about 3 items each, but test-period orders have about 1.3 (inputs only, no labels). A model leaning on size would be looking at a different kind of order, so I left order size and order value out.

Shipping mode does most of the work: about 72% of the model's importance, plus about 15% for the same-day-after-noon flag.

**Models.** Logistic regression is the baseline. I compared it with random forest, LightGBM, XGBoost and CatBoost. LightGBM and XGBoost were also tuned with grid search (192 combinations each, time-ordered cross-validation, scored on PR-AUC). The tuned LightGBM had the best validation PR-AUC, so that is the final model. Its margin over the default LightGBM is small (0.868 vs 0.864). Seeds are fixed (42).

## Choosing the cutoff

The model gives each order a score between 0 and 1 for how likely it is to be late. A cutoff turns that score into a yes or no flag. I assumed that missing a late order matters more than a false alarm, so I wanted high recall, but flagging everything is useless. I don't have real cost figures.

So the rule is: pick the cutoff with the highest recall on validation, as long as at least 70% of the flagged orders really are late. That gave a cutoff of **0.40**. The 70% is my assumption. A different business cost would give a different cutoff.

On validation, accuracy stays around 0.725 down to a cutoff of 0.44, then drops. At 0.40 it is 0.70. The default 0.50 is shown below for comparison.

## Results (test set)

Recall is the share of late orders that were flagged. Precision is the share of flagged orders that really were late.

The chosen model, tuned LightGBM, on the test set (5,429 late orders out of 9,435):

| Cutoff | Late orders caught | Late orders missed | False alarms | Precision | Accuracy | Orders flagged |
|---|---|---|---|---|---|---|
| 0.50 (default) | 3,413 (62.9%) | 2,016 | 561 | 0.859 | 0.727 | 42% |
| **0.40 (chosen)** | **4,478 (82.5%)** | **951** | **1,862** | **0.706** | **0.702** | **67%** |

AUC is 0.808 and PR-AUC is 0.874. Always guessing "late" would get 57.5% accuracy.

All models at the 0.40 cutoff, on test:

| Model | Accuracy | Precision | Recall | AUC |
|---|---|---|---|---|
| Logistic regression | 0.661 | 0.672 | 0.806 | 0.775 |
| Random forest | 0.680 | 0.692 | 0.800 | 0.788 |
| XGBoost | 0.678 | 0.687 | 0.808 | 0.789 |
| XGBoost, tuned* | 0.613 | 0.599 | 0.987 | 0.793 |
| CatBoost | 0.672 | 0.689 | 0.782 | 0.782 |
| LightGBM | 0.694 | 0.705 | 0.803 | 0.800 |
| **LightGBM, tuned (chosen)** | **0.702** | **0.706** | **0.825** | **0.808** |

\*Tuned XGBoost used a class weight that pushes its scores up, so 0.40 is not a sensible cutoff for it (it flags 95% of orders). Cutoffs are not comparable across models. The differences between the top models are small, and I haven't put confidence intervals on them, so I wouldn't read much into the ranking.

## Limitations

- The data only has what is known at order time, and it has no distance, carrier, dispatch load or weather. For Standard Class orders (5,646 of the 9,435 test orders, 40% late), shipping mode and order time are not enough to tell late from on-time, so that is where the errors are: all 951 missed late orders and 1,498 of the 1,862 false alarms. Adding that kind of data is the main way to improve the model.
- First Class and Same Day orders follow a very regular pattern in this dataset (no errors on the test set). It looks rule-based, which is probably a property of the data, and real deliveries are likely noisier, so results on other data may differ.
- The test set is one time period, and it was also scored in earlier versions of the project, so the numbers may be slightly optimistic.
- The 70% precision floor is my assumption. A different business cost would give a different cutoff.
- Next steps: widen the tuning search, add confidence intervals and a rolling backtest, and add order-time information such as distance.

## Project structure

```
.
├── data/
│   ├── raw/          # original CSV (not included)
│   ├── interim/      # cleaned data (notebook 02)
│   └── processed/    # train, validation and test tables (notebook 03)
├── notebooks/        # 01 to 04, run in order
└── README.md
```

## Run it yourself

1. Download the [DataCo Supply Chain dataset](https://www.kaggle.com/datasets/saicharankomati/dataco-supply-chain-dataset) and put `DataCoSupplyChainDataset.csv` in `data/raw/`.
2. Install the libraries:
   ```
   pip install pandas numpy matplotlib seaborn scikit-learn lightgbm xgboost catboost
   ```
3. Run the notebooks in order, `01` to `04`. Each one saves what the next needs.

Data files are not stored in the repository. Notebooks 02 and 03 recreate them.
