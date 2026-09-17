# Baseline Evaluation V1

## 1. Objective

The purpose of Baseline V1 is to establish reference predictive performance at prediction point $k=10$ before model tuning and explainable-AI (XAI) analysis. This reporting layer consolidates the frozen outputs from Parts 1–7; it does not retrain or refit anything.

## 2. Evaluation Population

- Total BPIC17 cases: 31,509
- Valid target cases: 31,411
- Unresolved cases: 98
- Cases with fewer than 10 events: 0
- Cases excluded because the outcome occurred at or before $k=10$: 35
- Eligible cases: 31,376
- Eligible Success: 17,228
- Eligible Unsuccessful: 14,148

All 35 outcome-at-or-before-$k$ exclusions came from the Unsuccessful class. This is confirmed by the stored valid-target counts: 17,228 Success and 14,183 Unsuccessful cases before the timing exclusion, compared with 17,228 Success and 14,148 Unsuccessful eligible cases afterward.

## 3. Leakage Controls

The recorded outcome-leakage audit and future-event-leakage audit both passed. Features were restricted to the observed prefix, events 1 through 10. Vocabulary and preprocessing were fitted on training data only, and test-only resources did not expand the fitted feature space. These statements summarize the existing audits; this report does not perform or claim additional leakage analyses.

## 4. Features

The model representations contain five numeric features, one-hot categories for `LoanGoal` and `ApplicationType`, and count features for activities, `Action`, `EventOrigin`, resources, and lifecycle transitions. The final transformed dimension is 165 features.

Vocabulary was learned from training data only. Nine resource keys occurred only in the test set and were ignored by the fitted vocabulary. One zero-variance training feature was retained: `activity::A_Create Application`.

## 5. Temporal Split

The split used no shuffle and no stratification, with a strict chronological boundary.

| Split | Cases | Start | End | Success (1) | Unsuccessful (0) |
|---|---:|---|---|---:|---:|
| Train | 25,100 | 2016-01-01T09:51:15.304000+00:00 | 2016-10-18T11:25:01.193000+00:00 | 13,778 | 11,322 |
| Test | 6,276 | 2016-10-18T11:25:16.948000+00:00 | 2016-12-31T21:37:53.216000+00:00 | 3,450 | 2,826 |

The small prevalence difference between these periods is descriptive and is not evidence of no drift.

## 6. Models

- **DummyClassifier:** `strategy="prior"`; it learns the training class prior and uses its standard prediction rule.
- **Logistic Regression:** fixed L2 semantics, $C=1.0$, `solver="lbfgs"`, and `class_weight=None`.
- **Random Forest:** 300 estimators, Gini criterion, `max_depth=None`, `min_samples_split=2`, `min_samples_leaf=1`, `max_features="sqrt"`, bootstrap sampling, `class_weight=None`, and `random_state=42`.

There was no hyperparameter tuning, cross-validation, class rebalancing, or threshold optimization.

## 7. Metrics

The primary metric is ROC-AUC. Secondary metrics are Average Precision, accuracy, precision, recall, F1, and the confusion matrix. The positive class is Success = 1.

Average Precision is reported as AP; it is not a trapezoidal area computed from the plotted precision-recall curve.

## 8. Results

| Model | ROC-AUC | Average Precision | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Dummy | 0.5000 | 0.5497 | 0.5497 | 0.5497 | 1.0000 | 0.7094 |
| Logistic Regression | 0.6003 | 0.6426 | 0.5811 | 0.5967 | 0.7345 | 0.6584 |
| Random Forest | 0.5876 | 0.6301 | 0.5679 | 0.5976 | 0.6551 | 0.6250 |

The full-precision values are preserved in [`baseline_metrics.csv`](baseline_metrics.csv) and [`baseline_metrics.json`](baseline_metrics.json).

- [`roc_curve.png`](roc_curve.png)
- [`pr_curve.png`](pr_curve.png)
- Confusion matrices: [`Dummy`](confusion_matrix_dummy.png), [`Logistic Regression`](confusion_matrix_logistic.png), and [`Random Forest`](confusion_matrix_random_forest.png)

All figures use the same 6,276-case frozen test population and were produced only from the saved prediction files.

## 9. Baseline Interpretation

The Dummy model predicts every test case as Success because Success is the majority training class. Its Success recall is therefore 1.0, and its F1 of 0.7094 can appear high. However, its constant scores provide no ranking discrimination, so ROC-AUC remains 0.5. Dummy F1 alone is not evidence of predictive quality.

Logistic Regression produced non-trivial predictions for both classes. Its ROC-AUC (0.6003) and Average Precision (0.6426) both exceed the Dummy values.

Random Forest also exceeds Dummy in ROC-AUC (0.5876) and Average Precision (0.6301). In this fixed baseline, its metrics differ descriptively from Logistic Regression; no tuning or model selection follows from that observation.

The untuned forest is highly complex: the previously validated tree depths range from 79 to 136, with a mean of 99.52, and each tree has approximately 9.3 thousand leaves on average (9,373.73). This is evidence of structural complexity, not proof of overfitting by itself.

These results are untuned baseline estimates on one frozen temporal holdout. They are not final optimized model performance, cross-validated estimates, SHAP results, or concept-drift conclusions. No final research model is designated here.

## 10. Limitations and Next Steps

Baseline V1 uses one temporal holdout and fixed untuned configurations. It has not performed cross-validation or tuning, has not evaluated temporal explanation stability, and has not run SHAP. Those activities are outside Part 8 and are not implemented by this reporting task.
