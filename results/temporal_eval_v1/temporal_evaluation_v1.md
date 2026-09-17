# Temporal Evaluation V1

## 1. Objective

Temporal Evaluation V1 asks two questions about the frozen BPIC 2017 test period: **RQ1**, how did prediction metrics and prediction distributions change across time; and **RQ2**, how did global SHAP feature importance and feature ranking change across time? The combined question is: **How did prediction and explanation change over time?**

This is an observational, preliminary analysis. It does not establish statistical significance, formal concept drift, or causality.

## 2. Experimental Setup

- **Dataset:** BPIC 2017.
- **Prediction point:** `k=10` events.
- **Model:** frozen Random Forest Baseline V1, an untuned baseline.
- **Prediction population:** all 6,276 frozen test cases.
- **Explanation population:** the frozen 1,000-case SHAP sample drawn from the same test population (T1: 325; T2: 332; T3: 343).
- **Time variable:** `case_start_time`.
- **Window design:** three chronological equal-case windows, each containing 2,092 prediction cases.

The frozen time ranges are:

- **T1:** 2016-10-18T11:25:16.948000+00:00 through 2016-11-08T17:05:15.051000+00:00 (2,092 cases).
- **T2:** 2016-11-08T17:28:09.702000+00:00 through 2016-12-02T18:50:47.710000+00:00 (2,092 cases).
- **T3:** 2016-12-02T19:21:29.337000+00:00 through 2016-12-31T21:37:53.216000+00:00 (2,092 cases).

The prediction and explanation analyses use the same frozen temporal boundaries. The populations differ: prediction results cover 6,276 cases, whereas explanation results describe the frozen 1,000-case sample.

## 3. Temporal Prediction Results

| Window | Cases | Success rate | Mean P(Success) | ROC-AUC | Average Precision | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| T1 | 2,092 | 0.555449 | 0.557764 | 0.577777 | 0.634067 | 0.555449 | 0.590343 | 0.652324 | 0.619787 |
| T2 | 2,092 | 0.552581 | 0.551845 | 0.577632 | 0.621998 | 0.565966 | 0.598257 | 0.653114 | 0.624483 |
| T3 | 2,092 | 0.541109 | 0.547631 | 0.607302 | 0.635569 | 0.582218 | 0.604369 | 0.659894 | 0.630912 |

From T1 to T2, ROC-AUC changed by -0.000145 and Average Precision (AP) by -0.012069. Success prevalence decreased by -0.002868, and mean predicted P(Success) decreased by -0.005919.

From T2 to T3, ROC-AUC changed by +0.029670 and AP by +0.013572. Success prevalence decreased by -0.011472, while mean predicted P(Success) decreased by -0.004214.

Across T1 to T3, ROC-AUC was higher by +0.029525, while AP was higher by +0.001503. Success prevalence decreased by -0.014340, and mean predicted P(Success) decreased by -0.010134. These are descriptive changes; no statistical significance testing was performed, and a higher value for one metric is not treated as general model improvement.

Figures: [temporal prediction metrics](temporal_prediction_metrics.png) and [temporal probability and prevalence](temporal_probability.png).

## 4. Temporal Explanation Results

| Rank | T1 | T2 | T3 |
|---:|---|---|---|
| 1 | `requested_amount` | `requested_amount` | `requested_amount` |
| 2 | `mean_event_gap_seconds` | `mean_event_gap_seconds` | `mean_event_gap_seconds` |
| 3 | `prefix_duration_seconds` | `prefix_duration_seconds` | `prefix_duration_seconds` |
| 4 | `resource::User_1` | `resource::User_1` | `activity::A_Submitted` |
| 5 | `activity::A_Submitted` | `activity::A_Submitted` | `resource::User_1` |
| 6 | `activity::W_Handle leads` | `activity::W_Handle leads` | `activity::W_Handle leads` |
| 7 | `loan_goal_Home improvement` | `n_unique_resources` | `n_unique_resources` |
| 8 | `n_unique_resources` | `loan_goal_Home improvement` | `loan_goal_Home improvement` |
| 9 | `loan_goal_Car` | `application_type_New credit` | `loan_goal_Car` |
| 10 | `application_type_New credit` | `application_type_Limit raise` | `application_type_New credit` |

The intersection of the top-10 sets across all three windows contains 9 features: `requested_amount`, `mean_event_gap_seconds`, `prefix_duration_seconds`, `resource::User_1`, `activity::A_Submitted`, `activity::W_Handle leads`, `loan_goal_Home improvement`, `n_unique_resources`, `application_type_New credit`. Relative to T1, T2 added `application_type_Limit raise` and omitted `loan_goal_Car`. From T2 to T3, `loan_goal_Car` re-entered and `application_type_Limit raise` left. The T1 and T3 top-10 sets are identical.

- **T1 to T2:** Jaccard@10 = 0.818181818, with 9 of 10 features overlapping; Spearman rho = 0.972852036 across all 165 features.
- **T2 to T3:** Jaccard@10 = 0.818181818, with 9 of 10 features overlapping; Spearman rho = 0.973634970 across all 165 features.
- **T1 to T3:** Jaccard@10 = 1.000000000, with all 10 features overlapping; Spearman rho = 0.956959122 across all 165 features.

The top-feature composition showed high overlap across the three windows, and the overall importance ranking remained strongly correlated across windows. No formal stable/unstable threshold is defined, and these results alone do not establish the absence of drift.

Figures: [temporal top SHAP features](temporal_shap_top_features.png) and [temporal SHAP heatmap](temporal_shap_heatmap.png).

## 5. Signed Attribution Changes

The union of temporal top-10 features contains 11 features. Three features changed average SHAP contribution direction around T2 while having the same T1 and T3 directions: `loan_goal_Car`, `application_type_New credit`, `application_type_Limit raise`. Each followed **negative in T1, positive in T2, negative in T3**.

This describes the frozen model's average attribution direction relative to its SHAP reference. It does not mean that the real-world effect of a feature reversed. SHAP explains model behavior, not causality.

## 6. Feature-Group Changes

| Group | Encoded features | T1 share | T2 share | T3 share | T1 mean/feature | T2 mean/feature | T3 mean/feature |
|---|---:|---:|---:|---:|---:|---:|---:|
| Numeric | 5 | 26.87% | 26.03% | 26.31% | 0.015451 | 0.014393 | 0.014345 |
| LoanGoal | 14 | 12.34% | 13.50% | 13.05% | 0.002534 | 0.002666 | 0.002541 |
| ApplicationType | 2 | 4.90% | 5.61% | 5.32% | 0.007049 | 0.007760 | 0.007248 |
| Activity | 14 | 13.77% | 14.32% | 15.30% | 0.002829 | 0.002828 | 0.002980 |
| Action | 5 | 5.13% | 4.99% | 4.94% | 0.002952 | 0.002757 | 0.002695 |
| EventOrigin | 3 | 3.98% | 3.89% | 4.17% | 0.003819 | 0.003584 | 0.003790 |
| Resource | 115 | 25.14% | 23.90% | 22.56% | 0.000629 | 0.000575 | 0.000535 |
| Lifecycle | 7 | 7.85% | 7.77% | 8.35% | 0.003227 | 0.003070 | 0.003252 |

Resource total importance decreased from 0.072301 to 0.066084 to 0.061513, while its share decreased from 25.14% to 23.90% to 22.56%. Activity total importance changed from 0.039606 to 0.039590 to 0.041716, while its share increased from 13.77% to 14.32% to 15.30%. Numeric total importance changed from 0.077253 to 0.071965 to 0.071726, and its share changed from 26.87% to 26.03% to 26.31%.

Group totals and shares must be read together with dimensionality and mean importance per feature. In particular, Resource contains 115 encoded features, so its total share does not by itself establish greater intrinsic importance.

Figure: [temporal feature-group importance](temporal_group_importance.png).

## 7. Prediction-Explanation Synthesis

The canonical synthesis is [prediction_explanation_synthesis.csv](prediction_explanation_synthesis.csv).

| Comparison | Delta ROC-AUC | Delta AP | Jaccard@10 | Spearman rho | Top-10 overlap |
|---|---:|---:|---:|---:|---:|
| T1 to T2 | -0.000145 | -0.012069 | 0.818181818 | 0.972852036 | 9 of 10 |
| T2 to T3 | +0.029670 | +0.013572 | 0.818181818 | 0.973634970 | 9 of 10 |
| T1 to T3 | +0.029525 | +0.001503 | 1.000000000 | 0.956959122 | 10 of 10 |

Across the three test windows, predictive metrics changed modestly, with ROC-AUC notably higher in T3 than in T1/T2. At the same time, SHAP explanations showed substantial continuity: adjacent windows shared 9 of their top 10 features, T1 and T3 shared all 10, and full-feature importance rankings remained highly correlated.

This suggests that the observed temporal variation in predictive performance was not accompanied by a comparably large reorganization of the model's global feature-importance structure in this preliminary analysis. This is a descriptive association: explanation continuity did not cause the prediction pattern, and no formal drift conclusion is made.

## 8. Main Findings

1. Success prevalence decreased from T1 through T3.
2. Mean predicted Success probability decreased from T1 through T3.
3. ROC-AUC was nearly unchanged from T1 to T2 and was higher in T3.
4. Average Precision decreased in T2 and recovered in T3.
5. Adjacent windows shared 9 of their top 10 SHAP features.
6. T1 and T3 had identical top-10 SHAP feature sets.
7. Full 165-feature importance rankings remained strongly correlated across windows.
8. Resource attribution share decreased while Activity share increased.
9. Three union-top-10 features had a temporary mean signed SHAP direction change in T2.
10. Predictive changes were not accompanied by a comparably large reorganization of global SHAP importance.

## 9. Limitations

- The Random Forest is an untuned baseline.
- Explanation analysis uses a 1,000-case sample rather than all 6,276 test cases.
- The three windows cover only the frozen test period from October to December 2016.
- Windows are equal-case windows, not fixed-duration windows.
- No statistical significance testing was performed.
- No formal concept-drift detector was used.
- SHAP explains model behavior, not causality.
- Results may change after model tuning or final model selection.

## 10. Conclusion

How did prediction and explanation change over time? Success prevalence and mean predicted P(Success) decreased across the three windows. ROC-AUC was nearly unchanged from T1 to T2 and higher in T3; AP decreased in T2 and recovered in T3. Meanwhile, global SHAP explanations retained substantial feature-set and ranking continuity, although three features showed temporary T2 changes in mean attribution direction and feature-group shares shifted descriptively. Thus, the observed predictive variation was not accompanied by a comparably large reorganization of global SHAP importance in this preliminary analysis. This conclusion is descriptive, non-causal, and not a formal claim about concept drift.

## 11. Next Steps

Possible future work includes a reproduction experiment, model tuning and final model selection, final SHAP evaluation, stronger temporal stability validation, and formal drift analysis if required. None of those tasks is part of Temporal Evaluation V1 Part 4.
