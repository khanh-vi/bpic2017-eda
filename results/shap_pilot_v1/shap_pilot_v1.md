# SHAP Pilot V1

## 1. Objective

This pilot validates and inspects SHAP explanations for the frozen, untuned Random Forest Baseline V1 at prediction point k=10. It is a reporting and interpretation layer over the canonical Part 3 values, not the final tuned-model SHAP evaluation.

## 2. Frozen Model and Data

- Model: Random Forest Baseline V1
- Model tuning: none
- Prediction point: k=10
- Feature count: 165
- SHAP background: 500 TRAIN cases
- Explained sample: 1,000 TEST cases
- Positive class: Success = 1
- SHAP version: 0.52.0

The explained rows were reconstructed directly from the canonical transformed TEST representation using the frozen `source_row_index` values. No preprocessing was fitted or applied in Part 4.

## 3. SHAP Configuration

Part 3 used TreeExplainer with interventional feature perturbation, probability model output, and `approximate=False`. The background contains only 500 frozen TRAIN cases. The 1,000-case TEST explained sample is frozen, and no resampling occurred. Part 4 only constructs `shap.Explanation` containers from saved values for plotting; it does not construct an explainer or recompute explanations.

## 4. Numerical Validation

The original project-defined absolute probability-space tolerance of 1e-6 failed, with a smoke-test maximum residual of 6.68378713334e-06 (approximately 6.67e-6). Diagnostic checks of model output, class mapping, base-value mapping, feature alignment, and frozen input identity passed. The project then adopted a revised absolute probability-space tolerance of 1e-5 before the full computation; this is a project-defined validation criterion, not a SHAP library default.

Across all 1,000 frozen cases, the maximum absolute additivity error was 6.688518622e-06, the mean was 6.66766127867e-06, and the median was 6.66724438875e-06. All 1,000 cases passed 1e-5, and the SHAP values, base values, model probabilities, and reconstructed feature matrix contained no NaN or infinity.

## 5. Global Feature Importance

Global importance is the mean absolute SHAP value over all 1,000 frozen explained cases. The complete canonical ranking is in [`global_shap_importance.csv`](global_shap_importance.csv).

| rank | feature | mean_abs_shap | share_of_total_importance |
| --- | --- | --- | --- |
| 1 | requested_amount | 0.026641277 | 9.556884% |
| 2 | mean_event_gap_seconds | 0.016407595 | 5.885809% |
| 3 | prefix_duration_seconds | 0.016038389 | 5.753366% |
| 4 | resource::User_1 | 0.011546641 | 4.142065% |
| 5 | activity::A_Submitted | 0.011405731 | 4.091517% |
| 6 | activity::W_Handle leads | 0.010695407 | 3.836707% |
| 7 | n_unique_resources | 0.009614712 | 3.449035% |
| 8 | loan_goal_Home improvement | 0.008855549 | 3.176704% |
| 9 | loan_goal_Car | 0.007706117 | 2.764375% |
| 10 | application_type_New credit | 0.007614443 | 2.731489% |

The top-20 magnitude ranking is visualized in [`shap_global_bar.png`](shap_global_bar.png), and the distribution of signed contributions is shown in [`shap_beeswarm.png`](shap_beeswarm.png). Beeswarm color represents the exact transformed feature value supplied to the frozen model. Many inputs are one-hot or count encoded, so these colors must not be read as raw business-unit values for every feature. These plots describe the frozen model and do not imply causality.

## 6. Feature-Group Importance

The complete grouped output is in [`global_shap_group_importance.csv`](global_shap_group_importance.csv).

| group | feature_count | group_total_importance | group_mean_importance_per_feature | share_of_total_importance |
| --- | --- | --- | --- | --- |
| Numeric | 5 | 0.073601634 | 0.014720327 | 26.402724% |
| LoanGoal | 14 | 0.036123039 | 0.002580217 | 12.958226% |
| ApplicationType | 2 | 0.014706432 | 0.007353216 | 5.275560% |
| Activity | 14 | 0.040324119 | 0.002880294 | 14.465257% |
| Action | 5 | 0.013996867 | 0.002799373 | 5.021022% |
| EventOrigin | 3 | 0.011193932 | 0.003731311 | 4.015540% |
| Resource | 115 | 0.066536964 | 0.000578582 | 23.868452% |
| Lifecycle | 7 | 0.022282322 | 0.003183189 | 7.993219% |

Group totals sum individual feature magnitudes and are descriptive. In particular, Resource contains 115 of the 165 encoded dimensions. Its total importance must therefore be interpreted together with its mean importance per feature; total group importance alone does not establish that Resource is intrinsically the most influential feature type.

## 7. Local Examples

For each TP, TN, FP, and FN category, the selected example is the case with the smallest `sample_order` in the frozen 1,000-case sample. Selection did not use confidence, SHAP magnitude, probability extremes, interesting features, or visual appearance. Complete metadata is in [`local_examples.csv`](local_examples.csv), and all 660 feature contributions are in [`local_feature_contributions.csv`](local_feature_contributions.csv).

Positive SHAP values mean that a feature contribution moves the frozen model output toward a higher predicted Success probability relative to the SHAP reference expectation for that case. Negative values move it toward a lower predicted Success probability. They are model contributions, not causal effects.

### TP

- Case ID: `Application_1716488937`
- Frozen sample order: 0
- True class: 1
- Predicted class: 1
- Predicted Success probability: 0.646666667
- Largest positive model contributions: `resource::User_12` (+0.040346); `requested_amount` (+0.025482); `loan_goal_Remaining debt home` (+0.020206); `n_unique_activities` (+0.009010); `lifecycle::suspend` (+0.007968)
- Largest negative model contributions: `resource::User_46` (-0.033074); `prefix_duration_seconds` (-0.017373); `activity::W_Handle leads` (-0.005805); `activity::A_Submitted` (-0.004634); `loan_goal_Home improvement` (-0.004070)

### TN

- Case ID: `Application_626689908`
- Frozen sample order: 5
- True class: 0
- Predicted class: 0
- Predicted Success probability: 0.416666667
- Largest positive model contributions: `resource::User_2` (+0.039713); `loan_goal_Home improvement` (+0.017770); `loan_goal_Car` (+0.005911); `loan_goal_Other, see explanation` (+0.002127); `lifecycle::start` (+0.001856)
- Largest negative model contributions: `requested_amount` (-0.042513); `resource::User_1` (-0.041739); `activity::W_Handle leads` (-0.031913); `mean_event_gap_seconds` (-0.012889); `activity::A_Submitted` (-0.011262)

### FP

- Case ID: `Application_1780975856`
- Frozen sample order: 1
- True class: 0
- Predicted class: 1
- Predicted Success probability: 0.730000000
- Largest positive model contributions: `mean_event_gap_seconds` (+0.041888); `prefix_duration_seconds` (+0.026616); `requested_amount` (+0.025697); `n_unique_resources` (+0.023797); `activity::W_Handle leads` (+0.015188)
- Largest negative model contributions: `resource::User_61` (-0.020033); `activity::O_Sent (mail and online)` (-0.007405); `loan_goal_Existing loan takeover` (-0.006901); `application_type_New credit` (-0.005431); `application_type_Limit raise` (-0.005379)

### FN

- Case ID: `Application_1606123452`
- Frozen sample order: 8
- True class: 1
- Predicted class: 0
- Predicted Success probability: 0.273333333
- Largest positive model contributions: `loan_goal_Car` (+0.005332); `action::statechange` (+0.001982); `loan_goal_Not speficied` (+0.001870); `resource::User_18` (+0.001441); `resource::User_37` (+0.001268)
- Largest negative model contributions: `requested_amount` (-0.065109); `resource::User_96` (-0.032613); `n_unique_resources` (-0.024269); `loan_goal_Other, see explanation` (-0.015807); `loan_goal_Home improvement` (-0.015185)

The corresponding waterfall plots are `local_tp_waterfall.png`, `local_tn_waterfall.png`, `local_fp_waterfall.png`, and `local_fn_waterfall.png`. Each uses the same 15-feature display limit and is constructed from the aligned frozen SHAP row, base value, and transformed feature row.

## 8. Interpretation Boundaries

SHAP explains the frozen model prediction; it does not establish causality. The Random Forest is an untuned baseline, and explanations may change after appropriate model validation or tuning. This pilot does not establish temporal explanation stability and includes no temporal-window, drift, Jaccard, Spearman, or attribution-sign stability analysis.

## 9. Pilot Outcome

- Canonical SHAP computation: completed in Part 3 and hash-validated in Part 4
- Global explanations: produced
- Local explanations: produced
- Feature and class alignment: valid
- SHAP recomputation in Part 4: none
- Deterministic Part 4 rerun comparison: PASS
- Protected Part 0--3 artifact integrity: PASS

## 10. Next Step

Temporal explanation-stability analysis is outside this Part 4 task and was not implemented.
