# Temporal Evaluation Design V1

## 1. Mục tiêu

Temporal Evaluation V1 được thiết kế để quan sát cách **prediction** và **SHAP explanation** của một mô hình Predictive Process Monitoring thay đổi theo thời gian trên tập test đã được đóng băng.

Mục tiêu chính là trả lời hai câu hỏi:

**RQ1 — Prediction change:**  
Hiệu năng dự đoán của mô hình thay đổi như thế nào giữa các giai đoạn thời gian khác nhau?

**RQ2 — Explanation change:**  
Global SHAP feature importance và thứ hạng feature thay đổi như thế nào giữa các giai đoạn thời gian khác nhau?

Sau đó, kết quả của hai phần được tổng hợp để quan sát:

> Khi predictive behavior thay đổi theo thời gian, explanation của mô hình có thay đổi tương ứng hay không?

Temporal Evaluation V1 là một **observational temporal analysis**. Kết quả không được diễn giải như bằng chứng chính thức của concept drift hay quan hệ nhân quả.

---

## 2. Phạm vi đánh giá

### 2.1. Mô hình

Sử dụng nguyên trạng **Random Forest Baseline V1** đã được freeze trong SHAP Pilot V1.

Không thực hiện:

- retraining;
- hyperparameter tuning;
- class rebalancing;
- threshold optimization;
- feature selection;
- preprocessing refit;
- thay đổi feature space.

Việc giữ nguyên mô hình nhằm bảo đảm rằng thay đổi quan sát được giữa các temporal windows không đến từ việc mô hình được thay đổi.

### 2.2. Prediction point

Prediction point được cố định tại:

`k = 10`

Mỗi case chỉ sử dụng thông tin khả dụng trong 10 event đầu tiên theo pipeline đã được kiểm chứng bởi leakage audit.

### 2.3. Positive class

Binary target:

- `Success = 1`
- `Unsuccessful = 0`

---

## 3. Dữ liệu đánh giá

### 3.1. Prediction population

Prediction analysis sử dụng toàn bộ frozen temporal test set:

- Tổng số test cases: **6,276**
- Success: **3,450**
- Unsuccessful: **2,826**

Khoảng thời gian test:

- Bắt đầu: `2016-10-18T11:25:16.948000+00:00`
- Kết thúc: `2016-12-31T21:37:53.216000+00:00`

Frozen Random Forest predictions đã tồn tại cho toàn bộ 6,276 cases và sẽ được tái sử dụng. Không chạy lại model để tạo một tập prediction khác.

### 3.2. Explanation population

Explanation analysis sử dụng frozen SHAP Pilot sample:

- Tổng số explained cases: **1,000**
- Nguồn: random sample từ frozen TEST set
- Sampling: uniform without replacement
- Seed: `42`
- SHAP feature count: **165**
- SHAP values đã được tính và freeze trong SHAP Pilot V1.

Không resample 1,000 cases và không chạy lại TreeSHAP trong Temporal Evaluation V1.

### 3.3. Khác biệt population

Prediction analysis sử dụng toàn bộ 6,276 test cases, trong khi explanation analysis sử dụng 1,000 cases được lấy mẫu từ cùng test population.

Do đó, explanation results chỉ được diễn giải cho frozen 1,000-case SHAP sample và không được mô tả như phép đo chính xác trên toàn bộ 6,276 test cases.

---

## 4. Biến thời gian

Temporal ordering sử dụng:

`case_start_time`

Lý do:

- đây là biến đã được sử dụng để tạo temporal train/test split;
- giữ cùng định nghĩa thời gian xuyên suốt pipeline;
- tránh thay đổi temporal reference giữa baseline và temporal evaluation.

Temporal Evaluation V1 không sử dụng timestamp của event thứ 10 làm biến chia window.

---

## 5. Thiết kế temporal windows

### 5.1. Số lượng windows

Chia frozen TEST population thành ba temporal windows:

- `T1`: early test period
- `T2`: middle test period
- `T3`: late test period

### 5.2. Quy tắc chia

Sắp xếp toàn bộ 6,276 test cases theo `case_start_time` tăng dần.

Thiết kế sử dụng **chronological equal-case windows**.

Nominal target:

- T1: 2,092 cases
- T2: 2,092 cases
- T3: 2,092 cases

### 5.3. Boundary tie policy

Nếu một nominal boundary rơi vào nhiều cases có cùng `case_start_time`, không chia các cases có cùng timestamp sang hai windows khác nhau.

Ưu tiên:

`max(T1.case_start_time) < min(T2.case_start_time)`

và:

`max(T2.case_start_time) < min(T3.case_start_time)`

thay vì ép số lượng mỗi window chính xác bằng 2,092.

Actual case count và actual time range của từng window phải được lưu sau khi boundary được freeze.

### 5.4. Gán SHAP sample vào windows

Temporal boundaries được xác định từ **toàn bộ 6,276 test cases**.

Sau khi T1/T2/T3 được freeze, 1,000 SHAP cases được gán vào window theo chính các boundaries đó.

Không chia riêng 1,000 SHAP cases thành `333/333/334`.

Không resample để làm số SHAP cases trong ba windows bằng nhau.

Điều này bảo đảm prediction analysis và explanation analysis sử dụng cùng định nghĩa thời gian.

---

## 6. Temporal Prediction Analysis

### 6.1. Metrics theo từng window

Đối với mỗi window T1, T2 và T3, báo cáo:

- `n_cases`
- Success count
- Unsuccessful count
- Success rate
- mean predicted `P(Success)`
- ROC-AUC
- Average Precision
- Accuracy
- Precision
- Recall
- F1

Primary predictive metrics:

- **ROC-AUC**
- **Average Precision**

Secondary metrics:

- Accuracy
- Precision
- Recall
- F1

Positive class luôn là `Success = 1`.

Frozen prediction labels và probabilities được tái sử dụng. Không thay đổi threshold riêng cho từng window.

### 6.2. Pairwise predictive change

Tính descriptive differences cho:

- T1 → T2
- T2 → T3
- T1 → T3

Ví dụ:

`Delta ROC-AUC(T1,T2) = ROC-AUC(T2) - ROC-AUC(T1)`

Tương tự cho:

- Average Precision
- F1
- Success rate
- mean `P(Success)`

Temporal Evaluation V1 không thực hiện significance testing cho các differences này.

---

## 7. Temporal SHAP Analysis

### 7.1. Global SHAP importance theo window

Với feature `j` trong temporal window `t`:

`Importance(j,t) = mean(|SHAP(i,j)|)` với mọi case `i` thuộc window `t`.

Mỗi window tạo một vector global importance gồm 165 features.

Kết quả phải cho phép so sánh:

- importance magnitude;
- feature ranking;
- top features;

giữa T1, T2 và T3.

### 7.2. Ranking rule

Feature ranking trong từng window được xác định bằng:

1. `mean_abs_shap` giảm dần;
2. nếu bằng nhau, dùng feature name tăng dần làm deterministic tie-break.

Không loại feature zero-variance chỉ để phục vụ ranking.

---

## 8. Explanation Stability Metrics

### 8.1. Jaccard@10

Primary top-feature stability metric:

`K = 10`

Với hai windows A và B:

`Jaccard@10(A,B) = |Top10(A) intersection Top10(B)| / |Top10(A) union Top10(B)|`

Tính:

- T1 vs T2
- T2 vs T3
- T1 vs T3

Jaccard@10 được dùng để quan sát mức độ overlap của các feature quan trọng nhất.

Temporal Evaluation V1 không đặt threshold định tính như "stable" hoặc "unstable" dựa trên một ngưỡng Jaccard tùy ý.

### 8.2. Spearman rank correlation

Primary full-ranking stability metric:

Spearman correlation được tính trên **toàn bộ 165 features** giữa các vector `mean_abs_shap` của hai windows.

Tính:

- Spearman(T1, T2)
- Spearman(T2, T3)
- Spearman(T1, T3)

Spearman được dùng để quan sát mức độ duy trì thứ hạng importance tổng thể, trong khi Jaccard@10 tập trung vào tập top features.

Hai metric được sử dụng bổ sung cho nhau.

---

## 9. Mean Signed SHAP Analysis

Mean signed SHAP là secondary descriptive analysis.

Với feature `j` trong window `t`:

`MeanSignedSHAP(j,t) = mean(SHAP(i,j))`

Diễn giải:

- giá trị dương: contribution trung bình của feature đang hướng model prediction về xác suất Success cao hơn so với SHAP reference;
- giá trị âm: contribution trung bình đang hướng model prediction về xác suất Success thấp hơn.

Không diễn giải mean signed SHAP như causal effect.

Trong report chính, signed analysis chỉ tập trung vào **union của top-10 features từ T1, T2 và T3**, thay vì trình bày toàn bộ 165 features.

Nếu mean signed SHAP đổi dấu giữa các windows, chỉ mô tả rằng hướng contribution trung bình của model đã thay đổi.

---

## 10. Temporal Feature-Group Analysis

Sử dụng cùng tám feature groups đã được xác định trong SHAP Pilot V1:

1. Numeric
2. LoanGoal
3. ApplicationType
4. Activity
5. Action
6. EventOrigin
7. Resource
8. Lifecycle

Trong mỗi window tính:

- `feature_count`
- `group_total_importance`
- `group_mean_importance_per_feature`
- `share_of_total_importance`

Feature-group analysis là secondary analysis.

Đặc biệt, nhóm Resource có số lượng dimensions lớn nên không được diễn giải chỉ dựa trên total importance. Cần báo cáo song song total importance và mean importance per feature.

---

## 11. Prediction-Explanation Synthesis

Kết quả prediction và explanation được tổng hợp theo ba comparisons:

- T1 → T2
- T2 → T3
- T1 → T3

Bảng synthesis tối thiểu:

| Comparison | Delta ROC-AUC | Delta AP | Delta Mean P(Success) | Jaccard@10 | Spearman |
|---|---:|---:|---:|---:|---:|
| T1 → T2 | | | | | |
| T2 → T3 | | | | | |
| T1 → T3 | | | | | |

Mục đích là quan sát các pattern như:

- predictive performance tương đối giống nhau nhưng explanation ranking thay đổi;
- predictive performance thay đổi cùng lúc với explanation ranking;
- top-feature composition thay đổi nhưng overall ranking vẫn tương đối tương quan.

Không kết luận rằng explanation change gây ra prediction change hoặc ngược lại.

---

## 12. Planned Visualizations

Temporal Evaluation V1 dự kiến tạo các visualization chính sau:

### 12.1. Temporal prediction metrics

File dự kiến:

`temporal_prediction_metrics.png`

Hiển thị ROC-AUC và Average Precision theo T1/T2/T3.

### 12.2. Temporal probability and prevalence

File dự kiến:

`temporal_probability.png`

Hiển thị:

- Success rate;
- mean predicted `P(Success)`;

theo từng window.

### 12.3. Temporal top SHAP features

File dự kiến:

`temporal_shap_top_features.png`

So sánh các global SHAP importance quan trọng giữa ba windows.

### 12.4. Temporal SHAP heatmap

File dự kiến:

`temporal_shap_heatmap.png`

Heatmap cho union của các top features qua T1/T2/T3.

### 12.5. Temporal feature-group importance

File dự kiến:

`temporal_group_importance.png`

So sánh feature-group importance/share giữa T1/T2/T3.

Không yêu cầu tạo thêm visualization nếu chúng không hỗ trợ trực tiếp research questions của V1.

---

## 13. Reproducibility và Leakage Controls

Temporal Evaluation V1 phải tuân thủ các constraints sau:

- sử dụng frozen Random Forest Baseline V1;
- không retrain model;
- không refit preprocessing;
- không xây lại feature vocabulary;
- không thay đổi target;
- không thay đổi prediction point;
- không thay đổi test split;
- không thay đổi frozen RF predictions;
- không resample SHAP cases;
- không recompute SHAP values;
- không thay đổi SHAP background;
- không thay đổi feature order;
- không dùng future data để train lại model;
- temporal boundaries phải được freeze trước khi tính temporal metrics.

Prediction và explanation analysis phải sử dụng cùng temporal window boundaries.

---

## 14. Không nằm trong phạm vi V1

Temporal Evaluation V1 không thực hiện:

- formal concept drift detection;
- PSI;
- KS test;
- ADWIN;
- statistical significance testing;
- bootstrap confidence intervals;
- retraining theo từng temporal window;
- temporal model adaptation;
- hyperparameter tuning;
- threshold optimization;
- new SHAP sampling;
- SHAP recomputation;
- final tuned-model SHAP;
- causal inference.

Các nội dung trên có thể được mở rộng ở phiên bản nghiên cứu sau nếu cần.

---

## 15. Interpretation Boundaries

Các kết quả cần được mô tả theo hướng:

- predictive performance changed across temporal windows;
- model explanation ranking changed across temporal windows;
- top important features showed different levels of overlap;
- mean model attribution direction changed for selected features.

Không sử dụng kết quả để khẳng định:

- concept drift đã được chứng minh;
- một feature gây ra outcome;
- explanation change gây ra performance change;
- kết quả trên 1,000 SHAP cases đại diện chính xác cho toàn bộ test population.

SHAP giải thích behavior của frozen model, không chứng minh causal relationship.

---

## 16. Planned Outputs

Các artifact dự kiến:

```text
results/temporal_eval_v1/
    temporal_windows.csv
    temporal_window_summary.json
    temporal_prediction_metrics.csv
    temporal_prediction_deltas.csv
    temporal_shap_importance.csv
    temporal_shap_top10.csv
    temporal_shap_stability.csv
    temporal_signed_shap.csv
    temporal_group_importance.csv
    prediction_explanation_synthesis.csv
    temporal_prediction_metrics.png
    temporal_probability.png
    temporal_shap_top_features.png
    temporal_shap_heatmap.png
    temporal_group_importance.png
    temporal_evaluation_v1.md
```

Các artifact cụ thể có thể được điều chỉnh nhẹ khi implementation, nhưng không được thay đổi evaluation semantics đã freeze trong tài liệu này.

---

## 17. Implementation Order

Temporal Evaluation V1 được triển khai theo thứ tự:

### Part 1 — Freeze temporal windows

- xác định T1/T2/T3 từ toàn bộ 6,276 test cases;
- áp dụng boundary tie policy;
- lưu window membership;
- gán 1,000 frozen SHAP cases vào cùng boundaries;
- chưa tính prediction hoặc SHAP stability metrics.

### Part 2 — Temporal prediction analysis

- tính prediction metrics theo T1/T2/T3;
- tính pairwise metric deltas;
- tạo prediction plots.

### Part 3 — Temporal SHAP analysis

- tính global SHAP importance theo window;
- top-10 features;
- Jaccard@10;
- Spearman trên 165 features;
- mean signed SHAP;
- feature-group temporal analysis.

### Part 4 — Synthesis and report

- kết hợp prediction và explanation changes;
- tạo final temporal plots;
- viết `temporal_evaluation_v1.md`;
- không mở rộng sang formal drift detection.

---

## 18. Definition of Done

Temporal Evaluation V1 được xem là hoàn thành khi:

1. T1/T2/T3 được freeze từ toàn bộ frozen test set.
2. Prediction và SHAP analysis sử dụng cùng temporal boundaries.
3. Toàn bộ 6,276 test cases được gán đúng một window.
4. Toàn bộ 1,000 frozen SHAP cases được gán đúng một window.
5. Prediction metrics được tính cho T1/T2/T3.
6. Global SHAP importance được tính cho T1/T2/T3.
7. Jaccard@10 được tính cho ba pairwise comparisons.
8. Spearman được tính trên toàn bộ 165 features cho ba pairwise comparisons.
9. Mean signed SHAP được báo cáo cho union top features.
10. Feature-group temporal analysis được hoàn thành.
11. Prediction-explanation synthesis được tạo.
12. Các visualization chính được tạo reproducibly.
13. Frozen model, frozen test set và frozen SHAP artifacts không bị thay đổi.
14. Không retraining, resampling hoặc SHAP recomputation xảy ra.
15. Báo cáo cuối không đưa ra causal hoặc formal concept-drift claims vượt quá phạm vi thiết kế.

---

## 19. Frozen Design Summary

| Thành phần | Quyết định |
|---|---|
| Model | Frozen Random Forest Baseline V1 |
| Prediction point | `k=10` |
| Prediction population | 6,276 frozen TEST cases |
| Explanation population | 1,000 frozen SHAP TEST cases |
| Time variable | `case_start_time` |
| Windows | T1, T2, T3 |
| Window design | Chronological equal-case |
| Nominal cases/window | 2,092 |
| Boundary rule | Không chia cùng timestamp qua hai windows |
| Prediction primary metrics | ROC-AUC, Average Precision |
| Prediction secondary metrics | Accuracy, Precision, Recall, F1 |
| Distribution descriptors | Success rate, mean `P(Success)` |
| SHAP importance | Mean absolute SHAP |
| Top-K | 10 |
| Stability metric 1 | Jaccard@10 |
| Stability metric 2 | Spearman trên 165 features |
| Signed SHAP | Secondary analysis |
| Feature-group analysis | Secondary analysis |
| Model retraining | Không |
| SHAP recomputation | Không |
| Formal drift claim | Không |
| Main comparisons | T1-T2, T2-T3, T1-T3 |
