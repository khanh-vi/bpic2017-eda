# BPIC 2017 Feature Availability

Feature Availability xác định những thuộc tính BPIC 2017 thực sự có thể sử dụng tại điểm dự đoán cố định mà không đưa thông tin tương lai vào mô hình (data leakage).

## 1. Điểm dự đoán (Prediction Point)

- **Prediction point:** `k = 10`
- **Observation prefix:** sự kiện 1–10
- **Future suffix:** sự kiện `> 10`

Mô hình chỉ được sử dụng thông tin thực sự quan sát được trong prediction prefix.

Tập dữ liệu tại biên dự đoán đã được kiểm chứng như sau:

```yaml
total_cases: 31,509
cases_with_outcome: 31,411
cases_with_one_distinct_outcome: 31,411
cases_with_ambiguous_outcome: 0
cases_with_repeated_outcome_events: 1
cases_without_outcome: 98
cases_with_outcome_by_k: 35
eligible_cases: 31,376
prefix_rows: 313,760
future_rows: 884,184
outcome_inside_prefix: 0
```

Mỗi hồ sơ đủ điều kiện đóng góp chính xác 10 sự kiện vào prefix.

Một ngoại lệ ngắn cần lưu ý là `Application_1116429997`: hồ sơ này có hai sự kiện `A_Denied` riêng biệt nhưng chỉ có một nhãn kết quả phân biệt. Vì vậy, hồ sơ không được xem là mơ hồ; cả hai dòng sự kiện vẫn được giữ trong log; lần xuất hiện `A_Denied` sớm nhất được dùng để xác định thời điểm kết quả trở nên đã biết.

## 2. Phương pháp xác định khả dụng

Mỗi thuộc tính thô được đánh giá bằng ba câu hỏi:

1. Thuộc tính đã được quan sát tại `k = 10` chưa?
2. Thông tin đó có thực tế được biết tại thời điểm dự đoán không?
3. Việc lấy giá trị có cần thông tin từ các sự kiện tương lai không?

Các quyết định về khả dụng được hiểu như sau:

- **Available:** thông tin quan sát được tại thời điểm dự đoán mà không cần dữ liệu tương lai.
- **Conditionally Available:** thông tin chỉ sử dụng được khi trạng thái quy trình tương ứng đã xảy ra trong sự kiện 1–10. Nếu chưa xuất hiện tại `k = 10`, giá trị phải giữ ở trạng thái thiếu/chưa quan sát; tuyệt đối không backfill từ future suffix.
- **Leakage Risk:** log sự kiện thô phơi bày thông tin có ngữ nghĩa phụ thuộc vào diễn biến quy trình về sau và không thể tin cậy là đã biết tại thời điểm dự đoán thực tế.

**Feature availability** khác với **model inclusion / model handling**. Một thuộc tính có thể khả dụng nhưng vẫn không phù hợp làm biến dự báo thô, điển hình là các định danh.

## 3. Kiểm toán thực nghiệm (Empirical Audits)

### 3.1. Thuộc tính cấp hồ sơ

Ba thuộc tính `case:LoanGoal`, `case:ApplicationType` và `case:RequestedAmount` đều có kết quả giống nhau:

- khả dụng ở 31,376/31,376 hồ sơ tại `k = 10` (100%);
- lần đầu xuất hiện tại sự kiện 1 trong mọi hồ sơ;
- 0 hồ sơ xuất hiện lần đầu sau `k`;
- 0 hồ sơ không bao giờ quan sát được;
- 0 trường hợp không nhất quán trong cùng hồ sơ.

Có 3,064 hồ sơ đủ điều kiện với `case:RequestedAmount == 0` tại sự kiện 1. Giá trị `0` được xem là đã quan sát, không phải dữ liệu thiếu; đây là vấn đề chất lượng dữ liệu tách biệt với tính khả dụng.

### 3.2. Thuộc tính sự kiện/quy trình

Sáu thuộc tính `concept:name`, `Action`, `EventOrigin`, `org:resource`, `lifecycle:transition` và `time:timestamp` đều hiện diện trên toàn bộ 313,760 dòng prefix:

- độ đầy đủ theo dòng: 100%;
- bao phủ đủ 31,376 hồ sơ: 100%;
- không có giá trị thiếu tại bất kỳ vị trí sự kiện nào từ 1 đến 10.

Đây là **khả dụng theo phạm vi prefix (prefix-scoped availability)**: một thuộc tính cấp sự kiện hợp lệ nếu giá trị thuộc về một sự kiện đã quan sát trong prefix. Việc cùng thuộc tính đó còn có các giá trị ở sự kiện sau không biến giá trị trước đó thành leakage. Cách mã hóa hoặc tổng hợp các sự kiện là quyết định mô hình hóa riêng, không phải quyết định về tính khả dụng.

### 3.3. Thuộc tính offer thưa

Bảy thuộc tính `FirstWithdrawalAmount`, `NumberOfTerms`, `Accepted`, `MonthlyCost`, `Selected`, `CreditScore` và `OfferedAmount` đều được ghi lần đầu trên `O_Create Offer`. Đối với bảy thuộc tính này:

```yaml
available_by_k: 20,905 / 31,376 = 66.63%
future_only:    10,471 / 31,376 = 33.37%
never_observed: 0
```

Một application có thể chứa nhiều offer, nên nhiều giá trị trong cùng hồ sơ là hợp lệ và không phải sự không nhất quán. `FirstWithdrawalAmount`, `NumberOfTerms`, `Accepted`, `MonthlyCost`, `CreditScore` và `OfferedAmount` được phân loại là **khả dụng có điều kiện (Conditionally Available)**: chỉ dùng giá trị nếu nó đã xuất hiện trong sự kiện 1–10.

`OfferID` có phạm vi quan sát khác:

```yaml
available_by_k: 16,761 / 31,376 = 53.42%
future_only:    14,615 / 31,376 = 46.58%
```

Thuộc tính này là **Conditionally Available**, nhưng cách xử lý mô hình là **loại định danh thô; chỉ giữ để nhóm các sự kiện offer**.

### 3.4. Định danh

`EventID` và `case:concept:name` đều khả dụng trên 100% dòng prefix và 100% hồ sơ. `EventID` có 313.760 giá trị phân biệt trên 313.760 dòng. `case:concept:name` có đúng 31.376 ID phân biệt và mỗi ID xuất hiện đúng 10 lần.

Cả hai được phân loại là **Available**, nhưng phải loại khỏi mô hình dự báo dưới dạng định danh thô. Đây là quyết định về model handling, không phải kết luận leakage.

## 4. Kiểm toán rò rỉ ngữ nghĩa (Semantic Leakage Audit)

`Accepted` có liên hệ với các kết quả offer xuất hiện về sau, nhưng không mã hóa một cách tất định trạng thái `O_Accepted` sau đó. Kiểm toán không cung cấp bằng chứng đủ mạnh để xem thuộc tính này là leakage phát sinh từ tương lai; vì vậy quyết định cuối cùng vẫn là **Conditionally Available**.

`Selected` được phân loại là **Leakage Risk**, với **Model handling: Exclude**. Bằng chứng thực nghiệm mạnh nhất là:

- không offer nào có `Selected=False` về sau đạt `O_Accepted`;
- mọi offer có `Selected=True` về sau đều đạt `O_Returned`;
- mọi offer có `Selected=True` về sau đều đạt `O_Sent`;
- `Selected` phản ánh mạnh hành vi vòng đời offer/lựa chọn của khách hàng xảy ra về sau.

Một giá trị “được ghi trên một sự kiện lịch sử sớm” không nhất thiết đồng nghĩa với “có thể biết tại thời điểm thực tế của sự kiện đó”. Do đó, đây là rủi ro leakage về ngữ nghĩa và thời gian.

## 5. Ma trận Feature Availability cuối cùng

| Feature | Group | Availability at k=10 | Availability Decision | Model Handling |
|---|---|---:|---|---|
| `case:LoanGoal` | case-level | 31,376/31,376 (100%) | Available | Candidate model feature |
| `case:ApplicationType` | case-level | 31,376/31,376 (100%) | Available | Candidate model feature |
| `case:RequestedAmount` | case-level | 31,376/31,376 (100%) | Available | Candidate model feature |
| `concept:name` | event-process | 31,376/31,376 (100%) | Available | Prefix-safe encoding or aggregation |
| `Action` | event-process | 31,376/31,376 (100%) | Available | Prefix-safe encoding or aggregation |
| `EventOrigin` | event-process | 31,376/31,376 (100%) | Available | Prefix-safe encoding or aggregation |
| `org:resource` | event-process | 31,376/31,376 (100%) | Available | Evaluate raw-resource generalization separately |
| `lifecycle:transition` | event-process | 31,376/31,376 (100%) | Available | Prefix-safe encoding or aggregation |
| `time:timestamp` | event-process | 31,376/31,376 (100%) | Available | Use prefix-derived temporal features only |
| `FirstWithdrawalAmount` | offer-sparse | 20,905/31,376 (66.63%) | Conditionally Available | Use only when observed within events 1-10 |
| `NumberOfTerms` | offer-sparse | 20,905/31,376 (66.63%) | Conditionally Available | Use only when observed within events 1-10 |
| `Accepted` | offer-sparse | 20,905/31,376 (66.63%) | Conditionally Available | Use only when observed within events 1-10 |
| `MonthlyCost` | offer-sparse | 20,905/31,376 (66.63%) | Conditionally Available | Use only when observed within events 1-10 |
| `Selected` | offer-sparse | 20,905/31,376 (66.63%) | Leakage Risk | Exclude |
| `CreditScore` | offer-sparse | 20,905/31,376 (66.63%) | Conditionally Available | Use only when observed within events 1-10 |
| `OfferedAmount` | offer-sparse | 20,905/31,376 (66.63%) | Conditionally Available | Use only when observed within events 1-10 |
| `OfferID` | offer-sparse | 16,761/31,376 (53.42%) | Conditionally Available | Exclude raw ID; retain for offer grouping |
| `EventID` | identifier | 31,376/31,376 (100%) | Available | Exclude as a raw identifier |
| `case:concept:name` | identifier | 31,376/31,376 (100%) | Available | Exclude as a raw identifier |

Tổng hợp quyết định: **Available: 11**, **Conditionally Available: 7**, **Leakage Risk: 1**.

## 6. Quy tắc ngăn leakage và khả năng tái lập

1. Chỉ các sự kiện có `event_nr <= 10` được đóng góp vào đầu vào mô hình.
2. Giá trị xuất hiện lần đầu sau `k = 10` không bao giờ được truyền ngược vào prefix.
3. `Selected` không được dùng làm biến dự báo.
4. `EventID`, `case:concept:name` và `OfferID` thô không được dùng làm định danh dự báo.
5. Đặc trưng thời gian chỉ được dùng timestamp của các sự kiện prefix đã quan sát.
6. Không được dùng timestamp cuối, tổng độ dài trace, tổng thời lượng hồ sơ, hoạt động cuối hoặc bất kỳ thông tin nào suy ra từ suffix.
7. Nhiều offer trong một hồ sơ là hợp lệ và không được diễn giải là bản ghi không nhất quán.

Phân tích được triển khai tại `scripts/feature_availability.py`. Ma trận máy đọc được cuối cùng được xuất tái lập tại `results/feature_availability_matrix.csv` khi chạy từ project root:

```text
python scripts/feature_availability.py
```
