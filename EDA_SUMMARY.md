# BPIC 2017 - Basic EDA Summary

## 1. Dataset Overview

Dataset: BPI Challenge 2017

- Cases: 31,509
- Events: 1,202,267
- Activities: 26
- Time range:
  - Start: 2016-01-01
  - End: 2017-02-01

The event log contains three main activity groups:

- Application (`A_`)
- Offer (`O_`)
- Workflow (`W_`)

## 2. Events per Case

- Minimum: 10
- Mean: 38.16
- Median: 35
- Maximum: 180
- 90th percentile: 60
- 99th percentile: 93
- Skewness: 1.44

Trace lengths vary considerably across cases.

## 3. Missing Values

No missing values were found in the core event-log attributes:

- Case ID
- Activity
- Timestamp

Offer-specific attributes such as:

- CreditScore
- OfferedAmount
- MonthlyCost
- NumberOfTerms
- Accepted
- Selected

contain approximately 96.42% missing values because these attributes are
only defined for `O_Create Offer` events.

Therefore, these missing values are considered structural missingness
rather than data-quality errors.

## 4. Duplicate Check

No duplicate events were detected:

- Duplicate EventID: 0
- Exact duplicate rows: 0
- Duplicate case + activity + timestamp: 0

## 5. Target / Ground Truth

Three outcome activities were used:

- `A_Pending`: 17,228 cases (54.85%)
- `A_Cancelled`: 10,431 cases (33.21%)
- `A_Denied`: 3,752 cases (11.94%)

Total labeled cases: 31,411.

98 cases do not contain one of the three target activities.

Their final application states are:

- `A_Complete`: 53
- `A_Incomplete`: 42
- `A_Validating`: 3

These cases are kept unresolved and are not assigned an artificial label.

## 6. Task-related Observations

### Observation 1 - Target imbalance

The target distribution is imbalanced, with `A_Denied` representing only
11.94% of labeled cases. Evaluation should therefore not rely only on
accuracy; class-specific Precision, Recall and F1-score should also be
considered.

### Observation 2 - Variable trace length

Cases contain between 10 and 180 events, while 99% contain at most 93
events. This variation should be considered when constructing prefixes
or sequence representations for Predictive Process Monitoring.

### Observation 3 - Workflow events dominate the log

Workflow (`W_`) activities account for 63.95% of all events. Several
workflow activities occur repeatedly within the same case. Predictive
models and XAI methods should therefore distinguish actual predictive
importance from simple activity frequency.

## 7. Additional Case-level Findings

### Application Type

- New credit: 89.24%
- Limit raise: 10.76%

### Requested Amount

- Mean: 16,233.74
- Median: 12,500
- Q1: 6,000
- Q3: 21,000
- Maximum: 450,000
- Skewness: 4.67

The distribution is strongly right-skewed.

Using the 1.5×IQR rule, 1,827 cases (5.80%) are statistical outliers.
They are retained because extreme loan values may still represent valid
business cases.