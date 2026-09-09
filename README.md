# BPIC 2017 Basic EDA

Basic Exploratory Data Analysis for the **BPI Challenge 2017** event log.

The purpose of this project is to inspect the dataset structure, validate data quality, summarize core process characteristics, and analyze the target distribution before building a Predictive Process Monitoring model.

## Project Structure

```text
bpic2017-eda/
│
├── data/
│   └── BPI Challenge 2017.xes
│
├── outputs/
│   └── eda/
│       ├── figures/
│       │   ├── activity_frequency.png
│       │   ├── activity_group_distribution.png
│       │   ├── application_type_distribution.png
│       │   ├── loan_goal_distribution.png
│       │   ├── requested_amount_boxplot.png
│       │   ├── requested_amount_histogram.png
│       │   ├── requested_amount_histogram_p99.png
│       │   └── target_distribution.png
│       │
│       ├── activity_frequency.csv
│       ├── application_type_distribution.csv
│       ├── loan_goal_distribution.csv
│       └── target_distribution.csv
│
├── scripts/
│   └── eda.py
│
├── EDA_SUMMARY.md
├── requirements.txt
├── .gitignore
└── README.md
```

> **Note:** The raw BPIC 2017 dataset is not included in the repository.

## Dataset

Dataset: **BPI Challenge 2017**

The dataset contains event logs from a personal loan application process.

### Main Characteristics

- Cases: **31,509**
- Events: **1,202,267**
- Activities: **26**
- Time range: **2016-01-01 to 2017-02-01**
- Activity groups:
  - `A_` — Application
  - `O_` — Offer
  - `W_` — Workflow

## Requirements

Recommended:

- Python 3.10+
- pandas
- pm4py
- matplotlib

Install the required dependencies with:

```powershell
python -m pip install -r requirements.txt
```

## Setup

### 1. Create a virtual environment

```powershell
python -m venv .venv
```

If `python` is unavailable:

```powershell
py -m venv .venv
```

### 2. Activate the virtual environment

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then activate the environment again:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

## Dataset Preparation

Download the **BPI Challenge 2017** dataset and place it in:

```text
data/BPI Challenge 2017.xes
```

The raw `.xes` file is intentionally excluded from the repository because of its size.

## Run the EDA

From the project root directory:

```powershell
python scripts/eda.py
```

The script performs:

- Dataset loading and schema inspection
- Case, event, and activity counting
- Time-range inspection
- Missing-value analysis
- Duplicate-event checks
- Case-attribute consistency checks
- Loan goal analysis
- Application type analysis
- Requested amount analysis
- Activity frequency analysis
- Activity-group analysis
- Events-per-case analysis
- Target / ground-truth analysis
- CSV result generation
- EDA figure generation

## Generated Outputs

All generated results are stored in:

```text
outputs/eda/
```

### CSV Files

- `activity_frequency.csv`
- `application_type_distribution.csv`
- `loan_goal_distribution.csv`
- `target_distribution.csv`

### Figures

The figures are stored in:

```text
outputs/eda/figures/
```

Generated figures:

- `activity_frequency.png`
- `activity_group_distribution.png`
- `application_type_distribution.png`
- `loan_goal_distribution.png`
- `requested_amount_boxplot.png`
- `requested_amount_histogram.png`
- `requested_amount_histogram_p99.png`
- `target_distribution.png`

## Key EDA Results

### Event Log Overview

| Metric | Value |
|---|---:|
| Cases | 31,509 |
| Events | 1,202,267 |
| Activities | 26 |
| Mean events per case | 38.16 |
| Median events per case | 35 |
| Minimum events per case | 10 |
| Maximum events per case | 180 |

### Data Quality

No missing values were found in the core event-log attributes:

- `case:concept:name`
- `concept:name`
- `time:timestamp`

Offer-specific attributes contain many missing values because they are only defined for offer-related events. These are considered **structural missing values**, rather than data-quality errors.

No duplicate events were detected:

- Duplicate `EventID`: **0**
- Exact duplicate rows: **0**
- Duplicate case + activity + timestamp: **0**

## Activity Groups

| Activity Group | Percentage of Events |
|---|---:|
| Workflow | 63.95% |
| Application | 19.93% |
| Offer | 16.12% |

Workflow activities make up the majority of the event log.

## Target / Ground Truth

Three activities are used as process outcomes:

| Target | Cases | Percentage |
|---|---:|---:|
| `A_Pending` | 17,228 | 54.85% |
| `A_Cancelled` | 10,431 | 33.21% |
| `A_Denied` | 3,752 | 11.94% |

Total labeled cases:

**31,411**

There are **98 cases** without one of the three target activities.

Their last application states are:

| Last Application State | Cases |
|---|---:|
| `A_Complete` | 53 |
| `A_Incomplete` | 42 |
| `A_Validating` | 3 |

These cases are left unresolved rather than being assigned an artificial target label.

## Additional Case-Level Findings

### Application Type

| Type | Percentage |
|---|---:|
| New credit | 89.24% |
| Limit raise | 10.76% |

### Requested Amount

| Statistic | Value |
|---|---:|
| Mean | 16,233.74 |
| Median | 12,500 |
| Q1 | 6,000 |
| Q3 | 21,000 |
| Maximum | 450,000 |
| Skewness | 4.67 |

The distribution is strongly right-skewed.

Using the 1.5×IQR rule, **1,827 cases (5.80%)** are statistical outliers. These cases are retained because unusually large requested amounts may still represent valid business cases.

## Main Task-Related Observations

### 1. Target Imbalance

The target classes are imbalanced. `A_Denied` represents only **11.94%** of labeled cases.

For future predictive modeling, evaluation should therefore not rely only on accuracy. Class-specific Precision, Recall, and F1-score should also be considered.

### 2. Variable Trace Length

Cases contain between **10 and 180 events**, with a median of **35 events**.

Additionally, 99% of cases contain no more than **93 events**.

This variation should be considered when constructing process prefixes or sequence representations for Predictive Process Monitoring.

### 3. Workflow Activities Dominate the Event Log

Workflow (`W_`) activities account for **63.95%** of all events.

Several workflow activities occur repeatedly within the same case. Predictive models and XAI methods should therefore distinguish actual predictive importance from simple activity frequency.

## Detailed EDA Summary

See [`EDA_SUMMARY.md`](EDA_SUMMARY.md) for the detailed EDA findings and analysis.