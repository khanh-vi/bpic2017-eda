from pathlib import Path

import pandas as pd
import pm4py
import matplotlib.pyplot as plt


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "BPI Challenge 2017.xes"
OUTPUT_DIR = Path("outputs/eda")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FIGURE_DIR = OUTPUT_DIR / "figures"

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CASE_ID = "case:concept:name"
ACTIVITY = "concept:name"
TIMESTAMP = "time:timestamp"
RESOURCE = "org:resource"
LIFECYCLE = "lifecycle:transition"

print("Reading dataset...")

df = pm4py.read_xes(str(DATA_PATH))

print("\nDataset read successfully")

print("\n=== SHAPE ===")
print(df.shape)

print("\n=== COLUMNS ===")
print(df.columns.tolist())

print("\n=== DTYPES ===")
print(df.dtypes)

print("\n=== FIRST 5 ROWS ===")
print(df.head().to_string())

print("\n=== BASIC EVENT LOG INFO ===")

print(f"Number of events: {len(df):,}")

print(
    f"Number of cases: "
    f"{df[CASE_ID].nunique():,}"
)

print(
    f"Number of activities: "
    f"{df[ACTIVITY].nunique():,}"
)

print("\n=== ACTIVITIES ===")

activities = sorted(
    df[ACTIVITY]
    .dropna()
    .unique()
)

for activity in activities:
    print(activity)

print("\n=== UNIQUE VALUES PER COLUMN ===")

for column in df.columns:
    unique_count = df[column].nunique(
        dropna=True
    )

    print(
        f"{column:<30} "
        f"{unique_count:>10,}"
    )

CATEGORICAL_COLUMNS = [
    "Action",
    "EventOrigin",
    "lifecycle:transition",
    "case:LoanGoal",
    "case:ApplicationType",
    "Accepted",
    "Selected",
]

for column in CATEGORICAL_COLUMNS:
    print(
        f"\n=== {column} ==="
    )

    print(
        df[column]
        .value_counts(
            dropna=False
        )
    )

print("\n=== TIME RANGE ===")

print(
    "Start:",
    df[TIMESTAMP].min()
)

print(
    "End:",
    df[TIMESTAMP].max()
)

print("\n=== MISSING VALUES ===")

missing_summary = pd.DataFrame({
    "missing_count": df.isna().sum(),
    "missing_percent": df.isna().mean() * 100,
})

missing_summary = missing_summary.sort_values(
    "missing_percent",
    ascending=False,
)

print(
    missing_summary.to_string(
        formatters={
            "missing_percent": "{:.2f}%".format
        }
    )
)

print("\n=== CRITICAL COLUMN MISSING VALUES ===")

critical_columns = [
    CASE_ID,
    ACTIVITY,
    TIMESTAMP,
]

for column in critical_columns:
    missing_count = df[column].isna().sum()

    print(
        f"{column:<25}: "
        f"{missing_count:,}"
    )

print("\n=== ACCEPTED NON-NULL EVENTS ===")

accepted_events = (
    df.loc[
        df["Accepted"].notna(),
        ACTIVITY,
    ]
    .value_counts()
)

print(accepted_events)


print("\n=== SELECTED NON-NULL EVENTS ===")

selected_events = (
    df.loc[
        df["Selected"].notna(),
        ACTIVITY,
    ]
    .value_counts()
)

print(selected_events)

print("\n=== OFFER ATTRIBUTE LOCATIONS ===")

offer_columns = [
    "FirstWithdrawalAmount",
    "NumberOfTerms",
    "Accepted",
    "MonthlyCost",
    "Selected",
    "CreditScore",
    "OfferedAmount",
    "OfferID",
]

for column in offer_columns:

    print(f"\n--- {column} ---")

    counts = (
        df.loc[
            df[column].notna(),
            ACTIVITY,
        ]
        .value_counts()
    )

    print(counts)

print("\n=== CASE ATTRIBUTE MISSING VALUES ===")

case_columns = [
    "case:LoanGoal",
    "case:ApplicationType",
    "case:RequestedAmount",
]

for column in case_columns:

    missing_count = df[column].isna().sum()

    print(
        f"{column:<30}: "
        f"{missing_count:,}"
    )

print("\n=== DUPLICATE EVENT ID CHECK ===")

duplicate_event_ids = df["EventID"].duplicated().sum()

print(
    f"Duplicate EventID count: "
    f"{duplicate_event_ids:,}"
)

print("\n=== EXACT DUPLICATE ROW CHECK ===")

exact_duplicates = df.duplicated().sum()

print(
    f"Exact duplicate rows: "
    f"{exact_duplicates:,}"
)

print(
    "\n=== SAME CASE + ACTIVITY + TIMESTAMP CHECK ==="
)

duplicate_event_keys = df.duplicated(
    subset=[
        CASE_ID,
        ACTIVITY,
        TIMESTAMP,
    ],
    keep=False,
)

duplicate_key_rows = df[
    duplicate_event_keys
]

print(
    f"Rows involved: "
    f"{len(duplicate_key_rows):,}"
)

if not duplicate_key_rows.empty:

    print(
        "\n=== SAMPLE SAME CASE + ACTIVITY + TIMESTAMP ==="
    )

    columns_to_show = [
        CASE_ID,
        ACTIVITY,
        TIMESTAMP,
        "lifecycle:transition",
        "Action",
        "EventID",
        "org:resource",
    ]

    print(
        duplicate_key_rows[
            columns_to_show
        ]
        .sort_values(
            [
                CASE_ID,
                TIMESTAMP,
                ACTIVITY,
            ]
        )
        .head(20)
        .to_string(index=False)
    )

print(
    "\n=== SAME CASE + TIMESTAMP CHECK ==="
)

same_case_timestamp = df.duplicated(
    subset=[
        CASE_ID,
        TIMESTAMP,
    ],
    keep=False,
)

same_case_timestamp_rows = df[
    same_case_timestamp
]

print(
    f"Rows sharing timestamp within same case: "
    f"{len(same_case_timestamp_rows):,}"
)

print("\n=== CASE ATTRIBUTE CONSISTENCY ===")

case_attributes = [
    "case:LoanGoal",
    "case:ApplicationType",
    "case:RequestedAmount",
]

for column in case_attributes:

    unique_per_case = (
        df.groupby(CASE_ID)[column]
        .nunique(dropna=False)
    )

    inconsistent_cases = unique_per_case[
        unique_per_case > 1
    ]

    print(f"\n--- {column} ---")

    print(
        f"Inconsistent cases: "
        f"{len(inconsistent_cases):,}"
    )

    if len(inconsistent_cases) > 0:

        sample_case_ids = (
            inconsistent_cases
            .head(5)
            .index
        )

        sample = (
            df.loc[
                df[CASE_ID].isin(sample_case_ids),
                [
                    CASE_ID,
                    TIMESTAMP,
                    column,
                ],
            ]
            .sort_values(
                [
                    CASE_ID,
                    TIMESTAMP,
                ]
            )
        )

        print(
            sample.to_string(
                index=False
            )
        )

case_data = (
    df[
        [
            CASE_ID,
            "case:LoanGoal",
            "case:ApplicationType",
            "case:RequestedAmount",
        ]
    ]
    .drop_duplicates(
        subset=[CASE_ID]
    )
    .reset_index(drop=True)
)

print("\n=== CASE-LEVEL DATA ===")

print(
    f"Rows: {len(case_data):,}"
)

print(
    case_data.head().to_string(
        index=False
    )
)

print("\n=== LOAN GOAL DISTRIBUTION ===")

loan_goal_distribution = (
    case_data["case:LoanGoal"]
    .value_counts()
    .rename_axis("loan_goal")
    .reset_index(name="case_count")
)

loan_goal_distribution["percent"] = (
    loan_goal_distribution["case_count"]
    / len(case_data)
    * 100
)

print(
    loan_goal_distribution.to_string(
        index=False,
        formatters={
            "percent": "{:.2f}%".format,
        },
    )
)

loan_goal_distribution.to_csv(
    OUTPUT_DIR / "loan_goal_distribution.csv",
    index=False,
)

print("\n=== APPLICATION TYPE DISTRIBUTION ===")

application_type_distribution = (
    case_data["case:ApplicationType"]
    .value_counts()
    .rename_axis("application_type")
    .reset_index(name="case_count")
)

application_type_distribution["percent"] = (
    application_type_distribution["case_count"]
    / len(case_data)
    * 100
)

print(
    application_type_distribution.to_string(
        index=False,
        formatters={
            "percent": "{:.2f}%".format,
        },
    )
)

application_type_distribution.to_csv(
    OUTPUT_DIR / "application_type_distribution.csv",
    index=False,
)

print("\n=== CASE DISTRIBUTION VALIDATION ===")

print(
    "LoanGoal total:",
    loan_goal_distribution["case_count"].sum(),
)

print(
    "ApplicationType total:",
    application_type_distribution["case_count"].sum(),
)

print(
    "Expected cases:",
    len(case_data),
)

print("\nCreating LoanGoal chart...")

loan_goal_plot = (
    loan_goal_distribution
    .sort_values(
        "case_count",
        ascending=True,
    )
)

plt.figure(
    figsize=(10, 7)
)

plt.barh(
    loan_goal_plot["loan_goal"],
    loan_goal_plot["case_count"],
)

plt.xlabel("Number of cases")
plt.ylabel("Loan goal")
plt.title(
    "Distribution of Loan Goals"
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "loan_goal_distribution.png",
    dpi=300,
)

plt.close()

print("Creating ApplicationType chart...")

plt.figure(
    figsize=(7, 5)
)

bars = plt.bar(
    application_type_distribution[
        "application_type"
    ],
    application_type_distribution[
        "case_count"
    ],
)

plt.xlabel("Application type")
plt.ylabel("Number of cases")
plt.title(
    "Distribution of Application Types"
)

for bar, percent in zip(
    bars,
    application_type_distribution[
        "percent"
    ],
):
    plt.text(
        bar.get_x()
        + bar.get_width() / 2,
        bar.get_height(),
        f"{percent:.1f}%",
        ha="center",
        va="bottom",
    )

plt.tight_layout()

plt.savefig(
    FIGURE_DIR
    / "application_type_distribution.png",
    dpi=300,
)

plt.close()

print("\n=== REQUESTED AMOUNT SUMMARY ===")

requested_amount = case_data[
    "case:RequestedAmount"
]

print(
    requested_amount
    .describe()
    .to_string()
)

print("\n=== REQUESTED AMOUNT KEY STATISTICS ===")

print(
    f"Minimum: "
    f"{requested_amount.min():,.2f}"
)

print(
    f"Maximum: "
    f"{requested_amount.max():,.2f}"
)

print(
    f"Mean: "
    f"{requested_amount.mean():,.2f}"
)

print(
    f"Median: "
    f"{requested_amount.median():,.2f}"
)

print(
    f"Standard deviation: "
    f"{requested_amount.std():,.2f}"
)

requested_amount_skew = (
    requested_amount.skew()
)

print(
    f"Skewness: "
    f"{requested_amount_skew:.4f}"
)

print("\n=== REQUESTED AMOUNT OUTLIER CHECK ===")

q1 = requested_amount.quantile(0.25)
q3 = requested_amount.quantile(0.75)

iqr = q3 - q1

lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

print(f"Q1: {q1:,.2f}")
print(f"Q3: {q3:,.2f}")
print(f"IQR: {iqr:,.2f}")

print(
    f"Lower bound: "
    f"{lower_bound:,.2f}"
)

print(
    f"Upper bound: "
    f"{upper_bound:,.2f}"
)

outlier_mask = (
    (requested_amount < lower_bound)
    |
    (requested_amount > upper_bound)
)

outlier_count = outlier_mask.sum()

outlier_percent = (
    outlier_count
    / len(requested_amount)
    * 100
)

print(
    f"Outlier cases: "
    f"{outlier_count:,}"
)

print(
    f"Outlier percentage: "
    f"{outlier_percent:.2f}%"
)

print(
    "\n=== TOP 10 REQUESTED AMOUNTS ==="
)

top_requested_amounts = (
    case_data[
        [
            CASE_ID,
            "case:LoanGoal",
            "case:ApplicationType",
            "case:RequestedAmount",
        ]
    ]
    .sort_values(
        "case:RequestedAmount",
        ascending=False,
    )
    .head(10)
)

print(
    top_requested_amounts
    .to_string(index=False)
)

print("\n=== ZERO REQUESTED AMOUNT CHECK ===")

zero_amount_cases = case_data[
    case_data["case:RequestedAmount"] == 0
]

print(
    f"Cases with RequestedAmount = 0: "
    f"{len(zero_amount_cases):,}"
)

print(
    f"Percentage: "
    f"{len(zero_amount_cases) / len(case_data) * 100:.2f}%"
)

if not zero_amount_cases.empty:

    print("\n=== ZERO AMOUNT BY APPLICATION TYPE ===")

    print(
        zero_amount_cases[
            "case:ApplicationType"
        ].value_counts()
    )

    print("\n=== ZERO AMOUNT BY LOAN GOAL ===")

    print(
        zero_amount_cases[
            "case:LoanGoal"
        ].value_counts()
    )

print("\n=== NEGATIVE REQUESTED AMOUNT CHECK ===")

negative_amount_count = (
    case_data["case:RequestedAmount"] < 0
).sum()

print(
    f"Negative RequestedAmount cases: "
    f"{negative_amount_count:,}"
)

print("\n=== REQUESTED AMOUNT PERCENTILES ===")

percentiles = requested_amount.quantile(
    [
        0.50,
        0.75,
        0.90,
        0.95,
        0.99,
        0.995,
    ]
)

for percentile, value in percentiles.items():

    print(
        f"{percentile * 100:>5.1f}%: "
        f"{value:,.2f}"
    )

print(
    "\n=== REQUESTED AMOUNT BY APPLICATION TYPE ==="
)

amount_by_application_type = (
    case_data
    .groupby("case:ApplicationType")[
        "case:RequestedAmount"
    ]
    .agg(
        [
            "count",
            "mean",
            "median",
            "min",
            "max",
        ]
    )
)

print(
    amount_by_application_type.to_string(
        float_format=lambda x: f"{x:,.2f}"
    )
)

print("\nCreating RequestedAmount histogram...")

plt.figure(
    figsize=(9, 5)
)

plt.hist(
    requested_amount,
    bins=50,
)

plt.xlabel("Requested amount")
plt.ylabel("Number of cases")
plt.title(
    "Distribution of Requested Amount"
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR
    / "requested_amount_histogram.png",
    dpi=300,
)

plt.close()

p99 = requested_amount.quantile(0.99)

requested_amount_99 = (
    requested_amount[
        requested_amount <= p99
    ]
)

print(
    "Creating zoomed RequestedAmount histogram..."
)

plt.figure(
    figsize=(9, 5)
)

plt.hist(
    requested_amount_99,
    bins=40,
)

plt.xlabel("Requested amount")
plt.ylabel("Number of cases")

plt.title(
    "Distribution of Requested Amount "
    "(Up to 99th Percentile)"
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR
    / "requested_amount_histogram_p99.png",
    dpi=300,
)

plt.close()

print(
    "Creating RequestedAmount boxplot..."
)

plt.figure(
    figsize=(9, 4)
)

plt.boxplot(
    requested_amount,
    vert=False,
)

plt.yticks([])

plt.xlabel("Requested amount")

plt.title(
    "Boxplot of Requested Amount"
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR
    / "requested_amount_boxplot.png",
    dpi=300,
)

plt.close()

print("\n=== ACTIVITY FREQUENCY ===")

activity_distribution = (
    df[ACTIVITY]
    .value_counts()
    .rename_axis("activity")
    .reset_index(name="event_count")
)

activity_distribution["percent"] = (
    activity_distribution["event_count"]
    / len(df)
    * 100
)

print(
    activity_distribution.to_string(
        index=False,
        formatters={
            "percent": "{:.2f}%".format,
        },
    )
)

print("\n=== ACTIVITY FREQUENCY VALIDATION ===")

print(
    "Activity event total:",
    activity_distribution[
        "event_count"
    ].sum(),
)

print(
    "Expected events:",
    len(df),
)

activity_distribution.to_csv(
    OUTPUT_DIR / "activity_frequency.csv",
    index=False,
)

activity_group = (
    df[ACTIVITY]
    .str.extract(
        r"^([A-Z])_",
        expand=False,
    )
)

activity_group_names = {
    "A": "Application",
    "O": "Offer",
    "W": "Workflow",
}

activity_group = activity_group.map(activity_group_names)

print("\n=== ACTIVITY GROUP DISTRIBUTION ===")

activity_group_distribution = (
    activity_group
    .value_counts()
    .rename_axis("activity_group")
    .reset_index(name="event_count")
)

activity_group_distribution[
    "percent"
] = (
    activity_group_distribution[
        "event_count"
    ]
    / len(df)
    * 100
)

print(
    activity_group_distribution.to_string(
        index=False,
        formatters={
            "percent": "{:.2f}%".format,
        },
    )
)

print(
    "\n=== ACTIVITY GROUP VALIDATION ==="
)

missing_group = (
    activity_group
    .isna()
    .sum()
)

print(
    f"Events without activity group: "
    f"{missing_group:,}"
)

print(
    "Grouped event total:",
    activity_group_distribution[
        "event_count"
    ].sum(),
)

print(
    "Expected events:",
    len(df),
)

print("\nCreating activity frequency chart...")

activity_plot = (
    activity_distribution
    .sort_values(
        "event_count",
        ascending=True,
    )
)

plt.figure(
    figsize=(11, 9)
)

plt.barh(
    activity_plot["activity"],
    activity_plot["event_count"],
)

plt.xlabel("Number of events")
plt.ylabel("Activity")
plt.title(
    "Frequency of Activities in BPIC 2017"
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR
    / "activity_frequency.png",
    dpi=300,
)

plt.close()

print(
    "Creating activity group distribution chart..."
)

plt.figure(
    figsize=(7, 5)
)

bars = plt.bar(
    activity_group_distribution[
        "activity_group"
    ],
    activity_group_distribution[
        "event_count"
    ],
)

plt.xlabel("Activity group")
plt.ylabel("Number of events")
plt.title(
    "Distribution of Activity Groups"
)

for bar, percent in zip(
    bars,
    activity_group_distribution[
        "percent"
    ],
):
    plt.text(
        bar.get_x()
        + bar.get_width() / 2,
        bar.get_height(),
        f"{percent:.1f}%",
        ha="center",
        va="bottom",
    )

plt.tight_layout()

plt.savefig(
    FIGURE_DIR
    / "activity_group_distribution.png",
    dpi=300,
)

plt.close()

print("\n=== TRACE LENGTH SUMMARY ===")

trace_length = (
    df.groupby(CASE_ID)
    .size()
    .rename("event_count")
)

print(
    trace_length.describe().to_string()
)

print("\n=== TRACE LENGTH KEY STATISTICS ===")

print(
    f"Minimum events per case: "
    f"{trace_length.min():,}"
)

print(
    f"Maximum events per case: "
    f"{trace_length.max():,}"
)

print(
    f"Mean events per case: "
    f"{trace_length.mean():.2f}"
)

print(
    f"Median events per case: "
    f"{trace_length.median():.2f}"
)

print(
    f"Standard deviation: "
    f"{trace_length.std():.2f}"
)

print(
    f"Skewness: "
    f"{trace_length.skew():.4f}"
)

print("\n=== TRACE LENGTH PERCENTILES ===")

trace_percentiles = trace_length.quantile(
    [
        0.50,
        0.75,
        0.90,
        0.95,
        0.99,
    ]
)

for percentile, value in trace_percentiles.items():

    print(
        f"{percentile * 100:>5.1f}%: "
        f"{value:.0f} events"
    )

print("\n=== TOP 10 LONGEST CASES ===")

longest_cases = (
    trace_length
    .sort_values(
        ascending=False
    )
    .head(10)
)

print(
    longest_cases.to_string()
)

print("\n=== TRACE LENGTH VALIDATION ===")

print(
    "Total events from traces:",
    trace_length.sum(),
)

print(
    "Expected events:",
    len(df),
)

print(
    "Total cases:",
    len(trace_length),
)

print(
    "Expected cases:",
    df[CASE_ID].nunique(),
)

print("\n=== TARGET / GROUND TRUTH CHECK ===")

target_activities = [
    "A_Pending",
    "A_Cancelled",
    "A_Denied",
]

# Mỗi case có bao nhiêu loại target activity
target_presence = (
    df[df[ACTIVITY].isin(target_activities)]
    .groupby(CASE_ID)[ACTIVITY]
    .nunique()
)

cases_with_no_target = (
    df[CASE_ID].nunique()
    - len(target_presence)
)

cases_with_one_target = (
    target_presence == 1
).sum()

cases_with_multiple_targets = (
    target_presence > 1
).sum()

print(
    f"Cases with no target activity: "
    f"{cases_with_no_target:,}"
)

print(
    f"Cases with exactly one target: "
    f"{cases_with_one_target:,}"
)

print(
    f"Cases with multiple target activities: "
    f"{cases_with_multiple_targets:,}"
)

print("\n=== TARGET DISTRIBUTION BY CASE ===")

target_distribution = (
    df[
        df[ACTIVITY].isin(target_activities)
    ]
    .groupby(ACTIVITY)[CASE_ID]
    .nunique()
    .sort_values(
        ascending=False
    )
)

print(target_distribution)

print("\n=== CASES WITHOUT TARGET ===")

cases_with_target = set(
    df.loc[
        df[ACTIVITY].isin(target_activities),
        CASE_ID,
    ]
)

all_cases = set(
    df[CASE_ID]
)

cases_without_target = (
    all_cases - cases_with_target
)

print(
    f"Number of cases without target: "
    f"{len(cases_without_target):,}"
)

if cases_without_target:

    no_target_events = (
        df[
            df[CASE_ID].isin(
                cases_without_target
            )
        ]
        .sort_values(
            [CASE_ID, TIMESTAMP]
        )
    )

    application_events = (
        no_target_events[
            no_target_events[
                ACTIVITY
            ].str.startswith("A_")
        ]
    )

    last_application_activity = (
        application_events
        .groupby(CASE_ID)
        .tail(1)
    )

    print(
        "\n=== LAST APPLICATION ACTIVITY "
        "FOR CASES WITHOUT TARGET ==="
    )

    print(
        last_application_activity[
            ACTIVITY
        ].value_counts()
    )

print("\n=== TARGET DISTRIBUTION ===")

target_distribution = (
    df[
        df[ACTIVITY].isin(target_activities)
    ]
    .groupby(ACTIVITY)[CASE_ID]
    .nunique()
    .rename("case_count")
    .reset_index()
    .rename(columns={
        ACTIVITY: "target"
    })
)

total_labeled_cases = (
    target_distribution["case_count"].sum()
)

target_distribution["percent"] = (
    target_distribution["case_count"]
    / total_labeled_cases
    * 100
)

target_distribution = (
    target_distribution
    .sort_values(
        "case_count",
        ascending=False,
    )
)

print(
    target_distribution.to_string(
        index=False,
        formatters={
            "percent": "{:.2f}%".format,
        },
    )
)

target_distribution.to_csv(
    OUTPUT_DIR / "target_distribution.csv",
    index=False,
)

print("\nCreating target distribution chart...")

plt.figure(
    figsize=(7, 5)
)

bars = plt.bar(
    target_distribution["target"],
    target_distribution["case_count"],
)

plt.xlabel("Target outcome")
plt.ylabel("Number of cases")
plt.title(
    "Distribution of Target Outcomes"
)

for bar, percent in zip(
    bars,
    target_distribution["percent"],
):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{percent:.1f}%",
        ha="center",
        va="bottom",
    )

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "target_distribution.png",
    dpi=300,
)

plt.close()

