import io

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import StrMethodFormatter
import pandas as pd
import seaborn as sns
import streamlit as st

ID_COLS = ["id"]
TARGET_COL = "Heart Disease"
CONTINUOUS_COLS = ["Age", "BP", "Cholesterol", "Max HR", "ST depression"]
CATEGORICAL_COLS = [
    "Sex",
    "Chest pain type",
    "FBS over 120",
    "EKG results",
    "Exercise angina",
    "Slope of ST",
    "Number of vessels fluro",
    "Thallium",
]

st.set_page_config(page_title="Heart Disease EDA", page_icon=":bar_chart:", layout="wide")
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_style("whitegrid")


@st.cache_data(show_spinner=False)
def load_data(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes))


def available_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [col for col in columns if col in df.columns]


def build_schema(df: pd.DataFrame) -> dict[str, list[str] | str | None]:
    id_cols = available_columns(df, ID_COLS)
    target_col = TARGET_COL if TARGET_COL in df.columns else None
    continuous_cols = available_columns(df, CONTINUOUS_COLS)
    categorical_cols = available_columns(df, CATEGORICAL_COLS)

    reserved = set(id_cols + continuous_cols + categorical_cols)
    if target_col:
        reserved.add(target_col)

    other_cols = [col for col in df.columns if col not in reserved]
    return {
        "id_cols": id_cols,
        "target_col": target_col,
        "continuous_cols": continuous_cols,
        "categorical_cols": categorical_cols,
        "other_cols": other_cols,
    }


def encode_binary_target(series: pd.Series) -> tuple[pd.Series | None, str | None, list[str] | None]:
    cleaned = series.astype(str).str.strip()
    labels = [label for label in pd.unique(cleaned) if label.lower() != "nan"]
    if len(labels) != 2:
        return None, None, None

    if set(labels) == {"Absence", "Presence"}:
        class_order = ["Absence", "Presence"]
        positive_label = "Presence"
    else:
        class_order = sorted(labels)
        positive_label = class_order[-1]

    binary_target = (cleaned == positive_label).astype(int)
    return binary_target, positive_label, class_order


def sample_for_plot(df: pd.DataFrame, max_rows: int = 100_000) -> pd.DataFrame:
    if len(df) <= max_rows:
        return df
    return df.sample(max_rows, random_state=42)


def format_column_list(columns: list[str]) -> str:
    return ", ".join(columns) if columns else "None"


def build_top_risk_patterns(
    df: pd.DataFrame,
    binary_target: pd.Series,
    positive_label: str,
    categorical_cols: list[str],
    continuous_cols: list[str],
    min_support_ratio: float = 0.005,
) -> tuple[pd.DataFrame, float, int]:
    baseline_rate = float(binary_target.mean())
    min_support = max(100, int(len(df) * min_support_ratio))
    patterns: list[dict[str, object]] = []

    for col in categorical_cols:
        stats = (
            pd.DataFrame({col: df[col], "_target": binary_target})
            .groupby(col, dropna=False)["_target"]
            .agg(["mean", "count"])
            .reset_index()
        )
        stats = stats[(stats["count"] >= min_support) & (stats["mean"] >= baseline_rate)]

        for _, row in stats.iterrows():
            patterns.append(
                {
                    "Type": "Categorical",
                    "Feature": col,
                    "Pattern": f"{col} = {row[col]}",
                    "Count": int(row["count"]),
                    "Support (%)": round(row["count"] / len(df) * 100, 2),
                    f"{positive_label} rate (%)": round(row["mean"] * 100, 2),
                    "Lift": round(row["mean"] / baseline_rate, 2),
                }
            )

    for col in continuous_cols:
        if df[col].nunique(dropna=True) < 3:
            continue

        bins = pd.qcut(df[col], q=5, duplicates="drop")
        stats = (
            pd.DataFrame({"bin": bins, "_target": binary_target})
            .groupby("bin", observed=False)["_target"]
            .agg(["mean", "count"])
            .reset_index()
        )
        stats = stats[(stats["count"] >= min_support) & (stats["mean"] >= baseline_rate)]

        for _, row in stats.iterrows():
            patterns.append(
                {
                    "Type": "Numeric bin",
                    "Feature": col,
                    "Pattern": f"{col} in {row['bin']}",
                    "Count": int(row["count"]),
                    "Support (%)": round(row["count"] / len(df) * 100, 2),
                    f"{positive_label} rate (%)": round(row["mean"] * 100, 2),
                    "Lift": round(row["mean"] / baseline_rate, 2),
                }
            )

    if not patterns:
        return pd.DataFrame(), baseline_rate, min_support

    patterns_df = pd.DataFrame(patterns).sort_values(
        [f"{positive_label} rate (%)", "Lift", "Count"],
        ascending=[False, False, False],
    )
    return patterns_df, baseline_rate, min_support


st.title("Heart Disease EDA Analyzer")
st.write("Upload a CSV file to inspect schema, target behavior, and feature signals.")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if not uploaded_file:
    st.info("Upload a CSV file to start the analysis.")
    st.stop()

try:
    df = load_data(uploaded_file.getvalue())
except Exception as exc:
    st.error(f"Error reading CSV: {exc}")
    st.stop()

schema = build_schema(df)
id_cols = schema["id_cols"]
target_col = schema["target_col"]
continuous_cols = schema["continuous_cols"]
categorical_cols = schema["categorical_cols"]
other_cols = schema["other_cols"]

binary_target = None
positive_label = None
class_order = None
if target_col:
    binary_target, positive_label, class_order = encode_binary_target(df[target_col])

st.header("1. Dataset Overview")
overview_cols = st.columns(4)
overview_cols[0].metric("Rows", f"{df.shape[0]:,}")
overview_cols[1].metric("Columns", f"{df.shape[1]:,}")
overview_cols[2].metric("Duplicates", f"{df.duplicated().sum():,}")
if target_col and binary_target is not None:
    overview_cols[3].metric(
        f"{positive_label} rate",
        f"{binary_target.mean() * 100:.2f}%",
    )
else:
    overview_cols[3].metric("Target", "Not binary")

st.write(f"Columns: {', '.join(df.columns)}")

st.header("2. Missing Values")
missing = df.isna().sum()
missing_df = pd.DataFrame(
    {
        "Missing Count": missing,
        "Missing %": (missing / len(df) * 100).round(2),
    }
)
missing_df = missing_df[missing_df["Missing Count"] > 0].sort_values("Missing Count", ascending=False)
if missing_df.empty:
    st.success("No missing values found.")
else:
    st.dataframe(missing_df, use_container_width=True)

st.header("3. Schema Summary")
schema_col1, schema_col2 = st.columns(2)
schema_col1.write(f"ID columns: {format_column_list(id_cols)}")
schema_col1.write(f"Target column: {target_col or 'None'}")
schema_col1.write(f"Numeric columns: {format_column_list(continuous_cols)}")
schema_col2.write(f"Categorical columns: {format_column_list(categorical_cols)}")
schema_col2.write(f"Other columns: {format_column_list(other_cols)}")

if target_col:
    st.subheader("Target Distribution")
    target_counts = df[target_col].value_counts(dropna=False)
    target_dist = (
        target_counts.rename_axis(target_col)
        .reset_index(name="Count")
        .assign(Percent=lambda x: (x["Count"] / len(df) * 100).round(2))
    )
    st.dataframe(target_dist, use_container_width=True)

if target_col and binary_target is not None:
    st.header("4. Feature Signal Ranking")

    signal_rows = []

    for col in continuous_cols:
        corr = df[col].corr(binary_target)
        signal_rows.append(
            {
                "Feature": col,
                "Type": "Numeric",
                "Signal": abs(corr),
                "Detail": f"corr={corr:.3f}",
            }
        )

    for col in categorical_cols:
        rate_by_level = pd.DataFrame({col: df[col], "_target": binary_target}).groupby(col)["_target"].mean()
        spread = rate_by_level.max() - rate_by_level.min()
        signal_rows.append(
            {
                "Feature": col,
                "Type": "Categorical",
                "Signal": float(spread),
                "Detail": f"rate spread={spread:.3f}",
            }
        )

    signal_df = pd.DataFrame(signal_rows).sort_values("Signal", ascending=False)
    st.dataframe(signal_df, use_container_width=True, hide_index=True)

    st.header("5. Top Risk Patterns")
    patterns_df, baseline_rate, min_support = build_top_risk_patterns(
        df,
        binary_target,
        positive_label,
        categorical_cols,
        continuous_cols,
    )
    st.caption(
        f"Ranked groups with at least {min_support:,} rows. "
        f"Baseline {positive_label} rate: {baseline_rate * 100:.2f}%."
    )

    if not patterns_df.empty:
        top_pattern = patterns_df.iloc[0]
        st.write(
            f"Strongest pattern: {top_pattern['Pattern']} "
            f"with {top_pattern[f'{positive_label} rate (%)']:.2f}% {positive_label} rate "
            f"({top_pattern['Lift']:.2f}x baseline)."
        )
        st.dataframe(patterns_df.head(12), use_container_width=True, hide_index=True)
    else:
        st.info("No stable high-risk patterns met the minimum support threshold.")

    st.header("6. Numeric Feature Analysis")
    if continuous_cols:
        selected_num = st.selectbox("Choose a numeric feature", continuous_cols)
        plot_df = sample_for_plot(df[[selected_num, target_col]].dropna())

        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        sns.histplot(
            data=plot_df,
            x=selected_num,
            hue=target_col,
            hue_order=class_order,
            bins=30,
            stat="density",
            common_norm=False,
            element="step",
            ax=axes[0],
        )
        axes[0].set_title(f"Distribution of {selected_num}")

        sns.boxplot(
            data=plot_df,
            x=target_col,
            y=selected_num,
            order=class_order,
            ax=axes[1],
        )
        axes[1].set_title(f"{selected_num} by {target_col}")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        continuous_summary = df.groupby(target_col)[selected_num].agg(["mean", "median", "std", "min", "max"])
        st.dataframe(continuous_summary.round(3), use_container_width=True)
    else:
        st.info("No configured numeric  columns were found in this file.")

    st.header("7. Categorical Feature Analysis")
    if categorical_cols:
        selected_cat = st.selectbox("Choose a categorical feature", categorical_cols)

        count_table = pd.crosstab(
            df[selected_cat].astype(str),
            df[target_col].astype(str),
        ).reindex(columns=class_order, fill_value=0)
        count_table.index.name = selected_cat
        count_table["Total"] = count_table.sum(axis=1)
        count_table[f"{positive_label} rate (%)"] = (
            count_table[positive_label] / count_table["Total"] * 100
        ).round(2)
        count_table = count_table.sort_values("Total", ascending=False)
        plot_order = count_table.index[::-1].tolist()

        plot_counts = (
            count_table[class_order]
            .reset_index()
            .melt(id_vars=selected_cat, var_name=target_col, value_name="Count")
        )
        plot_counts[selected_cat] = pd.Categorical(plot_counts[selected_cat], categories=plot_order, ordered=True)

        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        sns.barplot(
            data=plot_counts,
            y=selected_cat,
            x="Count",
            hue=target_col,
            hue_order=class_order,
            order=plot_order,
            palette="Set2",
            ax=axes[0],
        )
        axes[0].set_title(f"Count of {selected_cat} by {target_col}")
        axes[0].xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        axes[0].grid(axis="x", alpha=0.25)

        rate_frame = count_table.reset_index()
        rate_frame[selected_cat] = pd.Categorical(rate_frame[selected_cat], categories=plot_order, ordered=True)
        sns.barplot(
            data=rate_frame,
            y=selected_cat,
            x=f"{positive_label} rate (%)",
            order=plot_order,
            color="#d95f02",
            ax=axes[1],
        )
        axes[1].set_title(f"{positive_label} rate by {selected_cat}")
        axes[1].xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}%"))
        axes[1].grid(axis="x", alpha=0.25)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.dataframe(count_table, use_container_width=True)
    else:
        st.info("No configured categorical columns were found in this file.")

    st.header("8. Correlation Heatmap")
    cols_to_drop = [c for c in (id_cols + [target_col]) if c in df.columns]
    corr_df = df.drop(columns=cols_to_drop)
    corr_matrix = corr_df.corr()

    fig, ax = plt.subplots(figsize=(12, 9))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # chỉ hiện nửa dưới
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
        ax=ax,
        annot_kws={"size": 8},
    )

st.header("9. Dataset Preview")
st.dataframe(df.head(), use_container_width=True)
