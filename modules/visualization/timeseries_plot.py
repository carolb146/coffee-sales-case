import matplotlib.pyplot as plt
import seaborn as sns

def plot_coffee_trend_with_seasons(
    sales_df,
    season_df,
    date_col="date",
    value_col="transactions_7d_avg",
    coffee_col="coffee_name",
    season_col="season",
    title="7-Day Moving Average of Transactions by Top Coffee Types",
    ylabel="Transactions - 7-day moving average",
    figsize=(12, 6),
):
    fig, ax = plt.subplots(figsize=figsize)

    sns.lineplot(
        data=sales_df,
        x=date_col,
        y=value_col,
        hue=coffee_col,
        ax=ax
    )

    season_colors = {
        "winter": "#9EC5FE",
        "spring": "#A8E6A3",
        "summer": "#FFF3A3",
        "autumn": "#F5B7B1",
    }

    season_periods = (
        season_df[[date_col, season_col]]
        .drop_duplicates()
        .sort_values(date_col)
    )

    season_blocks = (
        season_periods
        .assign(
            block=(
                season_periods[season_col]
                != season_periods[season_col].shift()
            ).cumsum()
        )
        .groupby(["block", season_col])
        .agg(
            start_date=(date_col, "min"),
            end_date=(date_col, "max")
        )
        .reset_index()
    )

    for _, row in season_blocks.iterrows():
        ax.axvspan(
            row["start_date"],
            row["end_date"],
            color=season_colors.get(row[season_col], "#E5E7E9"),
            alpha=0.35,
            label=row[season_col]
        )

    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))

    ax.legend(
        unique.values(),
        unique.keys(),
        title="Coffee type / Season",
        bbox_to_anchor=(1.05, 1),
        loc="upper left"
    )

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel(ylabel)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()