import matplotlib.pyplot as plt
import seaborn as sns

def plot_transactions_by_cash_type(df, cash_type_col="cash_type"):
    plt.figure(figsize=(7, 4))

    ax = sns.countplot(
        data=df,
        x=cash_type_col,
        order=df[cash_type_col].value_counts().index,
        palette="Set2"
    )

    ax.set_title("Number of transactions by payment type")
    ax.set_xlabel("Payment type")
    ax.set_ylabel("Number of transactions")

    for container in ax.containers:
        ax.bar_label(container)

    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt
import seaborn as sns


def plot_payment_value_by_cash_type(
    df,
    cash_type_col="cash_type",
    value_col="money",
    palette="Set2",
    point_color="black",
    point_alpha=0.45,
    point_size=4,
):
    """
    Plot payment value distribution by payment type using a boxplot
    with individual transaction points overlaid.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe containing payment type and payment value columns.
    cash_type_col : str, default="cash_type"
        Column name with payment type categories, such as card and cash.
    value_col : str, default="money"
        Column name with payment values.
    palette : str or list, default="Set2"
        Color palette used for the boxplot.
    point_color : str, default="black"
        Color used for individual transaction points.
    point_alpha : float, default=0.45
        Transparency level for individual points.
    point_size : int or float, default=4
        Size of individual points.
    """
    plt.figure(figsize=(7, 4))

    ax = sns.boxplot(
        data=df,
        x=cash_type_col,
        y=value_col,
        palette=palette,
        showfliers=False,
    )

    sns.stripplot(
        data=df,
        x=cash_type_col,
        y=value_col,
        color=point_color,
        alpha=point_alpha,
        size=point_size,
        jitter=True,
        ax=ax,
    )

    ax.set_title("Payment value distribution by payment type")
    ax.set_xlabel("Payment type")
    ax.set_ylabel("Payment value")

    plt.tight_layout()
    plt.show()