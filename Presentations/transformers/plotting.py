import pandas as pd
from typing import Optional
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def better_scatter_plot(
    df: pd.DataFrame,
    label_col: str,
    x_col: str,
    y_col: str,
    hue: Optional[str],
    plot_title: str,
    ax: object = None,
    n_quantiles: int = 50,
    n_max_ticks: int = 30,
    apply_pca: bool = False,
    ticks_as_percentiles: bool = False,
    rescale_axis: bool = False,
) -> None:
    """Plot a scatter plot with labels.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to plot.
    label_col : str
        The name of the column to use for the labels.
    x_col : str
        The name of the column to use for the x-axis.
    y_col : str
        The name of the column to use for the y-axis.
    plot_title : str
        The title of the plot.
    ax : Optional[plt.Axes]
        The axes to plot on.
    n_quantiles : int, optional
        The number of quantiles to use for the rescaling, by default 50
        Has no effect if rescale_axis is False
    n_max_ticks : int, optional
        The maximum number of ticks to use for the rescaling, by default 30
        Has no effect if rescale_axis is False
    apply_pca : bool, optional
        Whether to apply PCA to the x and y columns, by default False
    ticks_as_percentiles : bool, optional
        Whether to use percentiles for the ticks, by default False
        Has no effect if rescale_axis is False
    rescale_axis : bool, optional
        Whether to rescale the axis, by default True

    """

    # assert n_quantiles % n_max_ticks == 0
    # And rescale the x and y
    # First, create a monotonically increasing function
    # that interpolates the quantiles to the range [0, 1]

    if apply_pca:
        try:
            from sklearn.decomposition import PCA

        except ImportError:
            raise ImportError(
                "You need to install sklearn to use the PCA functionality, "
                "try running `pip install scikit-learn`"
            )

        transformed_xy = PCA(n_components=2).fit_transform(
            df[[x_col, y_col]].values
        )
        df[x_col] = transformed_xy[:, 0]
        df[y_col] = transformed_xy[:, 1]

    # The new x and y should range from 0 to 1 showing in which quantile each
    # point is located within its axis' distribution
    def quantile_rescaler(axis: np.ndarray) -> (np.ndarray, np.ndarray):
        """Returns the rescaled axis as well as the new ticks for the plot"""
        rescaled = np.interp(
            axis,
            np.quantile(axis, np.linspace(0, 1, n_quantiles)),
            np.linspace(0, 1, n_quantiles),
        )

        if ticks_as_percentiles:
            # And get the new ticks
            tick_labels = np.linspace(
                0, 100, n_max_ticks, endpoint=True, dtype=int
            )

            ticks = np.interp(
                np.linspace(
                    np.min(axis), np.max(axis), n_max_ticks, endpoint=True
                ),
                np.quantile(
                    axis, np.linspace(0, 1, axis.shape[0], endpoint=True)
                ),
                np.linspace(0, 1, axis.shape[0], endpoint=True),
            )

        else:
            tick_labels = np.round(
                np.linspace(
                    np.min(axis), np.max(axis), n_max_ticks, endpoint=True
                ),
                1,
            )

            # And the equidistant version
            ticks = np.linspace(0, 1, n_max_ticks, endpoint=True)

        return rescaled, ticks, tick_labels

    if rescale_axis:
        df[x_col], x_ticks, x_tick_labels = quantile_rescaler(df[x_col])
        df[y_col], y_ticks, y_tick_labels = quantile_rescaler(df[y_col])

    plt.figure()
    p1 = sns.scatterplot(
        x=x_col,  # Horizontal axis
        y=y_col,  # Vertical axis
        data=df,  # Data source
        hue=df[hue] if hue else None,
        # size=8,
        # legend=False,
        ax=ax,
    )

    if rescale_axis:
        # And add the ticks
        plt.xticks(x_ticks, x_tick_labels)
        plt.tick_params(axis="x", rotation=45)
        plt.yticks(y_ticks, y_tick_labels)

    scale_x_axis: float = df[x_col].max() - df[x_col].min()

    for line in df.index:
        p1.text(
            df[x_col][line] + 0.01 * scale_x_axis,
            df[y_col][line],
            df[label_col][line],
            horizontalalignment="left",
            size="medium",
            color="black",
            weight="semibold",
        )

    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(
        plot_title
        if not rescale_axis
        else plot_title + " (rescaled to make quantiles equidistant)"
    )
