import pandas as pd
from matplotlib import pyplot as plt


def plot_keypoints(keypoint_df, ax=None, x_col="x", y_col="y", **plot_kwargs):
    """
    A function to plot keypoints from a dataframe. The dataframe is expected to have a simple index containing the
    keypoint names. The default assumption is that the dataframe has columns "x" and "y" for the x and y coordinates of
    the keypoints. As the plotting is done through the pandas plot method, additional plot_kwargs can be passed to
    customize the plot, potentially based on the columns of the dataframe.

    :param keypoint_df: A dataframe containing the keypoints to plot.
    :type keypoint_df: pd.DataFrame
    :param ax: The axis to plot on. If None, a new figure is created.
    :type ax: matplotlib.axes.Axes, optional
    :param plot_kwargs: Additional keyword arguments to pass to the plot method.
    :type plot_kwargs: dict
    :return: The axis the keypoints were plotted on.
    :rtype: matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()
    default_plot_kwargs = dict(x=x_col, y=y_col, kind="scatter", legend=False)
    extracted_plot_kwargs = {k: keypoint_df[v] for k, v in plot_kwargs.items() if
                             isinstance(v, str) and v in keypoint_df.columns}
    plot_kwargs = {**default_plot_kwargs, **plot_kwargs, **extracted_plot_kwargs}
    keypoint_df.plot(ax=ax, **plot_kwargs)
    return ax


def plot_keypoint_labels(keypoint_df, ax=None, **annotation_kwargs):
    """
    Plot labels for keypoints on a plot. The default assumption is that the dataframe has columns "x" and "y" for the x
    and y coordinates of the keypoints. Plotting is done through the matplotlib annotate method, so additional
    annotation_kwargs can be passed to customize the annotations.

    :param keypoint_df: A dataframe containing the keypoints to plot.
    :type keypoint_df: pd.DataFrame
    :param ax: The axis to plot on. If None, a new figure is created.
    :type ax: matplotlib.axes.Axes, optional
    :param annotation_kwargs: Additional keyword arguments to pass to the annotate method.
    :type annotation_kwargs: dict
    :return: The axis the keypoints were plotted on.
    :rtype: matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()
    default_annotation_kwargs = dict(x="x", y="y", xytext=(5, 5), textcoords="offset points")
    annotation_kwargs = {**default_annotation_kwargs, **annotation_kwargs}
    x_col, y_col = annotation_kwargs.pop("x"), annotation_kwargs.pop("y")
    for kp, row in keypoint_df.iterrows():
        ax.annotate(kp, (row[x_col], row[y_col]), **annotation_kwargs)


def plot_keypoint_skeleton(keypoint_df, skeleton_df, ax=None, x_col="x", y_col="y", **plot_kwargs):
    """
    Plots a skeleton based on the keypoints in the keypoint dataframe. The skeleton dataframe is expected to have a
    simple index containing the edge names or indices, and two columns indicating the start and end keypoint of the edge.
    Plotting happens through the pandas plot method, so additional plot_kwargs can be passed to customize the plot.
    The plot_kwargs can also be based on the columns of the skeleton dataframe.

    :param keypoint_df: A dataframe containing the keypoints to plot.
    :type keypoint_df: pd.DataFrame
    :param skeleton_df: A dataframe containing the skeleton edges to plot.
    :type skeleton_df: pd.DataFrame
    :param ax: The axis to plot on. If None, a new figure is created.
    :type ax: matplotlib.axes.Axes, optional
    :param plot_kwargs: Additional keyword arguments to pass to the plot method.
    :type plot_kwargs: dict
    """
    if ax is None:
        _, ax = plt.subplots()

    default_plot_kwargs = dict(x=x_col, y=y_col, kind="line", legend=False, color="black")

    edge_feature_cols = [col for col in skeleton_df.columns if not col.startswith("node_")]
    edge_feature_df = skeleton_df[edge_feature_cols]

    skeleton_df = skeleton_df[[col for col in skeleton_df.columns if col.startswith("node_")]]
    skeleton_keypoint_df = skeleton_df.stack("edge_feature").rename("keypoint_name")
    skeleton_keypoint_df = pd.merge(skeleton_keypoint_df, keypoint_df, left_on="keypoint_name", how="left",
                                    right_index=True)
    skeleton_keypoint_df = skeleton_keypoint_df.join(edge_feature_df, on="edge_index", lsuffix="_keypoint")

    for edge_index, edge_data in skeleton_keypoint_df.groupby(level="edge_index"):
        edge_features = edge_feature_df.loc[edge_index].to_dict()
        edge_plot_kwargs = {k: edge_features[v] for k, v in plot_kwargs.items() if
                            isinstance(v, str) and v in edge_features}
        edge_plot_kwargs = {**default_plot_kwargs, **plot_kwargs, **edge_plot_kwargs}
        edge_data.plot(ax=ax, **edge_plot_kwargs)
    return ax


def plot_keypoint_instance(keypoint_df, skeleton_df=None, plot_labels=False, ax=None, keypoint_kwargs=None,
                           skeleton_kwargs=None, label_kwargs=None, **shared_kwargs):
    """
    A function to plot all details of a single instance of keypoints, potentially with a skeleton overlay, and
    optionally with labels. See the documentation of plot_keypoints, plot_keypoint_skeleton, and plot_keypoint_labels
    for more details on the individual plotting functions. The skeleton will be plotted first, followed by the keypoints,
    and then the labels if requested. This order can be manipulated by passing the zorder parameter in the
    keypoint_kwargs, skeleton_kwargs, and label_kwargs dictionaries.

    :param keypoint_df: A dataframe containing the keypoints to plot.
    :type keypoint_df: pd.DataFrame
    :param skeleton_df: A dataframe containing the skeleton edges to plot. If None, no skeleton is plotted.
    :type skeleton_df: pd.DataFrame, optional
    :param plot_labels: Whether to plot labels for the keypoints. Default is False.
    :type plot_labels: bool
    :param ax: The axis to plot on. If None, a new figure is created.
    :type ax: matplotlib.axes.Axes, optional
    :param keypoint_kwargs: Additional keyword arguments to pass to the plot_keypoints function.
    :type keypoint_kwargs: dict, optional
    :param skeleton_kwargs: Additional keyword arguments to pass to the plot_keypoint_skeleton function.
    :type skeleton_kwargs: dict, optional
    :param label_kwargs: Additional keyword arguments to pass to the plot_keypoint_labels function.
    :type label_kwargs: dict, optional
    :return: The axis the keypoints were plotted on.
    :rtype: matplotlib.axes.Axes
    """

    if ax is None:
        _, ax = plt.subplots()

    if keypoint_df.empty:
        return ax

    if keypoint_kwargs is None:
        keypoint_kwargs = {}
    if skeleton_kwargs is None:
        skeleton_kwargs = {}
    if label_kwargs is None:
        label_kwargs = {}

    if skeleton_df is not None:
        plot_keypoint_skeleton(keypoint_df, skeleton_df, ax=ax, **skeleton_kwargs, **shared_kwargs)
    plot_keypoints(keypoint_df, ax=ax, **keypoint_kwargs, **shared_kwargs)
    if plot_labels:
        plot_keypoint_labels(keypoint_df, ax=ax, **label_kwargs, **shared_kwargs)
    return ax


def plot_keypoint_instances(multi_instance_keypoint_df, ax=None, *args, **kwargs):
    if ax is None:
        _, ax = plt.subplots()

    instance_identifier_levels = multi_instance_keypoint_df.index.names[:-1]
    for identifier, single_keypoint_df in multi_instance_keypoint_df.groupby(instance_identifier_levels):
        single_keypoint_df = single_keypoint_df.droplevel(instance_identifier_levels, axis=0)

        plot_keypoint_instance(single_keypoint_df, ax=ax, *args, **kwargs)
    return ax
