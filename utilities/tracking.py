import numpy as np
import pandas as pd


def track_to_centroid_df(track_df, aggregate_func="median"):
    track_x_df, track_y_df = (track_df.xs("x", level="keypoint_feature", axis=1),
                              track_df.xs("y", level="keypoint_feature", axis=1))
    agg_x, agg_y = track_x_df.agg(aggregate_func, axis=1), track_y_df.agg(aggregate_func, axis=1)
    return pd.DataFrame(dict(x=agg_x, y=agg_y))


def track_to_roi_df(track_df, size_override=None, aggregate_func="median", size_quantile=0.99, size_multiplier=1.05):
    track_x_df, track_y_df = (track_df.xs("x", level="keypoint_feature", axis=1),
                              track_df.xs("y", level="keypoint_feature", axis=1))
    centroid_df = track_to_centroid_df(track_df, aggregate_func=aggregate_func)
    centroid_x, centroid_y = centroid_df["x"], centroid_df["y"]

    if size_override is None:
        width = (track_x_df.T - centroid_x).T.abs().quantile(size_quantile).max()
        height = (track_y_df.T - centroid_y).T.abs().quantile(size_quantile).max()
        square_size = max(width, height) * size_multiplier
        width, height = square_size, square_size
    else:
        width, height = size_override

    return pd.DataFrame(dict(x=centroid_x - width//2, y=centroid_y - height//2, w=width, h=height))


def cart2pol(x, y):
    """Transform a cartesian coordinate pair to polar coordinates

    :param x: x-coordinate
    :type x: float
    :param y: y-coordinate
    :type y: float
    :return: tuple of rho and phi, describing length and angle of the polar vector
    :rtype: tuple
    """
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi


def track_to_distance_orientation_df(track_df, orientation_reference_point, position_reference_point,
                                     orientation_in_degrees=True):
    sliced_track_df = track_df.loc[:, pd.IndexSlice[:, ["x", "y"]]]

    if isinstance(orientation_reference_point, pd.DataFrame):
        orientation_reference_df = orientation_reference_point
    else:
        orientation_reference_df = sliced_track_df.loc[:, orientation_reference_point]

    if isinstance(position_reference_point, pd.DataFrame):
        position_reference_df = position_reference_point
    else:
        position_reference_df = sliced_track_df.loc[:, position_reference_point]

    normalized_orientation_df = orientation_reference_df - position_reference_df
    length_orientation_df = normalized_orientation_df.apply(lambda r: cart2pol(r["x"], r["y"]), axis=1).apply(pd.Series)
    length_orientation_df.columns = ["rho", "phi"]

    if orientation_in_degrees:
        length_orientation_df["phi"] = length_orientation_df["phi"].apply(np.rad2deg)

    return length_orientation_df
