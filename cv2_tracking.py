import cv2
import numpy as np
import pandas as pd

from .cv2_funcs import add_rgba_overlay_to_frame
from .processor_funcs import ensure_frame_index_func, func_modified_frames_generator


def cv2_keypoint_df_iterator(keypoint_df, x_col="x", y_col="y", default_kwargs=None, **kwargs):
    default_kwargs = {} if default_kwargs is None else default_kwargs
    kwargs = {**default_kwargs, **kwargs}

    for row_tuple in keypoint_df.itertuples():
        row_dict = row_tuple._asdict()
        row = pd.Series(row_dict, name=row_dict.pop("Index"))

        point = row[[x_col, y_col]]
        if point.isna().any():
            continue
        clean_row = row.drop([x_col, y_col]).dropna()

        extracted_kwargs = {k: clean_row[v] for k, v in kwargs.items() if
                            isinstance(v, str) and v in clean_row.index}
        complete_kwargs = {**kwargs, **extracted_kwargs}

        yield (point, complete_kwargs)


def plot_keypoints(base_frame, keypoint_df, x_col="x", y_col="y", **plot_kwargs):
    default_plot_kwargs = dict(radius=2, thickness=-1, color=(255, 255, 255))
    for point_series, complete_kwargs in cv2_keypoint_df_iterator(keypoint_df, x_col=x_col, y_col=y_col,
                                                           default_kwargs=default_plot_kwargs, **plot_kwargs):
        cv2.circle(base_frame, point_series.astype(int).values, **complete_kwargs)
    return base_frame


def plot_keypoint_labels(base_frame, keypoint_df, x_col="x", y_col="y", **annotation_kwargs):
    default_annotation_kwargs = dict(fontScale=1.0, fontFace=cv2.FONT_HERSHEY_PLAIN, color=(255, 255, 255))
    for point_series, complete_kwargs in cv2_keypoint_df_iterator(keypoint_df, x_col=x_col, y_col=y_col,
                                                                  default_kwargs=default_annotation_kwargs,
                                                                  **annotation_kwargs):
        kp_name = point_series.name
        cv2.putText(base_frame, text=kp_name, org=point_series.astype(int).values, **complete_kwargs)
    return base_frame


def plot_keypoint_skeleton(base_frame, keypoint_df, skeleton_df, x_col="x", y_col="y",
                           node0_col="node_0", node1_col="node_1", **plot_kwargs):
    default_plot_kwargs = dict(thickness=1, lineType=8, color=(255, 255, 255))

    for point_series, complete_kwargs in cv2_keypoint_df_iterator(skeleton_df, x_col=node0_col, y_col=node1_col,
                                                                  default_kwargs=default_plot_kwargs, **plot_kwargs):
        pt1 = keypoint_df.loc[point_series[node0_col], [x_col, y_col]]
        pt2 = keypoint_df.loc[point_series[node1_col], [x_col, y_col]]

        if pt1.isna().any() or pt2.isna().any():
            continue

        cv2.line(base_frame, pt1.astype(int).values, pt2.astype(int).values, **complete_kwargs)
    return base_frame


def plot_keypoint_instance(base_frame, keypoint_df, skeleton_df=None, plot_labels=False, ax=None, keypoint_kwargs=None,
                           skeleton_kwargs=None, label_kwargs=None, **shared_kwargs):
    if keypoint_df.empty:
        return base_frame

    keypoint_kwargs = {} if keypoint_kwargs is None else keypoint_kwargs
    skeleton_kwargs = {} if skeleton_kwargs is None else skeleton_kwargs
    label_kwargs = {} if label_kwargs is None else label_kwargs

    if skeleton_df is not None:
        plot_keypoint_skeleton(base_frame, keypoint_df, skeleton_df, **skeleton_kwargs, **shared_kwargs)
    plot_keypoints(base_frame, keypoint_df, **keypoint_kwargs, **shared_kwargs)
    if plot_labels:
        plot_keypoint_labels(base_frame, keypoint_df, **label_kwargs, **shared_kwargs)
    return base_frame


def plot_keypoint_instances(base_frame, multi_instance_keypoint_df, *args, **kwargs):
    instance_identifier_levels = multi_instance_keypoint_df.index.names[:-1]
    for identifier, single_keypoint_df in multi_instance_keypoint_df.groupby(instance_identifier_levels):
        single_keypoint_df = single_keypoint_df.droplevel(instance_identifier_levels, axis=0)
        plot_keypoint_instance(base_frame, single_keypoint_df, *args, **kwargs)
    return base_frame


def cv2_annotated_frames_generator(frame_source, cv2_func, base_frame, cv2_input=None, alpha=1.0, *args, **kwargs):
    cv2_input_func = ensure_frame_index_func(cv2_input)

    def generate_cv2_rgba_overlay(frame_index):
        local_args = args
        local_kwargs = kwargs

        input_dict_or_object = cv2_input_func(frame_index)

        if isinstance(input_dict_or_object, dict):
            local_kwargs = {**local_kwargs, **input_dict_or_object}
        else:
            local_args = [*local_args, input_dict_or_object]
        cv2_rgb_array = cv2_func(base_frame.copy(), *local_args, **local_kwargs)
        base_modified_mask = (base_frame[:, :, :3] != cv2_rgb_array[:, :, :3]).any(axis=2).astype("uint8")
        cv2_rgba_array = np.dstack([cv2_rgb_array, base_modified_mask * int(alpha*255)])
        return cv2_rgba_array

    yield from func_modified_frames_generator(frame_source, modification_func=add_rgba_overlay_to_frame,
                                              modification_input=generate_cv2_rgba_overlay,
                                              use_alpha_composite=True if alpha!=1 else False)
