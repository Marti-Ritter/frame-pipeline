import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .base_classes import FrameProcessor
from .cv2_tracking import cv2_annotated_frames_generator, plot_keypoint_instance, plot_keypoint_instances


class Cv2AnnotatedFramesGenerator(FrameProcessor):
    source_canvas_was_created = False
    def __init__(self, frame_source, cv2_func, base_frame=None, cv2_input=None, *args, **kwargs):
        if base_frame is None:
            example_index, example_frame = frame_source.example_index, frame_source.example_frame

            if example_frame is None:
                raise ValueError("No base_frame given and frame_source is empty")

            self.base_frame = np.zeros_like(example_frame, dtype="uint8")
        else:
            self.base_frame = base_frame

        super().__init__(frame_source, cv2_annotated_frames_generator, cv2_func=cv2_func,
                         base_frame=self.base_frame, cv2_input=cv2_input, *args, **kwargs)


class KeypointInstanceAnnotatedFramesGenerator(Cv2AnnotatedFramesGenerator):
    def __init__(self, frame_source, track_df, base_frame=None, *args, **kwargs):
        if isinstance(track_df.index, pd.MultiIndex):
            plot_func = plot_keypoint_instances
            index_length = len(track_df.index.names)
            reformed_track_df = track_df.reorder_levels([index_length-1] + list(range(index_length-1)))
        else:
            plot_func = plot_keypoint_instance
            reformed_track_df = track_df

        reformed_keypoint_df = reformed_track_df.stack("keypoint_name", future_stack=True)

        def keypoint_access_func(frame_index):
            if frame_index in reformed_keypoint_df.index.get_level_values(level=0):
                return reformed_keypoint_df.loc[frame_index]
            return pd.DataFrame()

        super().__init__(frame_source, cv2_func=plot_func, base_frame=base_frame,
                         cv2_input=keypoint_access_func, *args, **kwargs)
