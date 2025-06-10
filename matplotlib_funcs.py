import matplotlib as mpl
import matplotlib.axis
import matplotlib.pyplot as plt
import matplotlib.spines
import numpy as np
from PIL import Image

from .utilities.general import ensure_list


def get_all_matplotlib_children(matplotlib_object, excluded_types=None):
    excluded_types = [] if excluded_types is None else excluded_types
    children = [child for child in matplotlib_object.get_children() if
                not any(isinstance(child, excluded_type) for excluded_type in excluded_types)]
    if children:
        return [grandchild for child in children for grandchild in ensure_list(get_all_matplotlib_children(child))]
    else:
        return matplotlib_object


def get_plot_elements(matplotlib_object):
    return get_all_matplotlib_children(matplotlib_object,
                                       excluded_types=(matplotlib.axis.Axis, matplotlib.spines.Spine,))


def create_canvas(shape_x, shape_y, dpi=100, alpha=0., show_after_creation=False):
    """Generates an empty figure with x by y pixels

    The generated canvas can be used to assign markers and lines accurately to specific pixel locations.

    :param shape_x: Width of the canvas in pixels
    :type shape_x: int
    :param shape_y: Height of the canvas in pixels
    :type shape_y: int
    :param dpi: Resolution of the canvas. Determines the final width and height in inches.
    :type dpi: int
    :param alpha: How opaque the canvas will be. Default is transparent. Value between 0 and 1.
    :type alpha: float
    :param show_after_creation: Whether to show the canvas after creation. Default is False. This resets the matplotlib
    backend, so it might be necessary to restore the backend manually, even if it is reset here, by using %matplotlib
    inline in jupyter notebooks.
    :type show_after_creation: bool
    :return: plt.Figure of width x and height y (in pixels)
    :rtype: plt.Figure
    """
    if not show_after_creation:
        backend_ = mpl.get_backend()
        mpl.use("Agg")  # Prevent showing stuff

    fig = plt.figure(figsize=(shape_x / dpi, shape_y / dpi), dpi=dpi)
    fig.patch.set_alpha(alpha)

    ax = plt.Axes(fig, (0., 0., 1., 1.), xlim=(0, shape_x), ylim=(0, shape_y))
    ax.set_axis_off()
    ax.autoscale(False)
    ax.invert_yaxis()
    fig.add_axes(ax)

    if not show_after_creation:
        mpl.use(backend_)  # Reset backend

    return fig


# from https://stackoverflow.com/q/55703105
def fig2data(fig):
    """
    Convert a Matplotlib figure to a numpy array and return it.

    :param fig: a matplotlib figure
    :type fig: plt.Figure
    :return: a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)

    return buf


def fig2img(fig):
    """
    Convert a Matplotlib figure to a PIL Image in RGBA format and return it.

    :param fig: a matplotlib figure
    :type fig: plt.Figure
    :return: a Python Imaging Library (PIL) image
    """
    # put the figure pixmap into a numpy array
    buf = fig2data(fig)
    w, h, d = buf.shape
    return Image.frombuffer("RGBA", (w, h), buf)


def fig2cv2(fig):
    """Convert a Matplotlib figure to a RGBA numpy array that can be used as an image in OpenCV2

    :param fig: a matplotlib figure
    :type fig: plt.Figure
    :return: a NumPy array
    :rtype: np.ndarray
    """
    return np.array(fig2img(fig))


def get_function_added_artists(func, *args, reference_figure=None, return_func_return=False, **kwargs):
    """
    Calls a function with the given arguments and keyword arguments and returns the artists that were added to the
    reference_figure. Will only work if the function only adds artists to the reference_figure.
    To avoid unexpected output, the axis being plotted to should be already initialized before calling this function, 
    as otherwise there will also be all artists included that are generated on the first initialization of an axis.

    :param func: The function to call
    :type func: function
    :param args: The arguments to pass to the function
    :type args: tuple
    :param return_func_return: Whether to return the return value of the function. If True, the return value will be
    a tuple of the return value of the function and then the artists added to the figure. Default is False.
    :type return_func_return: bool
    :param kwargs: The keyword arguments to pass to the function
    :type kwargs: dict
    :return: A list of artists that were added to the current figure or a tuple of the return value of the function and
    the artists added to the figure
    :rtype: tuple or list
    """
    if reference_figure is None:
        reference_figure = plt.gcf()

    previous_elements = get_plot_elements(reference_figure)
    func_return = func(*args, **kwargs)
    added_elements = [element for element in get_plot_elements(reference_figure) if element not in previous_elements]

    if return_func_return:
        return func_return, added_elements
    else:
        return added_elements


