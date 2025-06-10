import inspect
from functools import wraps

import pickle
from io import BytesIO


def split_iterable(to_slice, slices):
    slices = [0, *slices, len(to_slice)]
    for i in range(1, len(slices)):
        yield to_slice[slices[i - 1]:slices[i]]


def iter_cumsum(input_iter):
    """
    Computes the cumulative sum of an iterable.

    :param input_iter: The iterable to generate the cumulative sum from.
    :type input_iter: collections.Iterable
    :return: A list containing the cumulative sum at each position in the iterable.
    :rtype: list of int or list of float
    """
    cum_sum = []
    total = 0
    for item in input_iter:
        total += item
        cum_sum.append(total)
    return cum_sum


def split_chunks(to_slice, chunk_size):
    slices = range(chunk_size, len(to_slice), chunk_size) if isinstance(chunk_size, int) else iter_cumsum(chunk_size)
    for chunk in split_iterable(to_slice, slices):
        yield chunk


def ensure_list(input_object):
    """

    :param input_object:
    :return:
    :rtype: list
    """
    return (input_object if isinstance(input_object, list) else [input_object])


def standardize_padding(padding):
    """
    Standardizes a given tuple of pad widths to adhere to the standard (top, bottom, left, right) format.
    Valid input formats are:
        (top, bottom, left, right)  -->  standard format
        (top & bottom, left & right)  -->  top = bottom, left = right
        (top & bottom & left & right)  -->  top = bottom = left = right
        top & bottom & left & right  -->  top = bottom = left = right (single value)
    :param padding: A tuple of pad widths or a single pad width.
    :type padding: tuple of (int or float) or int or float
    :return: A tuple of pad widths in the standard format.
    :rtype: tuple of (int or float)
    """

    assert isinstance(padding, (list, tuple, int, float)), "Padding must be a list, tuple, int or float"
    if isinstance(padding, (int, float)):
        padding = (padding, padding, padding, padding)
    if len(padding) == 4:
        pad_top, pad_bottom, pad_left, pad_right = padding
    elif len(padding) == 2:
        pad_top = pad_bottom = padding[0]
        pad_left = pad_right = padding[1]
    elif len(padding) == 1:
        pad_top = pad_bottom = pad_left = pad_right = padding[0]
    else:
        raise ValueError("Padding must be a list of length 1, 2, or 4.")
    return pad_top, pad_bottom, pad_left, pad_right


def copy_object(input_object):
    """
    An attempt at an universal copy function. The object is pickled and then unpickled. This should work for most
    objects, but may fail for some. An example of an object that this fails for is a matplotlib figure, as those are all
    registered with the backend and figure manager.
    From https://stackoverflow.com/a/45812071.

    :param input_object: An object
    :type input_object: Any
    :return: A copy of the object
    :rtype: Any
    """
    buffer = BytesIO()
    pickle.dump(input_object, buffer)
    buffer.seek(0)
    return pickle.load(buffer)


def copy_signature_from(signature_source_func, ensure_compatible_args=False, *wrap_args, **wrap_kwargs):
    """
    Based on https://stackoverflow.com/a/64605018.
    Replaces all signature features in a target_func with ones given by a signature_source_func.
    Any other arguments are passed to functools.wraps.

    It appears to be the only valid way to carry over information from an "inner" func, since merging seems to be rather
    complicated: see https://github.com/Kwpolska/merge_args/blob/master/merge_args.py or
    https://chriswarrick.com/blog/2018/09/20/python-hackery-merging-signatures-of-two-python-functions/ for an example.

    :param signature_source_func: A function that will be used to copy its signatures to a decorated func.
    :type signature_source_func: function
    :param ensure_compatible_args: If True, the decorated function's args will be checked against those of the source
    func.
    :type ensure_compatible_args: bool
    :return: A decorator that will replace target_func's signature with one from signature_source_func.
    :rtype: function
    """

    def signature_copy_decorator(target_func):
        @wraps(signature_source_func, *wrap_args, **wrap_kwargs)  # copy source function's docstring and annotations
        def wrapper(*args, **kwargs):
            if ensure_compatible_args:
                inspect.signature(signature_source_func).bind(*args, **kwargs)
            return target_func(*args, **kwargs)

        wrapper.__signature__ = inspect.signature(signature_source_func)  # copy signature from source func
        return wrapper

    return signature_copy_decorator


