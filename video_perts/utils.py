import os
import numpy as np
from IPython.display import HTML


def create_directory_if_needed(file_path=""):
    dir_name, file_name = os.path.split(file_path)
    if not os.path.isdir(dir_name): os.makedirs(dir_name)


def get_subarray_from_last_ndims(arr, ndims=4):
    np_arr = np.array(arr)
    current_ndims = len(np_arr.shape)

    if current_ndims > ndims:
        np_arr = get_subarray_from_last_ndims(np_arr[0], ndims)

    return np_arr


def embed_html_in_notebook(html=""):
    return HTML(data=html)