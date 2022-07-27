# -*- coding: utf-8 -*-
"""
| Utils function
| Author: Alexandre CARRE
| Created on: Jan 14, 2021
"""
import fnmatch
import logging
import os
from typing import List, Sequence, Union, Generator, Set, Optional
from typing import Tuple

import numpy as np
from monai.transforms import RandWeightedCrop
from numba import njit, prange
from skimage.filters import threshold_otsu
from skimage import exposure
from skimage.morphology import remove_small_objects

logger = logging.getLogger(__name__)
logging.getLogger('numba').setLevel(logging.WARNING)


def check_isdir(input_dir: str) -> str:
    """
    Check if a directory exist.

    :param input_dir: string of the path of the input directory.
    :return: string if exist, else raise NotADirectoryError.
    """
    if os.path.isdir(input_dir):
        return input_dir
    else:
        raise NotADirectoryError(input_dir)


def check_file_exist(input_file_path):
    """
    Check if the input_file_path exist.

    :param input_file_path: string of the input file path
    :raise: FileNotFoundError
    """
    if not os.path.exists(input_file_path):
        raise FileNotFoundError(f"{input_file_path} was not found, check if it's a valid file path")


def check_ext(input_file_path: str, extension: Union[Sequence[str], str]) -> str:
    """
    Check if a directory exist.

    :param input_file_path: string of the path of the nii or nii.gz.
    :param extension: List of str or str of extension
    :return: string if exist, else raise Error.
    :raise: FileExistsError
    """

    if not isinstance(extension, list):
        extension = [extension]

    pth, fnm, ext = split_filename(input_file_path)
    if ext not in extension:
        raise FileExistsError(f"extension of {input_file_path} need to be {extension}")
    return input_file_path


def safe_file_name(file_name: str) -> str:
    """
    Remove any potentially dangerous or confusing characters from
    the file name by mapping them to reasonable substitutes.

    :param file_name: name of the file.
    :return: name of the file corrected.
    """
    underscores = r"""+`~!?@#$%^&*(){}[]/=\|<>,.":' """
    safe_name = ""
    for c in file_name:
        if c in underscores:
            c = "_"
        safe_name += c
    return safe_name


def split_filename(file_name: str) -> Tuple[str, str, str]:
    """
    Split file_name into folder path name, basename, and extension name.

    :param file_name: full path
    :return: path name, basename, extension name
    """
    pth = os.path.dirname(file_name)
    f_name = os.path.basename(file_name)

    ext = None
    for special_ext in ['.nii.gz']:
        ext_len = len(special_ext)
        if f_name[-ext_len:].lower() == special_ext:
            ext = f_name[-ext_len:]
            f_name = f_name[:-ext_len] if len(f_name) > ext_len else ''
            break
    if not ext:
        f_name, ext = os.path.splitext(f_name)
    return pth, f_name, ext


def multi_filter(names: Sequence, patterns: Tuple[str]) -> Generator:
    """
    Generator function which yields the names that match one or more of the patterns.

    `<https://codereview.stackexchange.com/questions/74713/filtering-with-multiple-inclusion-and-exclusion-patterns>`_

    :param names: list to filter
    :param patterns: patterns uses for filter
    """
    for name in names:
        if any(fnmatch.fnmatch(name, pattern) for pattern in patterns):
            yield name


def super_filter(names: Sequence, inclusion_patterns: Tuple[str] = ("*",),
                 exclusion_patterns: Tuple[str] = ("",)) -> Set:
    """
    Enhanced version of fnmatch.filter() that accepts multiple inclusion and exclusion patterns.

    Filter the input names by choosing only those that are matched by
    some pattern in inclusion_patterns _and_ not by any in exclusion_patterns.

    `<https://codereview.stackexchange.com/questions/74713/filtering-with-multiple-inclusion-and-exclusion-patterns>`_

    :param names: list to filter
    :param inclusion_patterns: inclusion patterns uses for filter
    :param exclusion_patterns: exclusion patterns uses for filter
    :return: Set
    """
    included = multi_filter(names, inclusion_patterns)
    excluded = multi_filter(names, exclusion_patterns)
    return set(included) - set(excluded)


@njit(parallel=True)
def fill_mask(mask_arr: np.ndarray) -> np.ndarray:
    """
    # Function from cBrainMRIPrePro
    Fill a 3D mask array. Useful when use a threshold function and need to fill hole

    :param mask_arr: mask to fill
    :return: mask filled
    """
    assert mask_arr.ndim == 3, "Mask to fill need to be a 3d array"
    for z in prange(0, mask_arr.shape[0]):  # we use np convention -> z,y,x
        for x in prange(0, mask_arr.shape[2]):
            if np.max(mask_arr[z, :, x]) == 1:
                a0 = mask_arr.shape[1] - 1
                b0 = 0
                while mask_arr[z, a0, x] == 0:
                    if a0 != 0:
                        a0 = a0 - 1  # Top of the data. Above it is zero.

                while mask_arr[z, b0, x] == 0:
                    if b0 != mask_arr.shape[1] - 1:
                        b0 = b0 + 1  # Bottom of the data. Below it is zero.
                for k in prange(b0, a0 + 1):
                    mask_arr[z, k, x] = 1
        for y in prange(0, mask_arr.shape[1]):
            if np.max(mask_arr[z, y, :]) == 1:
                c0 = mask_arr.shape[2] - 1
                d0 = 0
                while mask_arr[z, y, c0] == 0:
                    if c0 != 0:
                        c0 = c0 - 1  # Top of the data. Above it is zero.

                while mask_arr[z, y, d0] == 0:
                    if d0 != mask_arr.shape[1] - 1:
                        d0 = d0 + 1  # Bottom of the data. Below it is zero.
                for j in prange(d0, c0 + 1):
                    mask_arr[z, y, j] = 1
    return mask_arr


def get_mask(input_array: np.ndarray) -> np.ndarray:
    """
    Function from cBrainMRIPrePro
    Get a mask. Based on Otsu threshold and noise reduced. Then result mask is holes filled.

    :param input_array: input image array
    :return: binary head mask
    """

    input_array = exposure.equalize_hist(input_array) * 255
    thresh = threshold_otsu(input_array)
    otsu_mask = input_array > thresh
    noise_reduced = remove_small_objects(otsu_mask, 10, )

    head_mask = fill_mask(noise_reduced.astype(np.uint8))
    return head_mask


def create_patch_3d_rand_weighted(img: np.ndarray, spatial_size: Union[Sequence[int], int], num_samples: int = 1,
                                  weight_map: Optional[np.ndarray] = None, random_state: Union[int, None] = 123) -> \
        List[np.ndarray]:
    """
    From monai: Samples a list of num_samples image patches according to the provided weight_map.

    :param img: input image to sample patches from. assuming img is a channel-first array.
    :param spatial_size: the spatial size of the image patch e.g. [224, 224, 128].
        If its components have non-positive values, the corresponding size of img will be used.
    :param num_samples: number of samples (image patches) to take in the returned list.
    :param weight_map: weight map used to generate patch samples. The weights must be non-negative.
        Each element denotes a sampling weight of the spatial location. 0 indicates no sampling.
        It should be a single-channel array in shape, for example, (1, spatial_dim_0, spatial_dim_1, …).
    :param random_state: int, RandomState instance or None, optional, default: 123
        If int, random_state is the seed used by the random number generator;
        If None, the random number generator is the RandomState instance used by `np.random`.
    :return: A list of image patches
    """

    patches = RandWeightedCrop(spatial_size, num_samples, weight_map)
    if random_state is not None:
        patches.set_random_state(random_state)
    return patches.__call__(img, weight_map)


def get_foreground_and_background_image(volume: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    get the foreground and background image

    :param volume: array-like
    :return: array-like foreground mask, array-like background mask,
        array-like foreground_intensity_voxels, array-like background_intensity_voxels (1D)
    """
    # create foreground mask
    foreground_mask = get_mask(volume)
    logical_mask = foreground_mask == 1  # force the mask to be logical type

    background_mask = ~logical_mask
    background_mask = background_mask.astype(np.int)
    foreground_mask = foreground_mask.astype(np.int)

    # create foreground intensity voxels
    foreground_intensity_voxels = np.copy(volume)
    foreground_intensity_voxels = foreground_intensity_voxels[logical_mask]

    # create background intensity voxels
    background_intensity_voxels = np.copy(volume)
    background_intensity_voxels = background_intensity_voxels[~logical_mask]

    return foreground_mask, background_mask, foreground_intensity_voxels, background_intensity_voxels
