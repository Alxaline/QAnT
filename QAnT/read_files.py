# -*- coding: utf-8 -*-
"""
| Function to read files. Dicom and Nii
| Author: Alexandre CARRE (alexandre.carre@gustaveroussy.fr)
| Created on: Dec 15, 2019
"""

import logging
from typing import List, Tuple, Optional, Union

import SimpleITK as sitk
import numpy as np
import pydicom

logger = logging.getLogger(__name__)


class InvalidFileFormat(Exception):
    pass


def rescale_values(header: Optional = None) -> Union[float, int]:
    """
    Calculate rescale slope and intercept if they are in the dicom headers, otherwise 1 is returned for slope and 0
    for intercept.

    :param header: header of the dicom (Pydicom format).
    :return: rescale slope and intercept.
    """
    # apply rescale slope and intercept to the image
    if hasattr(header, 'RealWorldValueMappingSequence'):
        slope = header.RealWorldValueMappingSequence[0].RealWorldValueSlope
    elif hasattr(header, 'RescaleSlope'):
        slope = 1  # sitk does rescaling header.RescaleSlope
    else:
        logger.warning('No rescale slope found in dicom header')
        slope = 1

    if hasattr(header, 'RealWorldValueMappingSequence'):
        intercept = header.RealWorldValueMappingSequence[0].RealWorldValueIntercept
    elif hasattr(header, 'RescaleIntercept'):
        intercept = 0  # sitk does rescaling header.RescaleIntercept
    else:
        logger.warning('No rescale intercept found in dicom header')
        intercept = 0

    return slope, intercept


def safe_sitk_read(img_list: List, *args, **kwargs) -> sitk.Image:
    """
    Since the default function just looks at images 0 and 1 to determine slice thickness
    and the images are often not correctly alphabetically sorted, much slower

    :param img_list: sitk.Image, Multidimensional array with pixel data, metadata.
    :return: sitk.Image
    """

    pimg_list = [(sitk.ReadImage(x).GetOrigin(), x) for x in img_list]

    s_img_list = [path for _, path in sorted(pimg_list, key=lambda x: x[0][2], reverse=False)]  # sort by z
    return sitk.ReadImage(s_img_list, *args, **kwargs)


def read_dicom_files(file_list: List) -> sitk.Image:
    """
    Read a file or list of files using SimpleTIK. A file list will be
    read as an image series in SimpleITK.

    :param file_list: Input files list where the dicom files are located
    :return: sitk.Image, Multidimensional array with pixel data, metadata
    """

    if isinstance(file_list, str):
        file_list = [file_list]
    elif not isinstance(file_list, (tuple, list)):
        raise NotImplemented("Need to be a tuple or list to be read by safe sitk read")
    try:
        image = safe_sitk_read(file_list)
    except IOError:
        raise InvalidFileFormat('cannot read file: {0}'.format(file_list))
    return image


def read_serie(files: List, rescale: bool = True) -> sitk.Image:
    """
    Read a single image serie from a dicom database to SimpleITK images.

    :param files: Input files list where the dicom files are located
    :param rescale: Set to True to calculate rescale slope and intercept.
    :return: sitk.Image, Multidimensional array with pixel data, metadata
    """

    image = read_dicom_files(files)
    image = sitk.Cast(image, sitk.sitkFloat32)
    header = pydicom.read_file(files[0], stop_before_pixels=True)
    if rescale:
        slope, intercept = rescale_values(header)

        image *= slope
        image += intercept

    return image


def load_nifty_volume_as_array(input_path_file: str) -> Tuple[np.ndarray, Tuple[Tuple, Tuple, Tuple]]:
    """
    Load nifty image into numpy array [z,y,x] axis order. The output array shape is like [Depth, Height, Width].

    :param input_path_file: input path file, should be '*.nii' or '*.nii.gz'
    :return: a numpy data array, (with header)
    """

    img = sitk.ReadImage(input_path_file)
    data = sitk.GetArrayFromImage(img)

    origin, spacing, direction = img.GetOrigin(), img.GetSpacing(), img.GetDirection()
    return data, (origin, spacing, direction)
