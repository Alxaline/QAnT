# -*- coding: utf-8 -*-
"""
| Author: Alexandre CARRE (alexandre.carre@gustaveroussy.fr)
| Created on: Jan 14, 2021
"""
import argparse
import collections.abc
import inspect
import logging
import os
import sys
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from glob import glob
from multiprocessing import cpu_count
from typing import Tuple, List, Union, Callable, Sequence

import SimpleITK as sitk
import numpy as np
import pandas as pd
import pydicom
import pykwalify.core

from QAnT import get_parameter_validation_files, set_main_logger
from QAnT.metrics import ALL_METRICS
from QAnT.read_files import load_nifty_volume_as_array, read_serie
from QAnT.utils import get_foreground_and_background_image, split_filename, check_ext, safe_file_name, check_isdir, \
    super_filter

logger = logging.getLogger(__name__)


def arg_parser():
    parser = argparse.ArgumentParser(description="QAnT: image Quality Assessment and dicom Tags extraction")
    required = parser.add_argument_group("Required")
    required.add_argument("-i", "--input_dir", nargs="+", type=str, required=True,
                          help="Input directories path with DICOM files to be parsed. "
                               "Can be a list of directory")
    required.add_argument("-o", "--output_filepath", type=str, required=True,
                          help="Output filepath for saving the content in csv files. "
                               "Need to have the .csv extensions")
    options = parser.add_argument_group("Options")
    required.add_argument("-p", "--param", type=str, default="",
                          help="Parameter file containing the settings to be used in extraction. "
                               "If not provided use default setting.")
    options.add_argument("-j", "--n_jobs", metavar="N", type=int, default=cpu_count(),
                         choices=range(1, cpu_count() + 1),
                         help="Specifies the number of threads to use for parallel processing (default: all)")
    required.add_argument("--inclusion_keywords", nargs="+", type=str, default="*",
                          help="Inclusion keywords to parse files. fnmatch style, i.e ['a*', 'b*']")
    required.add_argument("--exclusion_keywords", nargs="+", type=str, default="",
                          help="Exclusion keywords to parse files. fnmatch style, i.e ['a*', 'b*']")
    options.add_argument("-v", "--verbosity", action="count", default=0,
                         help="increase output verbosity (e.g., -vv is more than -v)")
    return parser


def extract_dicom_metadata(dcm_file: str, dicom_tags: str) -> OrderedDict:
    """
    Function to extract dicom metadata of a dicom files

    :param dcm_file: path of the dicom files
    :param dicom_tags: List of dicom tags (need to respect the Pydicom convention, ie "SeriesDescription")
    :return: OrderedDict with id files and tags
    """
    metadata = OrderedDict()
    pth, _, _ = split_filename(dcm_file)
    metadata["id"] = pth

    curr_dcm = pydicom.read_file(dcm_file, stop_before_pixels=True)
    for tag in dicom_tags:
        if hasattr(curr_dcm, tag):
            value = getattr(curr_dcm, tag)

            if isinstance(value, collections.abc.Iterable) and not isinstance(value, str):
                # split tag sequence into different columns
                metadata.update({f"{tag}_{idx}": val for idx, val in enumerate(value)})
            else:
                metadata[tag] = value
        else:
            logger.error(f"Unknown Dicom tag format: {tag} for {dcm_file}")
    return metadata


def extract_metric_information(files: Union[List, str], metrics_list: List) -> OrderedDict:
    """
    Function to extract quality metrics of a dicom or nii files

    :param files: A list of dicom files path or List with unique str / str of nii file path
    :param metrics_list: list of metrics
    :return: OrderedDict with id files and metrics
    """
    if isinstance(files, list):
        if len(files) == 1:
            pth, _, ext = split_filename(files[0])
            if ".nii" not in ext:
                raise ValueError("A unique file was given. .nii or .nii.gz is required")
            files = files[0]
        elif not any([split_filename(file)[-1] == ".dcm" for file in files]):
            raise ValueError("A file with an unexpected extension was found. .dcm is required ")
    if isinstance(files, str):
        pth, _, ext = split_filename(files)
        if ".nii" not in ext:
            raise ValueError("A unique file was given. .nii or .nii.gz is required")

    metrics = OrderedDict()
    volume = np.array([])
    if isinstance(files, str):
        volume, _ = load_nifty_volume_as_array(files)
        metrics["id"] = files
    elif isinstance(files, list):
        image = read_serie(files, rescale=False)
        volume = sitk.GetArrayFromImage(image)
        pth, _, _ = split_filename(files[0])
        metrics["id"] = pth

    foreground_mask, background_mask, foreground_intensity_voxels, background_intensity_voxels \
        = get_foreground_and_background_image(volume)

    args_dict = {
        "volume": volume,
        "foreground_mask": foreground_mask,
        "background_mask": background_mask,
        "foreground_intensity_voxels": foreground_intensity_voxels,
        "background_intensity_voxels": background_intensity_voxels,
        "kernel_size": (5, 5, 5)
    }

    for m in metrics_list:
        assert m in ALL_METRICS.keys(), f"Metric is incorrect, possible metric are: {', '.join(ALL_METRICS.keys())}"
        args_needed = inspect.getfullargspec(ALL_METRICS[m]).args
        metric_value = ALL_METRICS[m](**{k: args_dict[k] for k in args_needed})
        metrics[m] = metric_value
    return metrics


def get_files(input_dir: str) -> Tuple[List, List]:
    """
    Get DICOM or NIfTI files recursively

    :param input_dir: input directory
    :return: list of dicom files, list of nifti files
    """
    dcm_files = glob(os.path.join(input_dir, "**/*.dcm"), recursive=True)
    nii_files = glob(os.path.join(input_dir, "**/*.nii*"), recursive=True)

    return dcm_files, nii_files


def get_unique_scan_series_dicom(dcm_files: List) -> Tuple[List, List, List]:
    """
    Get unique scan series (one file) of dicom files. Based on seen directory. A entire series need to be in different
    folder

    :param dcm_files: list of dicom files
    :return: Directory with dicom_files, Dicom files in all directory, list of dicom files representing unique scan
    """
    seen_dirs = []
    dcm_seen_dirs = []
    scans_list = []
    for res in dcm_files:
        curr_dcm = res.strip()
        curr_dir = os.path.dirname(curr_dcm)
        if curr_dir not in seen_dirs:
            scans_list.append(curr_dcm)
            seen_dirs.append(curr_dir)
            dcm_seen_dirs.append(glob(os.path.join(curr_dir, "**/*.dcm"), recursive=True))
    return seen_dirs, dcm_seen_dirs, scans_list


def _multi(function: Callable, iterable: Sequence, arg, n_jobs: int = 1, prefix_logger: str = "") -> List:
    """
    Multiprocess a function with a iterable

    :param function: a function
    :param iterable: iterable to multiprocess
    :param arg: arg for the function (unique)
    :param n_jobs: The number of parallel jobs to run for function. Int can not exceed multiprocess.cpu_count()
    :param prefix_logger: prefix for logger
    :return: list of result
    """
    result = []
    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        jobs = [executor.submit(function, i, arg) for i in iterable]

        for i, out in enumerate(as_completed(jobs)):
            logger.info(f"{prefix_logger}{i + 1}/{len(iterable)}")
            try:
                result.append(out.result())
            except Exception as e:
                logger.exception(f"An exception was thrown in multiprocess {e}")

    return result


def process(input_directories: List, parameters_file: str, output_filepath: str, n_jobs=int,
            inclusion_keywords: Tuple[str] = ("*",), exclusion_keywords: Tuple[str] = ()):
    """
    Process input directories with parameters_file

    :param input_directories: input directories
    :param parameters_file: parameters file
    :param output_filepath: output file path of the result. Need to have the .csv extension
    :param n_jobs: The number of parallel jobs to run for function. Int can not exceed multiprocess.cpu_count()
    :param inclusion_keywords: inclusion patterns uses for filter files
    :param exclusion_keywords: exclusion patterns uses for filter files
    """

    # check input_directories
    list(map(check_isdir, input_directories))

    # check extension of output_filepath
    output_filepath = check_ext(output_filepath, extension=[".csv"])

    # create export dir if not exist
    if not os.path.exists(os.path.dirname(output_filepath)):
        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)

    # safe file name
    pth, fnm, ext = split_filename(output_filepath)
    fnm = safe_file_name(fnm)
    output_filepath = os.path.join(pth, fnm + ext)

    # verify the parameter file
    default_file, schema_file, schema_funcs = get_parameter_validation_files()
    if not os.path.isfile(parameters_file):
        logger.warning("Path for specified parameter file does not exist! Use the default parameter file")
        parameters_file = default_file
        if not os.path.isfile(parameters_file):
            raise FileNotFoundError("Path for default parameter file does not exist!")

    validate_params_file = pykwalify.core.Core(source_file=parameters_file, schema_files=[schema_file],
                                               extensions=[schema_funcs])
    try:
        validate_params_file.validate()
    except Exception as e:
        logger.error(f"Parameter validation failed! \n{e}", exc_info=True)

    # parse files
    dcm_files, nii_files = [], []
    for input_dir in input_directories:
        input_dir_dcm_files, input_dir_nii_files = get_files(input_dir)
        dcm_files.extend(input_dir_dcm_files), nii_files.extend(input_dir_nii_files)

    # filter files if inclusion keywords or exclusion keywords
    if exclusion_keywords or inclusion_keywords:
        dcm_files = list(super_filter(dcm_files, inclusion_patterns=tuple(inclusion_keywords),
                                      exclusion_patterns=tuple(exclusion_keywords)))
        nii_files = list(super_filter(nii_files, inclusion_patterns=tuple(inclusion_keywords),
                                      exclusion_patterns=tuple(exclusion_keywords)))

    seen_dirs, dcm_in_seen_dirs, dcm_scans_list = get_unique_scan_series_dicom(dcm_files)
    dcm_df, nii_metrics_df = pd.DataFrame([]), pd.DataFrame([])
    if dcm_files:
        if "DicomTags" in validate_params_file.source and validate_params_file.source["DicomTags"]:
            logger.info("Extracting metadata from dicom scan series...")
            dicom_metadata_list = _multi(extract_dicom_metadata, dcm_scans_list,
                                         validate_params_file.source["DicomTags"], n_jobs, "[DicomTags] ")
            print(dicom_metadata_list)
            dcm_df = pd.DataFrame(dicom_metadata_list)

        if "QualityMetrics" in validate_params_file.source and validate_params_file.source["QualityMetrics"]:
            logger.info("Extracting metrics from dicom scan series...")
            dicom_metrics_list = _multi(extract_metric_information, dcm_in_seen_dirs,
                                        validate_params_file.source["QualityMetrics"], n_jobs,
                                        "[Dicom QualityMetrics] ")
            dicom_metrics_df = pd.DataFrame(dicom_metrics_list)
            if not dcm_df.empty:
                dcm_df = pd.merge(dcm_df, dicom_metrics_df, on="id")
            else:
                dcm_df = dicom_metrics_df

    if nii_files:
        if "DicomTags" in validate_params_file.source and validate_params_file.source["DicomTags"]:
            logger.warning("DicomTags are specified but can not be run for NIfTI files")

        if "QualityMetrics" in validate_params_file.source and validate_params_file.source["QualityMetrics"]:
            logger.info("Extracting metrics from dicom scan series...")
            nii_metrics_list = _multi(extract_metric_information, nii_files,
                                      validate_params_file.source["QualityMetrics"], n_jobs, "[NIfTI QualityMetrics] ")
            nii_metrics_df = pd.DataFrame(nii_metrics_list)

    final_df = pd.DataFrame([])
    if not dcm_df.empty and not nii_metrics_df.empty:
        final_df = pd.concat([dcm_df, nii_metrics_df])
    elif not dcm_df.empty:
        final_df = dcm_df
    elif not nii_metrics_df.empty:
        final_df = nii_metrics_df

    # export to csv
    if not final_df.empty:
        logger.info(f"Export results to: {output_filepath} ")
        final_df.to_csv(output_filepath, sep=",", na_rep="", index=False)
    else:
        logger.critical("Final output is empty")


def main(args=None) -> Union[int, None]:
    """
    main function

    :param args: args from cli
    :return: sys exit if run or exception
    """
    args = arg_parser().parse_args(args)

    # set logger
    set_main_logger(log_file=False, verbosity_lvl=args.verbosity)

    try:
        process(args.input_dir, args.param, args.output_filepath, args.n_jobs, args.inclusion_keywords,
                args.exclusion_keywords)

        return 0
    except (KeyboardInterrupt, SystemError):
        logger.info("Canceling QAnT")
        return -1
    except Exception as e:
        logger.exception(e)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
