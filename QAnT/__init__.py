# -*- coding: utf-8 -*-
"""
| Author: Alexandre CARRE
| Created on: Jan 14, 2021
"""
import logging
import os
import sys
from typing import Optional, Tuple


def get_logger_lvl(verbosity_lvl: int = 0) -> int:
    """
    Get logging lvl for logger

    :param verbosity_lvl: verbosity level
    :return: logging verbosity level
    """
    if verbosity_lvl == 1:
        level = logging.getLevelName("INFO")
    elif verbosity_lvl >= 2:
        level = logging.getLevelName("DEBUG")
    else:
        level = logging.getLevelName("WARNING")
    return level


def set_main_logger(log_file: bool = True, filename: Optional[str] = "logfile.log",
                    verbosity_lvl: Optional[int] = 0) -> None:
    """
    Set the main logger

    :param log_file: True to generate a logfile
    :param filename: logfile name (Default is "logfile.log")
    :param verbosity_lvl: level of verbosity
    """
    file_handler = logging.FileHandler(filename=filename) if log_file else None
    stdout_handler = logging.StreamHandler(sys.stdout)
    handlers = [file_handler, stdout_handler] if log_file else [stdout_handler]

    level = get_logger_lvl(verbosity_lvl)

    logging.basicConfig(level=level,
                        format="%(asctime)s - %(levelname)s - %(message)s",
                        handlers=handlers)


def get_parameter_validation_files() -> Tuple[str, str, str]:
    """
    Returns file locations for the parameter schemas and custom validation functions, which are needed when validating
    a parameter file using ``PyKwalify.core``.

    :return: a tuple with the file location of the schemas as first and python script with custom validation
        functions as second element.
    """
    schema_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'schemas'))
    default_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'example_parameters'))
    schema_file = os.path.join(schema_dir, 'param_schema.yaml')
    default_file = os.path.join(default_dir, 'default_parameters.yaml')

    schema_funcs = os.path.join(schema_dir, 'schema_funcs.py')
    return default_file, schema_file, schema_funcs
