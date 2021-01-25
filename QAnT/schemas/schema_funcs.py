import pykwalify.rule
from pydicom._dicom_dict import DicomDictionary
from typing import Tuple, List
from QAnT.metrics import ALL_METRICS


def check_dicom_tags(*pykwalify_obj: Tuple[List, pykwalify.rule.Rule, str]) -> bool:
    """
    Check if the input dicom tags exist
    :param pykwalify_obj: value, rule_obj, path
    :return: True if ok
    """
    value, _, _ = pykwalify_obj
    unrecognized_tags = []
    for tag in value:
        if not any([i[-1] == tag for i in DicomDictionary.values()]):
            unrecognized_tags.append(tag)
    if unrecognized_tags:
        raise ValueError(f"DicomTags contains unrecognized tags: {unrecognized_tags}. "
                         f"Check String dicoms tags in: \n"
                         f" https://github.com/pydicom/pydicom/blob/master/pydicom/_dicom_dict.py ")

    return True


def check_quality_metrics(*pykwalify_obj: Tuple[List, pykwalify.rule.Rule, str]) -> bool:
    """
    Check if the input dicom tags exist
    :param pykwalify_obj: value, rule_obj, path
    :return: True if ok
    """
    value, _, _ = pykwalify_obj
    unrecognized_tags = []
    for qm in value:
        if qm not in ALL_METRICS.keys():
            unrecognized_tags.append(qm)
    if unrecognized_tags:
        raise ValueError(f"QualityMetrics contains unrecognized tags: {unrecognized_tags}.\n"
                         f"Possible metrics are: {', '.join(ALL_METRICS.keys())}")

    return True
