# -*- coding: utf-8 -*-
"""
| Quality Assessment metrics.
| Author: Alexandre CARRE (alexandre.carre@gustaveroussy.fr).
| Created on: Jan 14, 2021.
|
| Initial def:
| F is Foreground intensity voxels

.. math::
    F=\sum ^{n}_{i=1}\dfrac{v_{f_{i}}}{n}

with :math:`v_{f_{i}} i^{th}` foreground voxels

| B is Background intensity voxels

.. math::
    B=\sum ^{n}_{i=1}\dfrac{v_{b_{i}}}{n}

| with :math:`v_{f_{i}} i^{th}` background voxels
| with :math:`F_{P}` is Foreground random patch voxels (n=5000, with a 5x5x5 patch-size)
| with :math:`B_{P}` is Background random patch voxels (n=5000, with a 5x5x5 patch-size)
"""
from typing import Tuple

import numpy as np
from monai.transforms import CropForegroundd
from scipy.ndimage import median_filter
from scipy.signal import convolve2d as conv2d

from QAnT.utils import create_patch_3d_rand_weighted


def mean(foreground_intensity_voxels: np.ndarray) -> np.dtype(float):
    """
    mean of the foreground

    .. math::
        mean=\dfrac{F}{n}

    :param foreground_intensity_voxels: array representing the foreground intensity voxels
    :return: value
    """
    return np.nanmean(foreground_intensity_voxels)


def rang(foreground_intensity_voxels: np.ndarray) -> np.dtype(float):
    """
    range of the foreground

    .. math::
        range=\max (F)-\min (F)

    :param foreground_intensity_voxels: array representing the foreground intensity voxels
    :return: value
    """
    return np.ptp(foreground_intensity_voxels)


def variance(foreground_intensity_voxels: np.ndarray) -> np.dtype(float):
    """
    variance of the foreground

    .. math::
        variance=\sigma_{F}^{2}

    :param foreground_intensity_voxels: array representing the foreground intensity voxels
    :return: value
    """
    return np.nanvar(foreground_intensity_voxels)


def percent_coefficient_variation(foreground_intensity_voxels: np.ndarray) -> np.dtype(float):
    """
    coefficient of variation of the foreground for shadowing and inhomogeneity artifacts.

    .. math::
        pcv=\dfrac{\sigma_{F}}{\mu_{F}}

    :ref: Y. Wang, Y. Zhang, W. Xuan, E. Kao, P. Cao, B. Tian, K. Ordovas, D. Saloner, andJ. Liu,
        Fully automatic segmentation of 4D MRI for cardiac functional measurements,Medical Physics46, 180–189 (2019).

    :param foreground_intensity_voxels: array representing the foreground intensity voxels
    :return: value
    """
    return (np.nanstd(foreground_intensity_voxels) / np.nanmean(foreground_intensity_voxels)) * 100


def contrast_per_pixel(volume: np.ndarray, foreground_mask: np.ndarray) -> np.dtype(float):
    r"""
    contrast  per  pixel:  mean  of  the  foreground  filtered  by  a  3×3  2D  Laplacian  kernel  for
    shadowing artifacts.

    .. math::
        cpp=\operatorname{mean}(\operatorname{conv2} (\mathrm{F}, \mathrm{f}_{1})

    .. math:: \mathrm{f}_{1} = \begin{bmatrix}
                    - 1 & -1 & -1 \\
                    -1 & 8 & -1 \\
                    -1 & -1 & -1
                    \end{bmatrix}

    :ref: S.-J. Chang, S. Li, A. Andreasen, X.-Z. Sha, and X.-Y. Zhai,  A reference-free method for brightness
        compensation and contrast enhancement of micrographs of serial sections,PloS one10(2015).

    :param volume: 3D array representing the volume
    :param foreground_mask: 3D array representing the foreground_mask
    :return: value
    """
    crop = CropForegroundd(keys=["volume", "foreground_mask"], source_key="foreground_mask")
    foreground_volume = crop.__call__(data={"volume": volume, "foreground_mask": foreground_mask})['volume']
    laplacian_filter = np.array([[-1 / 8, -1 / 8, -1 / 8], [-1 / 8, 1, -1 / 8], [-1 / 8, -1 / 8, -1 / 8]])
    cpp = list(map(lambda x: np.nanmean(conv2d(x, laplacian_filter, mode='same')), foreground_volume))
    return np.nanmean(cpp)


def psnr(volume: np.ndarray, foreground_mask: np.ndarray) -> np.dtype(float):
    """
    peak signal to noise ratio of the foreground.

    .. math::
        psnr=10 \log \dfrac{\max ^{2}(F)}{\operatorname{MSE}(F, f_{2})}

    with :math:`f_{2}` is a 5x5x5 median filter

    :ref: D. Sage and M. Unser,  Teaching Image-Processing Programming in Java,
        IEEE SignalProcessing Magazine20, 43–52 (2003).

    :param volume: 3D array representing the volume
    :param foreground_mask: 3D array representing the foreground_mask
    :return: value
    """
    crop = CropForegroundd(keys=["volume", "foreground_mask"], source_key="foreground_mask")
    foreground_volume = crop.__call__(data={"volume": volume, "foreground_mask": foreground_mask})['volume']
    i_hat = median_filter(foreground_volume, size=5)
    mse = np.square(np.subtract(foreground_volume, i_hat)).mean()
    return 20 * np.log10(np.nanmax(foreground_volume) / np.sqrt(mse))


def snr1(foreground_intensity_voxels: np.ndarray, background_intensity_voxels: np.ndarray) -> np.dtype(float):
    """
    foreground standard deviation (SD) divided by background SD.

    .. math::
        snr1=\dfrac{\sigma_{F}}{\sigma_{B}}

    :ref: T. Bushberg, J. A. Seibert, E. M. Leidholdt, and J. M. Boone,The essential physicsof medical imaging,
        Lippincott Williams & Wilkins, 2011.

    :param foreground_intensity_voxels: array representing the foreground intensity voxels
    :param background_intensity_voxels: array representing the background intensity voxels
    :return: value
    """
    return np.nanstd(foreground_intensity_voxels) / np.nanstd(background_intensity_voxels)


def snr2(volume: np.ndarray, foreground_mask: np.ndarray, background_intensity_voxels: np.ndarray,
         kernel_size: Tuple[int, int, int] = (5, 5, 5)) -> np.dtype(float):
    """
    mean of the foreground patch divided by background SD.

    .. math::
        snr2 = \dfrac{\mu_{F_{P}}}{\sigma_{B}}.

    :ref: O. Esteban, D. Birman, M. Schaer, O. O. Koyejo, R. A. Poldrack, and K. J. Gorgolewski,
        MRIQC: Advancing the automatic prediction of image quality in MRI from unseen sites,PloS one 12(2017).

    :param volume: 3D array representing the volume
    :param foreground_mask: 3D array representing the foreground_mask
    :param background_intensity_voxels: array representing the background intensity voxels
    :param kernel_size: kernel size. Default is (5, 5, 5)

    :return: value
    """
    fore_patch = create_patch_3d_rand_weighted(img=volume[None], spatial_size=kernel_size, num_samples=5000,
                                               weight_map=foreground_mask[None], random_state=123)
    fore_patch = list(map(lambda x: x[0], fore_patch))
    return np.nanmean(fore_patch) / np.nanstd(background_intensity_voxels)


def snr3(volume: np.ndarray, foreground_mask: np.ndarray, kernel_size: Tuple[int, int, int] = (5, 5, 5)) -> np.dtype(
    float):
    """
    foreground patch SD divided by the centered foreground patch SD.

    .. math::
        snr3 = \dfrac{\mu_{F_{P}}}{\sigma_{F_{P}}\mu_{F_{P}}}

    :param volume: 3D array representing the volume
    :param foreground_mask: 3D array representing the foreground_mask
    :param kernel_size: kernel size. Default is (5, 5, 5)
    :return: value
    """
    fore_patch = create_patch_3d_rand_weighted(img=volume[None], spatial_size=kernel_size, num_samples=5000,
                                               weight_map=foreground_mask[None], random_state=123)
    fore_patch = list(map(lambda x: x[0], fore_patch))
    return np.nanmean(fore_patch) / np.nanstd(fore_patch - np.nanmean(fore_patch))


def snr4(volume: np.ndarray, foreground_mask: np.ndarray, background_mask: np.ndarray,
         kernel_size: Tuple[int, int, int] = (5, 5, 5)) -> np.dtype(float):
    """
    mean of the foreground patch divided by mean of the background patch.

    .. math::
        snr4 = \dfrac{\mu_{F_{P}}}{\sigma_{B_{P}}}

    :param volume: 3D array representing the volume
    :param foreground_mask: 3D array representing the foreground_mask
    :param background_mask: 3D array representing the background_mask
    :param kernel_size: kernel size. Default is (5, 5, 5)
    :return: value
    """
    fore_patch = create_patch_3d_rand_weighted(img=volume[None], spatial_size=kernel_size, num_samples=5000,
                                               weight_map=foreground_mask[None], random_state=123)
    fore_patch = list(map(lambda x: x[0], fore_patch))

    back_patch = create_patch_3d_rand_weighted(img=volume[None], spatial_size=kernel_size, num_samples=5000,
                                               weight_map=background_mask[None], random_state=123)
    back_patch = list(map(lambda x: x[0], back_patch))

    return np.nanmean(fore_patch) / np.nanstd(back_patch)


def cnr(volume: np.ndarray, foreground_mask: np.ndarray, background_mask: np.ndarray,
        kernel_size: Tuple[int, int, int] = (5, 5, 5)) -> np.dtype(float):
    """
    contrast  to  noise  ratio  for  shadowing  and  noise  artifacts14:  mean  of  the  foreground  and  background
    patches difference divided by background patch SD.

    .. math::
        cnr = \dfrac{\mu_{F_{P}}-B_{P}}{\sigma_{B_{P}}}

    :ref: T. Bushberg, J. A. Seibert, E. M. Leidholdt, and J. M. Boone,The essential physicsof medical imaging,
        Lippincott Williams & Wilkins, 2011

    :param volume: 3D array representing the volume
    :param foreground_mask: 3D array representing the foreground_mask
    :param background_mask: 3D array representing the background_mask
    :param kernel_size: kernel size. Default is (5, 5, 5)
    :return: value
    """
    fore_patch = create_patch_3d_rand_weighted(img=volume[None], spatial_size=kernel_size, num_samples=5000,
                                               weight_map=foreground_mask[None], random_state=123)
    fore_patch = list(map(lambda x: x[0], fore_patch))

    back_patch = create_patch_3d_rand_weighted(img=volume[None], spatial_size=kernel_size, num_samples=5000,
                                               weight_map=background_mask[None], random_state=123)
    back_patch = list(map(lambda x: x[0], back_patch))

    return np.nanmean(np.array(fore_patch) - np.array(back_patch)) / np.nanstd(back_patch)


def cvp(volume: np.ndarray, foreground_mask: np.ndarray, kernel_size: Tuple[int, int, int] = (5, 5, 5)) -> np.dtype(
    float):
    """
    coefficient of variation of the foreground patch for shading artifacts:  foreground patch SD divided by
    foreground patch mean.

    .. math::
        cvp = \dfrac{\sigma_{F_{P}}}{\mu_{F_{P}}}

    :param volume: 3D array representing the volume
    :param foreground_mask: 3D array representing the foreground_mask
    :param kernel_size: kernel size. Default is (5, 5, 5)
    :return: value
    """
    fore_patch = create_patch_3d_rand_weighted(img=volume[None], spatial_size=kernel_size, num_samples=5000,
                                               weight_map=foreground_mask[None], random_state=123)
    fore_patch = list(map(lambda x: x[0], fore_patch))
    return np.nanstd(fore_patch) / np.nanmean(fore_patch)


def cjv(foreground_intensity_voxels: np.ndarray, background_intensity_voxels: np.ndarray) -> np.dtype(float):
    """
    coefficient  of  joint  variation  between  the  foreground  and  background  for  aliasing  and  inhomogeneity
    artifacts.

    .. math::
        cjv = \dfrac{\sigma_{F}+\sigma_{B}}{|\mu_{F}-\mu_{B}|}

    :ref: C. Hui, Y. X. Zhou, and P. Narayana,  Fast algorithm for calculation of inhomogeneity gradient in
        magnetic resonance imaging data,  Journal of Magnetic Resonance Imaging32, 1197–1208 (2010).

    :param foreground_intensity_voxels: array representing the foreground intensity voxels
    :param background_intensity_voxels: array representing the background intensity voxels
    :return: value
    """
    return (np.nanstd(foreground_intensity_voxels) + np.nanstd(background_intensity_voxels)) / abs(
        np.nanmean(foreground_intensity_voxels) - np.nanmean(background_intensity_voxels))


def efc(foreground_intensity_voxels: np.ndarray) -> float:
    """
    entropy focus criterion for motion artifacts.

    .. math::
        E = - \sum_{i=1}^n \dfrac{F_i}{F_\text{max}} \ln [\dfrac{F_i}{F_\text{max}}]

    with :math:`F_{\mathrm{max}}=\sqrt{\sum_{i, j} F^{2}(i, j)}`

    :ref: O. Esteban, D. Birman, M. Schaer, O. O. Koyejo, R. A. Poldrack, and K. J. Gorgolewski,
        MRIQC: Advancing the automatic prediction of image quality in MRI from unseen sites,PloS one12(2017).

    :param foreground_intensity_voxels: array representing the foreground intensity voxels
    :return: value
    """
    n_vox = foreground_intensity_voxels.shape[0]
    efc_max = 1.0 * n_vox * (1.0 / np.sqrt(n_vox)) * np.log(1.0 / np.sqrt(n_vox))
    cc = (foreground_intensity_voxels ** 2).sum()
    b_max = np.sqrt(abs(cc))
    return float(
        (1.0 / abs(efc_max)) * np.sum(
            (foreground_intensity_voxels / b_max) * np.log((foreground_intensity_voxels + 1e16) / b_max)))


def fber(foreground_intensity_voxels: np.ndarray, background_intensity_voxels: np.ndarray) -> float:
    """
    foreground-background energy ratio for ringing artifacts.

    .. math::
        fber=\dfrac{\operatorname{median}(|F|^{2})}{\operatorname{median}(|B|^{2})}

    :ref: Z. Shehzad, S. Giavasis, Q. Li, Y. Benhajali, C. Yan, Z. Yang, M. Milham, P. Bellec,and C. Craddock,
        The Preprocessed Connectomes Project Quality Assessment Protocolaresource for measuring the quality of MRI data,
        Frontiers in neuroscience47(2015).

    :param foreground_intensity_voxels: array representing the foreground intensity voxels
    :param background_intensity_voxels: array representing the background intensity voxels
    :return: value
    """
    fg_mu = np.nanmedian(np.abs(foreground_intensity_voxels) ** 2)
    bg_mu = np.nanmedian(np.abs(background_intensity_voxels) ** 2)
    if bg_mu < 1.0e-3:
        return 0.
    return float(fg_mu / bg_mu)


ALL_METRICS = {
    "mean": mean,
    "range": rang,
    "variance": variance,
    "pcv": percent_coefficient_variation,
    "cpp": contrast_per_pixel,
    "psnr": psnr,
    "snr1": snr1,
    "snr2": snr2,
    "snr3": snr3,
    "snr4": snr4,
    "cnr": cnr,
    "cvp": cvp,
    "cjv": cjv,
    "efc": efc,
    "fber": fber,
}
