import math

from scipy.interpolate import splprep, splev, UnivariateSpline

from deconvolve import deconvolve
from mat_estimation.scd import est_using_scd
import numpy as np


def calculate_stats(pixels):
    if np.empty(pixels):
        return None, None, None
    else:
        return np.percentile(pixels, 5), np.median(pixels), np.percentile(pixels, 95)


def deconvolved_channel_stats(stain_img, stain_mask, bg_mask):
    other_mask = ~(stain_mask | bg_mask)

    stain_px = np.ma.masked_array(stain_img, mask=stain_mask)
    other_px = np.ma.masked_array(stain_img, mask=other_mask)
    bg_px = np.ma.masked_array(stain_img, mask=bg_mask)

    stain_stats = calculate_stats(stain_px)
    other_stats = calculate_stats(other_px)
    bg_stats = calculate_stats(bg_px)

    return stain_stats, other_stats, bg_stats


def fit_spline(src_stats, dst_stats):
    # Find any rows containing a NaN in either set of stats, and eliminate them
    NaNs = np.isnan(src_stats) & np.isnan(dst_stats)
    src_stats = src_stats[~NaNs]
    dst_stats = dst_stats[~NaNs]

    # % Sort the stats into an ascending order
    src_stats, idx = np.sort(src_stats)
    dst_stats = dst_stats(idx)

    # Append values at the extremes to make sure that the values of pixels with
    # very high or low intensity remain unchanged by spline mapping
    src_stats = np.stack((-100, src_stats[:], 1000))
    dst_stats = np.stack((-100, dst_stats[:], 1000))

    # Generate the smoothing spline
    # 19584
    spline = UnivariateSpline(np.array(src_stats), np.array(dst_stats), s=3400)

    return spline


def norm_scd(src, dst, trainer=None):
    io = 255
    src_mat, src_lbls = est_using_scd(src, trainer)
    dst_mat, dst_lbls = est_using_scd(dst, trainer)

    src_stain = deconvolve(src, src_mat)
    dst_stain = deconvolve(dst, dst_mat)

    src_stain = io / np.exp(src_stain)
    dst_stain = io / np.exp(dst_stain)

    src_b = (src_lbls == 0)
    src_h = (src_lbls == 2)
    src_e = (src_lbls == 1)

    dst_b = (dst_lbls == 0)
    dst_h = (dst_lbls == 2)
    dst_e = (dst_lbls == 1)

    max_value = 1000

    # Threshold any pixels that are over the predefined maximum value
    src_stain[np.all(src_stain > max_value)] = max_value
    dst_stain[np.all(dst_stain > max_value)] = max_value

    # Calculate the intensity statistics of the two stains in the Source Image
    src_stats_h = deconvolved_channel_stats(src_stain[:, :, 0], src_h, src_b)
    src_stats_e = deconvolved_channel_stats(src_stain[:, :, 1], src_e, src_b)

    # Calculate the intensity statistics of the two stains in the Target Image
    dst_stats_h = deconvolved_channel_stats(dst_stain[:, :, 0], dst_h, dst_b)
    dst_stats_e = deconvolved_channel_stats(dst_stain[:, :, 0], dst_e, dst_b)

    # Generate Splines from Stain Channel Statistics
    # Resultant splines allow us to map the stain intensity stats from the
    # Source Image to the Target Image
    spline1 = fit_spline(src_stats_h, dst_stats_h)  # Spline for Haematoxylin
    spline2 = fit_spline(src_stats_e, dst_stats_e)  # Spline for Eosin

    # Calculate Adjusted Stain Channels
    # Use splines to calculate adjusted intensities for the two stain channels
    # adj_src_stain1 = ppual(spline1, src_stain[:, :, 0])
    # adj_src_stain2 = ppual(spline2, src_stain[:, :, 1])
    adj_src_stain1 = spline1(src_stain[:, :, 0])
    adj_src_stain2 = spline2(src_stain[:, :, 1])

    # The original background channel is not adjusted
    src_bg = src_stain[:, :, 2].reshape(-1, 1)

    c_mat = np.stack((adj_src_stain1[:], adj_src_stain2[:], src_bg[:]), axis=0)

    # Threshold values that do not fall within the expected range
    c_mat[np.all(c_mat > 255)] = 255
    c_mat[np.all(c_mat < 0)] = 0

    # Convert the stain data back to OD space.
    c_od_mat = np.log(io / (c_mat + 0.0001))

    # %% Reconstruct the RGB image
    norm = io * np.exp(c_od_mat * -dst_mat)
    norm = norm.reshape(src.shape)
    norm = norm.astype(np.uint8)

    return norm
