import scipy.ndimage as scnd
import scipy.optimize as sio
import numpy as np
import numba
import warnings
import stemtool as st


@numba.jit
def fit_nbed_disks(corr_image, disk_size, positions, diff_spots):
    warnings.filterwarnings("ignore")
    positions = np.asarray(positions, dtype=np.float64)
    diff_spots = np.asarray(diff_spots, dtype=np.float64)
    fitted_disk_list = np.zeros_like(positions)
    disk_locations = np.zeros_like(positions)
    for ii in range(int(np.shape(positions)[0])):
        posx = positions[ii, 0]
        posy = positions[ii, 1]
        par = st.util.fit_gaussian2D_mask(corr_image, posx, posy, disk_size)
        fitted_disk_list[ii, 0] = par[0]
        fitted_disk_list[ii, 1] = par[1]
    disk_locations = np.copy(fitted_disk_list)
    disk_locations[:, 1] = 0 - disk_locations[:, 1]
    center = disk_locations[
        np.logical_and((diff_spots[:, 0] == 0), (diff_spots[:, 1] == 0)), :
    ]
    cx = center[0, 0]
    cy = center[0, 1]
    disk_locations[:, 0:2] = disk_locations[:, 0:2] - np.asarray(
        (cx, cy), dtype=np.float64
    )
    lcbed, _, _, _ = np.linalg.lstsq(diff_spots, disk_locations, rcond=None)
    cy = (-1) * cy
    return fitted_disk_list, np.asarray((cx, cy), dtype=np.float64), lcbed


def sobel_filter(image, med_filter=50):
    ls_image, _ = st.util.sobel(st.util.image_logarizer(image))
    ls_image[ls_image > (med_filter * np.median(ls_image))] = med_filter * np.median(
        ls_image
    )
    ls_image[ls_image < (np.median(ls_image) / med_filter)] = (
        np.median(ls_image) / med_filter
    )
    return ls_image


@numba.jit
def strain_and_disk(data4D, disk_size, pixel_list_xy, disk_list, ROI=1, med_factor=50):
    warnings.filterwarnings("ignore")

    if np.size(ROI) < 2:
        ROI = np.ones((data4D.shape[2], data4D.shape[3]), dtype=bool)

    # Calculate needed values
    scan_size = np.asarray(data4D.shape)[2:4]
    sy, sx = np.mgrid[0 : scan_size[0], 0 : scan_size[1]]
    scan_positions = (np.asarray((np.ravel(sy), np.ravel(sx)))).astype(int)
    cbed_size = np.asarray(data4D.shape)[0:2]
    yy, xx = np.mgrid[0 : cbed_size[0], 0 : cbed_size[1]]
    center_disk = (
        st.util.make_circle(cbed_size, cbed_size[1] / 2, cbed_size[0] / 2, disk_size)
    ).astype(np.float64)
    i_matrix = (np.eye(2)).astype(np.float64)
    sobel_center_disk, _ = st.util.sobel(center_disk)

    # Initialize matrices
    e_xx = np.zeros(scan_size, dtype=np.float64)
    e_xy = np.zeros(scan_size, dtype=np.float64)
    e_th = np.zeros(scan_size, dtype=np.float64)
    e_yy = np.zeros(scan_size, dtype=np.float64)
    disk_x = np.zeros(scan_size, dtype=np.float64)
    disk_y = np.zeros(scan_size, dtype=np.float64)
    COM_x = np.zeros(scan_size, dtype=np.float64)
    COM_y = np.zeros(scan_size, dtype=np.float64)

    # Calculate for mean CBED if no reference
    mean_cbed = np.mean(data4D, axis=(-1, -2), dtype=np.float64)
    mean_ls_cbed, _ = st.util.sobel(st.util.image_logarizer(mean_cbed))
    mean_ls_cbed[
        mean_ls_cbed > med_factor * np.median(mean_ls_cbed)
    ] = med_factor * np.median(mean_ls_cbed)
    mean_ls_cbed[mean_ls_cbed < np.median(mean_ls_cbed) / med_factor] = (
        np.median(mean_ls_cbed) / med_factor
    )
    mean_lsc = st.util.cross_corr_unpadded(mean_ls_cbed, sobel_center_disk)
    _, mean_center, mean_axes = fit_nbed_disks(
        mean_lsc, disk_size, pixel_list_xy, disk_list
    )
    axes_lengths = ((mean_axes[:, 0] ** 2) + (mean_axes[:, 1] ** 2)) ** 0.5
    beam_r = axes_lengths[1]
    inverse_axes = np.linalg.inv(mean_axes)

    for pp in range(np.size(sy)):
        ii = scan_positions[0, pp]
        jj = scan_positions[1, pp]
        pattern = data4D[:, :, ii, jj]
        pattern_ls, _ = st.util.sobel(st.util.image_logarizer(pattern))
        pattern_ls[pattern_ls > med_factor * np.median(pattern_ls)] = np.median(
            pattern_ls
        )
        pattern_lsc = st.util.cross_corr_unpadded(pattern_ls, sobel_center_disk)
        _, pattern_center, pattern_axes = fit_nbed_disks(
            pattern_lsc, disk_size, pixel_list_xy, disk_list
        )
        pcirc = (
            (((yy - pattern_center[1]) ** 2) + ((xx - pattern_center[0]) ** 2)) ** 0.5
        ) <= beam_r
        pattern_x = np.sum(pattern[pcirc] * xx[pcirc]) / np.sum(pattern[pcirc])
        pattern_y = np.sum(pattern[pcirc] * yy[pcirc]) / np.sum(pattern[pcirc])
        t_pattern = np.matmul(pattern_axes, inverse_axes)
        s_pattern = t_pattern - i_matrix
        e_xx[ii, jj] = -s_pattern[0, 0]
        e_xy[ii, jj] = -(s_pattern[0, 1] + s_pattern[1, 0])
        e_th[ii, jj] = -(s_pattern[0, 1] - s_pattern[1, 0])
        e_yy[ii, jj] = -s_pattern[1, 1]
        disk_x[ii, jj] = pattern_center[0] - mean_center[0]
        disk_y[ii, jj] = pattern_center[1] - mean_center[1]
        COM_x[ii, jj] = pattern_x - mean_center[0]
        COM_y[ii, jj] = pattern_y - mean_center[1]
    return e_xx, e_xy, e_th, e_yy, disk_x, disk_y, COM_x, COM_y


@numba.jit
def dpc_central_disk(data4D, disk_size, position, ROI=1, med_val=20):
    """
    DPC routine on only the central disk

    Parameters
    ----------
    data4D:     ndarray
                The 4 dimensional dataset that will be analyzed
                The first two dimensions are the Fourier space
                diffraction dimensions and the last two dimensions
                are the real space scanning dimensions
    disk_size:  float
                Size of the central disk
    position:   ndarray
                X and Y positions
                This is the initial guess that will be refined
    ROI:        ndarray
                The region of interest for the scanning region
                that will be analyzed. If no ROI is given then
                the entire scanned area will be analyzed
    med_val:    float
                Sometimes some pixels are either too bright in
                the diifraction patterns due to stray muons or
                are zero due to dead detector pixels. This removes
                the effect of such pixels before Sobel filtering

    Returns
    -------
    p_cen: ndarray
           P positions of the central disk
    q_cen: ndarray
           Q positions of the central disk
    p_com: ndarray
           P positions of the center of mass
           of the central disk
    q_com: ndarray
           Q positions of the center of mass
           of the central disk

    Notes
    -----
    This is when we want to perform DPC without bothering
    about the higher order disks. The ROI of the 4D dataset
    is calculated, and the central disk is fitted in each ROI
    point, and then a disk is calculated centered on the edge
    fitted center and then the COM inside that disk is also
    calculated.

    :Authors:
    Debangshu Mukherjee <mukherjeed@ornl.gov>
    """
    warnings.filterwarnings("ignore")

    if np.size(ROI) < 2:
        ROI = np.ones((data4D.shape[2], data4D.shape[3]), dtype=bool)

    yy, xx = np.mgrid[0 : data4D.shape[2], 0 : data4D.shape[3]]
    data4D_ROI = data4D[:, :, yy[ROI], xx[ROI]]
    pp, qq = np.mgrid[0 : data4D.shape[0], 0 : data4D.shape[1]]
    no_points = np.sum(ROI)
    fitted_pos = np.zeros((2, no_points), dtype=np.float64)
    fitted_com = np.zeros((2, no_points), dtype=np.float64)

    pos_p = position[0]
    pos_q = position[1]
    corr_disk = st.util.make_circle(
        np.asarray(data4D.shape[0:2]), pos_p, pos_q, disk_size
    )
    sobel_corr_disk, _ = st.util.sobel(corr_disk)

    p_cen = np.zeros((data4D.shape[2], data4D.shape[3]), dtype=np.float64)
    q_cen = np.zeros((data4D.shape[2], data4D.shape[3]), dtype=np.float64)
    p_com = np.zeros((data4D.shape[2], data4D.shape[3]), dtype=np.float64)
    q_com = np.zeros((data4D.shape[2], data4D.shape[3]), dtype=np.float64)

    for ii in numba.prange(int(no_points)):
        cbed_image = data4D_ROI[:, :, ii]
        slm_image, _ = st.util.sobel(
            scnd.gaussian_filter(st.util.image_logarizer(cbed_image), 3)
        )
        slm_image[slm_image > med_val * np.median(slm_image)] = med_val * np.median(
            slm_image
        )
        slm_image[slm_image < np.median(slm_image) / med_val] = (
            np.median(slm_image) / med_val
        )
        corr_image = st.util.cross_corr(slm_image, sobel_corr_disk, hybridizer=0.25)

        fitted_disk_list = st.util.fit_gaussian2D_mask(
            corr_image, pos_p, pos_q, disk_size
        )
        fitted_center = fitted_disk_list[0:2] + (
            np.asarray((pos_p, pos_q))
            - 0.5 * (np.flip(np.asarray(np.shape(cbed_image))))
        )
        fitted_pos[0:2, ii] = fitted_center

        fitted_circle = st.util.make_circle(
            np.asarray(cbed_image.shape), fitted_center[0], fitted_center[1], disk_size
        )
        fitted_circle = fitted_circle.astype(bool)
        image_sum = np.sum(cbed_image[fitted_circle])
        com_pos_p = np.sum(cbed_image[fitted_circle] * pp[fitted_circle]) / image_sum
        com_pos_q = np.sum(cbed_image[fitted_circle] * qq[fitted_circle]) / image_sum

        fitted_com[0:2, ii] = np.asarray((com_pos_p, com_pos_q))

    p_cen[yy[ROI], xx[ROI]] = fitted_pos[0, :]
    q_cen[yy[ROI], xx[ROI]] = fitted_pos[1, :]
    p_com[yy[ROI], xx[ROI]] = fitted_com[0, :]
    q_com[yy[ROI], xx[ROI]] = fitted_com[1, :]

    return p_cen, q_cen, p_com, q_com


def log_sobel(pattern, med_factor=30, gauss_val=3):
    """
    Take the Log-Sobel of a pattern.

    Parameters
    ----------
    pattern:    ndarray
                Image on which Log-Sobel is to be performed
    med_factor: float
                Due to detector noise, some stray pixels may often be brighter
                than the background. This is used for damping any such pixels.
                Default is 30
    gauss_val:  float
                The standard deviation of the Gaussian filter applied to the
                logarithm of the CBED pattern. Default is 3

    Returns
    -------
    lsb_pattern: ndarray
                 Log-Sobel Filtered pattern

    Notes
    -----
    Generate the Sobel filtered pattern of the logarithm of
    a dataset. Compared to running the Sobel filter back on
    a log dataset, this takes care of somethings - notably
    a Gaussian blur is applied to the image, and Sobel spikes
    are removed when any values are too higher or lower than
    the median of the image. This is because real detector
    images often are very noisy.

    See Also
    --------
    nbed.log_sobel4D
    """
    pattern = 1 + st.util.image_normalizer(pattern)
    lsb_pattern, _ = st.util.sobel(
        scnd.gaussian_filter(st.util.image_logarizer(pattern), gauss_val)
    )
    lsb_pattern[lsb_pattern > med_factor * np.median(lsb_pattern)] = (
        np.median(lsb_pattern) * med_factor
    )
    lsb_pattern[lsb_pattern < np.median(lsb_pattern) / med_factor] = (
        np.median(lsb_pattern) / med_factor
    )
    return lsb_pattern
