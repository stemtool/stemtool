import numpy as np
import numba
import scipy.ndimage as scnd
import scipy.optimize as sio
import scipy.signal as scisig
import skimage.feature as skfeat
import matplotlib.colors as mplc
import matplotlib.pyplot as plt
import matplotlib as mpl
import stemtool as st
import matplotlib.offsetbox as mploff
import matplotlib.gridspec as mpgs
import matplotlib_scalebar.scalebar as mpss
import dask
import dask.array as da
import dask.distributed as dd
import warnings


def rotation_finder(image_orig, axis=0):
    """
    Angle Finder

    Parameters
    ----------
    image_orig: (2,2) shape ndarray
                Input Image
    axis:       int, optional
                Axis along which to perform sum

    Returns
    -------
    min_x: float
           Angle by which if the image is rotated
           by, the sum of the image along the axis
           specified is maximum

    Notes
    -----
    Uses an internal `angle_fun` to sum the intensity along
    a particular axis. This will give the angle at which that
    sum is highest along the axis specified.
    """

    def angle_fun(angle, axis=axis):
        rotated_image = scnd.rotate(image_orig, angle, order=5, reshape=False)
        rotsum = (-1) * (np.sum(rotated_image, axis))
        rotmin = np.amin(rotsum)
        return rotmin

    x0 = 90
    x = sio.minimize(angle_fun, x0)
    min_x = x.x
    return min_x


def rotate_and_center_ROI(data4D_ROI, rotangle, xcenter, ycenter):
    """
    Rotation Corrector

    Parameters
    ----------
    data4D_ROI: ndarray
                Region of interest of the 4D-STEM dataset in
                the form of ROI pixels (scanning), CBED_Y, CBED_x
    rotangle:   float
                angle in counter-clockwise direction to
                rotate individual CBED patterns
    xcenter:    float
                X pixel co-ordinate of center of mean pattern
    ycenter:    float
                Y pixel co-ordinate of center of mean pattern

    Returns
    -------
    corrected_ROI: ndarray
                   Each CBED pattern from the region of interest
                   first centered and then rotated along the center

    Notes
    -----
    We start by centering each 4D-STEM CBED pattern
    and then rotating the patterns with respect to the
    pattern center
    """
    data_size = np.asarray(np.shape(data4D_ROI))
    corrected_ROI = np.zeros_like(data4D_ROI)
    for ii in range(data4D_ROI.shape[0]):
        cbed_pattern = data4D_ROI[ii, :, :]
        moved_cbed = np.abs(
            st.util.move_by_phase(
                cbed_pattern,
                (-xcenter + (0.5 * data_size[-1])),
                (-ycenter + (0.5 * data_size[-2])),
            )
        )
        rotated_cbed = scnd.rotate(moved_cbed, rotangle, order=5, reshape=False)
        corrected_ROI[ii, :, :] = rotated_cbed
    return corrected_ROI


def data4Dto2D(data4D):
    """
    Convert 4D data to 2D data

    Parameters
    ----------
    data4D: ndarray of shape (4,4)
            the first two dimensions are Fourier
            space, while the next two dimensions
            are real space

    Returns
    -------
    data2D: ndarray of shape (2,2)
            Raveled 2D data where the
            first two dimensions are positions
            while the next two dimensions are spectra
    """
    data2D = np.transpose(data4D, (2, 3, 0, 1))
    data_shape = data2D.shape
    data2D.shape = (data_shape[0] * data_shape[1], data_shape[2] * data_shape[3])
    return data2D


@numba.jit(parallel=True, cache=True)
def resizer1D_numbaopt(data, res, N):
    M = data.size
    carry = 0
    m = 0
    for n in range(int(N)):
        data_sum = carry
        while ((m * N) - (n * M)) < M:
            data_sum += data[m]
            m += 1
        carry = (m - (n + 1) * (M / N)) * data[m - 1]
        data_sum -= carry
        res[n] = data_sum * (N / M)
    return res


@numba.jit(parallel=True, cache=True)
def resizer2D_numbaopt(data2D, resampled_x, resampled_f, sampling):
    data_shape = np.asarray(data2D.shape)
    sampled_shape = (np.round(data_shape / sampling)).astype(int)
    for yy in range(data_shape[0]):
        resampled_x[yy, :] = resizer1D_numbaopt(
            data2D[yy, :], resampled_x[yy, :], sampled_shape[1]
        )
    for xx in range(sampled_shape[1]):
        resampled_f[:, xx] = resizer1D_numbaopt(
            resampled_x[:, xx], resampled_f[:, xx], sampled_shape[0]
        )
    return resampled_f


def resizer1D(data, N):
    M = data.size
    carry = 0
    m = 0
    res = np.zeros(int(N))
    for n in range(int(N)):
        data_sum = carry
        while ((m * N) - (n * M)) < M:
            data_sum += data[m]
            m += 1
        carry = (m - (n + 1) * (M / N)) * data[m - 1]
        data_sum -= carry
        res[n] = data_sum * (N / M)
    return res


def resizer2D(data2D, sampling):
    sampling = np.asarray(sampling)
    if len(sampling) < 2:
        sampling = np.asarray((sampling[0], sampling[0]))
    cluster = dd.LocalCluster()
    client = dd.Client(cluster)
    data2D_scat = client.scatter(data2D, broadcast=True)
    data2D_dask = da.from_delayed(data2D_scat, data2D.shape, data2D.dtype)
    data_shape = np.asarray(data2D.shape)
    res_shape = (data_shape / sampling).astype(int)
    resampled_Y = []
    resampledXY = []
    for xx in range(data_shape[1]):
        interY = dask.delayed(st.nbed.resizer1D)(data2D_dask[:, xx], res_shape[0])
        interYarr = da.from_delayed(interY, (1, res_shape[0]), data2D.dtype)
        resampled_Y.append(interYarr)
    resampled_Y = da.concatenate(resampled_Y, axis=0)

    resampled_Y = da.transpose(resampled_Y, (1, 0))
    for yy in range(res_shape[0]):
        interX = dask.delayed(st.nbed.resizer1D)(resampled_Y[yy, :], res_shape[1])
        interXarr = da.from_delayed(interX, (1, res_shape[1]), data2D.dtype)
        resampledXY.append(interXarr)
    resampledXY = da.concatenate(resampledXY, axis=0)
    resampled_f = resampledXY.compute()
    cluster.close()
    return resampled_f


@numba.jit
def bin4D(data4D, bin_factor):
    """
    Bin 4D data in spectral dimensions

    Parameters
    ----------
    data4D:     ndarray of shape (4,4)
                the first two dimensions are Fourier
                space, while the next two dimensions
                are real space
    bin_factor: int
                Value by which to bin data

    Returns
    -------
    binned_data: ndarray of shape (4,4)
                 Data binned in the spectral dimensions

    Notes
    -----
    The data is binned in the first two dimensions - which are
    the Fourier dimensions using the internal numba functions
    `resizer2D_numbaopt` and `resizer1D_numbaopt`

    See Also
    --------
    resizer1D_numbaopt
    resizer2D_numbaopt
    """
    data4D_flat = np.reshape(
        data4D, (data4D.shape[0], data4D.shape[1], data4D.shape[2] * data4D.shape[3])
    )
    datashape = np.asarray(data4D_flat.shape)
    res_shape = np.copy(datashape)
    res_shape[0:2] = np.round(datashape[0:2] / bin_factor)
    data4D_res = np.zeros(res_shape.astype(int), dtype=data4D_flat.dtype)
    resampled_x = np.zeros((datashape[0], res_shape[1]), data4D_flat.dtype)
    resampled_f = np.zeros(res_shape[0:2], dtype=data4D_flat.dtype)
    for zz in range(data4D_flat.shape[-1]):
        data4D_res[:, :, zz] = resizer2D_numbaopt(
            data4D_flat[:, :, zz], resampled_x, resampled_f, bin_factor
        )
    binned_data = np.reshape(
        data4D_res,
        (resampled_f.shape[0], resampled_f.shape[1], data4D.shape[2], data4D.shape[3]),
    )
    return binned_data


def bin4D_dask(data4D, bin_factor, workers=4):
    """
    Bin 4D data in spectral dimensions,
    implemented in dask

    Parameters
    ----------
    data4D:     ndarray of shape (4,4)
                the first two dimensions are Fourier
                space, while the next two dimensions
                are real space
    bin_factor: int
                Value by which to bin data
    workers:    int, optional
                Number of dask workers. Default is
                4.

    Returns
    -------
    binned_data: ndarray of shape (4,4)
                 Data binned in the spectral dimensions

    Notes
    -----
    The data is binned in the first two dimensions - which are
    the Fourier dimensions using the internal numba functions
    `resizer2D_numbaopt` and `resizer1D_numbaopt`

    See Also
    --------
    resizer1D
    """
    cluster = dd.LocalCluster(n_workers=workers)
    client = dd.Client(cluster)
    data4D_scat = client.scatter(data4D, broadcast=True)
    data4D_dask = da.from_delayed(data4D_scat, data4D.shape, data4D.dtype)
    data4D_flat = da.reshape(
        data4D_dask,
        (data4D.shape[0], data4D.shape[1], data4D.shape[2] * data4D.shape[3]),
    )
    datashape = np.asarray(data4D_flat.shape)
    res_shape = np.copy(datashape)
    res_shape[0:2] = (datashape[0:2] / bin_factor).astype(int)
    res4D = []
    for zz in range(datashape[-1]):
        resampled_Y = []
        resampledXY = []
        for xx in range(datashape[1]):
            interY = dask.delayed(st.nbed.resizer1D)(
                data4D_flat[:, xx, zz], res_shape[0]
            )
            interYarr = da.from_delayed(interY, (1, res_shape[0]), data4D.dtype)
            resampled_Y.append(interYarr)
        resampled_Y = da.concatenate(resampled_Y, axis=0)

        resampled_Y = da.transpose(resampled_Y, (1, 0))
        for yy in range(res_shape[0]):
            interX = dask.delayed(st.nbed.resizer1D)(resampled_Y[yy, :], res_shape[1])
            interXarr = da.from_delayed(interX, (1, res_shape[1]), data4D.dtype)
            resampledXY.append(interXarr)
        resampledXY = da.concatenate(resampledXY, axis=0)
        res4D.append(resampledXY)
    res4D = da.stack(res4D, axis=0)
    res4D = da.transpose(res4D, (1, 2, 0))
    data4D_res = da.reshape(
        res4D, (res_shape[0], res_shape[1], data4D.shape[2], data4D.shape[3])
    )
    binned_data = data4D_res.compute()
    cluster.close()
    return binned_data


def test_aperture(pattern, center, radius, showfig=True):
    """
    Test an aperture position for Virtual DF image

    Parameters
    ----------
    pattern: ndarray of shape (2,2)
             Diffraction pattern, preferably the
             mean diffraction pattern for testing out
             the aperture location
    center:  ndarray of shape (1,2)
             Center of the circular aperture
    radius:  float
             Radius of the circular aperture
    showfig: bool, optional
             If showfig is True, then the image is
             displayed with the aperture overlaid

    Returns
    -------
    aperture: ndarray of shape (2,2)
              A matrix of the same size of the input image
              with zeros everywhere and ones where the aperture
              is supposed to be

    Notes
    -----
    Use the showfig option to visually test out the aperture
    location with varying parameters
    """
    center = np.asarray(center)
    yy, xx = np.mgrid[0 : pattern.shape[0], 0 : pattern.shape[1]]
    yy = yy - center[1]
    xx = xx - center[0]
    rr = ((yy ** 2) + (xx ** 2)) ** 0.5
    aperture = np.asarray(rr <= radius, dtype=np.double)
    if showfig:
        plt.figure(figsize=(15, 15))
        plt.imshow(st.util.image_normalizer(pattern) + aperture, cmap="Spectral")
        plt.scatter(center[0], center[1], c="w", s=25)
    return aperture


def aperture_image(data4D, center, radius):
    """
    Generate Virtual DF image for a given aperture

    Parameters
    ----------
    data4D: ndarray of shape (4,4)
            the first two dimensions are Fourier
            space, while the next two dimensions
            are real space
    center: ndarray of shape (1,2)
            Center of the circular aperture
    radius: float
            Radius of the circular aperture

    Returns
    -------
    df_image: ndarray of shape (2,2)
              Generated virtual dark field image
              from the aperture and 4D data

    Notes
    -----
    We generate the aperture first, and then make copies
    of the aperture to generate a 4D dataset of the same
    size as the 4D data. Then we do an element wise
    multiplication of this aperture 4D data with the 4D data
    and then sum it along the two Fourier directions.
    """
    center = np.array(center)
    yy, xx = np.mgrid[0 : data4D.shape[0], 0 : data4D.shape[1]]
    yy = yy - center[1]
    xx = xx - center[0]
    rr = ((yy ** 2) + (xx ** 2)) ** 0.5
    aperture = np.asarray(rr <= radius, dtype=data4D.dtype)
    apt_copy = np.empty(
        (data4D.shape[2], data4D.shape[3]) + aperture.shape, dtype=data4D.dtype
    )
    apt_copy[:] = aperture
    apt_copy = np.transpose(apt_copy, (2, 3, 0, 1))
    apt_mult = apt_copy * data4D
    df_image = np.sum(np.sum(apt_mult, axis=0), axis=0)
    return df_image


def ROI_from_image(image, med_val, style="over", showfig=True):
    if style == "over":
        ROI = np.asarray(image > (med_val * np.median(image)), dtype=np.double)
    else:
        ROI = np.asarray(image < (med_val * np.median(image)), dtype=np.double)
    if showfig:
        plt.figure(figsize=(15, 15))
        plt.imshow(ROI + st.util.image_normalizer(image), cmap="viridis")
        plt.title("ROI overlaid")
    ROI = ROI.astype(bool)
    return ROI


def custom_detector(
    data4D, inner, outer=0, center=(0, 0), mrad_calib=0, get_detector=False
):
    """
    Return STEM image from detector values

    Parameters
    ----------
    data4D:       ndarray
                  the first two dimensions are Fourier
                  space, while the next two dimensions
                  are real space
    inner:        float
                  The inner collection angle in Fourier space
                  in pixels
    outer:        float, optional
                  The inner collection angle in Fourier space
                  in pixels. Default is 0
    center:       tuple, optional
                  The center of the 4D-STEM pattern in Fourier
                  space. Default is (0, 0)
    mrad_calib:   float, optional
                  Calibration of the Fourier space. Default
                  is 0.
    get_detector: bool, optional
                  Get the detector array if set to True. Default is
                  False

    Returns
    -------
    data_det: ndarray
              The STEM image from the detector region chosen
    detector: ndarray, optional
              The boolean ndarray if get_detector is True

    Notes
    -----
    Based on the inner and outer collection angle the a STEM
    image is generated from the 4D-STEM dataset. We assume that
    the first two dimensions are the Fourier space, while the
    next two dimensions are real space scanning dimensions.
    """
    center = np.asarray(center)
    if center[0] == 0:
        center[0] = 0.5 * data4D.shape[0]
    if center[1] == 0:
        center[1] = 0.5 * data4D.shape[1]
    yy, xx = np.mgrid[0 : data4D.shape[0], 0 : data4D.shape[1]]
    if mrad_calib > 0:
        inner = inner * mrad_calib
        outer = outer * mrad_calib
        center = center * mrad_calib
        yy = yy * mrad_calib
        xx = xx * mrad_calib
    yy = yy - center[0]
    xx = xx - center[1]
    rr = ((yy ** 2) + (xx ** 2)) ** 0.5
    if outer == 0:
        outer = np.amax(rr)
    detector = np.logical_and((rr > inner), (rr < outer))
    data_det = np.sum(data4D[detector, :, :], axis=0)
    if get_detector:
        return data_det, detector
    else:
        return data_det


@numba.jit
def colored_mcr(conc_data, data_shape):
    no_spectra = np.shape(conc_data)[1]
    color_hues = np.arange(no_spectra, dtype=np.float64) / no_spectra
    norm_conc = (conc_data - np.amin(conc_data)) / (
        np.amax(conc_data) - np.amin(conc_data)
    )
    saturation_matrix = np.ones(data_shape, dtype=np.float64)
    hsv_calc = np.zeros((data_shape[0], data_shape[1], 3), dtype=np.float64)
    rgb_calc = np.zeros((data_shape[0], data_shape[1], 3), dtype=np.float64)
    hsv_calc[:, :, 1] = saturation_matrix
    for ii in range(no_spectra):
        conc_image = (np.reshape(norm_conc[:, ii], data_shape)).astype(np.float64)
        hsv_calc[:, :, 0] = saturation_matrix * color_hues[ii]
        hsv_calc[:, :, 2] = conc_image
        rgb_calc = rgb_calc + mplc.hsv_to_rgb(hsv_calc)
    rgb_image = rgb_calc / np.amax(rgb_calc)
    return rgb_image


@numba.jit
def fit_nbed_disks(corr_image, disk_size, positions, diff_spots, nan_cutoff=0):
    """
    Disk Fitting algorithm for a single NBED pattern

    Parameters
    ----------
    corr_image: ndarray of shape (2,2)
                The cross-correlated image of the NBED that
                will be fitted
    disk_size:  float
                Size of each NBED disks in pixels
    positions:  ndarray of shape (n,2)
                X and Y positions where n is the number of positions.
                These are the initial guesses that will be refined
    diff_spots: ndarray of shape (n,2)
                a and b Miller indices corresponding to the
                disk positions
    nan_cutoff: float, optional
                Optional parameter that is used for thresholding disk
                fits. If the intensity ratio is below the threshold
                the position will not be fit. Default value is 0

    Returns
    -------
    fitted_disk_list: ndarray of shape (n,2)
                      Sub-pixel precision Gaussian fitted disk
                      locations. If nan_cutoff is greater than zero, then
                      only the positions that are greater than the threshold
                      are returned.
    center_position:  ndarray of shape (1,2)
                      Location of the central (0,0) disk
    fit_deviation:    ndarray of shape (1,2)
                      Standard deviation of the X and Y disk fits given as pixel
                      ratios
    lcbed:            ndarray of shape (2,2)
                      Matrix defining the Miller indices axes

    Notes
    -----
    Every disk position is fitted with a 2D Gaussian by cutting off a circle
    of the size of disk_size around the initial poistions. If nan-cutoff is above
    zero then only the locations inside this cutoff where the maximum pixel intensity
    is (1+nan_cutoff) times the median pixel intensity will be fitted. Use this
    parameter carefully, because in some cases this may result in no disks being fitted
    and the program throwing weird errors at you.
    """
    warnings.filterwarnings("ignore")
    no_pos = int(np.shape(positions)[0])
    diff_spots = np.asarray(diff_spots, dtype=np.float64)
    fitted_disk_list = np.zeros_like(positions)
    yy, xx = np.mgrid[0 : (corr_image.shape[0]), 0 : (corr_image.shape[1])]
    for ii in range(no_pos):
        posx = positions[ii, 0]
        posy = positions[ii, 1]
        reg = ((yy - posy) ** 2) + ((xx - posx) ** 2) <= (disk_size ** 2)
        peak_ratio = np.amax(corr_image[reg]) / np.median(corr_image[reg])
        if peak_ratio < (1 + nan_cutoff):
            fitted_disk_list[ii, 0:2] = np.nan
        else:
            par = st.util.fit_gaussian2D_mask(corr_image, posx, posy, disk_size)
            fitted_disk_list[ii, 0:2] = par[0:2]
    nancount = np.int(np.sum(np.isnan(fitted_disk_list)) / 2)
    if nancount == no_pos:
        center_position = np.nan * np.ones((1, 2))
        fit_deviation = np.nan
        lcbed = np.nan
    else:
        diff_spots = (diff_spots[~np.isnan(fitted_disk_list)]).reshape(
            (no_pos - nancount), 2
        )
        fitted_disk_list = (fitted_disk_list[~np.isnan(fitted_disk_list)]).reshape(
            (no_pos - nancount), 2
        )
        disk_locations = np.copy(fitted_disk_list)
        disk_locations[:, 1] = (-1) * disk_locations[:, 1]
        center = disk_locations[
            np.logical_and((diff_spots[:, 0] == 0), (diff_spots[:, 1] == 0)), :
        ]
        if center.shape[0] > 0:
            cx = center[0, 0]
            cy = center[0, 1]
            center_position = np.asarray((cx, -cy), dtype=np.float64)
            if (nancount / no_pos) < 0.5:
                disk_locations[:, 0:2] = disk_locations[:, 0:2] - np.asarray(
                    (cx, cy), dtype=np.float64
                )
                lcbed, _, _, _ = np.linalg.lstsq(diff_spots, disk_locations, rcond=None)
                calc_points = np.matmul(diff_spots, lcbed)
                stdx = np.std(
                    np.divide(
                        disk_locations[np.where(calc_points[:, 0] != 0), 0],
                        calc_points[np.where(calc_points[:, 0] != 0), 0],
                    )
                )
                stdy = np.std(
                    np.divide(
                        disk_locations[np.where(calc_points[:, 1] != 0), 1],
                        calc_points[np.where(calc_points[:, 1] != 0), 1],
                    )
                )
                fit_deviation = np.asarray((stdx, stdy), dtype=np.float64)
            else:
                fit_deviation = np.nan
                lcbed = np.nan
        else:
            center_position = np.nan
            fit_deviation = np.nan
            lcbed = np.nan
    return fitted_disk_list, center_position, fit_deviation, lcbed


@numba.jit
def strain_in_ROI(
    data4D,
    ROI,
    center_disk,
    disk_list,
    pos_list,
    reference_axes=0,
    med_factor=10,
    gauss_val=3,
    hybrid_cc=0.1,
    nan_cutoff=0.5,
):
    """
    Get strain from a region of interest

    Parameters
    ----------
    data4D:         ndarray
                    This is a 4D dataset where the first two dimensions
                    are the diffraction dimensions and the next two
                    dimensions are the scan dimensions
    ROI:            ndarray of dtype bool
                    Region of interest
    center_disk:    ndarray
                    The blank diffraction disk template where
                    it is 1 inside the circle and 0 outside
    disk_list:      ndarray of shape (n,2)
                    X and Y positions where n is the number of positions.
                    These are the initial guesses that will be refined
    pos_list:       ndarray of shape (n,2)
                    a and b Miller indices corresponding to the
                    disk positions
    reference_axes: ndarray, optional
                    The unit cell axes from the reference region. Strain is
                    calculated by comapring the axes at a scan position with
                    the reference axes values. If it is 0, then the average
                    NBED axes will be calculated and will be used as the
                    reference axes.
    med_factor:     float, optional
                    Due to detector noise, some stray pixels may often be brighter
                    than the background. This is used for damping any such pixels.
                    Default is 30
    gauss_val:      float, optional
                    The standard deviation of the Gaussian filter applied to the
                    logarithm of the CBED pattern. Default is 3
    hybrid_cc:      float, optional
                    Hybridization parameter to be used for cross-correlation.
                    Default is 0.1
    nan_cutoff:     float, optional
                    Parameter that is used for thresholding disk
                    fits. If the intensity ratio is below the threshold
                    the position will not be fit. Default value is 0.5

    Returns
    -------
    e_xx_map: ndarray
              Strain in the xx direction in the region of interest
    e_xy_map: ndarray
              Strain in the xy direction in the region of interest
    e_th_map: ndarray
              Angular strain in the region of interest
    e_yy_map: ndarray
              Strain in the yy direction in the region of interest
    fit_std:  ndarray
              x and y deviations in axes fitting for the scan points

    Notes
    -----
    At every scan position, the diffraction disk is filtered by first taking
    the log of the CBED pattern, and then by applying a Gaussian filter.
    Following this the Sobel of the filtered dataset is calculated.
    The intensity of the Sobel, Gaussian and Log filtered CBED data is then
    inspected for outlier pixels. If pixel intensities are higher or lower than
    a threshold of the median pixel intensity, they are replaced by the threshold
    value. This is then hybrid cross-correlated with the Sobel magnitude of the
    template disk. If the pattern axes return a numerical value, then the strain
    is calculated for that scan position, else it is NaN
    """
    warnings.filterwarnings("ignore")
    # Calculate needed values
    scan_y, scan_x = np.mgrid[0 : data4D.shape[2], 0 : data4D.shape[3]]
    data4D_ROI = data4D[:, :, scan_y[ROI], scan_x[ROI]]
    no_of_disks = data4D_ROI.shape[-1]
    disk_size = (np.sum(st.util.image_normalizer(center_disk)) / np.pi) ** 0.5
    i_matrix = (np.eye(2)).astype(np.float64)
    sobel_center_disk, _ = st.util.sobel(center_disk)
    # Initialize matrices
    e_xx_ROI = np.nan * (np.ones(no_of_disks, dtype=np.float64))
    e_xy_ROI = np.nan * (np.ones(no_of_disks, dtype=np.float64))
    e_th_ROI = np.nan * (np.ones(no_of_disks, dtype=np.float64))
    e_yy_ROI = np.nan * (np.ones(no_of_disks, dtype=np.float64))
    fit_std = np.nan * (np.ones((no_of_disks, 2), dtype=np.float64))
    e_xx_map = np.nan * np.ones_like(scan_y)
    e_xy_map = np.nan * np.ones_like(scan_y)
    e_th_map = np.nan * np.ones_like(scan_y)
    e_yy_map = np.nan * np.ones_like(scan_y)
    # Calculate for mean CBED if no reference
    # axes present
    if np.size(reference_axes) < 2:
        mean_cbed = np.mean(data4D_ROI, axis=-1)
        sobel_lm_cbed, _ = st.util.sobel(st.util.image_logarizer(mean_cbed))
        sobel_lm_cbed[
            sobel_lm_cbed > med_factor * np.median(sobel_lm_cbed)
        ] = np.median(sobel_lm_cbed)
        lsc_mean = st.util.cross_corr(
            sobel_lm_cbed, sobel_center_disk, hybridizer=hybrid_cc
        )
        _, _, _, mean_axes = fit_nbed_disks(lsc_mean, disk_size, disk_list, pos_list)
        inverse_axes = np.linalg.inv(mean_axes)
    else:
        inverse_axes = np.linalg.inv(reference_axes)
    for ii in range(int(no_of_disks)):
        pattern = data4D_ROI[:, :, ii]
        sobel_log_pattern, _ = st.util.sobel(
            scnd.gaussian_filter(st.util.image_logarizer(pattern), gauss_val)
        )
        sobel_log_pattern[
            sobel_log_pattern > med_factor * np.median(sobel_log_pattern)
        ] = (np.median(sobel_log_pattern) * med_factor)
        sobel_log_pattern[
            sobel_log_pattern < np.median(sobel_log_pattern) / med_factor
        ] = (np.median(sobel_log_pattern) / med_factor)
        lsc_pattern = st.util.cross_corr(
            sobel_log_pattern, sobel_center_disk, hybridizer=hybrid_cc
        )
        _, _, std, pattern_axes = fit_nbed_disks(
            lsc_pattern, disk_size, disk_list, pos_list, nan_cutoff
        )
        if ~(np.isnan(np.ravel(pattern_axes))[0]):
            fit_std[ii, :] = std
            t_pattern = np.matmul(pattern_axes, inverse_axes)
            s_pattern = t_pattern - i_matrix
            e_xx_ROI[ii] = -s_pattern[0, 0]
            e_xy_ROI[ii] = -(s_pattern[0, 1] + s_pattern[1, 0])
            e_th_ROI[ii] = s_pattern[0, 1] - s_pattern[1, 0]
            e_yy_ROI[ii] = -s_pattern[1, 1]
    e_xx_map[ROI] = e_xx_ROI
    e_xx_map[np.isnan(e_xx_map)] = 0
    e_xx_map = scnd.gaussian_filter(e_xx_map, 1)
    e_xy_map[ROI] = e_xy_ROI
    e_xy_map[np.isnan(e_xy_map)] = 0
    e_xy_map = scnd.gaussian_filter(e_xy_map, 1)
    e_th_map[ROI] = e_th_ROI
    e_th_map[np.isnan(e_th_map)] = 0
    e_th_map = scnd.gaussian_filter(e_th_map, 1)
    e_yy_map[ROI] = e_yy_ROI
    e_yy_map[np.isnan(e_yy_map)] = 0
    e_yy_map = scnd.gaussian_filter(e_yy_map, 1)
    return e_xx_map, e_xy_map, e_th_map, e_yy_map, fit_std


@numba.jit
def strain_log(
    data4D_ROI, center_disk, disk_list, pos_list, reference_axes=0, med_factor=10
):
    warnings.filterwarnings("ignore")
    # Calculate needed values
    no_of_disks = data4D_ROI.shape[-1]
    disk_size = (np.sum(center_disk) / np.pi) ** 0.5
    i_matrix = (np.eye(2)).astype(np.float64)
    # Initialize matrices
    e_xx_log = np.zeros(no_of_disks, dtype=np.float64)
    e_xy_log = np.zeros(no_of_disks, dtype=np.float64)
    e_th_log = np.zeros(no_of_disks, dtype=np.float64)
    e_yy_log = np.zeros(no_of_disks, dtype=np.float64)
    # Calculate for mean CBED if no reference
    # axes present
    if np.size(reference_axes) < 2:
        mean_cbed = np.mean(data4D_ROI, axis=-1)
        log_cbed, _ = st.util.image_logarizer(mean_cbed)
        log_cc_mean = st.util.cross_corr(log_cbed, center_disk, hybridizer=0.1)
        _, _, mean_axes = fit_nbed_disks(log_cc_mean, disk_size, disk_list, pos_list)
        inverse_axes = np.linalg.inv(mean_axes)
    else:
        inverse_axes = np.linalg.inv(reference_axes)
    for ii in range(int(no_of_disks)):
        pattern = data4D_ROI[:, :, ii]
        log_pattern, _ = st.util.image_logarizer(pattern)
        log_cc_pattern = st.util.cross_corr(log_pattern, center_disk, hybridizer=0.1)
        _, _, pattern_axes = fit_nbed_disks(
            log_cc_pattern, disk_size, disk_list, pos_list
        )
        t_pattern = np.matmul(pattern_axes, inverse_axes)
        s_pattern = t_pattern - i_matrix
        e_xx_log[ii] = -s_pattern[0, 0]
        e_xy_log[ii] = -(s_pattern[0, 1] + s_pattern[1, 0])
        e_th_log[ii] = s_pattern[0, 1] - s_pattern[1, 0]
        e_yy_log[ii] = -s_pattern[1, 1]
    return e_xx_log, e_xy_log, e_th_log, e_yy_log


@numba.jit
def strain_oldstyle(data4D_ROI, center_disk, disk_list, pos_list, reference_axes=0):
    warnings.filterwarnings("ignore")
    # Calculate needed values
    no_of_disks = data4D_ROI.shape[-1]
    disk_size = (np.sum(center_disk) / np.pi) ** 0.5
    i_matrix = (np.eye(2)).astype(np.float64)
    # Initialize matrices
    e_xx_ROI = np.zeros(no_of_disks, dtype=np.float64)
    e_xy_ROI = np.zeros(no_of_disks, dtype=np.float64)
    e_th_ROI = np.zeros(no_of_disks, dtype=np.float64)
    e_yy_ROI = np.zeros(no_of_disks, dtype=np.float64)
    # Calculate for mean CBED if no reference
    # axes present
    if np.size(reference_axes) < 2:
        mean_cbed = np.mean(data4D_ROI, axis=-1)
        cc_mean = st.util.cross_corr(mean_cbed, center_disk, hybridizer=0.1)
        _, _, mean_axes = fit_nbed_disks(cc_mean, disk_size, disk_list, pos_list)
        inverse_axes = np.linalg.inv(mean_axes)
    else:
        inverse_axes = np.linalg.inv(reference_axes)
    for ii in range(int(no_of_disks)):
        pattern = data4D_ROI[:, :, ii]
        cc_pattern = st.util.cross_corr(pattern, center_disk, hybridizer=0.1)
        _, _, pattern_axes = fit_nbed_disks(cc_pattern, disk_size, disk_list, pos_list)
        t_pattern = np.matmul(pattern_axes, inverse_axes)
        s_pattern = t_pattern - i_matrix
        e_xx_ROI[ii] = -s_pattern[0, 0]
        e_xy_ROI[ii] = -(s_pattern[0, 1] + s_pattern[1, 0])
        e_th_ROI[ii] = s_pattern[0, 1] - s_pattern[1, 0]
        e_yy_ROI[ii] = -s_pattern[1, 1]
    return e_xx_ROI, e_xy_ROI, e_th_ROI, e_yy_ROI


def ROI_strain_map(strain_ROI, ROI):
    """
    Convert the strain in the ROI array to a strain map
    """
    strain_map = np.zeros_like(ROI, dtype=np.float64)
    strain_map[ROI] = (strain_ROI).astype(np.float64)
    return strain_map


@numba.jit(cache=True, parallel=True)
def logarizer4D(data4D, scan_dims, bit_depth=32):
    """
    Take the Logarithm of a 4D dataset.

    Parameters
    ----------
    data4D:     ndarray
                4D dataset whose CBED patterns will be filtered
    scan_dims:  tuple
                Scan dimensions. If your scanning pixels are for
                example the first two dimensions specify it as (0,1)
                Will be converted to numpy array so pass tuple only
    med_factor: float, optional
                Due to detector noise, some stray pixels may often
                be brighter than the background. This is used for
                damping any such pixels. Default is 30

    Returns
    -------
    data_log: ndarray
              4D dataset where each CBED pattern has been log

    Notes
    -----
    Generate the logarithm of the 4D dataset.

    See Also
    --------
    log_sobel4D
    util.image_logarizer
    """
    bit_max = 2 ** bit_depth
    scan_dims = np.asarray(scan_dims)
    scan_dims[scan_dims < 0] = 4 + scan_dims[scan_dims < 0]
    sum_dims = np.sum(scan_dims)
    if sum_dims < 2:
        data4D = np.transpose(data4D, (2, 3, 0, 1))
    data_log = np.zeros_like(data4D, dtype=np.float32)
    for jj in range(data4D.shape[int(scan_dims[1])]):
        for ii in range(data4D.shape[int(scan_dims[0])]):
            pattern = st.util.image_normalizer(data4D[:, :, ii, jj])
            pattern = 1 + ((bit_max - 1) * pattern)
            data_log[:, :, ii, jj] = np.log2(pattern)
    if sum_dims < 2:
        data_log = np.transpose(data_log, (2, 3, 0, 1))
    return data_log


@numba.jit(cache=True, parallel=True)
def log_sobel4D(data4D, scan_dims, med_factor=30, gauss_val=3):
    """
    Take the Log-Sobel of a pattern.

    Parameters
    ----------
    data4D:     ndarray
                4D dataset whose CBED patterns will be filtered
    scan_dims:  tuple
                Scan dimensions. If your scanning pixels are for
                example the first two dimensions specify it as (0,1)
                Will be converted to numpy array so pass tuple only
    med_factor: float, optional
                Due to detector noise, some stray pixels may often
                be brighter than the background. This is used for
                damping any such pixels. Default is 30
    gauss_val:  float, optional
                The standard deviation of the Gaussian filter applied
                to the logarithm of the CBED pattern. Default is 3

    Returns
    -------
    data_lsb: ndarray
              4D dataset where each CBED pattern has been log
              Sobel filtered

    Notes
    -----
    Generate the Sobel filtered pattern of the logarithm of
    a dataset. Compared to running the Sobel filter back on
    a log dataset, this takes care of somethings - notably
    a Gaussian blur is applied to the image, and Sobel spikes
    are removed when any values are too higher or lower than
    the median of the image. This is because real detector
    images often are very noisy. This code generates the filtered
    CBED at every scan position, and is dimension agnostic, in
    that your CBED dimensions can either be the first two or last
    two - just specify the dimensions. Also if loops weirdly need
    to be outside the for loops - this is a numba feature (bug?)
    Small change - made the Sobel matrix order 5 rather than 3

    See Also
    --------
    logarizer4D
    dpc.log_sobel
    """
    scan_dims = np.asarray(scan_dims)
    scan_dims[scan_dims < 0] = 4 + scan_dims[scan_dims < 0]
    sum_dims = np.sum(scan_dims)
    if sum_dims < 2:
        data4D = np.transpose(data4D, (2, 3, 0, 1))
    data_lsb = np.zeros_like(data4D, dtype=np.float)
    for jj in range(data4D.shape[int(scan_dims[1])]):
        for ii in range(data4D.shape[int(scan_dims[0])]):
            pattern = data4D[:, :, ii, jj]
            pattern = 1000 * (1 + st.util.image_normalizer(pattern))
            lsb_pattern, _ = st.util.sobel(
                scnd.gaussian_filter(st.util.image_logarizer(pattern), gauss_val), 5
            )
            lsb_pattern[lsb_pattern > med_factor * np.median(lsb_pattern)] = (
                np.median(lsb_pattern) * med_factor
            )
            lsb_pattern[lsb_pattern < np.median(lsb_pattern) / med_factor] = (
                np.median(lsb_pattern) / med_factor
            )
            data_lsb[:, :, ii, jj] = lsb_pattern
    if sum_dims < 2:
        data_lsb = np.transpose(data_lsb, (2, 3, 0, 1))
    return data_lsb


def spectra_finder(data4D, yvals, xvals):
    spectra_data = np.ravel(
        np.mean(
            data4D[:, :, yvals[0] : yvals[1], xvals[0] : xvals[1]],
            axis=(-1, -2),
            dtype=np.float64,
        )
    )
    data_im = np.sum(data4D, axis=(0, 1))
    data_im = (data_im - np.amin(data_im)) / (np.amax(data_im) - np.amin(data_im))
    overlay = np.zeros_like(data_im)
    overlay[yvals[0] : yvals[1], xvals[0] : xvals[1]] = 1
    return spectra_data, 0.5 * (data_im + overlay)


def sort_edges(edge_map, edge_distance=5):
    yV, xV = np.mgrid[0 : np.shape(edge_map)[0], 0 : np.shape(edge_map)[1]]
    yy = yV[edge_map]
    xx = xV[edge_map]
    no_points = np.size(yy)
    points = np.arange(no_points)
    point_list = np.transpose(np.asarray((yV[edge_map], xV[edge_map])))
    truth_list = np.zeros((no_points, 2), dtype=bool)
    edge_list_1 = np.zeros((no_points, 2))
    point_number = 0
    edge_list_1[int(point_number), 0:2] = np.asarray((yy[0], xx[0]))
    truth_list[int(point_number), 0:2] = True
    edge_points = 1
    for _ in np.arange(no_points):
        last_yy = edge_list_1[int(edge_points - 1), 0]
        last_xx = edge_list_1[int(edge_points - 1), 1]
        other_points = np.reshape(
            point_list[~truth_list], (int(no_points - edge_points), 2)
        )
        dist_vals = (
            ((other_points[:, 0] - last_yy) ** 2)
            + ((other_points[:, 1] - last_xx) ** 2)
        ) ** 0.5
        min_dist = np.amin(dist_vals)
        if min_dist < edge_distance:
            n_yy = other_points[dist_vals == min_dist, 0][0]
            n_xx = other_points[dist_vals == min_dist, 1][0]
            point_number = points[
                (point_list[:, 0] == n_yy) & (point_list[:, 1] == n_xx)
            ][0]
            edge_list_1[int(edge_points), 0:2] = np.asarray((n_yy, n_xx))
            truth_list[int(point_number), 0:2] = True
            edge_points = edge_points + 1.0
    list_1 = np.reshape(point_list[truth_list], (int(edge_points), 2))
    list_2 = np.reshape(point_list[~truth_list], (int(no_points - edge_points), 2))
    edge1 = np.zeros_like(edge_map)
    edge1[list_1[:, 0], list_1[:, 1]] = 1
    edge2 = np.zeros_like(edge_map)
    edge2[list_2[:, 0], list_2[:, 1]] = 1
    edge1_sum = np.sum(edge1)
    edge2_sum = np.sum(edge2)
    if edge1_sum > edge2_sum:
        outer_edge = np.copy(edge1)
        inner_edge = np.copy(edge2)
    else:
        outer_edge = np.copy(edge2)
        inner_edge = np.copy(edge1)
    return outer_edge, inner_edge


@numba.jit
def get_inside(edges, cutoff=0.95):
    big_size = (2.5 * np.asarray(edges.shape)).astype(int)
    starter = (0.5 * (big_size - np.asarray(edges.shape))).astype(int)
    bigger_aa = np.zeros(big_size)
    bigger_aa[
        starter[0] : starter[0] + edges.shape[0],
        starter[1] : starter[1] + edges.shape[1],
    ] = edges
    aa1 = bigger_aa.astype(bool)
    aa2 = (np.fliplr(bigger_aa)).astype(bool)
    yy, xx = np.mgrid[0 : big_size[0], 0 : big_size[1]]
    positions = np.zeros((bigger_aa.size, 2), dtype=int)
    positions[:, 0] = np.ravel(yy)
    positions[:, 1] = np.ravel(xx)
    yy_aa1 = yy[aa1]
    xx_aa1 = xx[aa1]
    yy_aa2 = yy[aa2]
    xx_aa2 = xx[aa2]
    ang_range1 = np.zeros_like(yy, dtype=np.float)
    ang_range2 = np.zeros_like(yy, dtype=np.float)
    for ii in range(len(positions)):
        angles1 = (180 / np.pi) * np.arctan2(
            yy_aa1 - positions[ii, 0], xx_aa1 - positions[ii, 1]
        )
        ang_range1[positions[ii, 0], positions[ii, 1]] = np.amax(angles1) - np.amin(
            angles1
        )
    for jj in range(len(positions)):
        angles2 = (180 / np.pi) * np.arctan2(
            yy_aa2 - positions[jj, 0], xx_aa2 - positions[jj, 1]
        )
        ang_range2[positions[jj, 0], positions[jj, 1]] = np.amax(angles2) - np.amin(
            angles2
        )
    ang_range2 = np.fliplr(ang_range2)
    ang_range = np.logical_and(
        ang_range1 > cutoff * np.amax(ang_range1),
        ang_range2 > cutoff * np.amax(ang_range2),
    )
    real_ang_range = np.zeros_like(edges, dtype=bool)
    real_ang_range = ang_range[
        starter[0] : starter[0] + edges.shape[0],
        starter[1] : starter[1] + edges.shape[1],
    ]
    return real_ang_range


def sobel_filter(image, med_filter=50):
    ls_image, _ = st.util.sobel(st.util.image_logarizer(image))
    ls_image[ls_image > (med_filter * np.median(ls_image))] = med_filter * np.median(
        ls_image
    )
    ls_image[ls_image < (np.median(ls_image) / med_filter)] = (
        np.median(ls_image) / med_filter
    )
    return ls_image


def peak_prominence(peak_pos, peak_im, fit_radius):
    prom_y, prom_x = np.mgrid[0 : peak_im.shape[0], 0 : peak_im.shape[1]]
    prom_r = ((prom_y - peak_pos[0]) ** 2) + ((prom_x - peak_pos[1]) ** 2)
    prom_vals = peak_im[prom_r < (fit_radius ** 2)]
    prom_peak = peak_im[
        int(peak_pos[0] - 1) : int(peak_pos[0] + 1),
        int(peak_pos[1] - 1) : int(peak_pos[1] + 1),
    ]
    prominence = np.mean(prom_peak) - np.mean(prom_vals)
    return prominence


def strain4D_general(
    data4D,
    disk_radius,
    ROI=0,
    prom_val=0,
    disk_center=np.nan,
    max_radius=0,
    rotangle=0,
    med_factor=30,
    gauss_val=3,
    hybrid_cc=0.2,
    gblur=True,
    max_strain=0.1,
    take_median=True,
):
    """
    Get strain from a ROI without the need for
    specifying Miller indices of diffraction spots

    Parameters
    ----------
    data4D:      ndarray
                 This is a 4D dataset where the first two dimensions
                 are the diffraction dimensions and the next two
                 dimensions are the scan dimensions
    disk_radius: float
                 Radius in pixels of the diffraction disks
    ROI:         ndarray, optional
                 Region of interest. If no ROI is passed then the entire
                 scan region is the ROI
    prom_val:    float, optional
                 Minimum prominence value to use to use the peak for
                 strain mapping. Default is 0
    disk_center: tuple, optional
                 Location of the center of the diffraction disk - closest to
                 the <000> undiffracted beam
    max_radius:  float, optional
                 Maximum distance from the center to use for calculation. Default
                 is 0, when all distances are considered.
    rotangle:    float, optional
                 Angle of rotation of the CBED with respect to the optic axis
                 This must be in degrees
    med_factor:  float, optional
                 Due to detector noise, some stray pixels may often be brighter
                 than the background. This is used for damping any such pixels.
                 Default is 30
    gauss_val:   float, optional
                 The standard deviation of the Gaussian filter applied to the
                 logarithm of the CBED pattern. Default is 3
    hybrid_cc:   float, optional
                 Hybridization parameter to be used for cross-correlation.
                 Default is 0.1
    gblur:       bool, optional
                 If gblur is on, the strain maps are blurred by a single
                 pixel.
                 Default is True.
    max_strain:  float, optional
                 Tamp strain value above this value.
                 Default is 0.1
    take_median: bool, optional
                 If True, the Mean_CBED is the median diffraction pattern in the
                 ROI. If False, it is the mean diffraction patter.
                 Default is True

    Returns
    -------
    e_xx_map:  ndarray
               Strain in the xx direction in the region of interest
    e_xy_map:  ndarray
               Strain in the xy direction in the region of interest
    e_th_map:  ndarray
               Angular strain in the region of interest
    e_yy_map:  ndarray
               Strain in the yy direction in the region of interest
    newROI:    ndarray
               New reduced ROI where the prominence of the higher
               order diffraction disks is above prom_val
    new_pos:   ndarray
               List of all the higher order peak positions with
               respect to the central disk for all positions in the ROI
    distances: ndarray
               List of distances in the newly calculated ROI from the
               particle edge.

    Notes
    -----
    We first of all calculate the preconditioned data (log + Sobel filtered)
    for every CBED pattern in the ROI. Then the mean preconditioned
    pattern is calculated and cross-correlated with the Sobel template. The disk
    positions are as peaks in this cross-correlated pattern, with the central
    disk the one closest to the center of the CBED pattern. Using that insight
    the distances of the higher order diffraction disks are calculated with respect
    to the central transmitted beam. This is then performed for all other CBED
    patterns. The calculated higher order disk locations are then compared to the
    higher order disk locations for the median pattern to generate strain maps.
    However, sometimes the ROI may contain points where there is no diffraction
    pattern actually. To prevent picking such points and generating erroneous results,
    we calculate the peak prominence of every higher order diffraction spot, and only
    if they are more prominent than `prom_val` they will be chosen. If `prom_val` is
    zero, then all peaks are chosen.
    """

    rotangle = np.deg2rad(rotangle)
    rotmatrix = np.asarray(
        ((np.cos(rotangle), -np.sin(rotangle)), (np.sin(rotangle), np.cos(rotangle)))
    )
    diff_y, diff_x = np.mgrid[0 : data4D.shape[0], 0 : data4D.shape[1]]
    if np.isnan(np.mean(disk_center)):
        disk_center = np.asarray(np.shape(diff_y)) / 2
    else:
        disk_center = np.asarray(disk_center)

    radiating = ((diff_y - disk_center[0]) ** 2) + ((diff_x - disk_center[1]) ** 2)
    disk = np.zeros_like(radiating)
    disk[radiating < (disk_radius ** 2)] = 1
    sobel_disk, _ = st.util.sobel(disk)
    if np.sum(ROI) == 0:
        imROI = np.ones(data4D.shape[0:2], dtype=bool)
    else:
        imROI = ROI
    ROI_4D = data4D[:, :, imROI]
    no_of_disks = ROI_4D.shape[-1]
    LSB_ROI = np.zeros_like(ROI_4D, dtype=np.float)
    for ii in range(no_of_disks):
        cbed = ROI_4D[:, :, ii]
        cbed = 1 + st.util.image_normalizer(cbed)
        log_cbed = scnd.gaussian_filter(np.log10(cbed), 1)
        lsb_cbed, _ = st.util.sobel(log_cbed)
        lsb_cbed[lsb_cbed > (med_factor * np.median(lsb_cbed))] = (
            np.median(lsb_cbed) * med_factor
        )
        lsb_cbed[lsb_cbed < (np.median(lsb_cbed) / med_factor)] = (
            np.median(lsb_cbed) / med_factor
        )
        lsb_cbed = st.util.image_normalizer(lsb_cbed) + st.util.image_normalizer(
            sobel_disk
        )
        LSB_ROI[:, :, ii] = lsb_cbed
    if take_median:
        Mean_LSB = np.median(LSB_ROI, axis=(-1))
    else:
        Mean_LSB = np.mean(LSB_ROI, axis=(-1))
    LSB_CC = st.util.cross_corr(Mean_LSB, sobel_disk, hybrid_cc)
    data_peaks = skfeat.peak_local_max(
        LSB_CC, min_distance=int(2 * disk_radius), indices=False
    )
    peak_labels = scnd.measurements.label(data_peaks)[0]
    merged_peaks = np.asarray(
        scnd.measurements.center_of_mass(
            data_peaks, peak_labels, range(1, np.max(peak_labels) + 1)
        )
    )
    if max_radius > 0:
        dist_peaks = (
            ((merged_peaks[:, 0] - disk_center[0]) ** 2)
            + ((merged_peaks[:, 1] - disk_center[1]) ** 2)
        ) ** 0.5
        merged_peaks = merged_peaks[dist_peaks < max_radius, :]

    if merged_peaks.shape[0] == 1:
        e_xx_map = np.zeros_like(imROI, dtype=float)
        e_xy_map = np.zeros_like(imROI, dtype=float)
        e_th_map = np.zeros_like(imROI, dtype=float)
        e_yy_map = np.zeros_like(imROI, dtype=float)
        newROI = imROI
        new_list_pos = np.zeros(
            (no_of_disks, merged_peaks.shape[0], merged_peaks.shape[1])
        )
        return e_xx_map, e_xy_map, e_th_map, e_yy_map, newROI, new_list_pos

    fitted_mean = np.zeros_like(merged_peaks, dtype=np.float64)
    fitted_scan = np.zeros_like(merged_peaks, dtype=np.float64)

    for jj in range(merged_peaks.shape[0]):
        par = st.util.fit_gaussian2D_mask(
            LSB_CC, merged_peaks[jj, 1], merged_peaks[jj, 0], disk_radius
        )
        fitted_mean[jj, 0:2] = np.flip(par[0:2])
    distarr = (np.sum(((fitted_mean - disk_center) ** 2), axis=1)) ** 0.5
    peaks_mean = (
        fitted_mean[distarr != np.amin(distarr), :]
        - fitted_mean[distarr == np.amin(distarr), :]
    )
    central_disk_no = np.arange(merged_peaks.shape[0])[distarr == np.amin(distarr)][0]

    list_pos = np.zeros((int(np.sum(imROI)), peaks_mean.shape[0], peaks_mean.shape[1]))
    prominences = np.zeros((no_of_disks, merged_peaks.shape[0]), dtype=np.float)

    for kk in range(no_of_disks):
        scan_LSB = LSB_ROI[:, :, kk]
        scan_CC = st.util.cross_corr(scan_LSB, sobel_disk, hybrid_cc)
        for qq in range(merged_peaks.shape[0]):
            scan_par = st.util.fit_gaussian2D_mask(
                scan_CC, fitted_mean[qq, 1], fitted_mean[qq, 0], disk_radius
            )
            fitted_scan[qq, 0:2] = np.flip(scan_par[0:2])
            prominences[kk, qq] = st.nbed.peak_prominence(
                np.flip(scan_par[0:2]), scan_CC, disk_radius
            )
        peaks_scan = (
            fitted_scan[distarr != np.amin(distarr), :]
            - fitted_scan[distarr == np.amin(distarr), :]
        )
        list_pos[kk, :, :] = peaks_scan

    prominence_map = np.zeros((imROI.shape[0], imROI.shape[1], prominences.shape[-1]))
    for ii in np.arange(prominences.shape[-1]):
        prominence_map[imROI, ii] = prominences[:, ii]

    prom_disks = np.ones(prominence_map.shape[0:2], dtype=prominence_map.dtype)
    for ii in range(prominences.shape[-1]):
        if ii != central_disk_no:
            prom_disks = prom_disks * prominence_map[:, :, ii]
    prom_disks[prom_disks < 0] = 0
    prom_disks = prom_disks ** (1 / (prominences.shape[-1] - 1))
    prom_disksG = scnd.gaussian_filter(prom_disks, 1)

    promss = prom_disksG[imROI]
    promss2 = promss / np.amax(promss)
    promss2[promss2 < prom_val] = 0
    prom_disks2 = np.zeros_like(imROI, dtype=promss2.dtype)
    prom_disks2[imROI] = promss2

    newROI = prom_disks2 > 0

    exx_ROI = np.nan * np.ones(no_of_disks, dtype=np.float64)
    exy_ROI = np.nan * np.ones(no_of_disks, dtype=np.float64)
    eth_ROI = np.nan * np.ones(no_of_disks, dtype=np.float64)
    eyy_ROI = np.nan * np.ones(no_of_disks, dtype=np.float64)

    for kk in range(no_of_disks):
        peaks_scan = list_pos[kk, :, :]
        scan_strain, _, _, _ = np.linalg.lstsq(peaks_scan, peaks_mean, rcond=None)
        # In np.linalg.lstsq you get b/a
        scan_strain = np.matmul(scan_strain, rotmatrix)
        scan_strain = scan_strain - np.eye(2)
        exx_ROI[kk] = scan_strain[0, 0]
        exy_ROI[kk] = (scan_strain[0, 1] + scan_strain[1, 0]) / 2
        eth_ROI[kk] = (scan_strain[0, 1] - scan_strain[1, 0]) / 2
        eyy_ROI[kk] = scan_strain[1, 1]

    # make NaN values 0
    exx_ROI[np.isnan(exx_ROI)] = 0
    exy_ROI[np.isnan(exy_ROI)] = 0
    eth_ROI[np.isnan(eth_ROI)] = 0
    eyy_ROI[np.isnan(eyy_ROI)] = 0

    e_xx_map = np.zeros_like(imROI, dtype=exx_ROI.dtype)
    e_xy_map = np.zeros_like(imROI, dtype=exx_ROI.dtype)
    e_th_map = np.zeros_like(imROI, dtype=exx_ROI.dtype)
    e_yy_map = np.zeros_like(imROI, dtype=exx_ROI.dtype)

    min_strain = (-1) * max_strain

    e_xx_map[imROI] = exx_ROI
    e_xx_map -= np.median(e_xx_map[newROI])
    e_xx_map[e_xx_map > max_strain] = max_strain
    e_xx_map[e_xx_map < min_strain] = min_strain
    e_xx_map *= newROI.astype(float)

    e_xy_map[imROI] = exy_ROI
    e_xy_map -= np.median(e_xy_map[newROI])
    e_xy_map[e_xy_map > max_strain] = max_strain
    e_xy_map[e_xy_map < min_strain] = min_strain
    e_xy_map *= newROI.astype(float)

    e_th_map[imROI] = eth_ROI
    e_th_map -= np.median(e_th_map[newROI])
    e_th_map[e_th_map > max_strain] = max_strain
    e_th_map[e_th_map < min_strain] = min_strain
    e_th_map *= newROI.astype(float)

    e_yy_map[imROI] = eyy_ROI
    e_yy_map -= np.median(e_yy_map[newROI])
    e_yy_map[e_yy_map > max_strain] = max_strain
    e_yy_map[e_yy_map < min_strain] = min_strain
    e_yy_map *= newROI.astype(float)

    list_map = np.zeros(
        (imROI.shape[0], imROI.shape[1], list_pos.shape[1], list_pos.shape[2])
    )
    list_map[imROI, :, :] = list_pos
    new_pos = list_map[newROI, :, :]

    upsampling = 6
    roi_ups = st.util.resizer2D(newROI, 1 / upsampling)
    roi_edge, _ = st.util.sobel(roi_ups, 5)
    roi_dists = st.util.euclidean_dist(roi_edge)
    distances = st.util.resizer2D(roi_dists, upsampling)
    distances = distances[newROI]

    if gblur:
        e_xx_map = scnd.gaussian_filter(e_xx_map, 1)
        e_xy_map = scnd.gaussian_filter(e_xy_map, 1)
        e_th_map = scnd.gaussian_filter(e_th_map, 1)
        e_yy_map = scnd.gaussian_filter(e_yy_map, 1)

    return (
        e_xx_map,
        e_xy_map,
        e_th_map,
        e_yy_map,
        newROI,
        new_pos,
        distances,
    )


def bin_scan(data4D, bin_factor):
    """
    Bin the data in the scan dimensions

    Parameters
    ----------
    data4D:     ndarray
                This is a 4D dataset where the first two dimensions
                are the dffraction dimensions and the next two
                dimensions are the scan dimensions
    bin_factor: int or tuple
                Binning factor for scan dimensions

    Returns
    -------
    binned_4D: ndarray
               The data binned in the scanned dimensions.

    Notes
    -----
    You can specify the bin factor to be either an integer or
    a tuple. If you specify an integer the same binning will
    be used in both the scan X and scan Y dimensions, while if
    you specify a tuple then different binning factors for each
    dimensions.

    Examples
    --------
    Run as:

    >>> binned_4D = bin_scan(data4D, 4)

    This will bin the scan dimensions by 4. This is functionally
    identical to:

    >>> binned_4D = bin_scan(data4D, (4, 4))
    """
    bin_factor = np.array(bin_factor, ndmin=1)
    bf = np.copy(bin_factor)
    bin_factor = np.ones(4)
    bin_factor[2:4] = bf
    ini_shape = np.asarray(data4D.shape)
    fin_shape = (np.ceil(ini_shape / bin_factor)).astype(int)
    big_shape = (fin_shape * bin_factor).astype(int)
    binned_4D = np.zeros(fin_shape[0:4], dtype=data4D.dtype)
    big4D = np.zeros(big_shape[0:4], dtype=data4D.dtype)
    big4D[:, :, 0 : ini_shape[2], 0 : ini_shape[3]] = data4D
    for ii in range(fin_shape[2]):
        for jj in range(fin_shape[3]):
            starter_ii = int(bin_factor[2] * ii)
            stopper_ii = int(bin_factor[2] * (ii + 1))
            starter_jj = int(bin_factor[3] * jj)
            stopper_jj = int(bin_factor[3] * (jj + 1))
            summed_cbed = np.sum(
                big4D[:, :, starter_ii:stopper_ii, starter_jj:stopper_jj], axis=(-1, -2)
            )
            binned_4D[:, :, ii, jj] = summed_cbed
    binned_4D = binned_4D / (bin_factor[2] * bin_factor[3])
    return (binned_4D).astype(data4D.dtype)


def cbed_filter(
    image, beam_rad, med_val=50, sec_med=True, hybridizer=0.25, bit_depth=32
):
    """
    Generate the filtered cross-correlated image for locating disk
    positions

    Parameters
    ----------
    image:      ndarray
                The image to be filtered
    beam_rad:   float
                Radius of the circle. The circle used for
                cross-correlating is always centered at the image center.
    med_val:    float, optional
                Deviation from median value to accept in the
                Sobel filtered image. Default is 50
    sec_med:    bool, Optional
                Tamps out deviation from median values in the
                Sobel filtered image too if True
    hybridizer: float, optional
                The value to use for hybrid cross-correlation.
                Default is 0.25. 0 gives pure cross correlation,
                while 1 gives pure phase correlation
    bit_depth:  int, optional
                Maximum power of 2 to be used for scaling the image
                when taking logarithms. Default is 32

    Returns
    -------
    slm_image: ndarray
               The filtered image.

    lsc_image: ndarray
               The filtered image cross-correlated with the circle edge

    Notes
    -----
    We first generate the circle centered at the X and Y co-ordinates, with
    the radius given inside the circ_vals tuple. This generated circle is
    the Sobel filtered to generate an edge of the circle.

    Often due to detector issues, or stray muons a single pixel may be
    much brighter. Also dead pixels can cause individual pixels to be
    much darker. To remove such errors, and values in the image, we take
    the median value of the image and then throw any values that are med_val
    times larger or med_val times smaller than the median. Then we normalize
    the image from 1 to the 2^bit_depth and then take the log of that image.
    This generates an image whose scale is between 0 and the bit_depth. To
    further decrease detector noise, this scaled image is then Gaussian filtered
    with a single pixel blur, and then finally Sobel filtered. This Sobel
    filtered image is then cross-correlated with the Sobel filtered circle edge.

    If there are disks in the image whose size is close to the radius of the
    circle, then the locations of them now become 2D peaks. If the
    circle radius is however too small/large rather than 2D peaks at
    diffraction disk locations, we will observe circles.

    Examples
    --------
    This is extremely useful for locating NBED diffraction positions. If you know
    the size and of the central disk you use the on the Mean_CBED to calculate the
    disk positions from:

    >>> slm_reference, lsc_reference = st.nbed.cbed_filter(Mean_CBED, beam_r)

    """
    # Generating the circle edge
    center_disk = st.util.make_circle(
        np.asarray(image.shape), image.shape[1] / 2, image.shape[0] / 2, beam_rad
    )
    sobel_center_disk, _ = st.util.sobel(center_disk)

    # Throwing away stray pixel values
    med_image = np.copy(image)
    med_image[med_image > med_val * np.median(med_image)] = med_val * np.median(
        med_image
    )
    med_image[med_image < np.median(med_image) / med_val] = (
        np.median(med_image) / med_val
    )

    # Filtering the image
    slm_image, _ = st.util.sobel(
        scnd.gaussian_filter(st.util.image_logarizer(med_image, bit_depth), 1)
    )
    if sec_med:
        slm_image[slm_image > med_val * np.median(slm_image)] = med_val * np.median(
            slm_image
        )
        slm_image[slm_image < np.median(slm_image) / med_val] = (
            np.median(slm_image) / med_val
        )

    # Cross-correlating it
    lsc_image = st.util.cross_corr(slm_image, sobel_center_disk, hybridizer)
    return slm_image, lsc_image


def get_radius(cbed_image, ubound=0.2, tol=0.0001):
    """
    Find the size of the central disk from diffraction
    patterns.

    Parameters
    ----------
    cbed_image: ndarray
                The CBED image to be used for calculating
                the central convergence angle size from
    ubound:     float, optional
                The ratio of the size of the image to be used
                as the upper bound for the radius size. Default
                is 0.2. The lower bound is always 1 pixels
    tol:        float, optional
                The tolerance of the scipy.optimize fitting function
                Default is 0.01

    Returns
    -------
    rad: float
         The radius in pixels of the convergence aperture

    Notes
    -----
    This is based on the idea that if a Sobel filtered CBED pattern
    is cross-correlated with a circle edge, a sharp peak is obtained
    only if the radius of that circle is the same as the radius in
    pixels of the CBED pattern. If the circle is a lot smaller or larger,
    then a donut like ring is formed at the center of the cross-correlated
    pattern. If the radius is a bit off - then the peak is a bit diffuse.
    Thus the correct radius in pixels will give the sharpest peak, which
    is how we find the radius by using `scipy.optimize.minimize_scalar`

    See Also
    --------
    cbed_filter
    """
    imshape = np.asarray(cbed_image.shape)
    sobel_cbed, _ = st.util.sobel(cbed_image)
    test_precison = int((1 / tol) / 20)
    test_vals = np.zeros((test_precison, 2))
    test_vals[:, 0] = (1 + np.arange(test_precison)) * (
        (ubound * np.amax(imshape)) / test_precison
    )
    for ii in range(test_precison):
        center_disk = st.util.make_circle(
            np.asarray(imshape), imshape[1] / 2, imshape[0] / 2, test_vals[ii, 0]
        )
        sobel_center_disk, _ = st.util.sobel(center_disk)
        tc = st.util.cross_corr(sobel_cbed, sobel_center_disk)
        test_vals[ii, 1] = tc[int(tc.shape[0] / 2), int(tc.shape[1] / 2)]

    first_max = test_vals[test_vals[:, 1] == np.max(test_vals[:, 1]), 0][0]
    variation = (ubound * np.amax(imshape)) / 10
    lb = first_max - variation
    ub = first_max + variation

    def rad_func(radius):
        cd = st.util.make_circle(
            np.asarray(imshape), imshape[1] / 2, imshape[0] / 2, radius
        )
        sobel_cd, _ = st.util.sobel(cd)
        test_corr = st.util.cross_corr(sobel_cbed, sobel_cd)
        rad_val = test_corr[int(test_corr.shape[0] / 2), int(test_corr.shape[1] / 2)]
        return -rad_val

    res = sio.minimize_scalar(rad_func, bounds=(lb, ub), tol=tol, method="bounded")
    rad = res.x
    return rad


def bin_scan_test(big4D, bin_factor):
    bin_factor = np.array(bin_factor, ndmin=1)
    bf = np.copy(bin_factor)
    bin_factor = np.ones(4)
    bin_factor[2:4] = bf
    ini_shape = np.asarray(big4D.shape)
    fin_shape = (np.ceil(ini_shape / bin_factor)).astype(int)

    dbiny = ini_shape[2] / fin_shape[2]
    dbinx = ini_shape[3] / fin_shape[3]

    bin_y = np.zeros((fin_shape[2], 2))
    bin_x = np.zeros((fin_shape[3], 2))

    bin_y[:, 0] = (np.linspace(start=0, stop=ini_shape[2], num=(fin_shape[2] + 1)))[
        0 : fin_shape[2]
    ]
    bin_y[:, 1] = (np.linspace(start=0, stop=ini_shape[2], num=(fin_shape[2] + 1)))[
        1 : (fin_shape[2] + 1)
    ]
    bin_x[:, 0] = (np.linspace(start=0, stop=ini_shape[3], num=(fin_shape[3] + 1)))[
        0 : fin_shape[3]
    ]
    bin_x[:, 1] = (np.linspace(start=0, stop=ini_shape[3], num=(fin_shape[3] + 1)))[
        1 : (fin_shape[3] + 1)
    ]

    binned_4D = np.zeros(fin_shape[0:4], dtype=big4D.dtype)

    stopper_0i = int(np.floor(bin_y[0, 1]))
    topper_0i = np.mod(bin_y[0, 1], 1)
    stopper_0j = int(np.floor(bin_x[0, 1]))
    topper_0j = np.mod(bin_x[0, 1], 1)
    binned_4D[:, :, 0, 0] = np.sum(
        big4D[:, :, 0:stopper_0i, 0:stopper_0j], axis=(-1, -2)
    )
    binned_4D[:, :, 0, 0] = (
        binned_4D[:, :, 0, 0]
        + (topper_0i * big4D[:, :, int(stopper_0i + 1), stopper_0j])
        + (topper_0j * big4D[:, :, stopper_0i, int(stopper_0j + 1)])
        + (
            topper_0i
            * topper_0j
            * big4D[:, :, int(stopper_0i + 1), int(stopper_0j + 1)]
        )
    )

    starter_1i = int(np.ceil(bin_y[-1, 0]))
    lower_1i = 1 - np.mod(bin_y[-1, 0], 1)
    starter_1j = int(np.ceil(bin_y[-1, 0]))
    lower_1j = 1 - np.mod(bin_x[-1, 0], 1)
    binned_4D[:, :, -1, -1] = np.sum(
        big4D[:, :, starter_1i : int(bin_y[-1, 1]), starter_1j : int(bin_x[-1, 1])],
        axis=(-1, -2),
    )
    binned_4D[:, :, -1, -1] = (
        binned_4D[:, :, -1, -1]
        + (lower_1i * big4D[:, :, int(starter_1i - 1), starter_1j])
        + (lower_1j * big4D[:, :, starter_1i, int(starter_1j - 1)])
        + (lower_1i * lower_1j * big4D[:, :, int(starter_1i - 1), int(starter_1j - 1)])
    )

    for ii in np.arange(1, fin_shape[2] - 1):
        for jj in np.arange(1, fin_shape[3] - 1):
            starter_ii = int(np.ceil(bin_y[ii, 0]))
            stopper_ii = int(np.floor(bin_y[ii, 1]))
            lower_ii = 1 - np.mod(bin_y[ii, 0], 1)
            topper_ii = np.mod(bin_y[ii, 1], 1)

            starter_jj = int(np.ceil(bin_x[jj, 0]))
            stopper_jj = int(np.floor(bin_x[jj, 1]))
            lower_jj = 1 - np.mod(bin_x[jj, 0], 1)
            topper_jj = np.mod(bin_x[jj, 1], 1)

            summed_cbed = np.sum(
                big4D[:, :, starter_ii:stopper_ii, starter_jj:stopper_jj], axis=(-1, -2)
            )

            # get low vals
            summed_cbed = (
                summed_cbed
                + (lower_ii * big4D[:, :, int(starter_ii - 1), starter_jj])
                + (lower_jj * big4D[:, :, starter_ii, int(starter_jj - 1)])
                + (
                    lower_ii
                    * lower_jj
                    * big4D[:, :, int(starter_ii - 1), int(starter_jj - 1)]
                )
            )

            # get top vals
            summed_cbed = (
                summed_cbed
                + (topper_ii * big4D[:, :, int(stopper_ii + 1), stopper_jj])
                + (topper_jj * big4D[:, :, stopper_ii, int(stopper_jj + 1)])
                + (
                    topper_ii
                    * topper_jj
                    * big4D[:, :, int(stopper_ii + 1), int(stopper_jj + 1)]
                )
            )

            binned_4D[:, :, ii, jj] = summed_cbed

    binned_4D = binned_4D / (dbinx * dbiny)
    return binned_4D


def strain_figure(
    exx, exy, eth, eyy, ROI, vm=0, scale=0, scale_unit="nm", figsize=(22, 21)
):
    """
    Plot the strain maps from a given set of strains, where the strain is
    mapped only in the region of interest, while anything outside the region
    of interest is black.

    Parameters
    ----------
    exx:        ndarray
                2D array of e_xx strains
    eyy:        ndarray
                2D array of e_yy strains
    eth:        ndarray
                2D array of e_theta strains
    eyy:        ndarray
                2D array of e_yy strains
    ROI:        ndarray of type bool
                Region of interest
    vm:         float, optional
                Maximum value of strain map
                Default is 0, and when it's zero,
                it will calculate the maximum absolute
                strain value and use that.
    scale:      float, optional
                Pixel size. Default is 0
    scale_unit: char, optional
                Units of the calibration. Default is 'nm'
    figsize:    tuple, optional
                Size of the final figure

    Notes
    -----
    This is basically a nice function for plotting calculated strain maps
    from a region of interest. We use the RdBu_r, which is the RdBu reversed
    color scheme, where 0 is white, negative values are increasingly blue the
    more negative they are, while positive values are increasingly red the more
    positive they are. Anything outside the region of interest is black, and thus
    to an observer the ROI can be visualized clearly. Optionally, a scalebar can
    be assigned too.
    """

    def ROI_RdBu_map(valmap, roi, valrange):
        plot_col = np.zeros((256, 3), dtype=np.float)
        for ii in range(255):
            plot_col[ii, 0:3] = np.asarray(mpl.cm.RdBu_r(ii)[0:3])
        map_col = np.zeros((valmap.shape[0], valmap.shape[1], 3))
        colorvals = (255 * ((valmap[roi] + valrange) / (2 * valrange))).astype(int)
        colorvals[colorvals > 255] = 255
        colorvals[colorvals < 0] = 0
        map_col[roi, 0:3] = plot_col[colorvals, :]
        return map_col

    if vm == 0:
        vm = 100 * np.amax(
            np.asarray(
                (
                    np.amax(np.abs(exx)),
                    np.amax(np.abs(exy)),
                    np.amax(np.abs(eth)),
                    np.amax(np.abs(eyy)),
                )
            )
        )

    fontsize = int(1.25 * np.max(figsize))
    sc_font = {"weight": "bold", "size": fontsize}
    mpl.rc("font", **sc_font)
    plt.figure(figsize=figsize)

    gs = mpgs.GridSpec(21, 22)
    ax1 = plt.subplot(gs[0:10, 0:10])
    ax2 = plt.subplot(gs[0:10, 12:22])
    ax3 = plt.subplot(gs[10:20, 0:10])
    ax4 = plt.subplot(gs[10:20, 12:22])
    ax5 = plt.subplot(gs[20:21, :])

    ax1.imshow(ROI_RdBu_map(100 * exx, ROI, vm))
    at = mploff.AnchoredText(
        r"$\mathrm{\epsilon_{xx}}$",
        prop=dict(size=fontsize),
        frameon=True,
        loc="lower right",
    )
    at.patch.set_boxstyle("round, pad= 0., rounding_size= 0.2")
    ax1.add_artist(at)
    if scale > 0:
        scalebar = mpss.ScaleBar(scale, scale_unit)
        scalebar.location = "lower left"
        scalebar.box_alpha = 1
        scalebar.color = "k"
        ax1.add_artist(scalebar)
    ax1.axis("off")

    ax2.imshow(ROI_RdBu_map(100 * exy, ROI, vm))
    at = mploff.AnchoredText(
        r"$\mathrm{\epsilon_{xy}}$",
        prop=dict(size=fontsize),
        frameon=True,
        loc="lower right",
    )
    at.patch.set_boxstyle("round, pad= 0., rounding_size= 0.2")
    ax2.add_artist(at)
    if scale > 0:
        scalebar = mpss.ScaleBar(scale, scale_unit)
        scalebar.location = "lower left"
        scalebar.box_alpha = 1
        scalebar.color = "k"
        ax2.add_artist(scalebar)
    ax2.axis("off")

    ax3.imshow(ROI_RdBu_map(100 * eth, ROI, vm))
    at = mploff.AnchoredText(
        r"$\mathrm{\epsilon_{\theta}}$",
        prop=dict(size=fontsize),
        frameon=True,
        loc="lower right",
    )
    at.patch.set_boxstyle("round, pad= 0., rounding_size= 0.2")
    ax3.add_artist(at)
    if scale > 0:
        scalebar = mpss.ScaleBar(scale, scale_unit)
        scalebar.location = "lower left"
        scalebar.box_alpha = 1
        scalebar.color = "k"
        ax3.add_artist(scalebar)
    ax3.axis("off")

    ax4.imshow(ROI_RdBu_map(100 * eyy, ROI, vm))
    at = mploff.AnchoredText(
        r"$\mathrm{\epsilon_{yy}}$",
        prop=dict(size=fontsize),
        frameon=True,
        loc="lower right",
    )
    at.patch.set_boxstyle("round, pad= 0., rounding_size= 0.2")
    ax4.add_artist(at)
    if scale > 0:
        scalebar = mpss.ScaleBar(scale, scale_unit)
        scalebar.location = "lower left"
        scalebar.box_alpha = 1
        scalebar.color = "k"
        ax4.add_artist(scalebar)
    ax4.axis("off")

    sb = np.zeros((10, 1000), dtype=np.float)
    for ii in range(10):
        sb[ii, :] = np.linspace(-vm, vm, 1000)
    ax5.imshow(sb, cmap="RdBu_r")
    ax5.yaxis.set_visible(False)
    x1 = np.linspace(0, 1000, 8)
    ax5.set_xticks(x1)
    ax5.set_xticklabels(np.round(np.linspace(-vm, vm, 8), 2))
    for axis in ["top", "bottom", "left", "right"]:
        ax5.spines[axis].set_linewidth(2)
        ax5.spines[axis].set_color("black")
    ax5.xaxis.set_tick_params(width=2, length=6, direction="out", pad=10)
    ax5.set_title("Strain (%)", fontsize=25, fontweight="bold")

    plt.tight_layout()


def get_strain_plot(volume, roi, precision=(0.2, 0.001), upsampling=6):
    """
    Get strain maps from the volume calculation and region of
    interest

    Parameters
    ----------
    volume:     ndarray, float
                Calculated volume from strain
    roi:        ndarray, bool
                Region where the particle lies
    precision:  tuple
                The number of significant digits in each axes
    upsampling: int
                Upsampling factor for the region before calculation

    Returns
    -------
    rawvals: ndarray, float
             The raw calculated strain values from the region of interest
    strain:  ndarray, float
             Strain (second column) as a function of distance from particle
             surface (first column), with standard error (third column) and
             standard deviation (fourth column)
    maxdist: float
             Maximum distance from surface

    Notes
    -----
    The ROI is upsampled by the upsampled by the upsampling factor, and the
    cartesian distances inside that ROI from the nearest edge point are
    calculated. That's your x values, which is the distance from the surface,
    while the y values are the calculated volume strain at that position
    """
    roi_ups = st.util.resizer2D(roi, 1 / upsampling)
    roi_edge, _ = st.util.sobel(roi_ups, 5)
    ups_map = st.util.resizer2D(volume, 1 / upsampling)
    roi_dists = st.util.euclidean_dist(roi_edge)
    xvals = (np.max(roi_dists) - roi_dists[roi_dists > 1 / upsampling]) / upsampling
    yvals = ups_map[roi_dists > 1 / upsampling]
    maxdist = np.amax(np.round(xvals, 1))
    xvals = maxdist - xvals
    rawvals = np.transpose(np.asarray((xvals, yvals)))
    rawvals = st.util.reduce_precision_xy(rawvals, precision)
    strain = np.zeros((len(np.unique(rawvals[:, 0])), 4))
    strain[:, 0] = np.unique(rawvals[:, 0])
    yvals = rawvals[:, 1]
    for ii in np.arange(strain.shape[0]):
        strain[ii, 1] = np.median(yvals[rawvals[:, 0] == strain[ii, 0]])
        nn = len(yvals[rawvals[:, 0] == strain[ii, 0]])
        strain[ii, 2] = np.std(yvals[rawvals[:, 0] == strain[ii, 0]])
        strain[ii, 3] = (np.std(yvals[rawvals[:, 0] == strain[ii, 0]])) / (nn ** 0.5)
    return rawvals, strain, maxdist
