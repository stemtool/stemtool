import skimage.feature as skfeat
import matplotlib.pyplot as plt
import numpy as np
import numba
import scipy.ndimage as scnd
import scipy.optimize as spo
import scipy.interpolate as scinterp
import warnings
import matplotlib_scalebar.scalebar as mpss
import stemtool as st


def remove_close_vals(input_arr, limit):
    result = np.copy(input_arr)
    ii = 0
    newlen = len(result)
    while ii < newlen:
        dist = (np.sum(((result[:, 0:2] - result[ii, 0:2]) ** 2), axis=1)) ** 0.5
        distbool = dist > limit
        distbool[ii] = True
        result = np.copy(result[distbool, :])
        ii = ii + 1
        newlen = len(result)
    return result


def peaks_vis(data_image, dist=10, thresh=0.1, imsize=(20, 20)):
    """
    Find atom maxima pixels in images

    Parameters
    ----------
    data_image: ndarray
                Original atomic resolution image
    dist:       int
                Average distance between neighboring peaks
                Default is 10
    thresh:     float
                The cutoff intensity value below which a peak
                will not be detected
                Default is 0.1
    imsize:     ndarray
                Size of the display image
                Default is (20,20)

    Returns
    -------
    peaks: ndarray
           List of peak positions as y, x

    Notes
    -----
    This is a wrapper around the skimage peak finding
    module which finds the peaks with a given threshold
    value and an inter-peak separation. The function
    additionally plots the peak positions on the original
    image thus users can modify the input values of
    threshold and distance to ensure the right peaks are
    selected.

    :Authors:
    Debangshu Mukherjee <mukherjeed@ornl.gov>
    """
    data_image = (data_image - np.amin(data_image)) / (
        np.amax(data_image) - np.amin(data_image)
    )
    thresh_arr = np.array(data_image > thresh, dtype=np.float)
    data_thresh = (data_image * thresh_arr) - thresh
    data_thresh[data_thresh < 0] = 0
    data_thresh = data_thresh / (1 - thresh)
    data_peaks = skfeat.peak_local_max(
        data_thresh, min_distance=int(dist / 3), indices=False
    )
    peak_labels = scnd.measurements.label(data_peaks)[0]
    merged_peaks = scnd.measurements.center_of_mass(
        data_peaks, peak_labels, range(1, np.max(peak_labels) + 1)
    )
    peaks = np.array(merged_peaks)
    peaks = remove_close_vals(peaks, dist)
    plt.figure(figsize=imsize)
    plt.imshow(data_image)
    plt.scatter(peaks[:, 1], peaks[:, 0], c="b", s=15)
    return peaks


@numba.jit(parallel=True)
def refine_atoms(image_data, positions):
    """
    Single Gaussian Peak Atom Refinement

    Parameters
    ----------
    image_data: ndarray
                Original atomic resolution image
    positions:  ndarray
                Intensity minima/maxima list

    Returns
    -------
    ref_arr: ndarray
             List of refined peak positions as y, x

    Notes
    -----
    This is the single Gaussian peak fitting technique
    where the initial atom positions are fitted with a
    single 2D Gaussian function. The center of the Gaussian
    is returned as the refined atom position.
    """
    warnings.filterwarnings("ignore")
    no_pos = len(positions)
    dist = np.empty(no_pos, dtype=np.float)
    ccd = np.empty(no_pos, dtype=np.float)
    for ii in numba.prange(no_pos):
        ccd = np.sum(((positions[:, 0:2] - positions[ii, 0:2]) ** 2), axis=1)
        dist[ii] = (np.amin(ccd[ccd > 0])) ** 0.5
    med_dist = 0.5 * np.median(dist)
    ref_arr = np.empty((no_pos, 7), dtype=np.float)
    for ii in numba.prange(no_pos):
        pos_x = positions[ii, 1]
        pos_y = positions[ii, 0]
        refined_ii = st.util.fit_gaussian2D_mask(image_data, pos_x, pos_y, med_dist)
        ref_arr[ii, 0:2] = np.flip(refined_ii[0:2])
        ref_arr[ii, 2:7] = refined_ii[2:7]
    return ref_arr


@numba.jit
def mpfit(
    main_image,
    initial_peaks,
    peak_runs=16,
    cut_point=2 / 3,
    tol_val=0.01,
    peakparams=False,
):
    """
    Multi-Gaussian Peak Refinement (mpfit)

    Parameters
    ----------
    main_image:     ndarray
                    Original atomic resolution image
    initial_peaks:  ndarray
                    Y and X position of maxima/minima
    peak_runs:      int
                    Number of multi-Gaussian steps to run
                    Default is 16
    cut_point:      float
                    Ratio of distance to the median inter-peak
                    distance. Only Gaussian peaks below this are
                    used for the final estimation
                    Default is 2/3
    tol_val:        float
                    The tolerance value to use for a gaussian estimation
                    Default is 0.01
    peakparams:     boolean
                    If set to True, then the individual Gaussian peaks and
                    their amplitudes are also returned.
                    Default is False

    Returns
    -------
    mpfit_peaks: ndarray
                 List of refined peak positions as y, x

    Notes
    -----
    This is the multiple Gaussian peak fitting technique
    where the initial atom positions are fitted with a
    single 2D Gaussian function. The calculated Gaussian is
    then subsequently subtracted and refined again. The final
    refined position is the sum of all the positions scaled
    with the amplitude

    References:
    -----------
    1]_, Mukherjee, D., Miao, L., Stone, G. and Alem, N.,
         mpfit: a robust method for fitting atomic resolution images
         with multiple Gaussian peaks. Adv Struct Chem Imag 6, 1 (2020).
    """
    warnings.filterwarnings("ignore")
    dist = np.zeros(len(initial_peaks))
    for ii in np.arange(len(initial_peaks)):
        ccd = np.sum(((initial_peaks[:, 0:2] - initial_peaks[ii, 0:2]) ** 2), axis=1)
        dist[ii] = (np.amin(ccd[ccd > 0])) ** 0.5
    med_dist = np.median(dist)
    mpfit_peaks = np.zeros_like(initial_peaks, dtype=np.float)
    yy, xx = np.mgrid[0 : main_image.shape[0], 0 : main_image.shape[1]]
    cvals = np.zeros((peak_runs, 4), dtype=np.float)
    peak_vals = np.zeros((len(initial_peaks), peak_runs, 4), dtype=np.float)
    for jj in np.arange(len(initial_peaks)):
        ystart = initial_peaks[jj, 0]
        xstart = initial_peaks[jj, 1]
        sub_y = np.abs(yy - ystart) < med_dist
        sub_x = np.abs(xx - xstart) < med_dist
        sub = np.logical_and(sub_x, sub_y)
        xvals = xx[sub]
        yvals = yy[sub]
        zvals = main_image[sub]
        zcalc = np.zeros_like(zvals)
        for ii in np.arange(peak_runs):
            zvals = zvals - zcalc
            zgaus = (zvals - np.amin(zvals)) / (np.amax(zvals) - np.amin(zvals))
            mask_radius = med_dist
            xy = (xvals, yvals)
            initial_guess = st.util.initialize_gauss2D(xvals, yvals, zgaus)
            lower_bound = (
                (initial_guess[0] - med_dist),
                (initial_guess[1] - med_dist),
                -180,
                0,
                0,
                ((-2.5) * initial_guess[5]),
            )
            upper_bound = (
                (initial_guess[0] + med_dist),
                (initial_guess[1] + med_dist),
                180,
                (2.5 * mask_radius),
                (2.5 * mask_radius),
                (2.5 * initial_guess[5]),
            )
            popt, _ = spo.curve_fit(
                st.util.gaussian_2D_function,
                xy,
                zgaus,
                initial_guess,
                bounds=(lower_bound, upper_bound),
                ftol=tol_val,
                xtol=tol_val,
            )
            cvals[ii, 1] = popt[0]
            cvals[ii, 0] = popt[1]
            cvals[ii, -1] = popt[-1] * (np.amax(zvals) - np.amin(zvals))
            cvals[ii, 2] = (
                ((popt[0] - xstart) ** 2) + ((popt[1] - ystart) ** 2)
            ) ** 0.5
            zcalc = st.util.gaussian_2D_function(
                xy, popt[0], popt[1], popt[2], popt[3], popt[4], popt[5]
            )
            zcalc = (zcalc * (np.amax(zvals) - np.amin(zvals))) + np.amin(zvals)
        required_cvals = cvals[:, 2] < (cut_point * med_dist)
        total = np.sum(cvals[required_cvals, 3])
        y_mpfit = np.sum(cvals[required_cvals, 0] * cvals[required_cvals, 3]) / total
        x_mpfit = np.sum(cvals[required_cvals, 1] * cvals[required_cvals, 3]) / total
        mpfit_peaks[jj, 0:2] = np.asarray((y_mpfit, x_mpfit))
        peak_vals[jj, :, :] = cvals
    if peakparams:
        return mpfit_peaks, peak_vals
    else:
        return mpfit_peaks


@numba.jit
def mpfit_voronoi(
    main_image,
    initial_peaks,
    peak_runs=16,
    cut_point=2 / 3,
    tol_val=0.01,
    blur_factor=0.25,
):
    """
    Multi-Gaussian Peak Refinement (mpfit)

    Parameters
    ----------
    main_image:     ndarray
                    Original atomic resolution image
    initial_peaks:  ndarray
                    Y and X position of maxima/minima
    peak_runs:      int
                    Number of multi-Gaussian steps to run
                    Default is 16
    cut_point:      float
                    Ratio of distance to the median inter-peak
                    distance. Only Gaussian peaks below this are
                    used for the final estimation
                    Default is 2/3
    tol_val:        float
                    The tolerance value to use for a gaussian estimation
                    Default is 0.01
    blur_factor:    float
                    Make the Voronoi regions slightly bigger.
                    Default is 25% bigger

    Returns
    -------
    mpfit_peaks: ndarray
                 List of refined peak positions as y, x

    Notes
    -----
    This is the multiple Gaussian peak fitting technique
    where the initial atom positions are fitted with a
    single 2D Gaussian function. The calculated Gaussian is
    then subsequently subtracted and refined again. The final
    refined position is the sum of all the positions scaled
    with the amplitude. The difference with the standard mpfit
    code is that the masking region is actually chosen as a
    Voronoi region from the nearest neighbors

    References:
    -----------
    1]_, Mukherjee, D., Miao, L., Stone, G. and Alem, N.,
         mpfit: a robust method for fitting atomic resolution images
         with multiple Gaussian peaks. Adv Struct Chem Imag 6, 1 (2020).

    :Authors:
    Debangshu Mukherjee <mukherjeed@ornl.gov>
    """
    warnings.filterwarnings("ignore")
    distm = np.zeros(len(initial_peaks))
    for ii in np.arange(len(initial_peaks)):
        ccd = np.sum(((initial_peaks[:, 0:2] - initial_peaks[ii, 0:2]) ** 2), axis=1)
        distm[ii] = (np.amin(ccd[ccd > 0])) ** 0.5
    med_dist = np.median(distm)
    mpfit_peaks = np.zeros_like(initial_peaks, dtype=np.float)
    yy, xx = np.mgrid[0 : main_image.shape[0], 0 : main_image.shape[1]]
    cutoff = med_dist * 2.5
    for jj in np.arange(len(initial_peaks)):
        ypos, xpos = initial_peaks[jj, :]
        dist = (
            np.sum(((initial_peaks[:, 0:2] - initial_peaks[jj, 0:2]) ** 2), axis=1)
        ) ** 0.5
        distn = dist < cutoff
        distn[dist < 0.1] = False
        neigh = initial_peaks[distn, 0:2]
        sub = (((yy - ypos) ** 2) + ((xx - xpos) ** 2)) < (cutoff ** 2)
        xvals = xx[sub]
        yvals = yy[sub]
        zvals = main_image[sub]
        maindist = ((xvals - xpos) ** 2) + ((yvals - ypos) ** 2)
        dist_mat = np.zeros((len(xvals), len(neigh)))
        for ii in np.arange(len(neigh)):
            dist_mat[:, ii] = ((xvals - neigh[ii, 1]) ** 2) + (
                (yvals - neigh[ii, 0]) ** 2
            )
        neigh_dist = np.amin(dist_mat, axis=1)
        voronoi = maindist < ((1 + blur_factor) * neigh_dist)
        xvor = xvals[voronoi]
        yvor = yvals[voronoi]
        zvor = zvals[voronoi]
        vor_dist = np.amax((((xvor - xpos) ** 2) + ((yvor - ypos) ** 2)) ** 0.5)
        zcalc = np.zeros_like(zvor)
        xy = (xvor, yvor)
        cvals = np.zeros((peak_runs, 4), dtype=np.float)
        for ii in np.arange(peak_runs):
            zvor = zvor - zcalc
            zgaus = (zvor - np.amin(zvor)) / (np.amax(zvor) - np.amin(zvor))
            initial_guess = st.util.initialize_gauss2D(xvor, yvor, zgaus)
            lower_bound = (
                np.amin(xvor),
                np.amin(yvor),
                -180,
                0,
                0,
                ((-2.5) * initial_guess[5]),
            )
            upper_bound = (
                np.amax(xvor),
                np.amax(yvor),
                180,
                vor_dist,
                vor_dist,
                (2.5 * initial_guess[5]),
            )
            popt, _ = spo.curve_fit(
                st.util.gaussian_2D_function,
                xy,
                zgaus,
                initial_guess,
                bounds=(lower_bound, upper_bound),
                ftol=tol_val,
                xtol=tol_val,
            )
            cvals[ii, 1] = popt[0]
            cvals[ii, 0] = popt[1]
            cvals[ii, -1] = popt[-1] * (np.amax(zvor) - np.amin(zvor))
            cvals[ii, 2] = (((popt[0] - xpos) ** 2) + ((popt[1] - ypos) ** 2)) ** 0.5
            zcalc = st.util.gaussian_2D_function(
                xy, popt[0], popt[1], popt[2], popt[3], popt[4], popt[5]
            )
            zcalc = (zcalc * (np.amax(zvor) - np.amin(zvor))) + np.amin(zvor)
        required_cvals = cvals[:, 2] < (cut_point * vor_dist)
        total = np.sum(cvals[required_cvals, 3])
        y_mpfit = np.sum(cvals[required_cvals, 0] * cvals[required_cvals, 3]) / total
        x_mpfit = np.sum(cvals[required_cvals, 1] * cvals[required_cvals, 3]) / total
        mpfit_peaks[jj, 0:2] = np.asarray((y_mpfit, x_mpfit))
    return mpfit_peaks


def fourier_mask(original_image, center, radius, threshold=0.2):
    image_fourier = np.fft.fftshift(np.fft.fft2(original_image))
    pos_x = center[0]
    pos_y = center[1]
    blurred_image = scnd.filters.gaussian_filter(np.abs(image_fourier), 3)
    fitted_diff = st.util.fit_gaussian2D_mask(blurred_image, pos_x, pos_y, radius)
    new_x = fitted_diff[0]
    new_y = fitted_diff[1]
    new_center = np.asarray((new_x, new_y))
    size_image = np.asarray(np.shape(image_fourier), dtype=int)
    yV, xV = np.mgrid[0 : size_image[0], 0 : size_image[1]]
    sub = ((((yV - new_y) ** 2) + ((xV - new_x) ** 2)) ** 0.5) < radius
    circle = np.copy(sub)
    circle = circle.astype(np.float64)
    filtered_circ = scnd.filters.gaussian_filter(circle, 1)
    masked_image = np.multiply(image_fourier, filtered_circ)
    SAED_image = np.fft.ifft2(masked_image)
    mag_SAED = np.abs(SAED_image)
    mag_SAED = (mag_SAED - np.amin(mag_SAED)) / (np.amax(mag_SAED) - np.amin(mag_SAED))
    mag_SAED[mag_SAED < threshold] = 0
    mag_SAED[mag_SAED > threshold] = 1
    filtered_SAED = scnd.filters.gaussian_filter(mag_SAED, 3)
    filtered_SAED[filtered_SAED < threshold] = 0
    filtered_SAED[filtered_SAED > threshold] = 1
    fourier_selected_image = np.multiply(original_image, filtered_SAED)
    return fourier_selected_image, SAED_image, new_center, filtered_SAED


def find_diffraction_spots(image, circ_c, circ_y, circ_x):
    """
    Find the diffraction spots visually.

    Parameters
    ----------
    image:  ndarray
            Original image
    circ_c: ndarray
            Position of the central beam in
            the Fourier pattern
    circ_y: ndarray
            Position of the y beam in
            the Fourier pattern
    circ_x: ndarray
            Position of the x beam in
            the Fourier pattern


    Notes
    -----
    Put circles in red(central), y(blue) and x(green)
    on the diffraction pattern to approximately know
    the positions.

    :Authors:
    Debangshu Mukherjee <mukherjeed@ornl.gov>
    """
    image_ft = np.fft.fftshift(np.fft.fft2(image))
    log_abs_ft = scnd.filters.gaussian_filter(np.log10(np.abs(image_ft)), 3)
    f, ax = plt.subplots(figsize=(20, 20))
    circ_c_im = plt.Circle(circ_c, 15, color="red", alpha=0.33)
    circ_y_im = plt.Circle(circ_y, 15, color="blue", alpha=0.33)
    circ_x_im = plt.Circle(circ_x, 15, color="green", alpha=0.33)
    ax.imshow(log_abs_ft, cmap="gray")
    ax.add_artist(circ_c_im)
    ax.add_artist(circ_y_im)
    ax.add_artist(circ_x_im)
    plt.show()


def find_coords(image, fourier_center, fourier_y, fourier_x, y_axis, x_axis):
    """
    Convert the fourier positions to image axes.
    Do not use numba to accelerate as LLVM IR
    throws an error from the the if statements

    Parameters
    ----------
    image:  ndarray
            Original image
    four_c: ndarray
            Position of the central beam in
            the Fourier pattern
    four_y: ndarray
            Position of the y beam in
            the Fourier pattern
    four_x: ndarray
            Position of the x beam in
            the Fourier pattern

    Returns
    -------
    coords: ndarray
            Axes co-ordinates in the real image,
            as [y1 x1
                y2 x2]

    Notes
    -----
    Use the fourier coordinates to define the axes
    co-ordinates in real space, which will be used
    to assign each atom position to a axes position

    :Authors:
    Debangshu Mukherjee <mukherjeed@ornl.gov>
    """
    image_size = image.shape
    qx = (np.arange((-image_size[1] / 2), (image_size[1] / 2), 1)) / image_size[1]
    qy = (np.arange((-image_size[0] / 2), (image_size[0] / 2), 1)) / image_size[0]
    dx = np.mean(np.diff(qx))
    dy = np.mean(np.diff(qy))
    fourier_calib = (dy, dx)
    y_axis[y_axis == 0] = 1
    x_axis[x_axis == 0] = 1
    distance_y = fourier_y - fourier_center
    distance_y = np.divide(distance_y, y_axis[0:2])
    distance_x = fourier_x - fourier_center
    distance_x = np.divide(distance_x, x_axis[0:2])
    fourier_length_y = distance_y * fourier_calib
    fourier_length_x = distance_x * fourier_calib
    angle_y = np.degrees(
        np.arccos(fourier_length_y[0] / np.linalg.norm(fourier_length_y))
    )
    real_y_size = 1 / np.linalg.norm(fourier_length_y)
    angle_x = np.degrees(
        np.arccos(fourier_length_x[0] / np.linalg.norm(fourier_length_x))
    )
    real_x_size = 1 / np.linalg.norm(fourier_length_x)
    real_y = (
        (real_y_size * np.cos(np.deg2rad(angle_y - 90))),
        (real_y_size * np.sin(np.deg2rad(angle_y - 90))),
    )
    real_x = (
        (real_x_size * np.cos(np.deg2rad(angle_x - 90))),
        (real_x_size * np.sin(np.deg2rad(angle_x - 90))),
    )
    coords = np.asarray(((real_y[1], real_y[0]), (real_x[1], real_x[0])))
    if np.amax(np.abs(coords[0, :])) > np.amax(coords[0, :]):
        coords[0, :] = (-1) * coords[0, :]
    if np.amax(np.abs(coords[1, :])) > np.amax(coords[1, :]):
        coords[1, :] = (-1) * coords[1, :]
    return coords


def get_origin(image, peak_pos, coords):
    def origin_function(xyCenter, input_data=(peak_pos, coords)):
        peaks = input_data[0]
        coords = input_data[1]
        atom_coords = np.zeros((peaks.shape[0], 6))
        atom_coords[:, 0:2] = peaks[:, 0:2] - xyCenter[0:2]
        atom_coords[:, 2:4] = atom_coords[:, 0:2] @ np.linalg.inv(coords)
        atom_coords[:, 4:6] = np.round(atom_coords[:, 2:4])
        average_deviation = (
            ((np.mean(np.abs(atom_coords[:, 3] - atom_coords[:, 5]))) ** 2)
            + ((np.mean(np.abs(atom_coords[:, 2] - atom_coords[:, 4]))) ** 2)
        ) ** 0.5
        return average_deviation

    initial_x = image.shape[1] / 2
    initial_y = image.shape[0] / 2
    initial_guess = np.asarray((initial_y, initial_x))
    lower_bound = np.asarray(((initial_y - initial_y / 2), (initial_x - initial_x / 2)))
    upper_bound = np.asarray(((initial_y + initial_y / 2), (initial_x + initial_x / 2)))
    res = spo.minimize(
        fun=origin_function, x0=initial_guess, bounds=(lower_bound, upper_bound)
    )
    origin = res.x
    return origin


def get_coords(image, peak_pos, origin, current_coords):
    ang_1 = np.degrees(np.arctan2(current_coords[0, 1], current_coords[0, 0]))
    mag_1 = np.linalg.norm((current_coords[0, 1], current_coords[0, 0]))
    ang_2 = np.degrees(np.arctan2(current_coords[1, 1], current_coords[1, 0]))
    mag_2 = np.linalg.norm((current_coords[1, 1], current_coords[1, 0]))

    def coords_function(coord_vals, input_data=(peak_pos, origin, ang_1, ang_2)):
        mag_t = coord_vals[0]
        mag_b = coord_vals[1]
        peaks = input_data[0]
        rigin = input_data[1]
        ang_t = input_data[2]
        ang_b = input_data[3]
        xy_coords = np.asarray(
            (
                (mag_t * np.cos(np.deg2rad(ang_t)), mag_t * np.sin(np.deg2rad(ang_t))),
                (mag_b * np.cos(np.deg2rad(ang_b)), mag_b * np.sin(np.deg2rad(ang_b))),
            )
        )
        atom_coords = np.zeros((peaks.shape[0], 6))
        atom_coords[:, 0:2] = peaks[:, 0:2] - rigin[0:2]
        atom_coords[:, 2:4] = atom_coords[:, 0:2] @ np.linalg.inv(xy_coords)
        atom_coords[:, 4:6] = np.round(atom_coords[:, 2:4])
        average_deviation = (
            ((np.mean(np.abs(atom_coords[:, 3] - atom_coords[:, 5]))) ** 2)
            + ((np.mean(np.abs(atom_coords[:, 2] - atom_coords[:, 4]))) ** 2)
        ) ** 0.5
        return average_deviation

    initial_guess = np.asarray((mag_1, mag_2))
    lower_bound = initial_guess - (0.25 * initial_guess)
    upper_bound = initial_guess + (0.25 * initial_guess)
    res = spo.minimize(
        fun=coords_function, x0=initial_guess, bounds=(lower_bound, upper_bound)
    )
    mag = res.x
    new_coords = np.asarray(
        (
            (mag[0] * np.cos(np.deg2rad(ang_1)), mag[0] * np.sin(np.deg2rad(ang_1))),
            (mag[1] * np.cos(np.deg2rad(ang_2)), mag[1] * np.sin(np.deg2rad(ang_2))),
        )
    )
    return new_coords


def coords_of_atoms(peaks, coords, origin):
    """
    Convert atom positions to coordinates

    Parameters
    ----------
    peaks:  ndarray
            List of Gaussian fitted peaks
    coords: ndarray
            Co-ordinates of the axes

    Returns
    -------
    atom_coords: ndarray
                 Peak positions as the atom coordinates

    Notes
    -----
    One atom is chosen as the origin and the co-ordinates
    of all the atoms are calculated with respect to the origin

    :Authors:
    Debangshu Mukherjee <mukherjeed@ornl.gov>
    """
    atom_coords = np.zeros((peaks.shape[0], 8))
    atom_coords[:, 0:2] = peaks[:, 0:2] - origin[0:2]
    atom_coords[:, 2:4] = atom_coords[:, 0:2] @ np.linalg.inv(coords)
    atom_coords[:, 0:2] = peaks[:, 0:2]
    atom_coords[:, 4:6] = np.round(atom_coords[:, 2:4])
    atom_coords[:, 6:8] = atom_coords[:, 4:6] @ coords
    atom_coords[:, 6:8] = atom_coords[:, 6:8] + origin[0:2]
    return atom_coords


@numba.jit
def three_neighbors(peak_list, coords, delta=0.25):
    warnings.filterwarnings("ignore")
    no_atoms = peak_list.shape[0]
    atoms_neighbors = np.zeros((no_atoms, 8))
    atoms_distances = np.zeros((no_atoms, 4))
    for ii in range(no_atoms):
        atom_pos = peak_list[ii, 0:2]
        neigh_yy = atom_pos + coords[0, :]
        neigh_xx = atom_pos + coords[1, :]
        neigh_xy = atom_pos + coords[0, :] + coords[1, :]
        parnb_yy = ((peak_list[:, 0] - neigh_yy[0]) ** 2) + (
            (peak_list[:, 1] - neigh_yy[1]) ** 2
        )
        neigh_yy = (peak_list[parnb_yy == np.amin(parnb_yy), 0:2])[0]
        ndist_yy = np.linalg.norm(neigh_yy - atom_pos)
        parnb_xx = ((peak_list[:, 0] - neigh_xx[0]) ** 2) + (
            (peak_list[:, 1] - neigh_xx[1]) ** 2
        )
        neigh_xx = (peak_list[parnb_xx == np.amin(parnb_xx), 0:2])[0]
        ndist_xx = np.linalg.norm(neigh_xx - atom_pos)
        parnb_xy = ((peak_list[:, 0] - neigh_xy[0]) ** 2) + (
            (peak_list[:, 1] - neigh_xy[1]) ** 2
        )
        neigh_xy = (peak_list[parnb_xy == np.amin(parnb_xy), 0:2])[0]
        ndist_xy = np.linalg.norm(neigh_xy - atom_pos)
        atoms_neighbors[ii, :] = np.ravel(
            np.asarray((atom_pos, neigh_yy, neigh_xx, neigh_xy))
        )
        atoms_distances[ii, :] = np.ravel(np.asarray((0, ndist_yy, ndist_xx, ndist_xy)))
    yy_dist = np.linalg.norm(coords[0, :])
    yy_list = np.asarray(((yy_dist * (1 - delta)), (yy_dist * (1 + delta))))
    xx_dist = np.linalg.norm(coords[1, :])
    xx_list = np.asarray(((xx_dist * (1 - delta)), (xx_dist * (1 + delta))))
    xy_dist = np.linalg.norm(coords[0, :] + coords[1, :])
    xy_list = np.asarray(((xy_dist * (1 - delta)), (xy_dist * (1 + delta))))
    pp = atoms_distances[:, 1]
    pp[pp > yy_list[1]] = 0
    pp[pp < yy_list[0]] = 0
    pp[pp == 0] = np.nan
    atoms_distances[:, 1] = pp
    pp = atoms_distances[:, 2]
    pp[pp > xx_list[1]] = 0
    pp[pp < xx_list[0]] = 0
    pp[pp == 0] = np.nan
    atoms_distances[:, 2] = pp
    pp = atoms_distances[:, 3]
    pp[pp > xy_list[1]] = 0
    pp[pp < xy_list[0]] = 0
    pp[pp == 0] = np.nan
    atoms_distances[:, 3] = pp
    atoms_neighbors = atoms_neighbors[~np.isnan(atoms_distances).any(axis=1)]
    atoms_distances = atoms_distances[~np.isnan(atoms_distances).any(axis=1)]
    return atoms_neighbors, atoms_distances


@numba.jit
def relative_strain(n_list, coords):
    warnings.filterwarnings("ignore")
    identity = np.asarray(((1, 0), (0, 1)))
    axis_pos = np.asarray(((0, 0), (1, 0), (0, 1), (1, 1)))
    no_atoms = (np.shape(n_list))[0]
    coords_inv = np.linalg.inv(coords)
    cell_center = np.zeros((no_atoms, 2))
    e_xx = np.zeros(no_atoms)
    e_xy = np.zeros(no_atoms)
    e_yy = np.zeros(no_atoms)
    e_th = np.zeros(no_atoms)
    for ii in range(no_atoms):
        cc = np.zeros((4, 2))
        cc[0, :] = n_list[ii, 0:2] - n_list[ii, 0:2]
        cc[1, :] = n_list[ii, 2:4] - n_list[ii, 0:2]
        cc[2, :] = n_list[ii, 4:6] - n_list[ii, 0:2]
        cc[3, :] = n_list[ii, 6:8] - n_list[ii, 0:2]
        l_cc, _, _, _ = np.linalg.lstsq(axis_pos, cc, rcond=None)
        t_cc = np.matmul(l_cc, coords_inv) - identity
        e_yy[ii] = t_cc[0, 0]
        e_xx[ii] = t_cc[1, 1]
        e_xy[ii] = 0.5 * (t_cc[0, 1] + t_cc[1, 0])
        e_th[ii] = 0.5 * (t_cc[0, 1] - t_cc[1, 0])
        cell_center[ii, 0] = 0.25 * (
            n_list[ii, 0] + n_list[ii, 2] + n_list[ii, 4] + n_list[ii, 6]
        )
        cell_center[ii, 1] = 0.25 * (
            n_list[ii, 1] + n_list[ii, 3] + n_list[ii, 5] + n_list[ii, 7]
        )
    return cell_center, e_yy, e_xx, e_xy, e_th


def strain_map(centers, e_yy, e_xx, e_xy, e_th, mask):
    yr, xr = np.mgrid[0 : mask.shape[0], 0 : mask.shape[1]]
    cartcoord = list(zip(centers[:, 1], centers[:, 0]))

    e_yy[np.abs(e_yy) > 3 * np.median(np.abs(e_yy))] = 0
    e_xx[np.abs(e_xx) > 3 * np.median(np.abs(e_xx))] = 0
    e_xy[np.abs(e_xy) > 3 * np.median(np.abs(e_xy))] = 0
    e_th[np.abs(e_th) > 3 * np.median(np.abs(e_th))] = 0

    f_yy = scinterp.LinearNDInterpolator(cartcoord, e_yy)
    f_xx = scinterp.LinearNDInterpolator(cartcoord, e_xx)
    f_xy = scinterp.LinearNDInterpolator(cartcoord, e_xy)
    f_th = scinterp.LinearNDInterpolator(cartcoord, e_th)

    map_yy = f_yy(xr, yr)
    map_yy[np.isnan(map_yy)] = 0
    map_yy = np.multiply(map_yy, mask)

    map_xx = f_xx(xr, yr)
    map_xx[np.isnan(map_xx)] = 0
    map_xx = np.multiply(map_xx, mask)

    map_xy = f_xy(xr, yr)
    map_xy[np.isnan(map_xy)] = 0
    map_xy = np.multiply(map_xy, mask)

    map_th = f_th(xr, yr)
    map_th[np.isnan(map_th)] = 0
    map_th = np.multiply(map_th, mask)

    return map_yy, map_xx, map_xy, map_th


def create_circmask(image, center, radius, g_val=3, flip=True):
    """
    Use a Gaussian blurred image to fit
    peaks.

    Parameters
    ----------
    image:  ndarray
            2D array representing the image
    center: tuple
            Approximate location as (x,y) of
            the peak we are trying to fit
    radius: float
            Masking radius
    g_val:  float
            Value in pixels of the Gaussian
            blur applied. Default is 3
    flip:   bool
            Switch to flip refined center position
            from (x,y) to (y,x). Default is True
            which returns the center as (y,x)


    Returns
    -------
    masked_image: ndarray
                  Masked Image centered at refined
                  peak position
    new_center:   ndarray
                  Refined atom center as (y,x) if
                  flip switch is on, else the center
                  is returned as (x,y)

    Notes
    -----
    For some noisy datasets, a peak may be visible with
    the human eye, but getting a sub-pixel estimation of
    the peak position is often challenging, especially
    for FFT for diffraction patterns. This code Gaussian
    blurs the image, and returns the refined peak position

    See also
    --------
    st.util.fit_gaussian2D_mask

    :Authors:
    Debangshu Mukherjee <mukherjeed@ornl.gov>

    """
    blurred_image = scnd.filters.gaussian_filter(np.abs(image), g_val)
    fitted_diff = st.util.fit_gaussian2D_mask(
        blurred_image, center[0], center[1], radius
    )
    size_image = np.asarray(np.shape(image), dtype=int)
    yV, xV = np.mgrid[0 : size_image[0], 0 : size_image[1]]
    masked_image = np.zeros_like(blurred_image)
    if flip:
        new_center = np.asarray(np.flip(fitted_diff[0:2]))
        sub = (
            (((yV - new_center[0]) ** 2) + ((xV - new_center[1]) ** 2)) ** 0.5
        ) < radius
    else:
        new_center = np.asarray(fitted_diff[0:2])
        sub = (
            (((yV - new_center[1]) ** 2) + ((xV - new_center[0]) ** 2)) ** 0.5
        ) < radius
    masked_image[sub] = image[sub]
    return masked_image, new_center


@numba.jit(parallel=True)
def med_dist_numba(positions):
    warnings.filterwarnings("ignore")
    no_pos = len(positions)
    dist = np.empty(no_pos, dtype=np.float)
    ccd = np.empty(no_pos, dtype=np.float)
    for ii in numba.prange(no_pos):
        ccd = np.sum(((positions[:, 0:2] - positions[ii, 0:2]) ** 2), axis=1)
        dist[ii] = (np.amin(ccd[ccd > 0])) ** 0.5
    med_dist = 0.5 * np.median(dist)
    return med_dist


@numba.jit(parallel=True)
def refine_atoms_numba(image_data, positions, ref_arr, med_dist):
    warnings.filterwarnings("ignore")
    for ii in numba.prange(len(positions)):
        pos_x = positions[ii, 1]
        pos_y = positions[ii, 0]
        refined_ii = st.util.fit_gaussian2D_mask(1 + image_data, pos_x, pos_y, med_dist)
        ref_arr[ii, 0:2] = np.flip(refined_ii[0:2])
        ref_arr[ii, 2:6] = refined_ii[2:6]
        ref_arr[ii, -1] = refined_ii[-1] - 1


class atom_fit(object):
    """
    Locate atom columns in atomic resolution STEM images
    and then subsequently use gaussian peak fitting to
    refine the column location with sub-pixel precision.

    Parameters
    ----------
    image:       ndarray
                 The image from which the peak positions
                 will be ascertained.
    calib:       float
                 Size of an individual pixel
    calib_units: str
                 Unit of calibration

    References
    ----------
    1]_, Mukherjee, D., Miao, L., Stone, G. and Alem, N.,
         mpfit: a robust method for fitting atomic resolution images
         with multiple Gaussian peaks. Adv Struct Chem Imag 6, 1 (2020).

    Examples
    --------
    Run as:

    >>> atoms = st.afit.atom_fit(stem_image, calibration, 'nm')

    Then to check the image you just loaded, with the optional parameter
    `12` determining how many pixels of gaussian blur need to applied to
    calculate and separate a background. If in doubt, don't use it.

    >>> atoms.show_image(12)

    It is then optional to define a refernce region for the image.
    If such a region is defined, atom positions will only be ascertained
    grom the reference region. If you don't run this step, the entire image
    will be analyzed

    >>> atoms.define_reference((17, 7), (26, 7), (26, 24), (17, 24))

    Then, visualize the peaks:

    >>> atoms.peaks_vis(dist=0.1, thresh=0.1)

    `dist` indicates the distance between the
    peaks in calibration units. Play around with the numbers till
    you get a satisfactory result. Then run the gaussian peak refinement as:

    >>> atoms.refine_peaks()

    You can visualize your fitted peaks as:

    >>> atoms.show_peaks(style= 'separate')

    """

    def __init__(self, image, calib, calib_units):
        self.image = st.util.image_normalizer(image)
        self.imcleaned = np.copy(self.image)
        self.calib = calib
        self.calib_units = calib_units
        self.imshape = np.asarray(image.shape)
        self.peaks_check = False
        self.refining_check = False
        self.reference_check = False

    def show_image(self, gaussval=0, imsize=(15, 15), colormap="inferno"):
        """
        Parameters
        ----------
        gaussval: int, optional
                  Extent of Gaussian blurring in pixels
                  to generate a background image for
                  subtraction. Default is 0
        imsize:   tuple, optional
                  Size in inches of the image with the
                  diffraction spots marked. Default is (15, 15)
        colormap: str, optional
                  Colormap of the image. Default is inferno
        """
        self.gaussval = gaussval
        if gaussval > 0:
            self.gblur = scnd.gaussian_filter(self.image, gaussval)
            self.imcleaned = st.util.image_normalizer(self.image - self.gblur)
        self.gauss_clean = gaussval
        plt.figure(figsize=imsize)
        plt.imshow(self.imcleaned, cmap=colormap)
        scalebar = mpss.ScaleBar(self.calib, self.calib_units)
        scalebar.location = "lower right"
        scalebar.box_alpha = 1
        scalebar.color = "k"
        plt.gca().add_artist(scalebar)
        plt.axis("off")

    def define_reference(self, A_pt, B_pt, C_pt, D_pt, imsize=(15, 15), tColor="k"):
        """
        Locate the reference image.

        Parameters
        ----------
        A_pt:   tuple
                Top left position of reference region in (x, y)
        B_pt:   tuple
                Top right position of reference region in (x, y)
        C_pt:   tuple
                Bottom right position of reference region in (x, y)
        D_pt:   tuple
                Bottom left position of reference region in (x, y)
        imsize: tuple, optional
                Size in inches of the image with the
                diffraction spots marked. Default is
                (10, 10)
        tColor: str, optional
                Color of the text on the image. Default is black

        Notes
        -----
        Locates a reference region bounded by the four points given in
        length units. Choose the points in a clockwise fashion.
        """
        A = np.asarray(A_pt) / self.calib
        B = np.asarray(B_pt) / self.calib
        C = np.asarray(C_pt) / self.calib
        D = np.asarray(D_pt) / self.calib

        yy, xx = np.mgrid[0 : self.imshape[0], 0 : self.imshape[1]]
        yy = np.ravel(yy)
        xx = np.ravel(xx)
        ptAA = np.asarray((xx, yy)).transpose() - A
        ptBB = np.asarray((xx, yy)).transpose() - B
        ptCC = np.asarray((xx, yy)).transpose() - C
        ptDD = np.asarray((xx, yy)).transpose() - D
        angAABB = np.arccos(
            np.sum(ptAA * ptBB, axis=1)
            / (
                ((np.sum(ptAA ** 2, axis=1)) ** 0.5)
                * ((np.sum(ptBB ** 2, axis=1)) ** 0.5)
            )
        )
        angBBCC = np.arccos(
            np.sum(ptBB * ptCC, axis=1)
            / (
                ((np.sum(ptBB ** 2, axis=1)) ** 0.5)
                * ((np.sum(ptCC ** 2, axis=1)) ** 0.5)
            )
        )
        angCCDD = np.arccos(
            np.sum(ptCC * ptDD, axis=1)
            / (
                ((np.sum(ptCC ** 2, axis=1)) ** 0.5)
                * ((np.sum(ptDD ** 2, axis=1)) ** 0.5)
            )
        )
        angDDAA = np.arccos(
            np.sum(ptDD * ptAA, axis=1)
            / (
                ((np.sum(ptDD ** 2, axis=1)) ** 0.5)
                * ((np.sum(ptAA ** 2, axis=1)) ** 0.5)
            )
        )
        angsum = ((angAABB + angBBCC + angCCDD + angDDAA) / (2 * np.pi)).reshape(
            self.imcleaned.shape
        )
        self.ref_reg = np.isclose(angsum, 1)
        self.ref_reg = np.flipud(self.ref_reg)

        pixel_list = np.arange(0, self.calib * self.imshape[0], self.calib)
        no_labels = 10
        step_x = int(self.imshape[0] / (no_labels - 1))
        x_positions = np.arange(0, self.imshape[0], step_x)
        x_labels = np.round(pixel_list[::step_x], 1)
        fsize = int(1.5 * np.mean(np.asarray(imsize)))

        print(
            "Choose your points in a clockwise fashion, or else you will get a wrong result"
        )

        plt.figure(figsize=imsize)
        plt.imshow(
            np.flipud(self.imcleaned + 0.33 * self.ref_reg),
            cmap="magma",
            origin="lower",
        )
        plt.annotate(
            "A=" + str(A_pt),
            A / self.imshape,
            textcoords="axes fraction",
            size=fsize,
            color=tColor,
        )
        plt.annotate(
            "B=" + str(B_pt),
            B / self.imshape,
            textcoords="axes fraction",
            size=fsize,
            color=tColor,
        )
        plt.annotate(
            "C=" + str(C_pt),
            C / self.imshape,
            textcoords="axes fraction",
            size=fsize,
            color=tColor,
        )
        plt.annotate(
            "D=" + str(D_pt),
            D / self.imshape,
            textcoords="axes fraction",
            size=fsize,
            color=tColor,
        )
        plt.scatter(A[0], A[1], c="r")
        plt.scatter(B[0], B[1], c="r")
        plt.scatter(C[0], C[1], c="r")
        plt.scatter(D[0], D[1], c="r")
        plt.xticks(x_positions, x_labels, fontsize=fsize)
        plt.yticks(x_positions, x_labels, fontsize=fsize)
        plt.xlabel("Distance along X-axis (" + self.calib_units + ")", fontsize=fsize)
        plt.ylabel("Distance along Y-axis (" + self.calib_units + ")", fontsize=fsize)
        self.reference_check = True

    def peaks_vis(self, dist, thresh, gfilt=2, imsize=(15, 15), spot_color="c"):
        if not self.reference_check:
            self.ref_reg = np.ones_like(self.imcleaned, dtype=bool)
        pixel_dist = dist / self.calib
        self.imfilt = scnd.gaussian_filter(self.imcleaned, gfilt)
        self.threshold = thresh
        self.data_thresh = ((self.imfilt * self.ref_reg) - self.threshold) / (
            1 - self.threshold
        )
        self.data_thresh[self.data_thresh < 0] = 0
        data_peaks = skfeat.peak_local_max(
            self.data_thresh, min_distance=int(pixel_dist / 3), indices=False
        )
        peak_labels = scnd.measurements.label(data_peaks)[0]
        merged_peaks = scnd.measurements.center_of_mass(
            data_peaks, peak_labels, range(1, np.max(peak_labels) + 1)
        )
        peaks = np.array(merged_peaks)
        self.peaks = (st.afit.remove_close_vals(peaks, pixel_dist)).astype(np.float)
        spot_size = int(0.5 * np.mean(np.asarray(imsize)))
        plt.figure(figsize=imsize)
        plt.imshow(self.imfilt, cmap="magma")
        plt.scatter(self.peaks[:, 1], self.peaks[:, 0], c=spot_color, s=spot_size)
        scalebar = mpss.ScaleBar(self.calib, self.calib_units)
        scalebar.location = "lower right"
        scalebar.box_alpha = 1
        scalebar.color = "k"
        plt.gca().add_artist(scalebar)
        plt.axis("off")
        self.peaks_check = True

    def refine_peaks(self):
        """
        Calls the numba functions `med_dist_numba` and
        `refine_atoms_numba` to refine the peaks originally
        calculated.
        """
        if not self.peaks_check:
            raise RuntimeError("Please locate the initial peaks first as peaks_vis()")
        test = int(len(self.peaks) / 50)
        st.afit.med_dist_numba(self.peaks[0:test, :])
        md = st.afit.med_dist_numba(self.peaks)
        refined_peaks = np.empty((len(self.peaks), 7), dtype=np.float)

        # Run once on a smaller dataset to initialize JIT
        st.afit.refine_atoms_numba(
            self.imcleaned, self.peaks[0:test, :], refined_peaks[0:test, :], md
        )

        # Run the JIT compiled faster code on the full dataset
        st.afit.refine_atoms_numba(self.imcleaned, self.peaks, refined_peaks, md)
        self.refined_peaks = refined_peaks
        self.refining_check = True

    def show_peaks(self, imsize=(15, 15), style="together"):
        if not self.refining_check:
            raise RuntimeError("Please refine the atom peaks first as refine_peaks()")
        togsize = tuple(np.asarray((2, 1)) * np.asarray(imsize))
        spot_size = int(np.amin(np.asarray(imsize)))
        big_size = int(3 * spot_size)
        if style == "together":
            plt.figure(figsize=imsize)
            plt.imshow(self.imcleaned, cmap="magma")
            plt.scatter(
                self.peaks[:, 1],
                self.peaks[:, 0],
                c="c",
                s=big_size,
                label="Original Peaks",
            )
            plt.scatter(
                self.refined_peaks[:, 1],
                self.refined_peaks[:, 0],
                c="r",
                s=spot_size,
                label="Fitted Peaks",
            )
            plt.gca().legend(loc="upper left", markerscale=3, framealpha=1)
            scalebar = mpss.ScaleBar(self.calib, self.calib_units)
            scalebar.location = "lower right"
            scalebar.box_alpha = 1
            scalebar.color = "k"
            plt.gca().add_artist(scalebar)
            plt.axis("off")
        else:
            plt.figure(figsize=togsize)
            plt.subplot(1, 2, 1)
            plt.imshow(self.imcleaned, cmap="magma")
            plt.scatter(
                self.peaks[:, 1],
                self.peaks[:, 0],
                c="b",
                s=spot_size,
                label="Original Peaks",
            )
            plt.gca().legend(loc="upper left", markerscale=3, framealpha=1)
            scalebar = mpss.ScaleBar(self.calib, self.calib_units)
            scalebar.location = "lower right"
            scalebar.box_alpha = 1
            scalebar.color = "k"
            plt.gca().add_artist(scalebar)
            plt.axis("off")

            plt.subplot(1, 2, 2)
            plt.imshow(self.imcleaned, cmap="magma")
            plt.scatter(
                self.refined_peaks[:, 1],
                self.refined_peaks[:, 0],
                c="k",
                s=spot_size,
                label="Fitted Peaks",
            )
            plt.gca().legend(loc="upper left", markerscale=3, framealpha=1)
            scalebar = mpss.ScaleBar(self.calib, self.calib_units)
            scalebar.location = "lower right"
            scalebar.box_alpha = 1
            scalebar.color = "k"
            plt.gca().add_artist(scalebar)
            plt.axis("off")
