import numpy as np
import scipy.optimize as spo
import scipy.ndimage as scnd
import stemtool as st


def gaussian_2D_function(xy, x0, y0, theta, sigma_x, sigma_y, amplitude):
    """
    The underlying 2D Gaussian function

    Parameters
    ----------
    xy:        tuple
               x and y positions
    x0:        float
               x center of Gaussian peak
    y0:        float
               y center of Gaussian peak
    theta:     float
               Rotation of the 2D gaussian peak in radians
    sigma_x:   float
               Standard deviation of the 2D Gaussian along x
    sigma_y:   float
               Standard deviation of the 2D Gaussian along y
    amplitude: float
               Peak intensity

    Returns
    -------
    gaussvals: ndarray
               A Gausian peak centered at x0, y0 based on the
               parameters given

    Notes
    -----
    The Gaussian 2D function is calculated at every x and y position
    for a list of position values based on the parameters given.

    See also
    --------
    gauss2D

    :Authors:
    Debangshu Mukherjee <mukherjeed@ornl.gov>
    """
    x = xy[0] - x0
    y = xy[1] - y0
    term_1 = (((np.cos(theta)) ** 2) / (2 * (sigma_x ** 2))) + (
        ((np.sin(theta)) ** 2) / (2 * (sigma_y ** 2))
    )
    term_2 = ((np.sin(2 * theta)) / (2 * (sigma_x ** 2))) - (
        (np.sin(2 * theta)) / (2 * (sigma_y ** 2))
    )
    term_3 = (((np.sin(theta)) ** 2) / (2 * (sigma_x ** 2))) + (
        ((np.cos(theta)) ** 2) / (2 * (sigma_y ** 2))
    )
    expo_1 = term_1 * (x ** 2)
    expo_2 = term_2 * x * y
    expo_3 = term_3 * (y ** 2)
    gaussvals = np.ravel(amplitude * np.exp((-1) * (expo_1 + expo_2 + expo_3)))
    return gaussvals


def gauss2D(im_size, x0, y0, theta, sigma_x, sigma_y, amplitude):
    """
    Return a 2D Gaussian function centered at x0, y0

    Parameters
    ----------
    im_size:   tuple
               Size of the image where a Gaussian peak
               is to be generated
    x0:        float
               x center of Gaussian peak
    y0:        float
               y center of Gaussian peak
    theta:     float
               Rotation of the 2D gaussian peak in radians
    sigma_x:   float
               Standard deviation of the 2D Gaussian along x
    sigma_y:   float
               Standard deviation of the 2D Gaussian along y
    amplitude: float
               Peak intensity

    Returns
    -------
    gauss2D: ndarray
             2D Gaussian peak of shape im_size

    Notes
    -----
    This returns a 2D ndarray with the the size as im_size, with
    a 2D gaussian function centered at (x0,y0) defined by the
    input parameters.

    See also
    --------
    gauss2D_function

    :Authors:
    Debangshu Mukherjee <mukherjeed@ornl.gov>
    """
    yr, xr = np.mgrid[0 : im_size[0], 0 : im_size[1]]
    gauss2D = np.zeros(yr, dtype=np.float)
    x = xr - x0
    y = yr - y0
    term_1 = (((np.cos(theta)) ** 2) / (2 * (sigma_x ** 2))) + (
        ((np.sin(theta)) ** 2) / (2 * (sigma_y ** 2))
    )
    term_2 = ((np.sin(2 * theta)) / (2 * (sigma_x ** 2))) - (
        (np.sin(2 * theta)) / (2 * (sigma_y ** 2))
    )
    term_3 = (((np.sin(theta)) ** 2) / (2 * (sigma_x ** 2))) + (
        ((np.cos(theta)) ** 2) / (2 * (sigma_y ** 2))
    )
    expo_1 = term_1 * np.multiply(x, x)
    expo_2 = term_2 * np.multiply(x, y)
    expo_3 = term_3 * np.multiply(y, y)
    gauss2D[yr, xr] = amplitude * np.exp((-1) * (expo_1 + expo_2 + expo_3))
    return gauss2D


def initialize_gauss2D(xx, yy, zz, center_type="COM"):
    """
    Generate an approximate Gaussian based on image

    Parameters
    ----------
    xx:          ndarray
                 X positions
    yy:          ndarray
                 Y Positions
    zz:          ndarray
                 Image value at the positions
    center_type: str
                 Default is `COM` which uses the center
                 of mass of the given positions to generate
                 the starting gaussian center. The other option
                 is `maxima` which takes the maximum intensity
                 value as the starting point.

    Returns
    -------
    gauss_ini: tuple
               X_center, Y_center, Angle, X_std, Y_std, Amplitude

    Notes
    -----
    For a given list of x positions, y positions and corresponding
    intensity values, this code returns a first pass approximation
    of a Gaussian function. The center of the gaussian 2D function
    can either be the center of mass or the intensity maxima, and is
    user defined. The angle is always given as 0.

    :Authors:
    Debangshu Mukherjee <mukherjeed@ornl.gov>
    """
    if center_type == "maxima":
        x_com = xx[zz == np.amax(zz)]
        y_com = yy[zz == np.amax(zz)]
    elif center_type == "COM":
        total = np.sum(zz)
        x_com = np.sum(xx * zz) / total
        y_com = np.sum(yy * zz) / total
    else:
        raise ValueError("Invalid Center Type")

    zz_norm = st.util.image_normalizer(zz)
    x_fwhm = xx[zz_norm > 0.5]
    x_fwhm = np.abs(x_fwhm - x_com)
    y_fwhm = yy[zz_norm > 0.5]
    y_fwhm = np.abs(y_fwhm - y_com)

    sigma_x = np.amax(x_fwhm) / (2 * ((2 * np.log(2)) ** 0.5))
    sigma_y = np.amax(y_fwhm) / (2 * ((2 * np.log(2)) ** 0.5))
    height = np.amax(zz)
    gauss_ini = (x_com, y_com, 0, sigma_x, sigma_y, height)
    return gauss_ini


def fit_gaussian2D_mask(
    image_data, mask_x, mask_y, mask_radius, mask_type="circular", center_type="COM"
):
    """
    Fit a 2D gaussian to a masked image based on
    the location of the mask, size of the mask and
    the type of the mask

    Parameters
    ----------
    image_data:  ndarray
                 The image that will be fitted with the Gaussian
    mask_x:      float
                 x center of the mask
    mask_y:      float
                 y center of the mask
    mask_radius: float
                 The size of the mask. For a circulat mask this
                 refers to the mask radius, while for a square mask
                 this refers to half the side of the square
    mask_type:   str
                 Default is `circular`, while the other option is `square`
    center_type: str
                 Center location for the first pass of the Gaussian.
                 Default is `COM`, while the other options are `minima`
                 or `maxima`.

    Returns
    -------
    popt: tuple
          Refined X position, Refined Y Position, Rotation angle of
          2D Gaussian, Standard deviation(s), Amplitude

    Notes
    -----
    This code uses the `scipy.optimize.curve_fit` module to fit a 2D
    Gaussian peak to masked data. `mask_x` and `mask_y` refer to the
    initial starting positions. Also, this can take in `minima` as a
    string for initializing Gaussian peaks, which allows for atom column
    mapping in inverted contrast images too.

    See also
    --------
    gaussian_2D_function
    initialize_gauss2D

    :Authors:
    Debangshu Mukherjee <mukherjeed@ornl.gov>
    """
    p, q = np.shape(image_data)
    yV, xV = np.mgrid[0:p, 0:q]
    if mask_type == "circular":
        sub = ((((yV - mask_y) ** 2) + ((xV - mask_x) ** 2)) ** 0.5) < mask_radius
    elif mask_type == "square":
        sub = np.logical_and(
            (np.abs(yV - mask_y) < mask_radius), (np.abs(xV - mask_x) < mask_radius)
        )
    else:
        raise ValueError("Unknown Mask Type")
    x_pos = np.asarray(xV[sub], dtype=np.float64)
    y_pos = np.asarray(yV[sub], dtype=np.float64)
    masked_image = np.asarray(image_data[sub], dtype=np.float64)
    mi_min = np.amin(masked_image)
    mi_max = np.amax(masked_image)
    if center_type == "minima":
        calc_image = (masked_image - mi_max) / (mi_min - mi_max)
        initial_guess = initialize_gauss2D(x_pos, y_pos, calc_image, "maxima")
    else:
        calc_image = (masked_image - mi_min) / (mi_max - mi_min)
        initial_guess = initialize_gauss2D(x_pos, y_pos, calc_image, center_type)
    lower_bound = (
        (initial_guess[0] - mask_radius),
        (initial_guess[1] - mask_radius),
        -180,
        0,
        0,
        ((-2.5) * initial_guess[5]),
    )
    upper_bound = (
        (initial_guess[0] + mask_radius),
        (initial_guess[1] + mask_radius),
        180,
        (2.5 * mask_radius),
        (2.5 * mask_radius),
        (2.5 * initial_guess[5]),
    )
    xy = (x_pos, y_pos)
    popt, _ = spo.curve_fit(
        gaussian_2D_function,
        xy,
        calc_image,
        initial_guess,
        bounds=(lower_bound, upper_bound),
        ftol=0.01,
        xtol=0.01,
    )
    if center_type == "minima":
        popt[-1] = (popt[-1] * (mi_min - mi_max)) + mi_max
    popt[-1] = (popt[-1] * (mi_max - mi_min)) + mi_min
    return popt


def gaussian_1D_function(x, x0, sigma_x, amplitude):
    """
    The underlying 1D Gaussian function

    Parameters
    ----------
    x:         ndarray
               x positions
    x0:        float
               x center of Gaussian peak
    sigma_x:   float
               Standard deviation of the 2D Gaussian along x
    amplitude: float
               Peak intensity

    Returns
    -------
    gaussvals: ndarray
               A Gausian peak centered at x0 based on the
               parameters given

    Notes
    -----
    The Gaussian 1D function is calculated at every x position for
    a list of position values based on the parameters given.

    See also
    --------
    gaussian_2D_function
    initialize_gauss1D
    fit_gaussian1D_mask
    """
    x = x - x0
    term = (x ** 2) / (2 * (sigma_x ** 2))
    gaussvals = amplitude * np.exp((-1) * term)
    return gaussvals


def initialize_gauss1D(xx, yy, center_type="COM"):
    """
    Generate an approximate Gaussian based on signal

    Parameters
    ----------
    xx:          ndarray
                 X positions
    yy:          ndarray
                 Signal value at the positions
    center_type: str
                 Default is `COM` which uses the center
                 of mass of the given positions to generate
                 the starting gaussian center. The other option
                 is `maxima` which takes the maximum intensity
                 value as the starting point.

    Returns
    -------
    gauss_ini: tuple
               X_center, X_std, Amplitude

    Notes
    -----
    For a given list of x positions and corresponding signal
    values, this code returns a first pass approximation of a
    1D Gaussian function. The center of the function can either
    be the center of mass or the intensity maxima, and is user
    defined.
    """
    if center_type == "maxima":
        x_com = xx[yy == np.amax(yy)]
        x_com = x_com[0]
    elif center_type == "COM":
        total = np.sum(yy)
        x_com = np.sum(np.multiply(xx, yy)) / total
    else:
        raise ValueError("Invalid Center Type")
    yy_norm = st.util.image_normalizer(yy)
    x_fwhm = xx[yy_norm > 0.5]
    x_fwhm = np.abs(x_fwhm - x_com)
    sigma_x = np.amax(x_fwhm) / (2 * ((2 * np.log(2)) ** 0.5))
    height = np.amax(yy)
    return x_com, sigma_x, height


def fit_gaussian1D_mask(signal, position, mask_width, center_type="COM"):
    """
    Fit a 2D gaussian to a masked image based on
    the location of the mask, size of the mask and
    the type of the mask

    Parameters
    ----------
    signal:      ndarray
                 The 1D signal that will be fitted with the Gaussian
    position:    float
                 x center of the mask
    mask_width:  float
                 The size of the mask.
    center_type: str
                 Center location for the first pass of the Gaussian.
                 Default is `COM`, while the other options are `minima`
                 or `maxima`.

    Returns
    -------
    popt: tuple
          Refined X position, Standard deviation, Amplitude

    Notes
    -----
    This code uses the `scipy.optimize.curve_fit` module to fit a 1D
    Gaussian peak to masked data. `mask_x` refers to the initial
    starting position. Also, this can take in `minima` as a string
    for initializing Gaussian peaks.

    See also
    --------
    fit_gaussian2D_mask
    initialize_gauss1D
    gaussian_1D_function

    :Authors:
    Debangshu Mukherjee <mukherjeed@ornl.gov>
    """
    xV = np.arange(np.len(signal))
    sub = np.abs(xV - position) < mask_width
    x_pos = np.asarray(xV[sub], dtype=np.float)
    masked_signal = np.asarray(signal[sub], dtype=np.float)
    mi_min = np.amin(masked_signal)
    mi_max = np.amax(masked_signal)
    if center_type == "minima":
        calc_signal = (masked_signal - mi_max) / (mi_min - mi_max)
        initial_guess = initialize_gauss1D(x_pos, calc_signal, "maxima")
    else:
        calc_signal = (masked_signal - mi_min) / (mi_max - mi_min)
        initial_guess = initialize_gauss1D(x_pos, calc_signal, center_type)
    lower_bound = ((initial_guess[0] - mask_width), 0, ((-2.5) * initial_guess[5]))
    upper_bound = (
        (initial_guess[0] + mask_width),
        (2.5 * mask_width),
        (2.5 * initial_guess[5]),
    )
    popt, _ = spo.curve_fit(
        gaussian_1D_function,
        x_pos,
        calc_signal,
        initial_guess,
        bounds=(lower_bound, upper_bound),
        ftol=0.01,
        xtol=0.01,
    )
    if center_type == "minima":
        popt[-1] = (popt[-1] * (mi_min - mi_max)) + mi_max
    popt[-1] = (popt[-1] * (mi_max - mi_min)) + mi_min
    return popt
