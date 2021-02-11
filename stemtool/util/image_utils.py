import numpy as np
import matplotlib as mpl
import mpl_toolkits.axes_grid1 as mplax
import matplotlib.colors as mplc
import matplotlib.cm as mplcm
import numba
import warnings
import scipy.misc as scm
import scipy.optimize as spo
import scipy.ndimage as scnd
import scipy.signal as scsig
import skimage.color as skc
import stemtool as st


def move_by_phase(image_to_move, x_pixels, y_pixels):
    """
    Move Images with sub-pixel precision

    Parameters
    ----------
    image_to_move: ndarray
                   Original Image to be moved
    x_pixels: float
              Pixels to shift in X direction
    y_pixels: float
              Pixels to Shift in Y direction

    Returns
    -------
    moved_image: ndarray
                 Moved Image

    Notes
    -----
    The underlying idea is that a shift in the real space
    is phase shift in Fourier space. So we take the original
    image, and take it's Fourier transform. Also, we calculate
    how much the image shifts result in the phase change, by
    calculating the Fourier pixel dimensions. We then multiply
    the FFT of the image with the phase shift value and then
    take the inverse FFT.

    :Authors:
    Debangshu Mukherjee <mukherjeed@ornl.gov>
    """
    image_size = (np.asarray(image_to_move.shape)).astype(int)
    fourier_cal_y = np.linspace(
        (-image_size[0] / 2), ((image_size[0] / 2) - 1), image_size[0]
    )
    fourier_cal_y = fourier_cal_y / (image_size[0]).astype(np.float64)
    fourier_cal_x = np.linspace(
        (-image_size[1] / 2), ((image_size[1] / 2) - 1), image_size[1]
    )
    fourier_cal_x = fourier_cal_x / (image_size[1]).astype(np.float64)
    [fourier_mesh_x, fourier_mesh_y] = np.meshgrid(fourier_cal_x, fourier_cal_y)
    move_matrix = np.multiply(fourier_mesh_x, x_pixels) + np.multiply(
        fourier_mesh_y, y_pixels
    )
    move_phase = np.exp((-2) * np.pi * 1j * move_matrix)
    original_image_fft = np.fft.fftshift(np.fft.fft2(image_to_move))
    moved_in_fourier = np.multiply(move_phase, original_image_fft)
    moved_image = np.fft.ifft2(moved_in_fourier)
    return moved_image


def image_normalizer(image_orig):
    """
    Normalizing Image

    Parameters
    ----------
    image_orig: ndarray
                'image_orig' is the original input image to be normalized

    Returns
    -------
    image_norm: ndarray
                Normalized Image

    Notes
    -----
    We normalize a real valued image here
    so that it's values lie between o and 1.
    This is done by first subtracting the
    minimum value of the image from the
    image itself, and then subsequently
    dividing the image by the maximum value
    of the subtracted image.

    :Authors:
    Debangshu Mukherjee <mukherjeed@ornl.gov>

    """
    image_norm = np.zeros_like(image_orig, dtype=np.float64)
    image_norm = (image_orig - np.amin(image_orig)) / (
        np.amax(image_orig) - np.amin(image_orig)
    )
    return image_norm


def image_logarizer(image_orig, bit_depth=64):
    """
    Normalized log of image

    Parameters
    ----------
    image_orig: ndarray
                Numpy array of real valued image
    bit_depth: int
               Bit depth of output image
               Default is 32

    Returns
    -------
    image_log: ndarray
               Normalized log

    Notes
    -----
    Normalize the image, and scale it 2^0 to 2^bit_depth.
    Take log2 of the scaled image.

    :Authors:
    Debangshu Mukherjee <mukherjeed@ornl.gov>
    """
    bit_max = 2 ** bit_depth
    image_norm = st.util.image_normalizer(image_orig)
    image_scale = np.zeros_like(image_norm, dtype=np.float64)
    image_log = np.zeros_like(image_norm, dtype=np.float64)
    image_scale = 1 + ((bit_max - 1) * image_norm)
    image_log = np.log2(image_scale)
    return image_log


def remove_dead_pixels(image_orig, iter_count=1, level=10000):
    """
    Remove dead pixels

    Parameters
    ----------
    image_orig: ndarray
                Numpy array of real valued image
    iter_count: int
                Number of iterations to run
                the process. Default is 1
    level:      int,float
                Ratio of minima pixels to total
                pixels. Default is 10,000

    Returns
    -------
    image_orig: ndarray
                Image with dead pixels converted

    Notes
    -----
    Subtract the minima from the image, and if the
    number of pixels with minima values is less than
    the 1/level of the total pixels, then those are
    decided to be dead pixels. Iterate if necessary

    :Authors:
    Debangshu Mukherjee <mukherjeed@ornl.gov>
    """
    pp, _ = np.mgrid[0 : image_orig.shape[0], 0 : image_orig.shape[1]]
    no_points = np.size(pp)
    for _ in range(iter_count):
        original_min = np.amin(image_orig)
        image_pos = image_orig - original_min
        no_minima = np.size(pp[image_pos == 0])
        if no_minima < (no_points / level):
            new_minimum = np.amin(image_pos[image_pos > 0])
        image_pos = image_pos - new_minimum
        image_pos[image_pos < 0] = 0
        image_orig = image_pos + new_minimum + original_min
    return image_orig


def hanned_image(image):
    """
    2D hanning filter for images

    Parameters
    ----------
    image: ndarray
           Original Image on which the Hanning filter
           is to be applied

    Returns
    -------
    hanned_image: ndarray
                  Image with the hanning filter applied

    Notes
    -----


    :Authors:
    Debangshu Mukherjee <mukherjeed@ornl.gov>
    """
    size_image = np.asarray(np.shape(image), dtype=int)
    row_hann = np.zeros((size_image[0], 1))
    row_hann[:, 0] = np.hanning(size_image[0])
    col_hann = np.zeros((1, size_image[1]))
    col_hann[0, :] = np.hanning(size_image[1])
    hann_window = np.multiply(row_hann, col_hann)
    hanned_image = np.multiply(image, hann_window)
    return hanned_image


def sane_colorbar(mappable):
    ax = mappable.axes
    fig = ax.figure
    divider = mplax.make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    return fig.colorbar(mappable, cax=cax)


def phase_color(phase_image):
    size_im = np.asarray(np.shape(phase_image), dtype=int)
    hsv_im = np.ones((size_im[0], size_im[1], 3))
    hsv_im[:, :, 0] = (phase_image + (2 * np.pi)) / (2 * np.pi)
    hsv_im[:, :, 0] = hsv_im[:, :, 0] - np.floor(hsv_im[:, :, 0])
    rgb_im = mplc.hsv_to_rgb(hsv_im)
    r, g, b = rgb_im[:, :, 0], rgb_im[:, :, 1], rgb_im[:, :, 2]
    gray_im = (0.2989 * r) + (0.5870 * g) + (0.1140 * b)
    gray_im = gray_im
    hsv_im[:, :, 2] = np.divide(hsv_im[:, :, 2], gray_im)
    hsv_im[:, :, 2] = hsv_im[:, :, 2] / np.amax(hsv_im[:, :, 2])
    rgb_im = mplc.hsv_to_rgb(hsv_im)
    return rgb_im


def hsv_overlay(data, image, color, climit=None, bit_depth=8):
    bit_range = 2 ** bit_depth
    im_map = mplcm.get_cmap(color, bit_range)
    if climit == None:
        data_lim = np.amax(np.abs(data))
    else:
        data_lim = climit
    data = 0.5 + (data / (2 * data_lim))
    rgb_image = np.asarray(im_map(data)[:, :, 0:3])
    hsv_image = mplc.rgb_to_hsv(rgb_image)
    hsv_image[:, :, -1] = (image - np.amin(image)) / (np.amax(image) - np.amin(image))
    rgb_image = mplc.hsv_to_rgb(hsv_image)
    return rgb_image


def sparse_division(sparse_numer, sparse_denom, bit_depth=32):
    """
    Divide two sparse matrices element wise to prevent zeros

    Parameters
    ----------
    spase_numer: ndarray
                 Numpy array of real valued numerator
    sparse_denom: ndarray
                  Numpy array of real valued denominator
    bit_depth: int
               Bit depth of output image
               Default is 32

    Returns
    -------
    divided_matrix: ndarray
                    Quotient matrix

    Notes
    -----
    Decide on a bit depth below which
    the values in the denominator are
    just noise, as they are below the
    bit depth. Do the same for the
    numerator. Turn those values to 1 in
    the denominator and 0 in the numerator.
    Then in the quotient matrix, turn the
    denominator values below the threshold
    to 0 too.

    :Authors:
    Debangshu Mukherjee <mukherjeed@ornl.gov>
    """
    depth_ratio = 2 ** bit_depth
    denom_abs = np.abs(sparse_denom)
    numer_abs = np.abs(sparse_numer)
    threshold_denom = (np.amax(denom_abs)) / depth_ratio
    threshold_numer = (np.amax(numer_abs)) / depth_ratio
    threshold_ind_denom = denom_abs < threshold_denom
    threshold_ind_numer = numer_abs < threshold_numer
    sparse_denom[threshold_ind_denom] = 1
    sparse_numer[threshold_ind_numer] = 0
    divided_matrix = np.divide(sparse_numer, sparse_denom)
    divided_matrix[threshold_ind_denom] = 0
    return divided_matrix


def cross_corr_unpadded(image_1, image_2, normal=True):
    if normal:
        im1_norm = image_1 / (np.sum(image_1 ** 2) ** 0.5)
        im2_norm = image_2 / (np.sum(image_2 ** 2) ** 0.5)
        im1_fft = np.fft.fft2(im1_norm)
        im2_fft = np.conj(np.fft.fft2(im2_norm))
    else:
        im1_fft = np.fft.fft2(image_1)
        im2_fft = np.conj(np.fft.fft2(image_2))
    corr_fft = np.abs(np.fft.ifftshift(np.fft.ifft2(im1_fft * im2_fft)))
    return corr_fft


def cross_corr(image_1, image_2, hybridizer=0, normal=True):
    """
    Normalized Correlation, allowing for hybridization
    with cross correlation being the default output if
    no hybridization parameter is given

    Parameters
    ----------
    image_1: ndarray
             First image
    image_2: ndarray
             Second image
    hybridizer: float
                Hybridization parameter between 0 and 1
                0 is pure cross correlation
                1 is pure phase correlation

    Returns
    -------
    corr_hybrid: ndarray
                 Complex valued correlation output

    Notes
    -----
    The cross-correlation theorem can be stated as:
    .. math::
         G_c = G_1 \times G_2^*

    where :math:`G_c` is the Fourier transform of the cross
    correlated matrix and :math:`G_1` and :math:`G_2` are
    the Fourier transforms of :math:`g_1` and :math:`g_2`
    respectively, which are the original matrices. This is pure
    cross-correlation. Phase correlation can be expressed as:
    .. math::
         G_c = \frac{G_1 \times G_2^*}{\mid G_1 \times G_2^* \mid}

    Thus, we can now define a hybrid cross-correlation where
    .. math::
         G_c = \frac{G_1 \times G_2^*}{\mid G_1 \times G_2^* \mid ^n}

    If n is 0, we have cross correlation, and if n is 1 we
    have phase correlation.

    References
    ----------
    1]_, Pekin, T.C., Gammer, C., Ciston, J., Minor, A.M. and Ophus, C.,
         2017. Optimizing disk registration algorithms for nanobeam
         electron diffraction strain mapping. Ultramicroscopy, 176,
         pp.170-176.

    See Also
    --------
    sparse_division

    :Authors:
    Debangshu Mukherjee <mukherjeed@ornl.gov>
    """
    im_size = np.asarray(np.shape(image_1))
    pad_size = (np.round(im_size / 2)).astype(int)
    if normal:
        im1_norm = image_1 / (np.sum(image_1 ** 2) ** 0.5)
        im1_pad = np.pad(im1_norm, pad_width=pad_size, mode="median")
        im2_norm = image_2 / (np.sum(image_2 ** 2) ** 0.5)
        im2_pad = np.pad(im2_norm, pad_width=pad_size, mode="median")
        im1_fft = np.fft.fft2(im1_pad)
        im2_fft = np.conj(np.fft.fft2(im2_pad))
    else:
        im1_pad = np.pad(image_1, pad_width=pad_size, mode="median")
        im2_pad = np.pad(image_2, pad_width=pad_size, mode="median")
        im1_fft = np.fft.fft2(im1_pad)
        im2_fft = np.conj(np.fft.fft2(im2_pad))
    corr_fft = np.multiply(im1_fft, im2_fft)
    corr_abs = (np.abs(corr_fft)) ** hybridizer
    corr_hybrid_fft = sparse_division(corr_fft, corr_abs, 32)
    corr_hybrid = np.fft.ifft2(corr_hybrid_fft)
    corr_hybrid = np.abs(np.fft.ifftshift(corr_hybrid))
    corr_unpadded = corr_hybrid[
        pad_size[0] : pad_size[0] + im_size[0], pad_size[1] : pad_size[1] + im_size[1]
    ]
    return corr_unpadded


def make_circle(size_circ, center_x, center_y, radius):
    """
    Make a circle

    Parameters
    ----------
    size_circ: ndarray
               2 element array giving the size of the output matrix
    center_x: float
              x position of circle center
    center_y: float
              y position of circle center
    radius: float
            radius of the circle

    Returns
    -------
    circle: ndarray
            p X q sized array where the it is 1
            inside the circle and 0 outside

    :Authors:
    Debangshu Mukherjee <mukherjeed@ornl.gov>
    """
    p = size_circ[0]
    q = size_circ[1]
    yV, xV = np.mgrid[0:p, 0:q]
    sub = ((((yV - center_y) ** 2) + ((xV - center_x) ** 2)) ** 0.5) < radius
    circle = np.asarray(sub, dtype=np.float64)
    return circle


@numba.jit
def image_tiler(dataset_4D, reducer=4, bit_depth=8):
    """
    Generate a tiled pattern of the 4D-STEM dataset

    """
    size_data = (np.asarray(dataset_4D.shape)).astype(int)
    normalized_4D = (dataset_4D - np.amin(dataset_4D)) / (
        np.amax(dataset_4D) - np.amin(dataset_4D)
    )
    reduced_size = np.zeros(2)
    reduced_size[0:2] = np.round(size_data[0:2] * (1 / reducer))
    reduced_size = reduced_size.astype(int)
    tile_size = np.multiply(reduced_size, (size_data[2], size_data[3]))
    image_tile = np.zeros(tile_size)
    for jj in range(size_data[3]):
        for ii in range(size_data[2]):
            ronchi = normalized_4D[:, :, ii, jj]
            xRange = (ii * reduced_size[0]) + np.arange(reduced_size[0])
            xStart = int(xRange[0])
            xEnd = 1 + int(xRange[-1])
            yRange = (jj * reduced_size[1]) + np.arange(reduced_size[1])
            yStart = int(yRange[0])
            yEnd = 1 + int(yRange[-1])
            image_tile[xStart:xEnd, yStart:yEnd] = resizer2D((ronchi + 1), reducer) - 1
    image_tile = image_tile - np.amin(image_tile)
    image_tile = (2 ** bit_depth) * (image_tile / (np.amax(image_tile)))
    image_tile = image_tile.astype(int)
    return image_tile


@numba.jit
def flip_corrector(data4D):
    """
    Correcting Image Flip

    Parameters
    ----------
    image_orig: ndarray
                'data4D' is the original 4D dataset in the
                form DiffX,DiffY,ScanX,ScanY

    Returns
    -------
    image_norm: ndarray
                Flip corrected 4D dataset

    Notes
    -----
    The microscope lenses may add a flip in
    the X direction of the ronchigram. This
    corrects for that.

    :Authors:
    Debangshu Mukherjee <mukherjeed@ornl.gov>

    """
    warnings.filterwarnings("ignore")
    datasize = (np.asarray(data4D.shape)).astype(int)
    flipped4D = np.zeros((datasize[0], datasize[1], datasize[2], datasize[3]))
    for jj in range(datasize[3]):
        for ii in range(datasize[2]):
            ronchi = data4D[:, :, ii, jj]
            ronchi_flip = np.fliplr(ronchi)
            flipped4D[:, :, ii, jj] = ronchi_flip
    return flipped4D


def array_rms(arr):
    arr_sq = arr ** 2
    arr_mean = np.mean(arr_sq)
    arr_rms = (arr_mean) ** 0.5
    return arr_rms


def sobel_circle(image):
    sobel_image, _ = st.util.sobel(image)
    yy, xx = np.mgrid[0 : sobel_image.shape[0], 0 : sobel_image.shape[1]]
    center_y, center_x = np.asarray(scnd.measurements.center_of_mass(sobel_image))
    rr = (((yy - center_y) ** 2) + ((xx - center_x) ** 2)) ** 0.5
    initial_guess = st.util.initialize_gauss1D(
        np.ravel(rr), np.ravel(sobel_image), "maxima"
    )
    popt, _ = spo.curve_fit(
        st.util.gaussian_1D_function,
        xdata=np.ravel(rr),
        ydata=np.ravel(sobel_image),
        p0=initial_guess,
    )
    radius = popt[0]
    return center_x, center_y, radius


def circle_function(xy, x0, y0, radius):
    xx = xy[0] - x0
    yy = xy[1] - y0
    zz = ((xx ** 2) + (yy ** 2)) ** 0.5
    zz[zz > radius] = 0
    zz[zz > 0] = 1
    return zz


def fit_circle(image_data, med_factor=50):
    image_data = image_data.astype(np.float64)
    image_data[
        image_data > (med_factor * np.median(image_data))
    ] = med_factor * np.median(image_data)
    image_data[image_data < (np.median(image_data) / med_factor)] = (
        np.median(image_data) / med_factor
    )
    calc_image = (image_data - np.amin(image_data)) / (
        np.amax(image_data) - np.amin(image_data)
    )
    image_size = (np.asarray(np.shape(image_data))).astype(np.float64)
    initial_x = image_size[1] / 2
    initial_y = image_size[0] / 2
    initial_radius = (np.sum(calc_image) / np.pi) ** 0.5
    initial_guess = (initial_x, initial_y, initial_radius)
    yV, xV = np.mgrid[0 : int(image_size[0]), 0 : int(image_size[1])]
    xy = (np.ravel(xV), np.ravel(yV))
    popt, _ = spo.curve_fit(circle_function, xy, np.ravel(calc_image), initial_guess)
    return popt


@numba.jit
def resizer(data, N):
    """
    Downsample 1D array

    Parameters
    ----------
    data: ndarray
    N:    int
          New size of array

    Returns
    -------
    res: ndarray of shape N
         Data resampled

    Notes
    -----
    The data is resampled. Since this is a Numba
    function, compile it once (you will get errors)
    by calling %timeit

    :Authors:
    Debangshu Mukherjee <mukherjeed@ornl.gov>
    """
    warnings.filterwarnings("ignore")
    M = data.size
    data = (data).astype(np.float64)
    res = np.zeros(int(N), dtype=np.float64)
    carry = 0
    m = 0
    for n in range(int(N)):
        data_sum = carry
        while m * N - n * M < M:
            data_sum += data[m]
            m += 1
        carry = (m - (n + 1) * M / N) * data[m - 1]
        data_sum -= carry
        res[n] = data_sum * N / M
    return res


@numba.jit
def resizer2D(data, sampling):
    """
    Downsample 2D array

    Parameters
    ----------
    data:     ndarray
              (2,2) shape
    sampling: tuple
              Downsampling factor in each axisa

    Returns
    -------
    resampled: ndarray
              Downsampled by the sampling factor
              in each axis

    Notes
    -----
    The data is a 2D wrapper over the resizer function

    See Also
    --------
    resizer

    :Authors:
    Debangshu Mukherjee <mukherjeed@ornl.gov>
    """
    warnings.filterwarnings("ignore")
    sampling = np.asarray(sampling)
    data_shape = np.asarray(np.shape(data))
    sampled_shape = (np.round(data_shape / sampling)).astype(int)
    resampled_x = np.zeros((data_shape[0], sampled_shape[1]), dtype=np.float64)
    resampled = np.zeros(sampled_shape, dtype=np.float64)
    for yy in range(int(data_shape[0])):
        resampled_x[yy, :] = resizer(data[yy, :], sampled_shape[1])
    for xx in range(int(sampled_shape[1])):
        resampled[:, xx] = resizer(resampled_x[:, xx], sampled_shape[0])
    return resampled


def is_odd(num):
    return num % 2 != 0


def get_mean_std(xlist, ylist, resolution=25, style="median"):
    """
    Get mean and standard deviation of a list of x and y values

    Parameters
    ----------
    xlist:      ndarray
                (n,1) shape of x values
    ylist:      ndarray
                (n,1) shape of x values
    resolution: float
                Optional value, default if 50
                How finely sampled the final list is

    Returns
    -------
    stdvals: ndarray
             First column is x values sampled at resolution
             Second Column is mean or median of corresponding y values
             Third column is the standard deviation

    :Authors:
    Debangshu Mukherjee <mukherjeed@ornl.gov>
    """
    xround = np.copy(xlist)
    xround = np.round(resolution * xround) / resolution
    x_points = int((np.amax(xround) - np.amin(xround)) / (1 / resolution))
    stdvals = np.zeros((x_points, 3), dtype=np.float64)
    for ii in np.arange(x_points):
        ccval = np.amin(xround) + ii / resolution
        stdvals[ii, 0] = ccval
        if style == "median":
            stdvals[ii, 1] = np.median(ylist[xround == ccval])
        else:
            stdvals[ii, 1] = np.mean(ylist[xround == ccval])
        stdvals[ii, 2] = np.std(ylist[xround == ccval])
    stdvals = stdvals[~np.isnan(stdvals[:, 2]), :]
    return stdvals


def cp_image_sat(comp_image):
    img_size = (np.asarray(comp_image.shape)).astype(int)
    hsv_image = np.ones((img_size[0], img_size[1], 3))
    real_image = np.abs(comp_image)
    angle_image = (180 + np.angle(comp_image, deg=True)) / 360
    angle_image = angle_image - np.floor(angle_image)
    max_val = np.amax(real_image)
    real_image = real_image / max_val
    hsv_image[:, :, 0] = angle_image
    hsv_image[:, :, 1] = real_image
    colored_image = skc.hsv2rgb(hsv_image)
    return colored_image


def cp_image_val(comp_image):
    img_size = (np.asarray(comp_image.shape)).astype(int)
    hsv_image = np.ones((img_size[0], img_size[1], 3))
    real_image = np.abs(comp_image)
    angle_image = (180 + np.angle(comp_image, deg=True)) / 360
    angle_image = angle_image - np.floor(angle_image)
    max_val = np.amax(real_image)
    real_image = real_image / max_val
    hsv_image[:, :, 0] = angle_image
    hsv_image[:, :, 2] = real_image
    colored_image = skc.hsv2rgb(hsv_image)
    return colored_image


def euclidean_dist(binary_image):
    """
    Get Euclidean distance of every non-zero
    pixel from zero valued pixels

    Parameters
    ----------
    binary_image: ndarray
                  The image

    Returns
    -------
    dist_map: ndarray
              The Euclidean distance map
    """
    bi_ones = binary_image != 0
    bi_zero = binary_image == 0
    bi_yy, bi_xx = np.mgrid[0 : binary_image.shape[0], 0 : binary_image.shape[1]]
    ones_vals = np.asarray((bi_yy[bi_ones], bi_xx[bi_ones])).transpose()
    zero_vals = np.asarray((bi_yy[bi_zero], bi_xx[bi_zero])).transpose()
    dist_vals = np.zeros(ones_vals.shape[0])
    for ii in np.arange(ones_vals.shape[0]):
        dist_vals[ii] = np.amin(np.sum(((zero_vals - ones_vals[ii, 0:2]) ** 2), axis=1))
    dist_map = np.zeros_like(binary_image, dtype=np.float)
    dist_map[bi_ones] = dist_vals ** 0.5
    return dist_map
