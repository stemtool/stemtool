import numpy as np
import numba
import matplotlib as mpl
import pyfftw
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as mplc
import matplotlib.cm as mplcm
from scipy import misc as scm

@numba.jit(parallel=True,cache=True)
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
    pyfftw.interfaces.cache.enable()
    image_size = (np.asarray(image_to_move.shape)).astype(int)
    fourier_cal_y = np.linspace((-image_size[0] / 2), ((image_size[0] / 2) - 1), image_size[0])
    fourier_cal_y = fourier_cal_y / image_size[0]
    fourier_cal_x = np.linspace((-image_size[1] / 2), ((image_size[1] / 2) - 1), image_size[1])
    fourier_cal_x = fourier_cal_x / image_size[1]
    [fourier_mesh_x, fourier_mesh_y] = np.meshgrid(fourier_cal_x, fourier_cal_y)
    move_matrix = np.multiply(fourier_mesh_x,x_pixels) + np.multiply(fourier_mesh_y,y_pixels)
    move_phase = np.exp((-2) * np.pi * 1j * move_matrix)
    original_image_fft = pyfftw.interfaces.scipy_fftpack.fftshift(pyfftw.interfaces.scipy_fftpack.fft2(image_to_move))
    moved_in_fourier = np.multiply(move_phase,original_image_fft)
    moved_image = pyfftw.interfaces.scipy_fftpack.ifft2(moved_in_fourier)
    return moved_image

@numba.jit(parallel=True,cache=True)
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
    image_pos = image_orig - np.amin(image_orig)
    image_norm = image_pos / np.amax(image_pos)
    return image_norm

@numba.jit(parallel=True)
def image_logarizer(image_orig,bit_depth=32):
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
    Subtract the minima of the data from the 
    numpy array of the image and then divide 
    it by the maximmum value of the numpy array.
    Then multiply it by the bit depth raised to
    the power of 2, so that the value lies between
    1 and (2^{bit depth}). Then take the logarithm
    of the image with base 2, so the final output
    data lies between 0 and the bit depth.
                 
    :Authors:
    Debangshu Mukherjee <mukherjeed@ornl.gov>
    """
    bit_max = 2 ** bit_depth
    image_pos = image_orig - np.amin(image_orig)
    image_norm = 1 + ((bit_max - 1) * (image_pos / np.amax(image_pos)))
    image_log = np.log2(image_norm)
    return image_log

@numba.jit(parallel=True,cache=True)
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
    size_image = np.asarray(np.shape(image),dtype=int)
    row_hann = np.zeros((size_image[0],1))
    row_hann[:,0] = np.hanning(size_image[0])
    col_hann = np.zeros((1,size_image[1]))
    col_hann[0,:] = np.hanning(size_image[1])
    hann_window = np.multiply(row_hann,col_hann)
    hanned_image = np.multiply(image,hann_window)
    return hanned_image

def sane_colorbar(mappable):
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    return fig.colorbar(mappable, cax=cax)
 
@numba.jit(parallel=True,cache=True)
def phase_color(phase_image):
    size_im = np.asarray(np.shape(phase_image))
    hsv_im = np.ones((size_im[0],size_im[1],3))
    hsv_im[:,:,0] = (phase_image + (2*np.pi))/(2*np.pi)
    hsv_im[:,:,0] = hsv_im[:,:,0] - np.floor(hsv_im[:,:,0])
    rgb_im = mplc.hsv_to_rgb(hsv_im)
    r, g, b = rgb_im[:,:,0], rgb_im[:,:,1], rgb_im[:,:,2]
    gray_im = (0.2989 * r) + (0.5870 * g) + (0.1140 * b)
    gray_im = gray_im
    hsv_im[:,:,2] = np.divide(hsv_im[:,:,2],gray_im)
    hsv_im[:,:,2] = hsv_im[:,:,2]/np.amax(hsv_im[:,:,2])
    rgb_im = mplc.hsv_to_rgb(hsv_im)
    return rgb_im

@numba.jit(parallel=True,cache=True)
def hsv_overlay(data,image,color,climit=None,bit_depth = 8):
    bit_range = 2 ** bit_depth
    im_map = mplcm.get_cmap(color, bit_range)
    if climit == None:
        data_lim = np.amax(np.abs(data))
    else:
        data_lim = climit
    data = 0.5 + (data/(2*data_lim))
    rgb_image = np.asarray(im_map(data)[:,:,0:3])
    hsv_image = mplc.rgb_to_hsv(rgb_image)
    hsv_image[:,:,-1] = (image - np.amin(image))/(np.amax(image) - np.amin(image))
    rgb_image = mplc.hsv_to_rgb(hsv_image)
    return rgb_image

@numba.jit(parallel=True)
def sparse_division(sparse_numer,sparse_denom,bit_depth=32):
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
    threshold_denom = (np.amax(denom_abs))/depth_ratio
    threshold_numer = (np.amax(numer_abs))/depth_ratio
    threshold_ind_denom = denom_abs < threshold_denom
    threshold_ind_numer = numer_abs < threshold_numer
    sparse_denom[threshold_ind_denom] = 1
    sparse_numer[threshold_ind_numer] = 0
    divided_matrix = np.divide(sparse_numer,sparse_denom)
    divided_matrix[threshold_ind_denom] = 0
    return divided_matrix

@numba.jit(parallel=True)
def normalized_correlation(image_1,image_2,hybridizer=0):
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
         
    If n is 0, we have pure cross correlation, and if n is 1 we 
    have pure phase correlation.
    
    References
    ----------
    Pekin, T.C., Gammer, C., Ciston, J., Minor, A.M. and Ophus, C., 
    2017. Optimizing disk registration algorithms for nanobeam 
    electron diffraction strain mapping. Ultramicroscopy, 176, 
    pp.170-176.
    
    See Also
    --------
    image_normalizer
    sparse_division
    
    :Authors:
    Debangshu Mukherjee <mukherjeed@ornl.gov>
    """
    im1_norm = image_normalizer(image_1)
    im2_norm = image_normalizer(image_2)
    im1_fft = pyfftw.interfaces.numpy_fft.fftshift(pyfftw.interfaces.numpy_fft.fft2(im1_norm))
    im2_fft = pyfftw.interfaces.numpy_fft.fftshift(pyfftw.interfaces.numpy_fft.fft2(im2_norm))
    im2_fft_conj = np.conj(im2_fft)
    corr_fft = np.multiply(im1_fft,im2_fft_conj)
    corr_abs = np.abs(corr_fft)
    corr_angle = np.angle(corr_fft)
    corr_abs_hybrid = corr_abs ** hybridizer
    corr_hybrid_fft = sparse_division(corr_fft,corr_abs_hybrid,32)
    corr_hybrid = pyfftw.interfaces.numpy_fft.ifftshift(pyfftw.interfaces.numpy_fft.ifft2(corr_hybrid_fft))
    return corr_hybrid

def make_circle(size_circ,center_x,center_y,radius):
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
    circle = (np.asarray(sub)).astype('float')
    return circle

@numba.jit(parallel=True)
def image_tiler(dataset_4D,reducer=3,bit_depth=8):
    """
    Generate a tiled pattern of the 4D-STEM dataset
    
    """
    size_data = (np.asarray(dataset_4D.shape)).astype(int)
    normalized_4D = dataset_4D - np.amin(dataset_4D)
    normalized_4D = 2 * (normalized_4D/(np.amax(normalized_4D)))
    normalized_4D = normalized_4D - 1 #make the data values lie from -1 to +1
    reduced_size = (np.zeros(2)).astype(int)
    reduced_size[0] = int(round(size_data[0]*(1/reducer)))
    reduced_size[1] = int(round(size_data[1]*(1/reducer)))
    tile_size = np.multiply(reduced_size,(size_data[2],size_data[3]))
    image_tile = np.zeros(tile_size)
    for jj in numba.prange(size_data[3]):
        for ii in range(size_data[2]):
            ronchi = normalized_4D[:,:,ii,jj]
            reduced_shape = np.round(np.asarray(np.shape(ronchi))/reducer)
            reduced_shape = reduced_shape.astype(int)
            xRange = (ii * reduced_size[0]) + np.arange(reduced_size[0])
            xStart = int(xRange[0])
            xEnd = int(xRange[-1])
            yRange = (jj * reduced_size[1]) + np.arange(reduced_size[1])
            yStart = int(yRange[0])
            yEnd = int(yRange[-1])
            reduced_ronchi = scm.imresize(ronchi,reduced_shape,interp='lanczos')
            resized_shape = (np.asarray(reduced_ronchi.shape)).astype(int)
            if (np.mean(reduced_ronchi[-1,:]) == 0):
                reduced_ronchi = reduced_ronchi[0:(resized_shape[0] - 1),:]
            if (np.mean(reduced_ronchi[:,-1]) == 0):
                reduced_ronchi = reduced_ronchi[:,0:(resized_shape[1] - 1)]
            if (np.mean(reduced_ronchi[0,:]) == 0):
                reduced_ronchi = reduced_ronchi[1:resized_shape[0],:]
            if (np.mean(reduced_ronchi[:,0]) == 0):
                reduced_ronchi = reduced_ronchi[:,1:resized_shape[1]]
            image_tile[xStart:xEnd,yStart:yEnd] = reduced_ronchi
    image_tile = image_tile - np.amin(image_tile)
    image_tile = (2 ** bit_depth)*(image_tile/(np.amax(image_tile)))
    image_tile = image_tile.astype(int)
    return image_tile