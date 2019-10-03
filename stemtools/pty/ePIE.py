import numpy as np
import numba
import pyfftw
from scipy import ndimage as scnd
from ..proc import sobel_canny as sc
from ..util import gauss_utils as gt
from ..util import image_utils as iu

@numba.jit
def resize_rotate(original_4D,
                  final_size,
                  rotangle,
                  sampler=2,
                  mask_val=1.25,
                  masking=True):
    """
    Resize the 4D-STEM dataset
    
    Parameters
    ----------
    original_4D: ndarray
                 Experimental dataset
    final_size:  ndarray
                 Size of the final ronchigram
    rotangle:    float
                 Angle to rotate the CBED pattern in degrees
    sampler:     float
                 Upsampling or downsampling ratio for CBED pattern
    mask_val:    float
                 Cut off data as a ratio of the beam radius,
                 Default is 1.25
    masking:     bool
                 If true, apply masking     
    
    Returns
    -------
    processed_4D: ndarray
                  4D-STEM dataset where every CBED pattern
                  has been rotated and resized
    
    Notes
    -----
    Experimental 4D-STEM datasets are often flipped, or rotated. 
    It is recommended to use DPC module to check the beam rotation,
    and then rotate the patterns. Also, to maintain consistency 
    and faster processing of ptychography, make the real space and 
    Fourier space pixels consistent
    
    :Authors:
    Debangshu Mukherjee <mukherjeed@ornl.gov>
    """
    data_size = (np.asarray(original_4D.shape)).astype(int)
    processed_4D = (np.zeros((data_size[0],data_size[1],final_size[0],final_size[1])))
    _,_,original_radius = iu.fit_circle(np.mean(original_4D,axis=(0,1)))
    new_radius = original_radius*sampler
    circle_mask = iu.make_circle(final_size,final_size[1]/2,final_size[0]/2,new_radius*mask_val)
    for jj in range(data_size[1]):
        for ii in range(data_size[0]):
            ronchigram = original_4D[ii,jj,:,:]
            ronchi_size = (np.asarray(ronchigram.shape)).astype(int)
            resized_ronchigram = iu.resizer2D((ronchigram + 1),(1/sampler)) - 1
            resized_rotated_ronchigram = scnd.rotate(resized_ronchigram,rotangle)
            resized_shape = (np.asarray(resized_rotated_ronchigram.shape)).astype(int)
            pad_size = np.round((np.asarray(final_size) - resized_shape)/2)
            before_pad_size = np.copy(pad_size)
            after_pad_size = np.copy(pad_size)
            if (2*pad_size[0] + resized_shape[0]) < final_size[0]:
                after_pad_size[0] = pad_size[0] + 1
            if (2*pad_size[1] + resized_shape[1]) < final_size[1]:
                after_pad_size[1] = pad_size[1] + 1
            before_pad_size = (before_pad_size).astype(int)
            after_pad_size = (after_pad_size).astype(int)
            FullPadSize = ((before_pad_size[0],after_pad_size[0]),(before_pad_size[1],after_pad_size[1]))
            padded_ronchi = np.pad(resized_rotated_ronchigram, FullPadSize, 'constant', constant_values=(0, 0))
            processed_4D[ii,jj,:,:] = padded_ronchi
    if masking:
        processed_4D = np.multiply(processed_4D,circle_mask)
    return processed_4D

def move_probe(probe_im,
               x_pixels,
               y_pixels):
    """
    Move Images with sub-pixel precision
    
    Parameters
    ----------
    image_to_move: ndarray
                   Original Image to be moved
    x_pixels:      float
                   Pixels to shift in X direction
    y_pixels:      float
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
    image_size = np.asarray(probe_im.shape)
    fourier_cal_y = np.linspace((-image_size[0] / 2), ((image_size[0] / 2) - 1), image_size[0])
    fourier_cal_y = fourier_cal_y / (image_size[0])
    fourier_cal_x = np.linspace((-image_size[1] / 2), ((image_size[1] / 2) - 1), image_size[1])
    fourier_cal_x = fourier_cal_x / (image_size[1])
    [fourier_mesh_x, fourier_mesh_y] = np.meshgrid(fourier_cal_x, fourier_cal_y)
    move_matrix = np.multiply(fourier_mesh_x,x_pixels) + np.multiply(fourier_mesh_y,y_pixels)
    move_phase = np.exp((-2) * np.pi * 1j * move_matrix)
    original_image_fft = np.fft.fftshift(pyfftw.interfaces.numpy_fft.fft2(probe_im))
    moved_in_fourier = np.multiply(move_phase,original_image_fft)
    moved_image = pyfftw.interfaces.numpy_fft.ifft2(moved_in_fourier)
    return moved_image

@numba.jit
def update_function(objec_func,
                    probe_func,
                    data4D_sqrt,
                    pos_x,
                    pos_y,
                    alpha_val=0.1):
    """
    Ptychographic Iterative Engine Update Function
    
    Parameters
    ----------
    objec_func:  ndarray
                 Complex function of the image object
    probe_func:  ndarray
                 Complex function of the electron probe
    data4D_sqrt: ndarray
                 Square root of the resized data
    pos_x:       int
                 X-Position of the electron beam
    pos_y:       int
                 Y-Position of the electron beam
    alpha_val:   float
                 Update parameter, updates will be smaller
                 for smaller values. Default is 0.1
    
    Returns
    -------
    new_obj: ndarray
             Updated Object Function
    new_prb: ndarray
             Updated Probe Function
    
    Notes
    -----
    The complex probe is multiplied by the complex object
    and then propagated to the Fourier plane. The complex
    probe is generated by moving the probe using the 
    move_probe function. The Fourier of the exit wave is'
    compared with the experimental dataset, with the 
    amplitude replaced by the square root of the resized 
    data corresponding to that scan position. This new
    function is backpropagated to the image plane and the
    difference with the existing original exit wave is used
    to reconstruct beam and probe update functions
    
    References
    ----------
    Maiden, Andrew M., and John M. Rodenburg. "An improved ptychographical 
    phase retrieval algorithm for diffractive imaging." 
    Ultramicroscopy 109.10 (2009): 1256-1262.
    
    See Also
    --------
    move_probe
    
    :Authors:
    Debangshu Mukherjee <mukherjeed@ornl.gov>
    """
    pyfftw.interfaces.cache.enable()
    square_root_ronchi = data4D_sqrt[pos_y,pos_x,:,:]
    data_shape = np.asarray(np.shape(data4D_sqrt))
    moved_probe = move_probe(probe_func,(pos_y - (data_shape[0]/2)),(pos_x - (data_shape[1]/2)))
    Psi_Orig = objec_func * moved_probe
    Psi_FFT = pyfftw.interfaces.numpy_fft.fft2(Psi_Orig)
    Psi_New_FFT = square_root_ronchi * (np.exp(1j * np.angle(Psi_FFT)))
    Psi_New = pyfftw.interfaces.numpy_fft.ifft2(Psi_New_FFT)
    Psi_Diff = Psi_New - Psi_Orig
    
    max_probe_val = (np.amax(np.abs(moved_probe))) ** 2
    obj_updater = (np.multiply(np.conj(moved_probe),Psi_Diff)) * (alpha_val/max_probe_val)
    new_obj = objec_func + obj_updater
    
    moved_objec = move_probe(objec_func,((data_shape[0]/2) - pos_y),((data_shape[1]/2) - pos_x))
    max_objec_val = (np.amax(np.abs(moved_objec))) ** 2
    prb_updater = (np.multiply(np.conj(moved_objec),Psi_Diff)) * (alpha_val/max_objec_val)
    new_prb = probe_func + prb_updater
    
    return new_obj,new_prb

def Ptych_Engine(data4D_sqrt,
                 original_probe,
                 pad_pix,
                 iterations=2):
    """
    Ptychographic Iterative Engine Main Function
    
    Parameters
    ----------
    data4D_sqrt:    ndarray
                    Square root of the resized data
    original_probe: ndarray
                    Original 2D complex probe function
    pad_pix:        int
                    Number of pixels used for padding to
                    remove edge artifacts
    iterations:     int
                    No of ePIE iterations
    
    Returns
    -------
    object_function: ndarray
                     Calculated complex object function
    calc_probe:      ndarray
                     Calculated complex probe function
    
    Notes
    -----
    At each scan position, the probe and the object are updated
    using the update_function. Once all the scan positions have 
    been used, one ePIE iteration finishes. Be careful about using
    padding for scan positions. Your padded pixels are 0s, and 
    your reconstruction will be all NAN if your beam is made to update
    from a padded region where there is no signal in the data4D_sqrt
    
    References
    ----------
    Maiden, Andrew M., and John M. Rodenburg. "An improved ptychographical 
    phase retrieval algorithm for diffractive imaging." 
    Ultramicroscopy 109.10 (2009): 1256-1262.
    
    See Also
    --------
    update_function
    
    :Authors:
    Debangshu Mukherjee <mukherjeed@ornl.gov>
    """
    data_size = np.asarray(np.shape(data4D_sqrt))
    calc_probe = np.copy(original_probe)
    y_scan = data_size[0] - (2*pad_pix)
    x_scan = data_size[1] - (2*pad_pix)
    object_function = np.ones((data_size[0],data_size[1]),dtype=complex)
    for kk in np.arange(iterations):
        for ii in (pad_pix + np.arange(y_scan)):
            for jj in (pad_pix + np.arange(x_scan)):
                object_function, calc_probe = update_function(object_function,calc_probe,data4D_sqrt,jj,ii)
    return object_function, calc_probe