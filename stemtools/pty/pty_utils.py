import numpy as np
import numba
from ..util import image_utils as iu
from ..dpc import atomic_dpc as ad

def wavelength_pm(voltage_kV):
    """
    Calculates the relativistic electron wavelength
    in picometers based on the microscope accelerating 
    voltage
    
    Parameters
    ----------
    voltage_kV: float
                microscope operating voltage in kilo 
                electronVolts
    
    Returns
    -------
    wavelength: float
                relativistic electron wavelength in 
                picometers
    
    :Authors:
    Debangshu Mukherjee <mukherjeed@ornl.gov>
    """
    m = 9.109383 * (10 ** (-31))  # mass of an electron
    e = 1.602177 * (10 ** (-19))  # charge of an electron
    c = 299792458  # speed of light
    h = 6.62607 * (10 ** (-34))  # Planck's constant
    voltage = voltage_kV * 1000
    numerator = (h ** 2) * (c ** 2)
    denominator = (e * voltage) * ((2*m*(c ** 2)) + (e * voltage))
    wavelength = (10 ** 12) *((numerator/denominator) ** 0.5) #in angstroms
    return wavelength

def fourier_calib(realspace_calibration,
                  image_size):
    """
    Get inverse Fourier units
    
    Parameters
    ----------
    realspace_calibration: float
                           calibration in real space
    image_size:            tuple
                           Shape of the image matrix
    
    Returns
    -------
    (dL_y,dL_x): tuple
                 Fourier calibration along the co-ordinate axes
    
    :Authors:
    Debangshu Mukherjee <mukherjeed@ornl.gov>
    """
    image_size = np.asarray(image_size)
    FOV = image_size*realspace_calibration
    qy = (np.arange((-image_size[0]/2),((image_size[0]/2)),1))/FOV[0]
    qx = (np.arange((-image_size[1]/2),((image_size[1]/2)),1))/FOV[1]
    Ly = np.roll(qy,int(image_size[0]/2))
    Lx = np.roll(qx,int(image_size[1]/2))
    dL_y = np.median(np.diff(Ly))
    dL_x = np.median(np.diff(Lx))
    return (dL_y,dL_x)

def fourier_coords_1D(real_size,
                      dx, 
                      fft_shifted=False):
    qxa = np.fft.fftfreq(real_size[0], dx[0])
    qya = np.fft.fftfreq(real_size[1], dx[1])
    [qxn, qyn] = np.meshgrid(qxa, qya)
    if fft_shifted:
        qxn = np.fft.fftshift(qxn)
        qyn = np.fft.fftshift(qyn)
    return qxn, qyn

@numba.jit
def sample_4D(original_4D,
              sampling_ratio=2):
    """
    Resize the 4D-STEM CBED pattern
    
    Parameters
    ----------
    original_4D:    ndarray
                    Experimental 4D dataset
    sampling_ratio: float
                    Value by which to resample the CBED pattern
    
    Returns
    -------
    processed_4D: ndarray
                  4D-STEM dataset where every CBED pattern
                  has been resampled
    
    Notes
    -----
    To maintain consistency and faster processing of ptychography, 
    make the real space and Fourier space pixels consistent
    
    :Authors:
    Debangshu Mukherjee <mukherjeed@ornl.gov>
    """
    data_size = (np.asarray(original_4D.shape)).astype(int)
    processed_4D = (np.zeros((data_size[2],data_size[3],data_size[2],data_size[3])))
    for jj in range(data_size[3]):
        for ii in range(data_size[2]):
            ronchigram = original_4D[:,:,ii,jj]
            ronchi_size = (np.asarray(ronchigram.shape)).astype(int)
            resized_ronchigram = iu.resizer2D((ronchigram + 1),(1/sampling_ratio)) - 1
            resized_shape = (np.asarray(resized_ronchigram.shape)).astype(int)
            cut_shape = (np.asarray(resized_ronchigram.shape)).astype(int)
            BeforePadSize = ((0.5 * ((data_size[2],data_size[3]) - cut_shape)) - 0.25).astype(int)
            padCorrect = (data_size[2],data_size[3]) - (cut_shape + (2*BeforePadSize))
            AfterPadSize = BeforePadSize + padCorrect
            FullPadSize = ((BeforePadSize[0],AfterPadSize[0]),(BeforePadSize[1],AfterPadSize[1]))
            padValue = np.amin(resized_ronchigram)
            padded_ronchi = np.pad(resized_ronchigram, FullPadSize, 'constant', constant_values=(padValue, padValue))
            processed_4D[:,:,ii,jj] = padded_ronchi
    return processed_4D

@numba.jit
def sparse4D(numer4D,
             denom4D,
             bit_depth=32):
    """
    Divide one 4D dataset from the other,
    so that zeros don't overflow
    
    Parameters
    ----------
    numer4D:   ndarray
               Numerator
    denom4D:   ndarray
               Denominator
    bit_depth: int
               Highest power of 2 that will be tolerated
               before truncation 
    
    Returns
    -------
    sparse_divided: ndarray
                    Numerator divided by the denominator
    
    Notes
    -----
    Wrapper around the sparse_division in the utils module
    for dividing one 4D matrix by another, where there are
    zeros, or very close to zero values in the denominator
    
    See Also
    --------
    sparse_division
    
    :Authors:
    Debangshu Mukherjee <mukherjeed@ornl.gov>
    """
    data_size = (np.asarray(numer4D.shape)).astype(int)
    sparse_divided = (np.zeros((data_size[0],data_size[1],data_size[2],data_size[3]))).astype('complex')
    for jj in range(data_size[3]):
        for ii in range(data_size[2]):
            sparse_divided[:,:,ii,jj] = iu.sparse_division(numer4D[:,:,ii,jj],denom4D[:,:,ii,jj],bit_depth)
    return sparse_divided