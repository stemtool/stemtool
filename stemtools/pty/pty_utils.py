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