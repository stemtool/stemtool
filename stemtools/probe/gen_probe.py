import numpy as np
import numba

@numba.jit(cache=True)
def wavelength_ang(voltage_kV):
    """
    Calculates the relativistic electron wavelength
    in angstroms based on the microscope accelerating 
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
    wavelength = (10 ** 10) *((numerator/denominator) ** 0.5) #in angstroms
    return wavelength_ang

@numba.jit(cache=True)
def FourierCoords(calibration,sizebeam):
    FOV = sizebeam[0]*calibration
    qx = (np.arange((-sizebeam[0]/2),((sizebeam[0]/2)),1))/FOV
    shifter = int(sizebeam[0]/2)
    Lx = np.roll(qx,shifter)
    Lya, Lxa = np.meshgrid(Lx,Lx)
    L2 = np.multiply(Lxa,Lxa) + np.multiply(Lya,Lya)
    L1 = L2**0.5
    dL = Lx[1] - Lx[0]
    return dL,L1

@numba.jit(cache=True)
def make_probe(aperture,
               voltage,
               image_size,
               calibration_pm,
               defocus=0,
               c3=0,
               c5=0):
    """
    This calculates an electron probe based on the size and the estimated Fourier co-ordinates
    """ 
    
    aperture = aperture / 1000
    wavelength = wavelength_ang(voltage)
    LMax = aperture / wavelength
    x_FOV = image_x * 0.01 * calibration_pm
    y_FOV = image_y * 0.01 * calibration_pm
    qx = (np.arange((-image_x / 2),(image_x / 2), 1)) / x_FOV
    x_shifter = (round(image_x / 2))
    qy = (np.arange((-image_y / 2),(image_y / 2), 1)) / y_FOV
    y_shifter = (round(image_y / 2))
    Lx = np.roll(qx, x_shifter)
    Ly = np.roll(qy, y_shifter)
    Lya, Lxa = np.meshgrid(Lx, Ly)
    L2 = np.multiply(Lxa, Lxa) + np.multiply(Lya, Lya)
    inverse_real_matrix = L2 ** 0.5
    fourier_scan_coordinate = Lx[1] - Lx[0]
    Adist = ((LMax - inverse_real_matrix) /
             fourier_scan_coordinate) + 0.5
    Adist[Adist < 0] = 0
    Adist[Adist > 1] = 1
    ideal_probe = np.fft.fftshift(Adist)
    chi_probe = aberration(fourier_scan_coordinate,wavelength,defocus,c3,c5)
    return (np.fft.fftshift(Adist),fourier_scan_coordinate)

@numba.jit(cache=True)
def aberration(four_coord,
               wavelength_ang,
               defocus=0,
               c3=0,
               c5=0):
    """
    Calculates the aberration function chi as a function of diffraction space radial coordinates qr
    for an electron with wavelength lam.
    Note that this function only considers the rotationally symmetric terms of chi (i.e. spherical
    aberration) up to 5th order.  Non-rotationally symmetric terms (coma, stig, etc) and higher
    order terms (c7, etc) are not considered.
    Accepts:
        qr      (float or array) diffraction space radial coordinate(s), in inverse Angstroms
        lam     (float) wavelength of electron, in Angstroms
        df      (float) probe defocus, in Angstroms
        cs      (float) probe 3rd order spherical aberration coefficient, in mm
        c5      (float) probe 5th order spherical aberration coefficient, in mm
    Returns:
        chi     (float) the aberation function
    """
    p = lam*qr
    chi = df*np.square(p)/2.0 + cs*1e7*np.power(p,4)/4.0 + c5*1e7*np.power(p,6)/6.0
    chi = 2*np.pi*chi/lam
    return chi_probe