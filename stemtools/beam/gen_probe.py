import numpy as np

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
                angstroms
    
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
    return wavelength

def FourierCoords(calibration,
                  sizebeam):
    FOV = sizebeam[0]*calibration
    qx = (np.arange((-sizebeam[0]/2),((sizebeam[0]/2)),1))/FOV
    shifter = int(sizebeam[0]/2)
    Lx = np.roll(qx,shifter)
    Lya, Lxa = np.meshgrid(Lx,Lx)
    L2 = np.multiply(Lxa,Lxa) + np.multiply(Lya,Lya)
    L1 = L2**0.5
    dL = Lx[1] - Lx[0]
    return dL,L1

def FourierCalib(calibration,
                 sizebeam):
    FOV_y = sizebeam[0]*calibration
    FOV_x = sizebeam[1]*calibration
    qy = (np.arange((-sizebeam[0]/2),((sizebeam[0]/2)),1))/FOV_y
    qx = (np.arange((-sizebeam[1]/2),((sizebeam[1]/2)),1))/FOV_x
    shifter_y = int(sizebeam[0]/2)
    shifter_x = int(sizebeam[1]/2)
    Ly = np.roll(qy,shifter_y)
    Lx = np.roll(qx,shifter_x)
    dL_y = Ly[1] - Ly[0]
    dL_x = Lx[1] - Lx[0]
    return np.asarray((dL_y,dL_x))

def make_probe(aperture,
               voltage,
               image_size,
               calibration_pm,
               defocus=0,
               c3=0,
               c5=0):
    """
    This calculates an electron probe based on the 
    size and the estimated Fourier co-ordinates with
    the option of adding spherical aberration in the 
    form of defocus, C3 and C5
    """ 
    aperture = aperture / 1000
    wavelength = wavelength_ang(voltage)
    LMax = aperture / wavelength
    image_y = image_size[0]
    image_x = image_size[1]
    x_FOV = image_x * 0.01 * calibration_pm
    y_FOV = image_y * 0.01 * calibration_pm
    qx = (np.arange((-image_x / 2),(image_x / 2), 1)) / x_FOV
    x_shifter = int(round(image_x / 2))
    qy = (np.arange((-image_y / 2),(image_y / 2), 1)) / y_FOV
    y_shifter = int(round(image_y / 2))
    Lx = np.roll(qx, x_shifter)
    Ly = np.roll(qy, y_shifter)
    Lya, Lxa = np.meshgrid(Lx, Ly)
    L2 = np.multiply(Lxa, Lxa) + np.multiply(Lya, Lya)
    inverse_real_matrix = L2 ** 0.5
    fourier_scan_coordinate = Lx[1] - Lx[0]
    Adist = np.asarray(inverse_real_matrix<=LMax, dtype=complex)
    chi_probe = aberration(inverse_real_matrix,wavelength,defocus,c3,c5)
    Adist *= np.exp(-1j*chi_probe)
    probe_real_space = np.fft.ifftshift(np.fft.ifft2(Adist))
    return probe_real_space

def aberration(fourier_coord,
               wavelength_ang,
               defocus=0,
               c3=0,
               c5=0):
    p_matrix = wavelength_ang*fourier_coord
    chi = ((defocus*np.power(p_matrix,2))/2) + ((c3*(1e7)*np.power(p_matrix,4))/4) + ((c5*(1e7)*np.power(p_matrix,6))/6)
    chi_probe = (2*np.pi*chi)/wavelength_ang
    return chi_probe