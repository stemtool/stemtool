import scipy.ndimage as scnd
import scipy.optimize as sio
import numpy as np
import numba
import warnings
import stemtool as st


def cart2pol(xx, yy):
    rho = ((xx ** 2) + (yy ** 2)) ** 0.5
    phi = np.arctan2(yy, xx)
    return rho, phi


def pol2cart(rho, phi):
    x = np.multiply(rho, np.cos(phi))
    y = np.multiply(rho, np.sin(phi))
    return x, y


def angle_fun(angle, rho_dpc, phi_dpc):
    angle = angle * ((np.pi) / 180)
    new_phi = phi_dpc + angle
    x_dpc, y_dpc = pol2cart(rho_dpc, new_phi)
    charge = np.gradient(x_dpc)[1] + np.gradient(y_dpc)[0]
    angle_sum = np.sum(np.abs(charge))
    return angle_sum


def optimize_angle(rho_dpc, phi_dpc):
    x0 = 90
    x = sio.minimize(angle_fun, x0, args=(rho_dpc, phi_dpc))
    min_x = x.x
    sol1 = min_x - 90
    sol2 = min_x + 90
    return sol1, sol2


def data_rotator(cbed_pattern, rotangle, xcenter, ycenter, data_radius):
    data_size = np.shape(cbed_pattern)
    yV, xV = np.mgrid[0 : data_size[0], 0 : data_size[1]]
    mask = ((((yV - ycenter) ** 2) + ((xV - xcenter) ** 2)) ** 0.5) > (
        1.04 * data_radius
    )
    cbed_min = np.amin(scnd.median_filter(cbed_pattern, 15))
    moved_cbed = np.abs(
        st.util.move_by_phase(
            cbed_pattern,
            (xcenter - (0.5 * data_size[1])),
            (ycenter - (0.5 * data_size[0])),
        )
    )
    rotated_cbed = scnd.rotate(moved_cbed, rotangle, order=5, reshape=False)
    rotated_cbed[mask] = cbed_min
    return rotated_cbed


def integrate_dpc(xshift, yshift, fourier_calibration=1):
    """
    Integrate DPC shifts using Fourier transforms and 
    preventing edge effects
    
    Parameters
    ----------
    xshift:              ndarray 
                         Beam shift in the X dimension
    yshift:              ndarray
                         Beam shift in the X dimensions
    fourier_calibration: float
                         Pixel size of the Fourier space
    
    Returns
    -------
    integrand: ndarray
               Integrated DPC
    
    Notes
    -----
    This is based on two ideas - noniterative complex plane 
    integration and antisymmetric mirror integration. First 
    two antisymmetric matrices are generated for each of the
    x shift and y shifts. Then they are integrated in Fourier
    space as per the idea of complex integration. Finally, a
    sub-matrix is taken out from the antisymmetric integrand
    matrix to give the dpc integrand
    
    References
    ----------
    Bon, Pierre, Serge Monneret, and Benoit Wattellier. 
    "Noniterative boundary-artifact-free wavefront 
    reconstruction from its derivatives." Applied optics 
    51, no. 23 (2012): 5698-5704.
                 
    :Authors:
    Debangshu Mukherjee <mukherjeed@ornl.gov>
    """

    # Initialize matrices
    size_array = np.asarray(np.shape(xshift))
    x_mirrored = np.zeros(2 * size_array, dtype=np.float64)
    y_mirrored = np.zeros(2 * size_array, dtype=np.float64)

    # Generate antisymmetric X arrays
    x_mirrored[0 : size_array[0], 0 : size_array[1]] = np.fliplr(np.flipud(0 - xshift))
    x_mirrored[0 : size_array[0], size_array[1] : (2 * size_array[1])] = np.fliplr(
        0 - xshift
    )
    x_mirrored[size_array[0] : (2 * size_array[0]), 0 : size_array[1]] = np.flipud(
        xshift
    )
    x_mirrored[
        size_array[0] : (2 * size_array[0]), size_array[1] : (2 * size_array[1])
    ] = xshift

    # Generate antisymmetric Y arrays
    y_mirrored[0 : size_array[0], 0 : size_array[1]] = np.fliplr(np.flipud(0 - yshift))
    y_mirrored[0 : size_array[0], size_array[1] : (2 * size_array[1])] = np.fliplr(
        yshift
    )
    y_mirrored[size_array[0] : (2 * size_array[0]), 0 : size_array[1]] = np.flipud(
        0 - yshift
    )
    y_mirrored[
        size_array[0] : (2 * size_array[0]), size_array[1] : (2 * size_array[1])
    ] = yshift

    # Calculated Fourier transform of antisymmetric matrices
    x_mirr_ft = np.fft.fft2(x_mirrored)
    y_mirr_ft = np.fft.fft2(y_mirrored)

    # Calculated inverse Fourier space calibration
    qx = np.mean(
        np.diff(
            (np.arange(-size_array[1], size_array[1], 1))
            / (2 * fourier_calibration * size_array[1])
        )
    )
    qy = np.mean(
        np.diff(
            (np.arange(-size_array[0], size_array[0], 1))
            / (2 * fourier_calibration * size_array[0])
        )
    )

    # Calculate mirrored CPM integrand
    mirr_ft = (x_mirr_ft + ((1j) * y_mirr_ft)) / (qx + ((1j) * qy))
    mirr_int = np.fft.ifft2(mirr_ft)

    # Select integrand from antisymmetric matrix
    integrand = np.abs(
        mirr_int[
            size_array[0] : (2 * size_array[0]), size_array[1] : (2 * size_array[1])
        ]
    )

    return integrand


def potential_dpc(x_dpc, y_dpc, angle=0):
    if angle == 0:
        potential = integrate_dpc(x_dpc, y_dpc)
    else:
        rho_dpc, phi_dpc = cart2pol(x_dpc, y_dpc)
        x_dpc, y_dpc = pol2cart(rho_dpc, phi_dpc + (angle * ((np.pi) / 180)))
        potential = integrate_dpc(x_dpc, y_dpc)
    return potential


def charge_dpc(x_dpc, y_dpc, angle=0):
    if angle == 0:
        charge = np.gradient(x_dpc)[1] + np.gradient(y_dpc)[0]
    else:
        rho_dpc, phi_dpc = cart2pol(x_dpc, y_dpc)
        x_dpc, y_dpc = pol2cart(rho_dpc, phi_dpc + (angle * ((np.pi) / 180)))
        charge = np.gradient(x_dpc)[1] + np.gradient(y_dpc)[0]
    return charge
