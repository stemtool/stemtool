import scipy.ndimage as scnd
import scipy.optimize as sio
import numpy as np
import numba
import warnings
import stemtool as st
import numexpr as ne
import pyfftw.interfaces.numpy_fft as pfft


def cart2pol(x, y):
    rho = ne.evaluate("((x**2) + (y**2)) ** 0.5")
    phi = ne.evaluate("arctan2(y, x)")
    return (rho, phi)


def pol2cart(rho, phi):
    x = ne.evaluate("rho * cos(phi)")
    y = ne.evaluate("rho * sin(phi)")
    return (x, y)


def angle_fun(angle, rho_dpc, phi_dpc):
    angle = angle * ((np.pi) / 180)
    new_phi = phi_dpc + angle
    x_dpc, y_dpc = st.dpc.pol2cart(rho_dpc, new_phi)
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


def integrate_dpc(xshift, yshift, padf=4, lP=0.5, hP=100, stepsize=0.2, iter_count=100):
    """
    Integrate DPC shifts using Fourier transforms and
    preventing edge effects

    Parameters
    ----------
    xshift:     ndarray
                Beam shift in the X dimension
    yshift:     ndarray
                Beam shift in the X dimensions
    padf:       int, optional
                padding factor for accurate FFT,
                default is 4
    lP:         float, optional
                low pass filter, default is 0.5
    hP:         float, optional
                high pass filter, default is 100
    stepsize:   float, optional
                fraction of phase differnce to update every
                iteration. Default is 0.5. This is a dynamic
                factor, and is reduced if the error starts
                increasing
    iter_count: int, optional
                Number of iterations to run. Default is 100

    Returns
    -------
    phase_final: ndarray
                 Phase of the matrix that leads to the displacement

    Notes
    -----
    This is based on two ideas - iterative complex plane
    integration and antisymmetric mirror integration. First
    two antisymmetric matrices are generated for each of the
    x shift and y shifts. Then they are integrated in Fourier
    space as per the idea of complex integration. Finally, a
    sub-matrix is taken out from the antisymmetric integrand
    matrix to give the dpc integrand

    References
    ----------
    .. [1] Ishizuka, Akimitsu, Masaaki Oka, Takehito Seki, Naoya Shibata,
        and Kazuo Ishizuka. "Boundary-artifact-free determination of
        potential distribution from differential phase contrast signals."
        Microscopy 66, no. 6 (2017): 397-405.

    :Authors:
    Debangshu Mukherjee <mukherjeed@ornl.gov>
    """
    imshape = np.asarray(xshift.shape)
    padshape = (imshape * padf).astype(int)
    qx = np.fft.fftfreq(padshape[0])
    qy = np.fft.rfftfreq(padshape[1])
    qr2 = qx[:, None] ** 2 + qy[None, :] ** 2

    denominator = qr2 + hP + ((qr2 ** 2) * lP)
    _ = np.seterr(divide="ignore")
    denominator = 1.0 / denominator
    denominator[0, 0] = 0
    _ = np.seterr(divide="warn")
    f = 1j * 0.25 * stepsize
    qxOperator = f * qx[:, None] * denominator
    qyOperator = f * qy[None, :] * denominator

    padded_phase = np.zeros(padshape)
    update = np.zeros_like(padded_phase)
    dx = np.zeros_like(padded_phase)
    dy = np.zeros_like(padded_phase)
    error = np.zeros(iter_count)
    mask = np.zeros_like(padded_phase, dtype=bool)
    mask[
        int(0.5 * (padshape[0] - imshape[0])) : int(0.5 * (padshape[0] + imshape[0])),
        int(0.5 * (padshape[1] - imshape[1])) : int(0.5 * (padshape[1] + imshape[1])),
    ] = True
    maskInv = mask == False
    for ii in range(iter_count):
        dx[mask] -= xshift.ravel()
        dy[mask] -= yshift.ravel()
        dx[maskInv] = 0
        dy[maskInv] = 0
        update = pfft.irfft2(pfft.rfft2(dx) * qxOperator + pfft.rfft2(dy) * qyOperator)
        padded_phase += scnd.gaussian_filter((stepsize * update), 1)
        dx = (
            np.roll(padded_phase, (-1, 0), axis=(0, 1))
            - np.roll(padded_phase, (1, 0), axis=(0, 1))
        ) / 2.0
        dy = (
            np.roll(padded_phase, (0, -1), axis=(0, 1))
            - np.roll(padded_phase, (0, 1), axis=(0, 1))
        ) / 2.0
        xDiff = dx[mask] - xshift.ravel()
        yDiff = dy[mask] - yshift.ravel()
        error[ii] = np.sqrt(
            np.mean((xDiff - np.mean(xDiff)) ** 2 + (yDiff - np.mean(yDiff)) ** 2)
        )
        if ii > 0:
            if error[ii] > error[ii - 1]:
                stepsize /= 2
    phase_final = np.reshape(padded_phase[mask], imshape)
    return phase_final


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
