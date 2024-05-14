import scipy.ndimage as scnd
import scipy.optimize as sio
import numpy as np
import stemtool as st
from tqdm.auto import trange
from typing import Any, Tuple
from nptyping import NDArray, Shape, Int, Float, Bool, Complex


def cart2pol(
    x: NDArray[Shape["*"], Any], y: NDArray[Shape["*"], Any]
) -> Tuple[NDArray[Shape["*"], Float], NDArray[Shape["*"], Float]]:
    rho: NDArray[Shape["*"], Float] = ((x**2) + (y**2)) ** 0.5
    phi: NDArray[Shape["*"], Float] = np.arctan2(y, x)
    return rho, phi


def pol2cart(
    rho: NDArray[Shape["*"], Any], phi: NDArray[Shape["*"], Any]
) -> Tuple[NDArray[Shape["*"], Float], NDArray[Shape["*"], Float]]:
    x: NDArray[Shape["*"], Float] = rho * np.cos(phi)
    y: NDArray[Shape["*"], Float] = rho * np.sin(phi)
    return x, y


def angle_fun(
    angle_d: Float,
    rho_dpc: NDArray[Shape["*, *"], Float],
    phi_dpc: NDArray[Shape["*, *"], Float],
) -> Float:
    angle_r: Float = angle_d * ((np.pi) / 180)
    new_phi: NDArray[Shape["*, *"], Float] = phi_dpc + angle_r
    x_dpc_rvl, y_dpc_rvl = st.dpc.pol2cart(rho_dpc.ravel(), new_phi.ravel())
    x_dpc: NDArray[Shape["*, *"], Float] = np.reshape(x_dpc_rvl, newshape=rho_dpc.shape)
    y_dpc: NDArray[Shape["*, *"], Float] = np.reshape(y_dpc_rvl, newshape=rho_dpc.shape)
    charge: NDArray[Shape["*, *"], Float] = (
        np.gradient(x_dpc)[1] + np.gradient(y_dpc)[0]
    )
    angle_sum: Float = np.sum(np.abs(charge))
    return angle_sum


def optimize_angle(
    rho_dpc: NDArray[Shape["*, *"], Float], phi_dpc: NDArray[Shape["*, *"], Float]
) -> Tuple[Float, Float]:
    x0: Float = 90
    x: Any = sio.minimize(angle_fun, x0, args=(rho_dpc, phi_dpc))
    min_x: Float = x.x
    sol1: Float = min_x - 90
    sol2: Float = min_x + 90
    return sol1, sol2


def data_rotator(
    cbed_pattern: NDArray[Shape["*, *"], Any],
    rotangle: Float,
    xcenter: Float,
    ycenter: Float,
    data_radius: Float,
) -> NDArray[Shape["*, *"], Float]:
    data_size: NDArray[Shape["1, 2"], Int] = np.shape(cbed_pattern)
    yV, xV = np.mgrid[0 : data_size[0], 0 : data_size[1]]
    mask: NDArray[Shape["*, *"], Bool] = (
        (((yV - ycenter) ** 2) + ((xV - xcenter) ** 2)) ** 0.5
    ) > (1.04 * data_radius)
    cbed_min: Float = np.amin(scnd.median_filter(cbed_pattern, 15))
    moved_cbed: NDArray[Shape["*, *"], Float] = np.abs(
        st.util.move_by_phase(
            cbed_pattern,
            (xcenter - (0.5 * data_size[1])),
            (ycenter - (0.5 * data_size[0])),
        )
    )
    rotated_cbed: NDArray[Shape["*, *"], Float] = scnd.rotate(
        moved_cbed, rotangle, order=5, reshape=False
    )
    rotated_cbed[mask] = cbed_min
    return rotated_cbed


def integrate_dpc(
    xshift: NDArray[Shape["*, *"], Float],
    yshift: NDArray[Shape["*, *"], Float],
    padf: Int = 4,
    lP: Float = 0.5,
    hP: Float = 100,
    stepsize: Float = 0.2,
    iter_count: Int = 100,
) -> NDArray[Shape["*, *"], Float]:
    """
    Integrate DPC shifts using Fourier transforms and
    preventing edge effects

    Parameters
    ----------
    xshift:     Beam shift in the X dimension
    yshift:     Beam shift in the X dimensions
    padf:       padding factor for accurate FFT,
                default is 4
    lP:         low pass filter, default is 0.5
    hP:         high pass filter, default is 100
    stepsize:   fraction of phase differnce to update every
                iteration. Default is 0.5. This is a dynamic
                factor, and is reduced if the error starts
                increasing
    iter_count: Number of iterations to run. Default is 100

    Returns
    -------
    phase_final: Phase of the matrix that leads to the displacement

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
    imshape: NDArray[Shape["1, 2"], Int] = np.asarray(xshift.shape)
    padshape: NDArray[Shape["1, 2"], Int] = (imshape * padf).astype(int)
    qx: NDArray[Shape["*"], Float] = np.fft.fftfreq(padshape[0])
    qy: NDArray[Shape["*"], Float] = np.fft.rfftfreq(padshape[1])
    qr2: NDArray[Shape["*, *"], Float] = qx[:, None] ** 2 + qy[None, :] ** 2

    denominator: NDArray[Shape["*, *"], Float] = qr2 + hP + ((qr2**2) * lP)
    _ = np.seterr(divide="ignore")
    denominator = 1.0 / denominator
    denominator[0, 0] = 0
    _ = np.seterr(divide="warn")
    f: Complex = 1j * 0.25 * stepsize
    qxOperator: NDArray[Shape["*, *"], Complex] = f * qx[:, None] * denominator
    qyOperator: NDArray[Shape["*, *"], Complex] = f * qy[None, :] * denominator

    padded_phase: NDArray[Shape["*, *"], Float] = np.zeros(
        shape=padshape, dtype=np.float64
    )
    update: NDArray[Shape["*, *"], Float] = np.zeros_like(padded_phase)
    dx: NDArray[Shape["*, *"], Float] = np.zeros_like(padded_phase)
    dy: NDArray[Shape["*, *"], Float] = np.zeros_like(padded_phase)
    error: NDArray[Shape["*"], Float] = np.zeros(iter_count)
    mask: NDArray[Shape["*, *"], Bool] = np.zeros_like(padded_phase, dtype=bool)
    mask[
        int(0.5 * (padshape[0] - imshape[0])) : int(0.5 * (padshape[0] + imshape[0])),
        int(0.5 * (padshape[1] - imshape[1])) : int(0.5 * (padshape[1] + imshape[1])),
    ] = True
    maskInv: NDArray[Shape["*, *"], Bool] = mask == False
    for ii in trange(iter_count):
        dx[mask] -= xshift.ravel()
        dy[mask] -= yshift.ravel()
        dx[maskInv] = 0
        dy[maskInv] = 0
        update += np.fft.irfft2(
            np.fft.rfft2(dx) * qxOperator + np.fft.rfft2(dy) * qyOperator
        )
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
    phase_final: NDArray[Shape["*, *"], Float] = np.reshape(
        padded_phase[mask], newshape=imshape
    )
    return phase_final


def potential_dpc(
    x_dpc: NDArray[Shape["*, *"], Float],
    y_dpc: NDArray[Shape["*, *"], Float],
    angle: Float = 0,
) -> NDArray[Shape["*, *"], Float]:
    potential: NDArray[Shape["*, *"], Float] = np.zeros_like(x_dpc)
    if angle == 0:
        potential = st.dpc.integrate_dpc(x_dpc, y_dpc)
    else:
        rho_dpc, phi_dpc = cart2pol(x_dpc, y_dpc)
        x_dpc, y_dpc = pol2cart(rho_dpc, phi_dpc + (angle * ((np.pi) / 180)))
        potential = st.dpc.integrate_dpc(x_dpc, y_dpc)
    return potential


def charge_dpc(
    x_dpc: NDArray[Shape["*, *"], Float],
    y_dpc: NDArray[Shape["*, *"], Float],
    angle: Float = 0,
) -> NDArray[Shape["*, *"], Float]:
    charge: NDArray[Shape["*, *"], Float] = np.zeros_like(x_dpc)
    if angle == 0:
        charge += np.gradient(x_dpc)[1] + np.gradient(y_dpc)[0]
    else:
        rho_dpc_rv, phi_dpc_rv = st.dpc.cart2pol(x_dpc.ravel(), y_dpc.ravel())
        x_dpc_rv, y_dpc_rv = st.dpc.pol2cart(
            rho_dpc_rv, phi_dpc_rv + (angle * ((np.pi) / 180))
        )
        x_dpc_rot: NDArray[Shape["*, *"], Float] = np.reshape(
            x_dpc_rv, newshape=x_dpc.shape
        )
        y_dpc_rot: NDArray[Shape["*, *"], Float] = np.reshape(
            y_dpc_rv, newshape=y_dpc.shape
        )
        charge += np.gradient(x_dpc_rot)[1] + np.gradient(y_dpc_rot)[0]
    return charge
