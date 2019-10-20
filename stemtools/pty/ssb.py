import numpy as np
import numba
from scipy import ndimage as scnd
from ..util import image_utils as iu
from ..beam import gen_probe as gp
from ..pty import pty_utils as pu

def single_side_band(processed_data4D,
                     aperture_mrad,
                     voltage,
                     image_size,
                     calibration_pm):
    e_wavelength_pm = pu.wavelength_pm(voltage)
    alpha_rad = aperture_mrad/1000
    eps = 1e-3
    k_max = dc.metadata.calibration['K_pix_size']
    dxy = dc.metadata.calibration['R_pix_size']
    theta = np.deg2rad(dc.metadata.calibration['R_to_K_rotation_degrees'])
    ny, nx, nky, nkx = processed_data4D.shape
    Kx, Ky = pu.fourier_coords_1D([nkx, nky], k_max, fft_shifted=True)
    Qx, Qy = pu.fourier_coords_1D([nx, ny], dxy)
    Kplus = np.sqrt((Kx + Qx[:, :, None, None]) ** 2 + (Ky + Qy[:, :, None, None]) ** 2)
    Kminus = np.sqrt((Kx - Qx[:, :, None, None]) ** 2 + (Ky - Qy[:, :, None, None]) ** 2)
    K = np.sqrt(Kx ** 2 + Ky ** 2)
    
    A_KplusQ = np.zeros_like(G)
    A_KminusQ = np.zeros_like(G)
    
    for ix, qx in enumerate(Qx[0]):
        for iy, qy in enumerate(Qy[:, 0]):
            x = Kx + qx
            y = Ky + qy
            A_KplusQ[iy, ix] = np.exp(1j * cartesian_aberrations(x, y, lam, C)) * aperture_xp(x, y, lam, alpha_rad,
                                                                                                  edge=0)
            
            x = Kx - qx
            y = Ky - qy
            A_KminusQ[iy, ix] = np.exp(1j * cartesian_aberrations(x, y, lam, C)) * aperture_xp(x, y, lam, alpha_rad,
                                                                                                   edge=0)
    
    Gamma = np.conj(A) * A_KminusQ - A * np.conj(A_KplusQ)
    double_overlap1 = (Kplus < alpha_rad / lam) * (K < alpha_rad / lam) * (Kminus > alpha_rad / lam)
    double_overlap2 = (Kplus > alpha_rad / lam) * (K < alpha_rad / lam) * (Kminus < alpha_rad / lam)
    Psi_Qp = np.zeros((ny, nx), dtype=np.complex64)
    Psi_Qp_left_sb = np.zeros((ny, nx), dtype=np.complex64)
    Psi_Qp_right_sb = np.zeros((ny, nx), dtype=np.complex64)
    
    for y in range(ny):
        for x in range(nx):
            Gamma_abs = np.abs(Gamma[y, x])
            take = Gamma_abs > eps
            Psi_Qp[y, x] = np.sum(G[y, x][take] * Gamma[y, x][take].conj())
            Psi_Qp_left_sb[y, x] = np.sum(G[y, x][double_overlap1[y, x]])
            Psi_Qp_right_sb[y, x] = np.sum(G[y, x][double_overlap2[y, x]])
            if x == 0 and y == 0:
                Psi_Qp[y, x] = np.sum(np.abs(G[y, x]))
                Psi_Qp_left_sb[y, x] = np.sum(np.abs(G[y, x]))
                Psi_Qp_right_sb[y, x] = np.sum(np.abs(G[y, x]))


    Psi_Rp = xp.fft.ifft2(Psi_Qp, norm='ortho')
    Psi_Rp_left_sb = xp.fft.ifft2(Psi_Qp_left_sb, norm='ortho')
    Psi_Rp_right_sb = xp.fft.ifft2(Psi_Qp_right_sb, norm='ortho')

    return Psi_Rp, Psi_Rp_left_sb, Psi_Rp_right_sb