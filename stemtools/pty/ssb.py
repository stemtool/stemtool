import numpy as np
import numba
from scipy import ndimage as scnd
from ..util import image_utils as iu
from ..beam import gen_probe as gp
from ..pty import pty_utils as pu

def single_side_band(data4D,
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
    ny, nx, nky, nkx = data4D.shape
    

    if cuda_is_available:
        M = cp.array(M, dtype=M.dtype)

    xp = cp.get_array_module(M)

    Kx, Ky = get_qx_qy_1D([nkx, nky], k_max, M.dtype, fft_shifted=True)
    Qx, Qy = get_qx_qy_1D([nx, ny], dxy, M.dtype, fft_shifted=False)

    pacbed = xp.mean(M, (0, 1))
    mean_intensity = xp.sum(pacbed)
    print(mean_intensity)
    ap = aperture3(Kx, Ky, lam, alpha_rad).astype(xp.float32)
    aperture_intensity = float(xp.sum(ap))
    print(aperture_intensity)
    scale = 1  # math.sqrt(mean_intensity / aperture_intensity)
    ap *= scale

        Kx, Ky = pu.fourier_coords_1D([nkx, nky], k_max, fft_shifted=True)
        # reciprocal in scanning space
        Qx, Qy = pu.fourier_coords_1D([nx, ny], dxy)

        Kplus = np.sqrt((Kx + Qx[:, :, None, None]) ** 2 + (Ky + Qy[:, :, None, None]) ** 2)
        Kminus = np.sqrt((Kx - Qx[:, :, None, None]) ** 2 + (Ky - Qy[:, :, None, None]) ** 2)
        K = np.sqrt(Kx ** 2 + Ky ** 2)

        A_KplusQ = np.zeros_like(G)
        A_KminusQ = np.zeros_like(G)

        a20 = th.Tensor([20])

        C = np.zeros((12))
        A = np.exp(1j * cartesian_aberrations(Kx, Ky, lam, C)) * aperture_xp(Kx, Ky, lam, alpha_rad, edge=0)

        print('Creating aperture overlap functions')
        for ix, qx in enumerate(Qx[0]):
            print(f"{ix} / {Qx[0].shape}")
            for iy, qy in enumerate(Qy[:, 0]):
                x = Kx + qx
                y = Ky + qy
                A_KplusQ[iy, ix] = np.exp(1j * cartesian_aberrations(x, y, lam, C)) * aperture_xp(x, y, lam, alpha_rad,
                                                                                                  edge=0)
                # A_KplusQ *= 1e4

                x = Kx - qx
                y = Ky - qy
                A_KminusQ[iy, ix] = np.exp(1j * cartesian_aberrations(x, y, lam, C)) * aperture_xp(x, y, lam, alpha_rad,
                                                                                                   edge=0)
                # A_KminusQ *= 1e4

        # [1] Equ. (4): Γ = A*(Kf)A(Kf-Qp) - A(Kf)A*(Kf+Qp)
        Γ = A.conj() * A_KminusQ - A * A_KplusQ.conj()

        double_overlap1 = (Kplus < alpha_rad / lam) * (K < alpha_rad / lam) * (Kminus > alpha_rad / lam)
        double_overlap2 = (Kplus > alpha_rad / lam) * (K < alpha_rad / lam) * (Kminus < alpha_rad / lam)

        Ψ_Qp = np.zeros((ny, nx), dtype=np.complex64)
        Ψ_Qp_left_sb = np.zeros((ny, nx), dtype=np.complex64)
        Ψ_Qp_right_sb = np.zeros((ny, nx), dtype=np.complex64)
        print(f"Now summing over K-space.")
        for y in trange(ny):
            for x in range(nx):
                Γ_abs = np.abs(Γ[y, x])
                take = Γ_abs > eps
                Ψ_Qp[y, x] = np.sum(G[y, x][take] * Γ[y, x][take].conj())
                Ψ_Qp_left_sb[y, x] = np.sum(G[y, x][double_overlap1[y, x]])
                Ψ_Qp_right_sb[y, x] = np.sum(G[y, x][double_overlap2[y, x]])

                # direct beam at zero spatial frequency
                if x == 0 and y == 0:
                    Ψ_Qp[y, x] = np.sum(np.abs(G[y, x]))
                    Ψ_Qp_left_sb[y, x] = np.sum(np.abs(G[y, x]))
                    Ψ_Qp_right_sb[y, x] = np.sum(np.abs(G[y, x]))


    Ψ_Rp = xp.fft.ifft2(Ψ_Qp, norm='ortho')
    Ψ_Rp_left_sb = xp.fft.ifft2(Ψ_Qp_left_sb, norm='ortho')
    Ψ_Rp_right_sb = xp.fft.ifft2(Ψ_Qp_right_sb, norm='ortho')

    return Ψ_Rp, Ψ_Rp_left_sb, Ψ_Rp_right_sb