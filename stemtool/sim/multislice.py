import numpy as np
import scipy.special as s2
import PIL
import stemtool as st


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
    denominator = (e * voltage) * ((2 * m * (c ** 2)) + (e * voltage))
    wavelength_ang = (10 ** 10) * ((numerator / denominator) ** 0.5)  # in angstroms
    return wavelength_ang


def transmission_func(pot_slice, voltage_kV):
    """
    Calculates the complex transmission function from
    a single potential slice at a given e;ectron accelerating
    voltage

    Parameters
    ----------
    pot_slice:  ndarray
                potential slice in Kirkland units
    voltage_kV: float
                microscope operating voltage in kilo
                electronVolts

    Returns
    -------
    trans: ndarray
           The transmission function of a single
           crystal slice

    :Authors:
    Debangshu Mukherjee <mukherjeed@ornl.gov>
    """
    m_e = 9.109383 * (10 ** (-31))  # electron mass
    e_e = 1.602177 * (10 ** (-19))  # electron charge
    c = 299792458  # speed of light
    h = 6.62607 * (10 ** (-34))  # planck's constant
    numerator = (h ** 2) * (c ** 2)
    denominator = (e_e * voltage_kV * 1000) * (
        (2 * m_e * (c ** 2)) + (e_e * voltage_kV * 1000)
    )
    wavelength_ang = (10 ** 10) * (
        (numerator / denominator) ** 0.5
    )  # wavelength in angstroms
    sigma = (
        (2 * np.pi / (wavelength_ang * voltage_kV * 1000))
        * ((m_e * c * c) + (e_e * voltage_kV * 1000))
    ) / ((2 * m_e * c * c) + (e_e * voltage_kV * 1000))
    trans = np.exp(1j * sigma * pot_slice)
    return trans.astype(np.complex64)


def propagation_func(imsize, thickness_ang, voltage_kV, calib_ang):
    """
    Calculates the complex propgation function that results
    in the phase shift of the exit wave when it travels from
    one slice to the next in the multislice algorithm

    Parameters
    ----------
    imsize:        tuple
                   Size of the image of the propagator
    thickness_ang: float
                   Distance between the slices in angstroms
    voltage_kV:    float
                   Accelerating voltage in kilovolts
    calib_ang:     float
                   Calibration or pixel size in angstroms

    Returns
    -------
    prop_shift:  ndarray
                 This is of the same size given by imsize

    :Authors:
    Debangshu Mukherjee <mukherjeed@ornl.gov>
    """
    FOV_y = imsize[0] * calib_ang
    FOV_x = imsize[1] * calib_ang
    qy = (np.arange((-imsize[0] / 2), ((imsize[0] / 2)), 1)) / FOV_y
    qx = (np.arange((-imsize[1] / 2), ((imsize[1] / 2)), 1)) / FOV_x
    shifter_y = int(imsize[0] / 2)
    shifter_x = int(imsize[1] / 2)
    Ly = np.roll(qy, shifter_y)
    Lx = np.roll(qx, shifter_x)
    Lya, Lxa = np.meshgrid(Lx, Ly)
    L2 = np.multiply(Lxa, Lxa) + np.multiply(Lya, Lya)
    wavelength_ang = st.sim.wavelength_ang(voltage_kV)
    prop = np.exp((-1j) * np.pi * wavelength_ang * thickness_ang * L2)
    prop_shift = np.fft.fftshift(prop)  # FFT shift the propagator
    return prop_shift.astype(np.complex64)


def FourierCoords(calibration, sizebeam):
    FOV = sizebeam[0] * calibration
    qx = (np.arange((-sizebeam[0] / 2), ((sizebeam[0] / 2)), 1)) / FOV
    shifter = int(sizebeam[0] / 2)
    Lx = np.roll(qx, shifter)
    Lya, Lxa = np.meshgrid(Lx, Lx)
    L2 = np.multiply(Lxa, Lxa) + np.multiply(Lya, Lya)
    L1 = L2 ** 0.5
    dL = Lx[1] - Lx[0]
    return dL, L1


def FourierCalib(calibration, sizebeam):
    FOV_y = sizebeam[0] * calibration
    FOV_x = sizebeam[1] * calibration
    qy = (np.arange((-sizebeam[0] / 2), ((sizebeam[0] / 2)), 1)) / FOV_y
    qx = (np.arange((-sizebeam[1] / 2), ((sizebeam[1] / 2)), 1)) / FOV_x
    shifter_y = int(sizebeam[0] / 2)
    shifter_x = int(sizebeam[1] / 2)
    Ly = np.roll(qy, shifter_y)
    Lx = np.roll(qx, shifter_x)
    dL_y = Ly[1] - Ly[0]
    dL_x = Lx[1] - Lx[0]
    return np.asarray((dL_y, dL_x))


def make_probe(aperture, voltage, image_size, calibration_pm, defocus=0, c3=0, c5=0):
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
    qx = (np.arange((-image_x / 2), (image_x / 2), 1)) / x_FOV
    x_shifter = int(round(image_x / 2))
    qy = (np.arange((-image_y / 2), (image_y / 2), 1)) / y_FOV
    y_shifter = int(round(image_y / 2))
    Lx = np.roll(qx, x_shifter)
    Ly = np.roll(qy, y_shifter)
    Lya, Lxa = np.meshgrid(Lx, Ly)
    L2 = np.multiply(Lxa, Lxa) + np.multiply(Lya, Lya)
    inverse_real_matrix = L2 ** 0.5
    fourier_scan_coordinate = Lx[1] - Lx[0]
    Adist = np.asarray(inverse_real_matrix <= LMax, dtype=complex)
    chi_probe = aberration(inverse_real_matrix, wavelength, defocus, c3, c5)
    Adist *= np.exp(-1j * chi_probe)
    probe_real_space = np.fft.ifftshift(np.fft.ifft2(Adist))
    return probe_real_space.astype(np.complex64)


def aberration(fourier_coord, wavelength_ang, defocus=0, c3=0, c5=0):
    p_matrix = wavelength_ang * fourier_coord
    chi = (
        ((defocus * np.power(p_matrix, 2)) / 2)
        + ((c3 * (1e7) * np.power(p_matrix, 4)) / 4)
        + ((c5 * (1e7) * np.power(p_matrix, 6)) / 6)
    )
    chi_probe = (2 * np.pi * chi) / wavelength_ang
    return chi_probe


def atomic_potential(
    atom_no,
    pixel_size,
    sampling=16,
    potential_extent=4,
    datafile="Kirkland_Potentials.npy",
):
    """
    Calculate the projected potential of a single atom

    Parameters
    ----------
    atom_no:          int
                      Atomic number of the atom whose potential is being calculated.
    pixel_size:       float
                      Real space pixel size
    datafile:         string
                      Load the location of the npy file of the Kirkland scattering factors
    sampling:         int, float
                      Supersampling factor for increased accuracy. Matters more with big
                      pixel sizes. The default value is 16.
    potential_extent: float
                      Distance in angstroms from atom center to which the projected
                      potential is calculated. The default value is 4 angstroms.

    Returns
    -------
    potential: ndarray
               Projected potential matrix

    Notes
    -----
    We calculate the projected screened potential of an
    atom using the Kirkland formula. Keep in mind however
    that this potential is for independent atoms only!
    No charge distribution between atoms occure here.

    References
    ----------
    Kirkland EJ. Advanced computing in electron microscopy.
    Springer Science & Business Media; 2010 Aug 12.

    :Authors:
    Debangshu Mukherjee <mukherjeed@ornl.gov>

    """
    a0 = 0.5292
    ek = 14.4
    term1 = 4 * (np.pi ** 2) * a0 * ek
    term2 = 2 * (np.pi ** 2) * a0 * ek
    kirkland = np.load(datafile)
    xsub = np.arange(-potential_extent, potential_extent, (pixel_size / sampling))
    ysub = np.arange(-potential_extent, potential_extent, (pixel_size / sampling))
    kirk_fun = kirkland[atom_no - 1, :]
    ya, xa = np.meshgrid(ysub, xsub)
    r2 = np.power(xa, 2) + np.power(ya, 2)
    r = np.power(r2, 0.5)
    part1 = np.zeros_like(r)
    part2 = np.zeros_like(r)
    sspot = np.zeros_like(r)
    part1 = term1 * (
        np.multiply(
            kirk_fun[0],
            s2.kv(0, (np.multiply((2 * np.pi * np.power(kirk_fun[1], 0.5)), r))),
        )
        + np.multiply(
            kirk_fun[2],
            s2.kv(0, (np.multiply((2 * np.pi * np.power(kirk_fun[3], 0.5)), r))),
        )
        + np.multiply(
            kirk_fun[4],
            s2.kv(0, (np.multiply((2 * np.pi * np.power(kirk_fun[5], 0.5)), r))),
        )
    )
    part2 = term2 * (
        (kirk_fun[6] / kirk_fun[7]) * np.exp(-((np.pi ** 2) / kirk_fun[7]) * r2)
        + (kirk_fun[8] / kirk_fun[9]) * np.exp(-((np.pi ** 2) / kirk_fun[9]) * r2)
        + (kirk_fun[10] / kirk_fun[11]) * np.exp(-((np.pi ** 2) / kirk_fun[11]) * r2)
    )
    sspot = part1 + part2
    finalsize = (np.asarray(sspot.shape) / sampling).astype(int)
    sspot_im = PIL.Image.fromarray(sspot)
    potential = np.array(sspot_im.resize(finalsize, resample=PIL.Image.LANCZOS))
    return potential


def find_uc_pos(atom_pos, cell_dim):
    uc_pos = np.zeros_like(atom_pos)
    for ii in numba.prange(len(uc_pos)):
        for jj in range(len(cell_dim)):
            cc = atom_pos[ii, :] / cell_dim[jj, :]
            cc[cc < 0] += 1
            cc[cc == np.inf] = 0
            cc[cc > 0.001]
            uc_pos[ii, jj] = cc[jj]
    uc_nonzero = uc_pos != 0
    uc_inv = 1 / uc_pos[uc_nonzero]
    uc_inv[np.abs(uc_inv - np.round(uc_inv)) < 0.001] = np.round(
        uc_inv[np.abs(uc_inv - np.round(uc_inv)) < 0.001]
    )
    uc_pos[uc_nonzero] = 1 / uc_inv
    uc_pos[uc_pos == 1] = 0
    return uc_pos


def miller_inverse(miller):
    miller_inv = np.empty_like(miller, dtype=np.float)
    miller_inv[miller == 0] = 0
    miller_inv[miller != 0] = 1 / miller[miller != 0]
    return miller_inv


def get_number_cells(miller_dir, length, cell_dim):
    minverse = miller_inverse(miller_dir)
    slicedir = np.matmul(np.transpose(cell_dim), minverse)
    no_cells = np.round(minverse * (length / np.linalg.norm(slicedir)))
    return no_cells


def slabbing_2D(miller_dir, no_cells, max_hdist):
    yy, xx = np.meshgrid(
        np.arange(0, int(no_cells[1]), 1), np.arange(0, int(no_cells[0]), 1)
    )
    yy = yy.ravel()
    xx = xx.ravel()
    xp = np.arange(np.amax((np.amax(yy), np.max(xx))))
    yp = xp * (miller_dir[0] / miller_dir[1])
    point_distances = np.abs((miller_dir[1] * yy) - (miller_dir[0] * xx)) / (
        ((miller_dir[1] ** 2) + (miller_dir[0] ** 2)) ** 0.5
    )
    yy_new, xx_new = np.meshgrid(
        np.arange(0 - np.ceil(max_hdist), int(no_cells[1]) + np.ceil(max_hdist), 1),
        np.arange(0 - np.ceil(max_hdist), int(no_cells[0]) + np.ceil(max_hdist), 1),
    )
    yy_new = yy_new.ravel()
    xx_new = xx_new.ravel()
    dists = np.abs((miller_dir[1] * yy_new) - (miller_dir[0] * xx_new)) / (
        ((miller_dir[1] ** 2) + (miller_dir[0] ** 2)) ** 0.5
    )
    xx_firstpass = xx_new[dists < max_hdist]
    yy_firstpass = yy_new[dists < max_hdist]
    dist_angles = np.abs(
        np.arctan2((yy_firstpass - 0), (xx_firstpass - 0))
        - np.arctan2(miller_dir[0], miller_dir[1])
    )
    xx_secondpass = xx_firstpass[dist_angles < (np.pi / 2)]
    yy_secondpass = yy_firstpass[dist_angles < (np.pi / 2)]
    dist_angles2 = np.abs(
        np.arctan2((yy_secondpass - 81), (xx_secondpass - 40))
        - np.arctan2(miller_dir[0], miller_dir[1])
    )
    xx_thirdpass = xx_secondpass[dist_angles2 > (np.pi / 2)]
    yy_thirdpass = yy_secondpass[dist_angles2 > (np.pi / 2)]
    vals = np.asarray((yy_thirdpass, xx_thirdpass))
    return vals.transpose()


def fwxm(probe2D, psize, x=0.5):
    p2 = np.abs(probe2D)
    pmax = np.amax(p2)
    pr = p2 > (x * pmax)
    radius = (np.sum(pr) / np.pi) ** 0.5
    return radius * psize


def move_probe_mesh(probe, pos, fmeshy, fmeshx):
    image_size = np.asarray(probe.shape)
    rel_pos = pos - (image_size / 2)
    move_mat = np.exp((fmeshx * rel_pos[1]) + (fmeshy * rel_pos[0]))
    moved_probe = np.fft.ifft2(move_mat * np.fft.fft2(probe))
    return moved_probe


def diff_to_im(cbed_pattern, detector):
    val = np.zeros(detector.shape[-1], dtype=np.float32)
    for ii in np.arange(detector.shape[-1]):
        val[ii] = np.sum(cbed_pattern[detector[:, :, ii]])
    return val


def annular_stem(probe, trans_array, prop, pos, coll_angles, pixel_size, voltage_kv):
    ypos = pos[0]
    xpos = pos[1]
    [xpos, ypos] = np.meshgrid(ypos, xpos)
    fcal_y = (
        np.linspace((-probe.shape[0] / 2), ((probe.shape[0] / 2) - 1), probe.shape[0])
    ) / probe.shape[0]
    fcal_x = (
        np.linspace((-probe.shape[1] / 2), ((probe.shape[1] / 2) - 1), probe.shape[1])
    ) / probe.shape[1]
    [fmesh_x, fmesh_y] = np.meshgrid(fcal_x, fcal_y)
    mmesh_x = ((-2) * np.pi * 1j * fmesh_x).astype(np.complex64)
    mmesh_y = ((-2) * np.pi * 1j * fmesh_y).astype(np.complex64)
    fmesh_r = (((fmesh_x ** 2) + (fmesh_y ** 2)) ** 0.5) / pixel_size
    radians_r = fmesh_r * st.sim.wavelength_ang(voltage_kv)
    detectors = np.zeros(
        (radians_r.shape[0], radians_r.shape[1], coll_angles.shape[0]), dtype=bool
    )

    for cc in np.arange(coll_angles.shape[0]):
        inner_angle = coll_angles[cc, 0]
        outer_angle = coll_angles[cc, 1]
        detectors[:, :, cc] = np.logical_and(
            ((inner_angle / 1000) < radians_r), (radians_r < (outer_angle / 1000))
        )

    ypos_da = da.from_array(ypos)
    xpos_da = da.from_array(xpos)
    probe_sc = client.scatter(probe, broadcast=True)
    prop_sc = client.scatter(prop, broadcast=True)
    mmesh_y_sc = client.scatter(mmesh_y, broadcast=True)
    mmesh_x_sc = client.scatter(mmesh_x, broadcast=True)
    trans_array_sc = client.scatter(trans_array, broadcast=True)
    detectors_sc = client.scatter(detectors, broadcast=True)
    stem_image = []

    with tqdm(total=ypos.shape[0]) as pbar:
        for ii in np.arange(ypos.shape[0]):
            im_vals = []
            for jj in np.arange(ypos.shape[1]):
                moved_probe = dask.delayed(move_probe_mesh)(
                    probe_sc, (ypos_da[ii, jj], xpos_da[ii, jj]), mmesh_y_sc, mmesh_x_sc
                )
                cbed_pattern = dask.delayed(cbed)(moved_probe, trans_array_sc, prop_sc)
                aperture_vals = dask.delayed(diff_to_im)(cbed_pattern, detectors_sc)
                im_vals.append(aperture_vals)
            im_vals = np.asarray(dask.compute(*im_vals))
            stem_image.append(im_vals)
            pbar.update(1)
    stem_image = np.asarray(stem_image)
    return stem_image


def pixellated_stem(probe, trans_array, prop, pos):
    ypos = pos[0]
    xpos = pos[1]
    probeft = np.fft.fft2(probe)
    probeft /= np.sum(np.abs(probeft))
    probe = np.fft.ifft2(probeft)
    [xpos, ypos] = np.meshgrid(ypos, xpos)
    fcal_y = (
        np.linspace((-probe.shape[0] / 2), ((probe.shape[0] / 2) - 1), probe.shape[0])
    ) / probe.shape[0]
    fcal_x = (
        np.linspace((-probe.shape[1] / 2), ((probe.shape[1] / 2) - 1), probe.shape[1])
    ) / probe.shape[1]
    [fmesh_x, fmesh_y] = np.meshgrid(fcal_x, fcal_y)
    mmesh_x = ((-2) * np.pi * 1j * fmesh_x).astype(np.complex64)
    mmesh_y = ((-2) * np.pi * 1j * fmesh_y).astype(np.complex64)

    ypos_da = da.from_array(ypos)
    xpos_da = da.from_array(xpos)
    probe_sc = client.scatter(probe, broadcast=True)
    prop_sc = client.scatter(prop, broadcast=True)
    mmesh_y_sc = client.scatter(mmesh_y, broadcast=True)
    mmesh_x_sc = client.scatter(mmesh_x, broadcast=True)
    trans_array_sc = client.scatter(trans_array, broadcast=True)
    stem_4D = []

    with tqdm(total=ypos.shape[0]) as pbar:
        for ii in np.arange(ypos.shape[0]):
            im_vals = []
            for jj in np.arange(ypos.shape[1]):
                moved_probe = dask.delayed(move_probe_mesh)(
                    probe_sc, (ypos_da[ii, jj], xpos_da[ii, jj]), mmesh_y_sc, mmesh_x_sc
                )
                cbd = dask.delayed(cbed)(moved_probe, trans_array_sc, prop_sc)
                im_vals.append(cbd)
            im_vals = np.asarray(dask.compute(*im_vals))
            stem_4D.append(im_vals)
            pbar.update(1)
    stem_4D = np.asarray(stem_4D)
    return stem_4D
