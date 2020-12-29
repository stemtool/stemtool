import numpy as np
import skimage.restoration as skr
import scipy.ndimage as scnd
import matplotlib as mpl
import matplotlib.pyplot as plt
import stemtool as st
import matplotlib.offsetbox as mploff
import matplotlib.gridspec as mpgs
import matplotlib_scalebar.scalebar as mpss
import numba


def phase_diff(angle_image):
    """
    Differentiate a complex phase image while
    ensuring that phase wrapping doesn't
    distort the differentiation.

    Parameters
    ----------
    angle_image:  ndarray
                  Wrapped phase image

    Returns
    -------
    diff_x: ndarray
            X differential of the phase image
    diff_y: ndarray
            Y differential of the phase image

    Notes
    -----
    The basic idea of this is that we differentiate the
    complex exponential of the phase image, and then obtain the
    differentiation result by multiplying the differential with
    the conjugate of the complex phase image.

    Reference
    ---------
    .. [1] Hÿtch, M. J., E. Snoeck, and R. Kilaas. "Quantitative measurement
       of displacement and strain fields from HREM micrographs."
       Ultramicroscopy 74.3 (1998): 131-146.
    """
    imaginary_image = np.exp(1j * angle_image)
    diff_imaginary_x = np.zeros(imaginary_image.shape, dtype="complex_")
    diff_imaginary_x[:, 0:-1] = np.diff(imaginary_image, axis=1)
    diff_imaginary_y = np.zeros(imaginary_image.shape, dtype="complex_")
    diff_imaginary_y[0:-1, :] = np.diff(imaginary_image, axis=0)
    conjugate_imaginary = np.conj(imaginary_image)
    diff_complex_x = np.multiply(conjugate_imaginary, diff_imaginary_x)
    diff_complex_y = np.multiply(conjugate_imaginary, diff_imaginary_y)
    diff_x = np.imag(diff_complex_x)
    diff_y = np.imag(diff_complex_y)
    return diff_x, diff_y


def phase_subtract(matrix_1, matrix_2):
    """
    Subtract one complex phase image from
    another without causing phase wrapping.

    Parameters
    ----------
    matrix_1:  ndarray
               First phase image
    matrix_2:  ndarray
               Second phase image

    Returns
    -------
             : ndarray
               Difference of the phase images

    Notes
    -----
    The basic idea of this is that we subtract the
    phase images from each other, then transform
    that to a complex phase, and take the angle of
    the complex image.
    """
    return np.angle(np.exp(1j * (matrix_1 - matrix_2)))


def circ_to_G(circ_pos, image):
    """
    Convert a pixel position to g vectors in
    Fourier space.

    Parameters
    ----------
    circ_pos:  tuple
               First phase image
    image:     ndarray
               The image matrix

    Returns
    -------
    g_vec: ndarray
           Shape is (2, 1) which is the
           corresponding g-vector in inverse pixels

    See Also
    --------
    G_to_circ
    """
    g_vec = np.zeros(2)
    g_vec[0:2] = np.divide(np.flip(np.asarray(circ_pos)), np.asarray(image.shape)) - 0.5
    return g_vec


def G_to_circ(g_vec, image):
    """
    Convert g vectors in Fourier space to
    pixel positions in real space.

    Parameters
    ----------
    g_vec: ndarray
           Shape is (2, 1) which is the G
           vector in Fourier space in inverse pixels
    image: ndarray
           The image matrix

    Returns
    -------
    circ_pos: ndarray
           Shape is (2, 1) which is the
           corresponding pixel position in
           real space.

    See Also
    --------
    circ_to_G
    """
    circ_pos = np.zeros(2)
    circ_pos[1] = (g_vec[0] * image.shape[0]) + (0.5 * image.shape[0])
    circ_pos[0] = (g_vec[1] * image.shape[1]) + (0.5 * image.shape[1])
    return circ_pos


def g_matrix(g_vector, image):
    """
    Multiply g vector with Fourier coordinates
    to generate a corresponding phase matrix

    Parameters
    ----------
    g_vec: ndarray
           Shape is (2, 1) which is the G
           vector in Fourier space in inverse pixels
    image: ndarray
           The image matrix

    Returns
    -------
    G_r: ndarray
         Same size as the image originally
         and gives the phase map for a given
         g vector
    """
    r_y = np.arange(start=-image.shape[0] / 2, stop=image.shape[0] / 2, step=1)
    r_x = np.arange(start=-image.shape[1] / 2, stop=image.shape[1] / 2, step=1)
    R_x, R_y = np.meshgrid(r_x, r_y)
    G_r = 2 * np.pi * ((R_x * g_vector[1]) + (R_y * g_vector[0]))
    return G_r


def phase_matrix(gvec, image, circ_size=0, g_blur=True):
    """
    Use the g vector in Fourier coordinates
    to select only the subset of phases
    associated with that diffraction spot,
    a.k.a. the lattice parameter.

    Parameters
    ----------
    g_vec:     ndarray
               Shape is (2, 1) which is the G
               vector in Fourier space in inverse pixels
    image:     ndarray
               The image matrix
    circ_size: float, optional
               Size of the circle in pixels
    g_blur:    bool, optional

    Returns
    -------
    P_matrix: ndarray
              Same size as the image originally
              and gives a real space phase matrix
              for a given real image and a g vector

    Notes
    -----
    We put an aperture around a single diffraction
    spot, given by the g vector that generates the
    phase matrix associated with that diffraction
    spot. If the g vector is already refined, then
    in the reference region, the difference between
    this phase matrix and that given by `g_matrix`
    should be zero.

    See Also
    --------
    g_matrix
    """
    imshape = np.asarray(np.shape(image))
    if circ_size == 0:
        circ_rad = np.amin(0.01 * np.asarray(imshape))
    else:
        circ_rad = circ_size
    yy, xx = np.mgrid[0 : imshape[0], 0 : imshape[1]]
    circ_pos = np.multiply(np.flip(gvec), imshape) + (0.5 * imshape)
    circ_mask = (
        st.util.make_circle(imshape, circ_pos[0], circ_pos[1], circ_rad)
    ).astype(bool)
    ham = np.sqrt(np.outer(np.hamming(imshape[0]), np.hamming(imshape[1])))

    if g_blur:
        sigma2 = np.sum((0.5 * gvec * imshape) ** 2)
        zz = (
            ((yy[circ_mask] - circ_pos[1]) ** 2) + ((xx[circ_mask] - circ_pos[0]) ** 2)
        ) / sigma2
        four_mask = np.zeros_like(yy, dtype=np.float)
        four_mask[circ_mask] = np.exp((-0.5) * zz)
        P_matrix = np.angle(
            np.fft.ifft2(four_mask * np.fft.fftshift(np.fft.fft2(image * ham)))
        )
    else:
        P_matrix = np.angle(
            np.fft.ifft2(circ_mask * np.fft.fftshift(np.fft.fft2(image * ham)))
        )
    return P_matrix


@numba.jit(cache=True, parallel=True)
def numba_strain_P(P_1, P_2, a_matrix):
    """
    Use the refined phase matrices and lattice matrix
    to calculate the strain matrices.

    Parameters
    ----------
    P_1:      ndarray
              Refined Phase matrix from first lattice spot
    P_2:      ndarray
              Refined Phase matrix from first lattice spot
    a_matrix: ndarray
              ndarray of shape (2, 2) that represents
              the lattice parameters in real space

    Returns
    -------
    e_xx: ndarray
          Strain along X direction
    e_yy: ndarray
          Strain along Y direction
    e_th: ndarray
          Rotational strain
    e_dg: ndarray
          Diagonal Strain

    Notes
    -----
    This is a numba accelerated JIT compiled
    version of the method `gen_strain()` in the
    where a for loop is used to refine the strain
    at every pixel position.

    See Also
    --------
    phase_diff
    GPA.gen_strain()
    """
    P1_x, P1_y = phase_diff(P_1)
    P2_x, P2_y = phase_diff(P_2)
    P_shape = np.shape(P_1)
    yy, xx = np.mgrid[0 : P_shape[0], 0 : P_shape[1]]
    yy = np.ravel(yy)
    xx = np.ravel(xx)
    P_mat = np.zeros((2, 2), dtype=np.float)
    e_xx = np.zeros_like(P_1)
    e_xy = np.zeros_like(P_1)
    e_yx = np.zeros_like(P_1)
    e_yy = np.zeros_like(P_1)
    for ii in range(len(yy)):
        ypos = yy[ii]
        xpos = xx[ii]
        P_mat[0, 0] = P1_x[ypos, xpos]
        P_mat[0, 1] = P1_y[ypos, xpos]
        P_mat[1, 0] = P2_x[ypos, xpos]
        P_mat[1, 1] = P2_y[ypos, xpos]
        e_mat = ((1) / (2 * np.pi)) * np.matmul(a_matrix, P_mat)
        e_xx[ypos, xpos] = e_mat[0, 0]
        e_xy[ypos, xpos] = e_mat[0, 1]
        e_yx[ypos, xpos] = e_mat[1, 0]
        e_yy[ypos, xpos] = e_mat[1, 1]
    e_th = 0.5 * (e_xy - e_yx)
    e_dg = 0.5 * (e_xy + e_yx)
    return e_xx, e_yy, e_th, e_dg


class GPA(object):
    """
    Use Geometric Phase Analysis (GPA) to measure strain in an
    electron micrograph by locating the diffraction spots and
    identifying a reference region

    Parameters
    ----------
    image:       ndarray
                 The image from which the strain will
                 be measured from
    calib:       float
                 Size of an individual pixel
    calib_units: str
                 Unit of calibration
    ref_iter:    int, optional
                 Number of iterations to run for refining
                 the G vectors and the phase matrixes.
                 Default is 20.
    use_blur:    bool, optional
                 Use a Gaussian blur to generate the
                 phase matrix from a g vector. Default is True

    References
    ----------
    .. [1] Hÿtch, M. J., E. Snoeck, and R. Kilaas. "Quantitative measurement
       of displacement and strain fields from HREM micrographs."
       Ultramicroscopy 74.3 (1998): 131-146.

    Examples
    --------
    Run as:

    >>> im_gpa = gpa(image=imageDC, calib=calib1, calib_units= calib1_units)

    Then to check the image you just loaded

    >>> im_gpa.show_image()

    Then, select the diffraction spots in inverse units that you
    want to be used for GPA. They must not be collinear.

    >>> im_gpa.find_spots((5, 0), (0, -5))

    where (5, 0) and (0, -5) are two diffraction spot
    locations. You can run the `find_spots` method manually
    multiple times till you locate the spots closely. After
    you have located the spots, you need to define a reference
    region for the image - with respect to which the strain
    will be calculated.

    >>> im_gpa.define_reference((6.8, 6.9), (10.1, 6.8), (10.2, 9.5), (7.0, 9.6))

    where (6.8, 6.9), (10.1, 6.8), (10.2, 9.5) and (7.0, 9.6) are
    the corners of the reference region you are defining.

    >>> im_gpa.refine_phase()
    >>> e_xx, e_yy, e_theta, e_diag = im_gpa.get_strain()

    To plot the obtained strain maps:

    >>> im_gpa.plot_gpa_strain()

    """

    def __init__(
        self, image, calib, calib_units, ref_iter=20, use_blur=True, max_strain=0.4
    ):
        self.image = image
        self.calib = calib
        self.calib_units = calib_units
        self.blur = use_blur
        self.ref_iter = int(ref_iter)
        self.imshape = np.asarray(image.shape)
        inv_len = 1 / (self.calib * self.imshape)
        if inv_len[0] == inv_len[1]:
            self.inv_calib = np.mean(inv_len)
        else:
            raise RuntimeError("Please ensure that the image is a square image")
        self.circ_0 = 0.5 * self.imshape
        self.inv_cal_units = "1/" + calib_units
        self.max_strain = max_strain
        self.spots_check = False
        self.reference_check = False
        self.refining_check = False

    def show_image(self, imsize=(15, 15), colormap="inferno"):
        """
        Parameters
        ----------
        imsize:   tuple, optional
                  Size in inches of the image with the
                  diffraction spots marked. Default is (15, 15)
        colormap: str, optional
                  Colormap of the image. Default is inferno
        """
        plt.figure(figsize=imsize)
        plt.imshow(self.image, cmap=colormap)
        scalebar = mpss.ScaleBar(self.calib, self.calib_units)
        scalebar.location = "lower right"
        scalebar.box_alpha = 1
        scalebar.color = "k"
        plt.gca().add_artist(scalebar)
        plt.axis("off")

    def find_spots(self, circ1, circ2, circ_size=15, imsize=(10, 10)):
        """
        Locate the diffraction spots visually.

        Parameters
        ----------
        circ1:     ndarray
                   Position of the first beam in
                   the Fourier pattern
        circ2:     ndarray
                   Position of the second beam in
                   the Fourier pattern
        circ_size: float
                   Size of the circle in pixels
        imsize:    tuple, optional
                   Size in inches of the image with the
                   diffraction spots marked. Default is
                   (10, 10)

        Notes
        -----
        Put circles in red(central), y(blue) and x(green)
        on the diffraction pattern to approximately know
        the positions. We also convert the circle locations
        to G vectors by calling the static method `circ_to_G`.
        We use the G vector locations to also generate the
        initial phase matrices.

        See Also
        --------
        circ_to_G
        phase_matrix
        """
        self.circ_1 = (self.imshape / 2) + (np.asarray(circ1) / self.inv_calib)
        self.circ_2 = (self.imshape / 2) + (np.asarray(circ2) / self.inv_calib)
        self.circ_size = circ_size
        self.ham = np.sqrt(
            np.outer(np.hamming(self.imshape[0]), np.hamming(self.imshape[1]))
        )
        self.image_ft = np.fft.fftshift(np.fft.fft2(self.image * self.ham))
        log_abs_ft = scnd.filters.gaussian_filter(np.log10(np.abs(self.image_ft)), 3)

        pixel_list = np.arange(
            -0.5 * self.inv_calib * self.imshape[0],
            0.5 * self.inv_calib * self.imshape[0],
            self.inv_calib,
        )
        no_labels = 9
        step_x = int(self.imshape[0] / (no_labels - 1))
        x_positions = np.arange(0, self.imshape[0], step_x)
        x_labels = np.round(pixel_list[::step_x], 1)

        _, ax = plt.subplots(figsize=imsize)
        circ_0_im = plt.Circle(self.circ_0, self.circ_size, color="red", alpha=0.75)
        circ_1_im = plt.Circle(self.circ_1, self.circ_size, color="blue", alpha=0.75)
        circ_2_im = plt.Circle(self.circ_2, self.circ_size, color="green", alpha=0.75)
        ax.imshow(log_abs_ft, cmap="gray")
        ax.add_artist(circ_0_im)
        ax.add_artist(circ_1_im)
        ax.add_artist(circ_2_im)
        plt.xticks(x_positions, x_labels)
        plt.yticks(x_positions, x_labels)
        plt.xlabel("Distance along X-axis (" + self.inv_cal_units + ")")
        plt.ylabel("Distance along Y-axis (" + self.inv_cal_units + ")")

        self.gvec_1_ini = st.gpa.circ_to_G(self.circ_1, self.image)
        self.gvec_2_ini = st.gpa.circ_to_G(self.circ_2, self.image)
        self.P_matrix1_ini = st.gpa.phase_matrix(
            self.gvec_1_ini, self.image, self.circ_size, self.blur
        )
        self.P_matrix2_ini = st.gpa.phase_matrix(
            self.gvec_2_ini, self.image, self.circ_size, self.blur
        )
        self.spots_check = True

    def define_reference(self, A_pt, B_pt, C_pt, D_pt, imsize=(10, 10), tColor="k"):
        """
        Locate the reference image.

        Parameters
        ----------
        A_pt:   tuple
                Top left position of reference region in (x, y)
        B_pt:   tuple
                Top right position of reference region in (x, y)
        C_pt:   tuple
                Bottom right position of reference region in (x, y)
        D_pt:   tuple
                Bottom left position of reference region in (x, y)
        imsize: tuple, optional
                Size in inches of the image with the
                diffraction spots marked. Default is
                (10, 10)
        tColor: str, optional
                Color of the text on the image

        Notes
        -----
        Locates a reference region bounded by the four points given in
        length units. Choose the points in a clockwise fashion.
        """
        if not self.spots_check:
            raise RuntimeError(
                "Please locate the diffraction spots first as find_spots()"
            )

        A = np.asarray(A_pt) / self.calib
        B = np.asarray(B_pt) / self.calib
        C = np.asarray(C_pt) / self.calib
        D = np.asarray(D_pt) / self.calib

        yy, xx = np.mgrid[0 : self.imshape[0], 0 : self.imshape[1]]
        yy = np.ravel(yy)
        xx = np.ravel(xx)
        ptAA = np.asarray((xx, yy)).transpose() - A
        ptBB = np.asarray((xx, yy)).transpose() - B
        ptCC = np.asarray((xx, yy)).transpose() - C
        ptDD = np.asarray((xx, yy)).transpose() - D
        angAABB = np.arccos(
            np.sum(ptAA * ptBB, axis=1)
            / (
                ((np.sum(ptAA ** 2, axis=1)) ** 0.5)
                * ((np.sum(ptBB ** 2, axis=1)) ** 0.5)
            )
        )
        angBBCC = np.arccos(
            np.sum(ptBB * ptCC, axis=1)
            / (
                ((np.sum(ptBB ** 2, axis=1)) ** 0.5)
                * ((np.sum(ptCC ** 2, axis=1)) ** 0.5)
            )
        )
        angCCDD = np.arccos(
            np.sum(ptCC * ptDD, axis=1)
            / (
                ((np.sum(ptCC ** 2, axis=1)) ** 0.5)
                * ((np.sum(ptDD ** 2, axis=1)) ** 0.5)
            )
        )
        angDDAA = np.arccos(
            np.sum(ptDD * ptAA, axis=1)
            / (
                ((np.sum(ptDD ** 2, axis=1)) ** 0.5)
                * ((np.sum(ptAA ** 2, axis=1)) ** 0.5)
            )
        )
        angsum = ((angAABB + angBBCC + angCCDD + angDDAA) / (2 * np.pi)).reshape(
            self.image.shape
        )
        self.ref_reg = np.isclose(angsum, 1)
        self.ref_reg = np.flipud(self.ref_reg)

        pixel_list = np.arange(0, self.calib * self.imshape[0], self.calib)
        no_labels = 10
        step_x = int(self.imshape[0] / (no_labels - 1))
        x_positions = np.arange(0, self.imshape[0], step_x)
        x_labels = np.round(pixel_list[::step_x], 1)
        fsize = int(1.5 * np.mean(np.asarray(imsize)))

        print(
            "Choose your points in a clockwise fashion, or else you will get a wrong result"
        )

        plt.figure(figsize=imsize)
        plt.imshow(
            np.flipud(st.util.image_normalizer(self.image) + 0.33 * self.ref_reg),
            cmap="magma",
            origin="lower",
        )
        plt.annotate(
            "A=" + str(A_pt),
            A / self.imshape,
            textcoords="axes fraction",
            size=fsize,
            color=tColor,
        )
        plt.annotate(
            "B=" + str(B_pt),
            B / self.imshape,
            textcoords="axes fraction",
            size=fsize,
            color=tColor,
        )
        plt.annotate(
            "C=" + str(C_pt),
            C / self.imshape,
            textcoords="axes fraction",
            size=fsize,
            color=tColor,
        )
        plt.annotate(
            "D=" + str(D_pt),
            D / self.imshape,
            textcoords="axes fraction",
            size=fsize,
            color=tColor,
        )
        plt.scatter(A[0], A[1], c="r")
        plt.scatter(B[0], B[1], c="r")
        plt.scatter(C[0], C[1], c="r")
        plt.scatter(D[0], D[1], c="r")
        plt.xticks(x_positions, x_labels, fontsize=fsize)
        plt.yticks(x_positions, x_labels, fontsize=fsize)
        plt.xlabel("Distance along X-axis (" + self.calib_units + ")", fontsize=fsize)
        plt.ylabel("Distance along Y-axis (" + self.calib_units + ")", fontsize=fsize)
        self.reference_check = True

    def refine_phase(self):
        """
        Refine the phase matrices and the G vectors
        from their initial values using the reference
        region location.

        Notes
        -----
        Iteratively refine the G vector and the phase matrices,
        so that the phase variation in the reference region is
        minimized.

        See Also
        --------
        phase_diff
        phase_matrix
        """
        if not self.reference_check:
            raise RuntimeError(
                "Please locate the reference region first as define_reference()"
            )
        ry = np.arange(start=-self.imshape[0] / 2, stop=self.imshape[0] / 2, step=1)
        rx = np.arange(start=-self.imshape[1] / 2, stop=self.imshape[1] / 2, step=1)
        R_x, R_y = np.meshgrid(rx, ry)
        self.gvec_1_fin = self.gvec_1_ini
        self.gvec_2_fin = self.gvec_2_ini
        self.P_matrix1_fin = self.P_matrix1_ini
        self.P_matrix2_fin = self.P_matrix2_ini
        for _ in range(int(self.ref_iter)):
            G1_x, G1_y = st.gpa.phase_diff(self.P_matrix1_fin)
            G2_x, G2_y = st.gpa.phase_diff(self.P_matrix2_fin)
            g1_r = (G1_x + G1_y) / (2 * np.pi)
            g2_r = (G2_x + G2_y) / (2 * np.pi)
            del_g1 = np.asarray(
                (
                    np.median(g1_r[self.ref_reg] / R_y[self.ref_reg]),
                    np.median(g1_r[self.ref_reg] / R_x[self.ref_reg]),
                )
            )
            del_g2 = np.asarray(
                (
                    np.median(g2_r[self.ref_reg] / R_y[self.ref_reg]),
                    np.median(g2_r[self.ref_reg] / R_x[self.ref_reg]),
                )
            )
            self.gvec_1_fin += del_g1
            self.gvec_2_fin += del_g2
            self.P_matrix1_fin = st.gpa.phase_matrix(
                self.gvec_1_fin, self.image, self.circ_size, self.blur
            )
            self.P_matrix2_fin = st.gpa.phase_matrix(
                self.gvec_2_fin, self.image, self.circ_size, self.blur
            )
        self.refining_check = True

    def get_strain(self):
        """
        Use the refined phase matrix and g vectors to calculate
        the strain matrices.

        Returns
        -------
        e_xx: ndarray
              Strain along X direction
        e_yy: ndarray
              Strain along Y direction
        e_th: ndarray
              Rotational strain
        e_dg: ndarray
              Diagonal Strain

        Notes
        -----
        Use the refined G vectors to generate a matrix
        of the lattice parameters, which is stored as the
        class attribute `a_matrix`. This is multiplied by the
        refined phase matrix, and the multiplicand is subsequently
        differentiated to get the strain parameters.

        See Also
        --------
        phase_diff
        """
        if not self.reference_check:
            raise RuntimeError(
                "Please refine the phase and g vectors first as refine_phase()"
            )
        g_matrix = np.zeros((2, 2), dtype=np.float64)
        g_matrix[0, :] = np.flip(np.asarray(self.gvec_1_fin))
        g_matrix[1, :] = np.flip(np.asarray(self.gvec_2_fin))
        self.a_matrix = np.linalg.inv(np.transpose(g_matrix))
        P1 = skr.unwrap_phase(self.P_matrix1_fin)
        P2 = skr.unwrap_phase(self.P_matrix1_fin)
        rolled_p = np.asarray((np.reshape(P1, -1), np.reshape(P2, -1)))
        u_matrix = np.matmul(self.a_matrix, rolled_p)
        u_x = np.reshape(u_matrix[0, :], P1.shape)
        u_y = np.reshape(u_matrix[1, :], P2.shape)
        self.e_xx, e_xy = st.gpa.phase_diff(u_x)
        e_yx, self.e_yy = st.gpa.phase_diff(u_y)
        self.e_th = 0.5 * (e_xy - e_yx)
        self.e_dg = 0.5 * (e_xy + e_yx)
        self.e_yy -= np.median(self.e_yy[self.ref_reg])
        self.e_dg -= np.median(self.e_dg[self.ref_reg])
        self.e_th -= np.median(self.e_th[self.ref_reg])
        self.e_xx -= np.median(self.e_xx[self.ref_reg])

        if self.max_strain > 0:
            self.e_yy[self.e_yy > self.max_strain] = self.max_strain
            self.e_yy[self.e_yy < -self.max_strain] = -self.max_strain
            self.e_dg[self.e_dg > self.max_strain] = self.max_strain
            self.e_dg[self.e_dg < -self.max_strain] = -self.max_strain
            self.e_th[self.e_th > self.max_strain] = self.max_strain
            self.e_th[self.e_th < -self.max_strain] = -self.max_strain
            self.e_xx[self.e_xx > self.max_strain] = self.max_strain
            self.e_xx[self.e_xx < -self.max_strain] = -self.max_strain
        return self.e_xx, self.e_yy, self.e_th, self.e_dg

    def plot_gpa_strain(self, mval=0, imwidth=15):
        """
        Use the calculated strain matrices to plot the strain maps

        Parameters
        ----------
        mval:    float, optional
                 The maximum strain value that will be plotted.
                 Default is 0, upon which the maximum strain
                 percentage will be calculated, which will be used
                 for plotting.
        imwidth: int, optional
                 Size in inches of the image with the
                 diffraction spots marked. Default is 15

        Notes
        -----
        Uses `matplotlib.gridspec` to plot the strain maps of the
        four types of strain calculated through geometric phase
        analysis.
        """
        fontsize = int(imwidth)
        if mval == 0:
            vm = 100 * np.amax(
                np.abs(
                    np.concatenate((self.e_yy, self.e_xx, self.e_dg, self.e_th), axis=1)
                )
            )
        else:
            vm = mval
        sc_font = {"weight": "bold", "size": fontsize}
        mpl.rc("font", **sc_font)
        imsize = (int(imwidth), int(imwidth * 1.1))

        plt.figure(figsize=imsize)

        gs = mpgs.GridSpec(11, 10)
        ax1 = plt.subplot(gs[0:5, 0:5])
        ax2 = plt.subplot(gs[0:5, 5:10])
        ax3 = plt.subplot(gs[5:10, 0:5])
        ax4 = plt.subplot(gs[5:10, 5:10])
        ax5 = plt.subplot(gs[10:11, :])

        ax1.imshow(-100 * self.e_xx, vmin=-vm, vmax=vm, cmap="RdBu_r")
        scalebar = mpss.ScaleBar(self.calib, self.calib_units)
        scalebar.location = "lower right"
        scalebar.box_alpha = 1
        scalebar.color = "k"
        ax1.add_artist(scalebar)
        at = mploff.AnchoredText(
            r"$\mathrm{\epsilon_{xx}}$",
            prop=dict(size=fontsize),
            frameon=True,
            loc="upper left",
        )
        at.patch.set_boxstyle("round, pad= 0., rounding_size= 0.2")
        ax1.add_artist(at)
        ax1.axis("off")

        ax2.imshow(-100 * self.e_dg, vmin=-vm, vmax=vm, cmap="RdBu_r")
        scalebar = mpss.ScaleBar(self.calib, self.calib_units)
        scalebar.location = "lower right"
        scalebar.box_alpha = 1
        scalebar.color = "k"
        ax2.add_artist(scalebar)
        at = mploff.AnchoredText(
            r"$\mathrm{\epsilon_{xy}}$",
            prop=dict(size=fontsize),
            frameon=True,
            loc="upper left",
        )
        at.patch.set_boxstyle("round, pad= 0., rounding_size= 0.2")
        ax2.add_artist(at)
        ax2.axis("off")

        ax3.imshow(-100 * self.e_th, vmin=-vm, vmax=vm, cmap="RdBu_r")
        scalebar = mpss.ScaleBar(self.calib, self.calib_units)
        scalebar.location = "lower right"
        scalebar.box_alpha = 1
        scalebar.color = "k"
        ax3.add_artist(scalebar)
        at = mploff.AnchoredText(
            r"$\mathrm{\epsilon_{\theta}}$",
            prop=dict(size=fontsize),
            frameon=True,
            loc="upper left",
        )
        at.patch.set_boxstyle("round, pad= 0., rounding_size= 0.2")
        ax3.add_artist(at)
        ax3.axis("off")

        ax4.imshow(-100 * self.e_yy, vmin=-vm, vmax=vm, cmap="RdBu_r")
        scalebar = mpss.ScaleBar(self.calib, self.calib_units)
        scalebar.location = "lower right"
        scalebar.box_alpha = 1
        scalebar.color = "k"
        ax4.add_artist(scalebar)
        at = mploff.AnchoredText(
            r"$\mathrm{\epsilon_{yy}}$",
            prop=dict(size=fontsize),
            frameon=True,
            loc="upper left",
        )
        at.patch.set_boxstyle("round, pad= 0., rounding_size= 0.2")
        ax4.add_artist(at)
        ax4.axis("off")

        sb = np.zeros((10, 1000), dtype=np.float)
        for ii in range(10):
            sb[ii, :] = np.linspace(-vm, vm, 1000)
        ax5.imshow(sb, cmap="RdBu_r")
        ax5.yaxis.set_visible(False)
        no_labels = 9
        x1 = np.linspace(0, 1000, no_labels)
        ax5.set_xticks(x1)
        ax5.set_xticklabels(np.round(np.linspace(-vm, vm, no_labels), 4))
        for axis in ["top", "bottom", "left", "right"]:
            ax5.spines[axis].set_linewidth(2)
            ax5.spines[axis].set_color("black")
        ax5.xaxis.set_tick_params(width=2, length=6, direction="out", pad=10)
        ax5.set_title("Strain (%)", **sc_font)

        plt.autoscale()
