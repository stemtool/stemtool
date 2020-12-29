import numpy as np
import numba
import warnings
import scipy.signal as scisig
import scipy.ndimage as scnd
import stemtool as st
import matplotlib.pyplot as plt
import math


def sobel(im, order=3):
    """
    Sobel Filter an Input Image

    Parameters
    ----------
    im:    ndarray
           the original input image to be filtered
    order: int
           3 is the default but if 5 is specified
           then a 5x5 Sobel filter is run

    Returns
    -------
    mag: ndarray
         Sobel Filter Magnitude
    and: ndarray
         Sobel Filter Angle

    Notes
    -----
    We define the two differentiation matrices - g_x and g_y
    and then move along our dataset - to perform the matrix
    operations on 5x5 or 3x3 sections of the input image. The
    magnitude of the Sobel filtered image is the absolute
    of the multiplied matrices - squared and summed and square
    rooted.

    References
    ----------
    .. [3] Sobel, I. and Feldman, G., 1968. A 3x3 isotropic gradient
           operator for image processing. a talk at the Stanford
           Artificial Project in, pp.271-272.

    """
    im = im.astype(np.float64)
    if order == 3:
        k_x = np.asarray(((-1, 0, 1), (-2, 0, 2), (-1, 0, 1)), dtype=np.float64)
        k_y = np.asarray(((-1, -2, -1), (0, 0, 0), (1, 2, 1)), dtype=np.float64)
    else:
        k_x = np.asarray(
            (
                (-1, -2, 0, 2, 1),
                (-4, -8, 0, 8, 4),
                (-6, -12, 0, 12, 6),
                (-4, -8, 0, 8, 4),
                (-1, -2, 0, 2, 1),
            ),
            dtype=np.float64,
        )
        k_y = np.asarray(
            (
                (1, 4, 6, 4, 1),
                (2, 8, 12, 8, 2),
                (0, 0, 0, 0, 0),
                (-2, -8, -12, -8, -2),
                (-1, -4, -6, -4, -1),
            ),
            dtype=np.float64,
        )
    g_x = scisig.convolve2d(im, k_x, mode="same", boundary="symm", fillvalue=0)
    g_y = scisig.convolve2d(im, k_y, mode="same", boundary="symm", fillvalue=0)
    mag = ((g_x ** 2) + (g_y ** 2)) ** 0.5
    ang = np.arctan2(g_y, g_x)
    return mag, ang


def circle_fit(edge_image):
    """
    Fit circle to data points algebraically

    Parameters
    ----------
    edge_image: boolean array
                Boolean data where the edge is 1

    Returns
    -------
    x_center:      float
                   X pixel of circle center
    y_center:      float
                   Y pixel of circle center
    radius_circle: float
                   Radius of the circle

    Notes
    -----
    We calculate the mean of all the points
    as the initial estimate of the circle center
    and then solve a set of linear equations
    to calculate radius and the disk center.

    See Also
    --------
    util.Canny
    """
    size_image = np.asarray(np.shape(edge_image)).astype(int)
    yV, xV = np.mgrid[0 : size_image[0], 0 : size_image[1]]
    xValues = np.asarray(xV[edge_image], dtype=np.float64)
    yValues = np.asarray(yV[edge_image], dtype=np.float64)

    xCentroid = np.mean(xValues)
    yCentroid = np.mean(yValues)

    uValues = xValues - xCentroid
    vValues = yValues - yCentroid

    Suv = np.sum(uValues * vValues)
    Suu = np.sum(uValues ** 2)
    Svv = np.sum(vValues ** 2)
    Suuv = np.sum((uValues ** 2) * vValues)
    Suvv = np.sum(uValues * (vValues ** 2))
    Suuu = np.sum(uValues ** 3)
    Svvv = np.sum(vValues ** 3)

    A = np.array([[Suu, Suv], [Suv, Svv]])
    B = np.array([(Suuu + Suvv), (Svvv + Suuv)]) / 2
    uc, vc = np.linalg.solve(A, B)

    xCenter = xCentroid + uc
    yCenter = yCentroid + vc

    Rvalues = np.sqrt(((xValues - xCenter) ** 2) + ((yValues - yCenter) ** 2))
    radius = np.mean(Rvalues)

    return xCenter, yCenter, radius


@numba.jit(parallel=True, cache=True)
def numba_thinner(pos, edge, mag, ang):
    """
    Numba JIT Function for thinning Sobel
    Filtered Edges

    Parameters
    ----------
    pos:  ndarray
          List of pixel positions
    edge: ndarray
          Thinned Edge
    mag:  ndarray
          Magnitude after Sobel Filtering
    ang:  ndarray
          ANgle of Sobel Filter, in degrees

    Notes
    -----
    We use the direction of the Sobel gradient to
    determine whether a pixel belongs to the edge
    or not. If the gradient is perpendicular to the
    pixels then it is an edge pixel or else it is not.
    """
    for pp in numba.prange(len(pos)):
        ii = pos[pp, 0]
        jj = pos[pp, 1]
        q = 1
        r = 1
        if (0 <= ang[ii, jj] < 22.5) or (157.5 <= ang[ii, jj] <= 180):
            q = mag[ii, jj + 1]
            r = mag[ii, jj - 1]
        elif 22.5 <= ang[ii, jj] < 67.5:
            q = mag[ii + 1, jj - 1]
            r = mag[ii - 1, jj + 1]
        elif 67.5 <= ang[ii, jj] < 112.5:
            q = mag[ii + 1, jj]
            r = mag[ii - 1, jj]
        elif 112.5 <= ang[ii, jj] < 157.5:
            q = mag[ii - 1, jj - 1]
            r = mag[ii + 1, jj + 1]

        if (mag[ii, jj] >= q) and (mag[ii, jj] >= r):
            edge[ii, jj] = mag[ii, jj]
        else:
            edge[ii, jj] = 0


@numba.jit(parallel=True, cache=True)
def numba_joiner(pos, edge, upper, lower):
    """
    Numba JIT function for joining of Edges

    Parameters
    ----------
    pos:   ndarray
           List of pixel positions
    edge:  ndarray
           Thinned Edge, that will be joined
    upper: float
           Upper threshold
    lower: float
           Lower threshold

    Notes
    -----
    The input image now consists of strong and weak
    thresholds. The final step of the algorithm is to
    look at some of the measured edges and join/link them.
    The idea is that if a edge pixel is classified as a
    weak edge but one of its neighbors is a strong edge then
    it is a strong edge pixel.
    """
    for pp in numba.prange(len(pos)):
        ii = pos[pp, 0]
        jj = pos[pp, 1]
        if edge[ii, jj] == lower:
            if (
                (edge[ii - 1, jj - 1] == upper)  # top left
                or (edge[ii - 1, jj] == upper)  # top
                or (edge[ii - 1, jj + 1] == upper)  # top right
                or (edge[ii, jj - 1] == upper)  # left
                or (edge[ii, jj + 1] == upper)  # right
                or (edge[ii + 1, jj - 1] == upper)  # bottom left
                or (edge[ii + 1, jj] == upper)  # bottom
                or (edge[ii + 1, jj + 1] == upper)
            ):  # bottom right
                edge[ii, jj] = upper
            else:
                edge[ii, jj] = 0


class Canny(object):
    """
    Canny Edge Detection

    Parameters
    ----------
    image:         ndarray
                   Image on which Canny edge detection is to be performed
    lowThreshold:  float
                   Lower threshold value, could be user defined or auto
                   generated from Otsu thresholding
    highThreshold: float
                   Upper threshold value, could be user defined or auto
                   generated from Otsu thresholding

    Notes
    -----
    The input image is first blurred with a [5 x 5] Gaussian kernel to
    prevent pixel jumps being incorrectly registered as edges. Then a Sobel
    filter is run on the image. The edges from the Sobel filter are
    subsequently thinned by determining whether the gradient of the Sobel
    filter is perpendicular to the edge. Subsequently the edges are sorted
    into two bins - weak edges and strong edges. The weak edges are
    reclassified as strong edges if they are connected to strong edges.

    Examples
    --------
    Run as:

    >>> im_canny = Canny(image,0.2,0.7)

    Where 0.2 and 0.7 are the parameters obtained through Otsu thresholding image.
    After setting the class, then generate the edges first as:

    >>> im_canny.edge_thinning()

    This Sobel filters the image, and generates the image edge. The magnitude of
    the Sobel filter can be accessed as `im_canny.sobel_mag`, while the Sobel angle
    in degrees is stored as `im_canny.sobel_ang`. The generated edge is stored as
    `im_canny.thin_edge`. This is the thresholded edge, however this is broken and
    discontinuous and non-uniform in value. To fix that run:

    >>> im_canny.edge_thresholding

    After that, generate the final canny edge as:

    >>> canny_image = im_canny.edge_joining()

    References
    ----------
    .. [4] John Canny, "A computational approach to edge detection."
       Readings in computer vision. Morgan Kaufmann, 1987. 184-203
    """

    def __init__(self, image, lowThreshold, highThreshold, plot_steps=True, blurVal=5):
        self.image = image
        self.im_size = (np.asarray(image.shape)).astype(int)
        self.thresh_lower = lowThreshold
        self.thresh_upper = highThreshold
        self.imblur = st.util.image_normalizer(scnd.gaussian_filter(image, blurVal))
        self.sobel_mag = np.empty_like(image)
        self.sobel_ang = np.empty_like(image)
        self.thin_edge = np.empty_like(image)
        self.residual = np.empty_like(image)
        self.cannyEdge = np.empty_like(image)
        self.edge_check = False
        self.thresh_check = False
        self.plot_steps = plot_steps

    def edge_thinning(self):
        """
        Thinning Sobel Filtered Edges

        Notes
        -----
        We use the direction of the Sobel gradient to
        determine whether a pixel belongs to the edge
        or not. If the gradient is perpendicular to the
        pixels then it is an edge pixel or else it is not.

        See Also
        --------
        numba_thinner
        """
        self.sobel_mag, self.sobel_ang = st.util.sobel(self.imblur, order=5)
        self.sobel_ang = self.sobel_ang * (180 / np.pi)
        self.sobel_ang[self.sobel_ang < 0] += 180
        yRange, xRange = np.mgrid[1 : self.im_size[0] - 1, 1 : self.im_size[1] - 1]
        thin_pos = np.asarray((np.ravel(yRange), np.ravel(xRange))).transpose()
        thin_edge = np.copy(self.thin_edge)

        # initialize JIT
        numba_thinner(
            thin_pos[0 : int(len(thin_pos) / 10), :],
            thin_edge,
            self.sobel_mag,
            self.sobel_ang,
        )

        # now run on full dataset
        numba_thinner(thin_pos, self.thin_edge, self.sobel_mag, self.sobel_ang)
        if self.plot_steps:
            plt.figure(figsize=(20, 10))
            plt.subplot(1, 2, 1)
            plt.imshow(self.sobel_mag)
            plt.axis("off")
            plt.subplot(1, 2, 2)
            plt.imshow(self.thin_edge)
            plt.axis("off")
        self.edge_check = True

    def edge_thresholding(self):
        """
        Thresholding of Edges

        Notes
        -----
        Sort edges measured as either strong edges or weak edges,
        depending on the intensity of the edge. The thresholds are
        user defined parameters and while they can be played around
        with it is recommended to use a thresholding algorithm like
        Otsu thresholding to robustly determine edges.
        """
        if not self.edge_check:
            raise RuntimeError("Please thin the edges first by calling edge_thinner()")
        highT = np.amax(self.thin_edge) * self.thresh_upper
        lowT = highT * self.thresh_lower
        self.residual[np.where(self.thin_edge > highT)] = highT
        self.residual[
            np.where((self.thin_edge <= highT) & (self.thin_edge > lowT))
        ] = lowT
        self.residual[self.residual > self.thresh_upper] = self.thresh_upper
        if self.plot_steps:
            plt.figure(figsize=(10, 10))
            plt.imshow(self.residual)
            plt.axis("off")
        self.thresh_check = True

    def edge_joining(self):
        """
        Joining of Edges

        Returns
        -------
        cannyEdge: boolean array
                   Boolean matrix with the same size as the original
                   image, where the edges are 1, while the rest is 0

        Notes
        -----
        The input image now consists of strong and weak
        thresholds. The final step of the algorithm is to
        look at some of the measured edges and join/link them.
        The idea is that if a edge pixel is classified as a
        weak edge but one of its neighbors is a strong edge then
        it is a strong edge pixel.

        See Also
        --------
        numba_joiner
        """
        if not self.thresh_check:
            raise RuntimeError(
                "Please threshold the edges first by calling canny_threshold()"
            )
        yRange, xRange = np.mgrid[1 : self.im_size[0] - 1, 1 : self.im_size[1] - 1]
        edge_pos = np.asarray((np.ravel(yRange), np.ravel(xRange))).transpose()
        cannyEdge = np.copy(self.residual)
        self.cannyEdge = np.copy(self.residual)

        # initialize JIT
        numba_joiner(
            edge_pos[0 : int(len(edge_pos) / 10), :],
            cannyEdge,
            self.thresh_upper,
            self.thresh_lower,
        )

        # now run on full dataset, twice
        numba_joiner(edge_pos, self.cannyEdge, self.thresh_upper, self.thresh_lower)
        numba_joiner(edge_pos, self.cannyEdge, self.thresh_upper, self.thresh_lower)
        if self.plot_steps:
            plt.figure(figsize=(10, 10))
            plt.imshow(self.cannyEdge)
            plt.axis("off")
        self.cannyEdge = (self.cannyEdge / np.amax(self.cannyEdge)).astype(bool)
        return self.cannyEdge
