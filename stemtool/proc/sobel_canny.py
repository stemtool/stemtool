import numpy as np
import numba
import warnings
from scipy import signal as scisig
from scipy import ndimage as scnd
import stemtool as st
import math

def sobel(im,
          order=3):
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
    Sobel, I. and Feldman, G., 1968. A 3x3 isotropic gradient 
    operator for image processing. a talk at the Stanford 
    Artificial Project in, pp.271-272.
                 
    :Authors:
    Debangshu Mukherjee <mukherjeed@ornl.gov>
    
    """
    im = im.astype(np.float64)
    if (order==3):
        k_x = np.asarray(((-1,0,1),
                          (-2,0,2),
                          (-1,0,1)),dtype=np.float64)
        k_y = np.asarray(((-1,-2,-1),
                          (0,0,0),
                          (1,2,1)),dtype=np.float64)
    else:
        k_x = np.asarray(((-1, -2, 0, 2, 1), 
                          (-4, -8, 0, 8, 4), 
                          (-6, -12, 0, 12, 6),
                          (-4, -8, 0, 8, 4),
                          (-1, -2, 0, 2, 1)), dtype = np.float64)
        k_y = np.asarray(((1, 4, 6, 4, 1), 
                          (2, 8, 12, 8, 2),
                          (0, 0, 0, 0, 0), 
                          (-2, -8, -12, -8, -2),
                          (-1, -4, -6, -4, -1)), dtype = np.float64)
    g_x = scisig.convolve2d(im, k_x, mode='same', boundary = 'symm', fillvalue=0)
    g_y = scisig.convolve2d(im, k_y, mode='same', boundary = 'symm', fillvalue=0)
    mag = ((g_x**2) + (g_y**2))**0.5
    ang = np.arctan2(g_y,g_x)
    return mag, ang

@numba.jit
def edge_thinner(sobel_mag,
                 sobel_angle):
    """
    Thinning Sobel Filtered Edges
    
    Parameters
    ----------
    sobel_mag: ndarray
               Sobel Filtered Magnitude
    sobel_angle: ndarray
                 Sobel Filtered Angle
                     
    Returns
    -------
    thinned_edge: ndarray
                  Thinned Image
    
    Notes
    -----
    We use the direction of the Sobel gradient to 
    determine whether a pixel belongs to the edge 
    or not. If the gradient is perpendicular to the 
    pixels then it is an edge pixel or else it is not.
    
    :Authors:
    Debangshu Mukherjee <mukherjeed@ornl.gov>
    """
    warnings.filterwarnings('ignore')
    thinned_edge = np.zeros(sobel_mag.shape)
    sobel_degree = sobel_angle*(180/np.pi)
    sobel_degree[sobel_degree < 0] += 180
    matrix_size = (np.asarray(sobel_mag.shape)).astype(int)
    for ii in range(1,matrix_size[0]-1):
        for jj in range(1,matrix_size[1]-1):
            q = 1
            r = 1
            if (0 <= sobel_degree[ii,jj] < 22.5) or (157.5 <= sobel_degree[ii,jj] <= 180):
                q = sobel_mag[ii, jj+1]
                r = sobel_mag[ii, jj-1]
            elif (22.5 <= sobel_degree[ii,jj] < 67.5):
                q = sobel_mag[ii+1, jj-1]
                r = sobel_mag[ii-1, jj+1]
            elif (67.5 <= sobel_degree[ii,jj] < 112.5):
                q = sobel_mag[ii+1, jj]
                r = sobel_mag[ii-1, jj]
            elif (112.5 <= sobel_degree[ii,jj] < 157.5):
                q = sobel_mag[ii-1, jj-1]
                r = sobel_mag[ii+1, jj+1]

            if (sobel_mag[ii,jj] >= q) and (sobel_mag[ii,jj] >= r):
                thinned_edge[ii,jj] = sobel_mag[ii,jj]
            else:
                thinned_edge[ii,jj] = 0
    return thinned_edge

@numba.jit
def canny_threshold(thinned_edge,
                    lowThreshold,
                    highThreshold):
    """
    Thresholding of Edges
    
    Parameters
    ----------
    thinned_edge: ndarray
                  Thinned Image image obtained by running edge_thinner on a 
                  Sobel Filtered Image
    lowThreshold: float
                  Lower threshold value, could be user defined or auto
                  generated from Otsu thresholding
    highThreshold: float
                   Upper threshold value, could be user defined or auto
                   generated from Otsu thresholding
                     
    Returns
    -------
    residual: ndarray
              Thresholded Image
    
    Notes
    -----
    Sort edges measured as either strong edges or weak edges,
    depending on the intensity of the edge. The thresholds are 
    user defined parameters and while they can be played around 
    with it is recommended to use a thresholding algorithm like 
    Otsu thresholding to robustly determine edges.
    
    See Also
    --------
    edge_thinner: Generates thinned edges for thresholding
    
    :Authors:
    Debangshu Mukherjee <mukherjeed@ornl.gov>
    """
    warnings.filterwarnings('ignore')
    highT = np.amax(thinned_edge) * highThreshold
    lowT = highT * lowThreshold
    residual = np.zeros(thinned_edge.shape)
    residual[np.where(thinned_edge > highT)] = highT
    residual[np.where((thinned_edge <= highT) & (thinned_edge > lowT))] = lowT
    return residual

@numba.jit
def edge_joining(thresholded,
                 lowThreshold,
                 highThreshold):
    """
    Joining of Edges
    
    Parameters
    ----------
    thresholded: ndarray
                 Thresholded image where the edge values are sorted,
                 obtained after running canny_threshold
    lowThreshold: float
                  Lower threshold value, could be user defined or auto
                  generated from Otsu thresholding
    highThreshold: float
                   Upper threshold value, could be user defined or auto
                   generated from Otsu thresholding
                     
    Returns
    -------
    joined_edge: ndarray
                 Image where the edges with strong thresholds are now 
                 joined together
    
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
    canny_threshold: Returns image sorted by strong and weak thresholds
    
    :Authors:
    Debangshu Mukherjee <mukherjeed@ornl.gov>
    """
    warnings.filterwarnings('ignore')
    matrix_size = (np.asarray(thresholded.shape)).astype(int)
    thresholded[thresholded > highThreshold] = highThreshold
    for ii in range(1, matrix_size[0]-1):
        for jj in range(1, matrix_size[1]-1):
            if (thresholded[ii,jj] == lowThreshold):
                if ((thresholded[ii-1, jj-1] == highThreshold) #top left
                    or (thresholded[ii-1, jj] == highThreshold) #top
                    or (thresholded[ii-1, jj+1] == highThreshold) #top right
                    or (thresholded[ii, jj-1] == highThreshold) #left
                    or (thresholded[ii, jj+1] == highThreshold) #right
                    or (thresholded[ii+1, jj-1] == highThreshold) #bottom left
                    or (thresholded[ii+1, jj] == highThreshold) #bottom
                    or (thresholded[ii+1, jj+1] == highThreshold)): #bottom right
                    thresholded[ii, jj] = highThreshold
                else:
                    thresholded[ii, jj] = 0
    joined_edge = thresholded /  np.amax(thresholded)
    return joined_edge

@numba.jit
def canny_edge(input_image,
               lowThreshold,
               highThreshold):
    """
    Canny Edge Detection
    
    Parameters
    ----------
    thresholded: ndarray
                 Image on which Canny edge detection is to be performed
    lowThreshold: float
                  Lower threshold value, could be user defined or auto
                  generated from Otsu thresholding
    highThreshold: float
                   Upper threshold value, could be user defined or auto
                   generated from Otsu thresholding
                     
    Returns
    -------
    joined_bool: boolean array
                 Boolean matrix with the same size as the original
                 image, where the edges are 1, while the rest is 0
     
    Notes
    -----
    The input image is first blurred with a [5 x 5]
    Gaussian kernel to prevent pixel jumps being 
    incorrectly registered as edges. Then a Sobel 
    filter is run on the image. The edges from the 
    Sobel filter are subsequently thinned by determining 
    whether the gradient of the Sobel filter is perpendicular
    to the edge. Subsequently the edges are sorted into two 
    bins - weak edges and strong edges. The weak edges are 
    reclassified as strong edges if they are connected to 
    strong edges.
    
    References
    ----------
    John Canny, "A computational approach to edge detection." 
    Readings in computer vision. Morgan Kaufmann, 1987. 184-203
    
    See Also
    --------
    sobel_filter
    edge_thinner
    canny_threshold
    edge_joining
    
    :Authors:
    Debangshu Mukherjee <mukherjeed@ornl.gov>
    """
    blurred_input = scnd.gaussian_filter(input_image,5)
    normalized_blurred = st.util.image_normalizer(blurred_input)
    sobel_mag, sobel_angle = sobel_filter(normalized_blurred)
    thinned_edge = edge_thinner(sobel_mag,sobel_angle)
    thresholded_edge = canny_threshold(thinned_edge, lowThreshold, highThreshold)
    joined_edge = edge_joining(thresholded_edge, lowThreshold, highThreshold)
    joined_bool = joined_edge.astype(bool)
    return joined_bool

@numba.jit
def circle_fit(edge_image):
    """
    Fit circle to data points algebraically
    
    Parameters
    ----------
    edge_image: boolean array
                Boolean data where the edge is 1
    
    Returns
    -------
    x_center: float
              X pixel of circle center
    y_center: float
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
    canny_edge
    
    :Authors:
    Debangshu Mukherjee <mukherjeed@ornl.gov>
    """
    warnings.filterwarnings('ignore')
    size_image = np.asarray(np.shape(edge_image)).astype(int)
    yV, xV = np.mgrid[0:size_image[0], 0:size_image[1]]
    xValues = np.asarray(xV[edge_image],dtype=np.float64)
    yValues = np.asarray(yV[edge_image],dtype=np.float64)
    
    #coordinates of the barycenter
    xCentroid = np.mean(xValues)
    yCentroid = np.mean(yValues)

    # calculation of the reduced coordinates
    uValues = xValues - xCentroid
    vValues = yValues - yCentroid

    # linear system defining the center in reduced coordinates (uc, vc):
    #    Suu * uc +  Suv * vc = (Suuu + Suvv)/2
    #    Suv * uc +  Svv * vc = (Suuv + Svvv)/2

    Suv  = np.sum(uValues * vValues)
    Suu  = np.sum(uValues ** 2)
    Svv  = np.sum(vValues ** 2)
    Suuv = np.sum((uValues ** 2) * vValues)
    Suvv = np.sum(uValues * (vValues ** 2))
    Suuu = np.sum(uValues ** 3)
    Svvv = np.sum(vValues ** 3)

    # Solving the linear system
    A = np.array([ [ Suu, Suv ], [Suv, Svv]])
    B = np.array([ Suuu + Suvv, Svvv + Suuv ])/2.0
    uc, vc = np.linalg.solve(A, B)

    xCenter = xCentroid + uc
    yCenter = yCentroid + vc

    # Calculation of all distances from the center (xc_1, yc_1)
    Rvalues     = np.sqrt(((xValues-xCenter) ** 2) + ((yValues-yCenter) ** 2))
    radius      = np.mean(Rvalues)
    
    return xCenter, yCenter, radius