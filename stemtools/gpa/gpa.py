import numpy as np
from skimage import restoration as skr
from scipy import ndimage as scnd
import matplotlib.pyplot as plt
from ..util import image_utils as iu

def find_diffraction_spots(image,
                           circ_0,
                           circ_1,
                           circ_2):
    """
    Locate the diffraction spots visually.
    
    Parameters
    ----------
    image:  ndarray
            Original image
    circ_0: ndarray
            Position of the central beam in
            the Fourier pattern
    circ_1: ndarray
            Position of the first beam in
            the Fourier pattern
    circ_2: ndarray
            Position of the second beam in
            the Fourier pattern
    
    Notes
    -----
    Put circles in red(central), y(blue) and x(green) 
    on the diffraction pattern to approximately know
    the positions.
    
    Returns
    -------
    Circle Positions
    
    :Authors:
    Debangshu Mukherjee <mukherjeed@ornl.gov>
    """
    image_ft = np.fft.fftshift(np.fft.fft2(image))
    log_abs_ft = scnd.filters.gaussian_filter(np.log10(np.abs(image_ft)),3)
    f, ax = plt.subplots(figsize=(20, 20))
    circ_0_im = plt.Circle(circ_0, 15,color="red",alpha=0.33)
    circ_1_im = plt.Circle(circ_1, 15,color="blue",alpha=0.33)
    circ_2_im = plt.Circle(circ_2, 15,color="green",alpha=0.33)
    ax.imshow(log_abs_ft,cmap='gray')
    ax.add_artist(circ_0_im)
    ax.add_artist(circ_1_im)
    ax.add_artist(circ_2_im)
    plt.show()
    return (circ_0,circ_1,circ_2)

def define_reference(image,
                     A,B,C,D):
    """
    Locate the reference image.
    
    Parameters
    ----------
    image:  ndarray
            Original image
    A:      Top left position of reference region in (x,y)
    B:      Top right position of reference region in (x,y)
    C:      Bottom right position of reference region in (x,y)
    D:      Bottom left position of reference region in (x,y)
    
    Returns
    -------
    ref_reg: ndarray
             Boolean indices marking the reference region
    
    Notes
    -----
    Locate visually the unstrained reference region.
    
    :Authors:
    Debangshu Mukherjee <mukherjeed@ornl.gov>
    """
    xx,yy = np.meshgrid(np.arange(imageDC.shape[1]),np.arange(imageDC.shape[0]))
    m_AB = (A[1] - B[1])/(A[0] - B[0])
    c_AB = A[1] - (m_AB*A[0])
    m_BC = (B[1] - C[1])/(B[0] - C[0])
    c_BC = B[1] - (m_BC*B[0])
    m_CD = (C[1] - D[1])/(C[0] - D[0])
    c_CD = C[1] - (m_CD*C[0])
    m_DA = (D[1] - A[1])/(D[0] - A[0])
    c_DA = D[1] - (m_DA*D[0])
    ref_reg = np.logical_and(np.logical_and((yy > (m_AB*xx) + c_AB),((yy - c_BC)/m_BC > xx)),
                             np.logical_and((yy < (m_CD*xx) + c_CD),((yy - c_DA)/m_DA < xx)))
    plt.figure(figsize=(15,15))
    plt.imshow(iu.image_normalizer(image)+0.33*ref_reg)
    plt.annotate(A, (A[0]/image.shape[0], (1 - A[1]/imageDC.shape[1])), textcoords='axes fraction', size=15,color='w')
    plt.annotate(B, (B[0]/image.shape[0], (1 - B[1]/imageDC.shape[1])), textcoords='axes fraction', size=15,color='w')
    plt.annotate(C, (C[0]/image.shape[0], (1 - C[1]/imageDC.shape[1])), textcoords='axes fraction', size=15,color='w')
    plt.annotate(D, (D[0]/image.shape[0], (1 - D[1]/imageDC.shape[1])), textcoords='axes fraction', size=15,color='w')
    plt.scatter(A[0],A[1])
    plt.scatter(B[0],B[1])
    plt.scatter(C[0],C[1])
    plt.scatter(D[0],D[1])
    plt.axis('off')
    return ref_reg

def phase_diff(angle_image):
    """
    Locate the diffraction spots visually.
    
    Parameters
    ----------
    angle_image:  ndarray
                  Wrapped phase image 
    
    Returns
    -------
    diff_x: ndarray
            X difference of the phase image
    diff_y: ndarray
            Y difference of the phase image
    
    Notes
    -----
    The basic idea of this is that we differentiate the 
    complex exponential of the phase image, and then obtain the 
    differentiation result by multiplying the differential with 
    the conjugate of the complex phase image.
    
    Reference
    ---------
    Hÿtch, M. J., E. Snoeck, and R. Kilaas. "Quantitative measurement 
    of displacement and strain fields from HREM micrographs." 
    Ultramicroscopy 74.3 (1998): 131-146.
    
    :Authors:
    Debangshu Mukherjee <mukherjeed@ornl.gov>
    """
    imaginary_image = np.exp(1j * angle_image)
    diff_imaginary_x = np.zeros(imaginary_image.shape,dtype='complex_')
    diff_imaginary_x[:,0:-1] = np.diff(imaginary_image,axis=1)
    diff_imaginary_y = np.zeros(imaginary_image.shape,dtype='complex_')
    diff_imaginary_y[0:-1,:] = np.diff(imaginary_image,axis=0)
    conjugate_imaginary = np.conj(imaginary_image)
    diff_complex_x = np.multiply(conjugate_imaginary,diff_imaginary_x)
    diff_complex_y = np.multiply(conjugate_imaginary,diff_imaginary_y)
    diff_x = np.imag(diff_complex_x)
    diff_y = np.imag(diff_complex_y)
    return diff_x,diff_y

def phase_subtract(matrix_1,
                   matrix_2):
    return np.angle(np.exp(1j*(matrix_1-matrix_2)))

def circ_to_G(circ_pos,image):
    g_vec = np.zeros(2)
    g_vec[0] = ((circ_pos[1] - (0.5*image.shape[0]))/image.shape[0])
    g_vec[1] = ((circ_pos[0] - (0.5*image.shape[1]))/image.shape[1])
    return g_vec

def G_to_circ(g_vec,image):
    circ_pos = np.zeros(2)
    circ_pos[1] = (g_vec[0]*image.shape[0]) + (0.5*image.shape[0])
    circ_pos[0] = (g_vec[1]*image.shape[1]) + (0.5*image.shape[1])
    return circ_pos

def g_matrix(g_vector,image):
    ry = (np.arange((-image.shape[0]/2),((image.shape[0]/2)),1))
    rx = (np.arange((-image.shape[1]/2),((image.shape[1]/2)),1))
    Rx, Ry = np.meshgrid(rx, ry)
    gr = 2*np.pi*((Rx*g_vector[1]) + (Ry*g_vector[0]))
    return gr

def p_matrix(g_vector,image):
    circ_pos = G_to_circ(g_vector,image)
    circ_rad = np.amin(0.01*np.asarray(image.shape))
    circ_mask = iu.make_circle(image.shape,circ_pos[0],circ_pos[1],circ_rad)
    G_matrix = np.angle(np.fft.ifft2(circ_mask*np.fft.fftshift(np.fft.fft2(image))))
    G_matrix = skr.unwrap_phase(G_matrix)
    P_matrix = G_matrix - g_matrix(g_vector,image)
    return P_matrix

def g_diff(P_matrix,ref_matrix):
    pdiff_x,pdiff_y = phase_diff(P_matrix)
    gdiff_y = (np.mean(pdiff_y[ref_matrix]))/(2*np.pi)
    gdiff_x = (np.mean(pdiff_x[ref_matrix]))/(2*np.pi)
    return (gdiff_y,gdiff_x)

def refined_P(P_matrix,old_g,ref_matrix,image,iter_count=10):
    new_g = old_g
    new_p = P_matrix
    for _ in range(iter_count): 
        g_delta = g_diff(new_p,ref_matrix)
        new_g = new_g - g_delta
        new_p = p_matrix(new_g,image)
    return new_g,new_p

def get_a_matrix(g_vector_1,
                 g_vector_2):
    g_matrix = np.zeros((2,2),dtype=np.float64)
    g_matrix[0,:] = np.flip(np.asarray(g_vector_1))
    g_matrix[1,:] = np.flip(np.asarray(g_vector_2))
    a_matrix = np.linalg.inv(np.transpose(g_matrix))
    return a_matrix

def get_u_matrices(P1,P2,a_matrix):
    rolled_p = np.asarray((np.reshape(P1,-1),np.reshape(P2,-1)))
    u_matrix = np.matmul(a_matrix,rolled_p)
    u_x = np.reshape(u_matrix[0,:],P1.shape)
    u_y = np.reshape(u_matrix[1,:],P2.shape)
    return u_x,u_y

def get_strain(U_x,U_y):
    e_xx,e_xy = phase_diff(U_x)
    e_yx,e_yy = phase_diff(U_y)
    e_theta = 0.5*(e_xy - e_yx)
    e_diag = 0.5*(e_xy + e_yx)
    return e_xx,e_yy,e_theta,e_diag