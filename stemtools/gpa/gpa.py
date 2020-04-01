import numpy as np
from skimage import restoration as skr
from scipy import ndimage as scnd
import matplotlib.pyplot as plt
from ..util import image_utils as iu
import numba

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
    ham = np.sqrt(np.outer(np.hamming(image.shape[0]),np.hamming(image.shape[1])))
    image_ft = np.fft.fftshift(np.fft.fft2(image*ham))
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
    xx,yy = np.meshgrid(np.arange(image.shape[1]),np.arange(image.shape[0]))
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
    plt.annotate(A, (A[0]/image.shape[0], (1 - A[1]/image.shape[1])), textcoords='axes fraction', size=15,color='w')
    plt.annotate(B, (B[0]/image.shape[0], (1 - B[1]/image.shape[1])), textcoords='axes fraction', size=15,color='w')
    plt.annotate(C, (C[0]/image.shape[0], (1 - C[1]/image.shape[1])), textcoords='axes fraction', size=15,color='w')
    plt.annotate(D, (D[0]/image.shape[0], (1 - D[1]/image.shape[1])), textcoords='axes fraction', size=15,color='w')
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
    ry = np.arange(start=-image.shape[0]/2,stop=image.shape[0]/2,step=1)
    rx = np.arange(start=-image.shape[1]/2,stop=image.shape[1]/2,step=1)
    rx,ry = np.meshgrid(rx,ry)
    gr = 2*np.pi*((rx*g_vector[1]) + (ry*g_vector[0]))
    return gr

def phase_matrix(g_vector,image,gauss_blur=True):
    ham = np.sqrt(np.outer(np.hamming(image.shape[0]),np.hamming(image.shape[1])))
    circ_pos = G_to_circ(g_vector,image)
    circ_rad = np.amin(0.01*np.asarray(image.shape))
    circ_mask = (iu.make_circle(image.shape,circ_pos[0],circ_pos[1],circ_rad)).astype(bool)
    yy,xx = np.mgrid[0:image.shape[0],0:image.shape[1]]
    sigma2 = np.sum((g_vector*0.5*np.asarray(image.shape))**2)
    zz = (((yy[circ_mask] - circ_pos[1])**2) + ((xx[circ_mask] - circ_pos[0])**2))/sigma2
    mask = np.exp((-0.5)*zz)
    four_mask = np.zeros_like(image,dtype=np.float) 
    four_mask[circ_mask] = mask
    if gauss_blur:
        G_matrix = np.angle(np.fft.ifft2(four_mask*np.fft.fftshift(np.fft.fft2(image*ham))))
    else:
        G_matrix = np.angle(np.fft.ifft2(circ_mask*np.fft.fftshift(np.fft.fft2(image*ham))))
    return G_matrix

def refined_phase(old_p,old_g,ref_matrix,image,iter_count=10,gauss_blur=True):
    ry = np.arange(start=-image.shape[0]/2,stop=image.shape[0]/2,step=1)
    rx = np.arange(start=-image.shape[1]/2,stop=image.shape[1]/2,step=1)
    rx,ry = np.meshgrid(rx,ry)
    new_g = old_g
    new_p = old_p
    for _ in range(int(iter_count)):
        G_x,G_y = phase_diff(new_p)
        G_nabla = G_x + G_y
        g_r = G_nabla/(2*np.pi)
        del_g = np.asarray((np.median(g_r[ref_matrix]/ry[ref_matrix]),np.median(g_r[ref_matrix]/rx[ref_matrix])))
        new_g = new_g + del_g
        new_p = phase_matrix(new_g,image,gauss_blur)
    return new_g,new_p

def get_a_matrix(g_vector_1,
                 g_vector_2):
    g_matrix = np.zeros((2,2),dtype=np.float64)
    g_matrix[0,:] = np.flip(np.asarray(g_vector_1))
    g_matrix[1,:] = np.flip(np.asarray(g_vector_2))
    a_matrix = np.linalg.inv(np.transpose(g_matrix))
    return a_matrix

def get_u_matrices(P1,P2,a_matrix):
    P1 = skr.unwrap_phase(P1)
    P2 = skr.unwrap_phase(P2)
    rolled_p = np.asarray((np.reshape(P1,-1),np.reshape(P2,-1)))
    u_matrix = np.matmul(a_matrix,rolled_p)
    u_x = np.reshape(u_matrix[0,:],P1.shape)
    u_y = np.reshape(u_matrix[1,:],P2.shape)
    return u_x,u_y

@numba.jit(cache=True,parallel=True)
def gen_strain(P1,P2,a_matrix):
    P1_x, P1_y = phase_diff(P1)
    P2_x, P2_y = phase_diff(P2)
    yy, xx = np.mgrid[0:P1.shape[0],0:P1.shape[1]]
    yy = np.ravel(yy)
    xx = np.ravel(xx)
    P_mat = np.zeros((2,2),dtype=np.float)
    e_xx = np.zeros_like(P1)
    e_xy = np.zeros_like(P1)
    e_yx = np.zeros_like(P1)
    e_yy = np.zeros_like(P1)
    for ii in numba.prange(len(yy)):
        ypos = yy[ii]
        xpos = xx[ii]
        P_mat[0,0] = P1_x[ypos,xpos]
        P_mat[0,1] = P1_y[ypos,xpos]
        P_mat[1,0] = P2_x[ypos,xpos]
        P_mat[1,1] = P2_y[ypos,xpos]
        e_mat = ((1)/(2*np.pi))*np.matmul(a_matrix,P_mat)
        e_xx[ypos,xpos] = e_mat[0,0]
        e_xy[ypos,xpos] = e_mat[0,1]
        e_yx[ypos,xpos] = e_mat[1,0]
        e_yy[ypos,xpos] = e_mat[1,1]
    e_th = 0.5*(e_xy - e_yx)
    e_dg = 0.5*(e_xy + e_yx)
    return e_xx,e_yy,e_th,e_dg

def get_strain_fromU(U_x,U_y):
    e_xx,e_xy = phase_diff(U_x)
    e_yx,e_yy = phase_diff(U_y)
    e_theta = 0.5*(e_xy - e_yx)
    e_diag = 0.5*(e_xy + e_yx)
    return e_xx,e_yy,e_theta,e_diag