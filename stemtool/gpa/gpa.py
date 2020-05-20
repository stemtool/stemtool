import numpy as np
from skimage import restoration as skr
from scipy import ndimage as scnd
import matplotlib.pyplot as plt
import stemtool as st
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
    [1]_, Hÿtch, M. J., E. Snoeck, and R. Kilaas. "Quantitative measurement 
          of displacement and strain fields from HREM micrographs." 
          Ultramicroscopy 74.3 (1998): 131-146.
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
    return np.angle(np.exp(1j*(matrix_1-matrix_2)))

def circ_to_G(circ_pos,
              image):
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
           Shape is (2,1) which is the 
           corresponding g-vector in inverse pixels
           
    See Also
    --------
    G_to_circ
    """
    g_vec = np.zeros(2)
    g_vec[0:2] = np.divide(np.flip(np.asarray(circ_pos)),np.asarray(image.shape)) - 0.5
    return g_vec

def G_to_circ(g_vec,
              image):
    """
    Convert g vectors in Fourier space to
    pixel positions in real space.
    
    Parameters
    ----------
    g_vec: ndarray
           Shape is (2,1) which is the G
           vector in Fourier space in inverse pixels
    image: ndarray
           The image matrix

    Returns
    -------
    circ_pos: ndarray
           Shape is (2,1) which is the 
           corresponding pixel position in
           real space.
           
    See Also
    --------
    circ_to_G
    """
    circ_pos = np.zeros(2)
    circ_pos[1] = (g_vec[0]*image.shape[0]) + (0.5*image.shape[0])
    circ_pos[0] = (g_vec[1]*image.shape[1]) + (0.5*image.shape[1])
    return circ_pos

def g_matrix(g_vector,
             image):
    """
    Multiply g vector with Fourier coordinates
    to generate a corresponding phase matrix
    
    Parameters
    ----------
    g_vec: ndarray
           Shape is (2,1) which is the G
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
    r_y = np.arange(start=-image.shape[0]/2,stop=image.shape[0]/2,step=1)
    r_x = np.arange(start=-image.shape[1]/2,stop=image.shape[1]/2,step=1)
    R_x,R_y = np.meshgrid(r_x,r_y)
    G_r = 2*np.pi*((R_x*g_vector[1]) + (R_y*g_vector[0]))
    return G_r

def phase_matrix(gvec,
                 image,
                 g_blur=True):
    """
    Use the g vector in Fourier coordinates
    to select only the subset of phases
    associated with that diffraction spot,
    a.k.a. the lattice parameter.
    
    Parameters
    ----------
    g_vec:  ndarray
            Shape is (2,1) which is the G
            vector in Fourier space in inverse pixels
    image:  ndarray
            The image matrix
    g_blur: bool, optional

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
    circ_rad = np.amin(0.01*np.asarray(imshape))
    yy,xx = np.mgrid[0:imshape[0],0:imshape[1]]
    circ_pos = np.multiply(np.flip(gvec),imshape) + (0.5*imshape)
    circ_mask = (st.util.make_circle(imshape,circ_pos[0],circ_pos[1],circ_rad)).astype(bool)
    ham = np.sqrt(np.outer(np.hamming(imshape[0]), np.hamming(imshape[1])))

    if g_blur:
        sigma2 = np.sum((0.5*gvec*imshape)**2)
        zz = (((yy[circ_mask] - circ_pos[1])**2) + ((xx[circ_mask] - circ_pos[0])**2))/sigma2
        four_mask = np.zeros_like(yy,dtype=np.float) 
        four_mask[circ_mask] = np.exp((-0.5)*zz)
        P_matrix = np.angle(np.fft.ifft2(four_mask*np.fft.fftshift(np.fft.fft2(image*ham))))
    else:
        P_matrix = np.angle(np.fft.ifft2(circ_mask*np.fft.fftshift(np.fft.fft2(image*ham))))
    return P_matrix

@numba.jit(cache=True, parallel=True)
def numba_strain_P(P_1,
                   P_2,
                   a_matrix):
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
              ndarray of shape (2,2) that represents
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
    
class GPA(object):
    """
    Use Geometric Phase Analysis (GPA) to measure strain in an 
    electron micrograph by locating the diffraction spots and 
    identifying a reference region
    
    Parameters
    ----------
    image:    ndarray
              The image from which the strain will
              be measured from
    ref_iter: int, optional
              Number of iterations to run for refining
              the G vectors and the phase matrixes. 
              Default is 20.
    use_blur: bool, optional
              Use a Gaussian blur to generate the 
              phase matrix from a g vector. Default is True
                 
    References
    ----------
    [1]_, Hÿtch, M. J., E. Snoeck, and R. Kilaas. "Quantitative measurement 
          of displacement and strain fields from HREM micrographs." 
          Ultramicroscopy 74.3 (1998): 131-146.
    
    Examples
    --------
    Run as:
    
    >>> im_gpa = gpa(imageDC)
    >>> im_gpa.find_spots((983,905),(1066,984))
    where (983,905) and (1066,984) are two diffraction spot
    locations. You can run the `find_spots` method manually
    multiple times till you locate the spots closely,
    >>> im_gpa.define_reference((870, 750), (1300, 740), (1310, 1080), (900, 1100))
    where (870, 750), (1300, 740), (1310, 1080) and (900, 1100) are
    the corners of the reference region you are defining. 
    >>> im_gpa.refine_phase()
    >>> e_xx,e_yy,e_theta,e_diag = im_gpa.get_strain()
    
    """
    def __init__(self,
                 image,
                 ref_iter=20,
                 use_blur=True):
        self.image = image
        self.blur = use_blur
        self.ref_iter = int(ref_iter)
        self.image_ft = np.empty_like(image,dtype=np.complex)
        self.P_matrix1_ini = np.empty_like(image,dtype=np.float)
        self.P_matrix2_ini = np.empty_like(image,dtype=np.float)
        self.P_matrix1_fin = np.empty_like(image,dtype=np.float)
        self.P_matrix2_fin = np.empty_like(image,dtype=np.float)
        self.ham = np.empty_like(image)
        self.imshape = np.asarray(image.shape)
        self.circ_0 = 0.5*np.asarray(image.shape)
        self.circ_1 = np.empty(2,dtype=np.float)
        self.circ_2 = np.empty(2,dtype=np.float)
        self.gvec_1_ini = np.empty(2,dtype=np.float)
        self.gvec_2_ini = np.empty(2,dtype=np.float)
        self.gvec_1_fin = np.empty(2,dtype=np.float)
        self.gvec_2_fin = np.empty(2,dtype=np.float)
        self.a_matrix = np.empty((2,2),dtype=np.float)
        self.e_xx = np.empty_like(image,dtype=np.float)
        self.e_dg = np.empty_like(image,dtype=np.float)
        self.e_th = np.empty_like(image,dtype=np.float)
        self.e_yy = np.empty_like(image,dtype=np.float)
        self.spots_check = False
        self.reference_check = False
        self.refining_check = False
        
    def find_spots(self,
                   circ1,
                   circ2):
        """
        Locate the diffraction spots visually.

        Parameters
        ----------
        circ1: ndarray
               Position of the first beam in
               the Fourier pattern
        circ2: ndarray
               Position of the second beam in
               the Fourier pattern

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
        self.circ_1 = np.asarray(circ1)
        self.circ_2 = np.asarray(circ2)
        self.ham = np.sqrt(np.outer(np.hamming(self.imshape[0]), np.hamming(self.imshape[1])))
        self.image_ft = np.fft.fftshift(np.fft.fft2(self.image*self.ham))
        log_abs_ft = scnd.filters.gaussian_filter(np.log10(np.abs(self.image_ft)),3)
        
        f, ax = plt.subplots(figsize=(15, 15))
        circ_0_im = plt.Circle(self.circ_0, 15,color="red",alpha=0.33)
        circ_1_im = plt.Circle(self.circ_1, 15,color="blue",alpha=0.33)
        circ_2_im = plt.Circle(self.circ_2, 15,color="green",alpha=0.33)
        ax.imshow(log_abs_ft,cmap='gray')
        ax.add_artist(circ_0_im)
        ax.add_artist(circ_1_im)
        ax.add_artist(circ_2_im)
        plt.show()
        
        self.gvec_1_ini = circ_to_G(self.circ_1,self.image)
        self.gvec_2_ini = circ_to_G(self.circ_2,self.image)
        self.P_matrix1_ini = phase_matrix(self.gvec_1_ini,self.image,self.blur)
        self.P_matrix2_ini = phase_matrix(self.gvec_2_ini,self.image,self.blur)
        self.spots_check = True
    
    def define_reference(self,
                         A,
                         B,
                         C,
                         D):
        """
        Locate the reference image.

        Parameters
        ----------
        A:      Top left position of reference region in (x,y)
        B:      Top right position of reference region in (x,y)
        C:      Bottom right position of reference region in (x,y)
        D:      Bottom left position of reference region in (x,y)

        Notes
        -----
        Locate visually the unstrained reference region. To prevent 
        division by zero errors, make sure that all of the individual
        *x* or *y* of the corners of the reference region are unique.
        """
        if (not self.spots_check):
            raise RuntimeError('Please locate the diffraction spots first as find_spots()')
        xx,yy = np.meshgrid(np.arange(self.imshape[1]),np.arange(self.imshape[0]))
        m_AB = (A[1] - B[1])/(A[0] - B[0])
        c_AB = A[1] - (m_AB*A[0])
        m_BC = (B[1] - C[1])/(B[0] - C[0])
        c_BC = B[1] - (m_BC*B[0])
        m_CD = (C[1] - D[1])/(C[0] - D[0])
        c_CD = C[1] - (m_CD*C[0])
        m_DA = (D[1] - A[1])/(D[0] - A[0])
        c_DA = D[1] - (m_DA*D[0])
        self.ref_reg = np.logical_and(np.logical_and((yy > (m_AB*xx) + c_AB),((yy - c_BC)/m_BC > xx)),
                                      np.logical_and((yy < (m_CD*xx) + c_CD),((yy - c_DA)/m_DA < xx)))
        
        plt.figure(figsize=(15,15))
        plt.imshow(st.util.image_normalizer(self.image)+0.33*self.ref_reg)
        plt.annotate(A, (A[0]/self.imshape[0], (1 - A[1]/self.imshape[1])), textcoords='axes fraction', size=15, color='w')
        plt.annotate(B, (B[0]/self.imshape[0], (1 - B[1]/self.imshape[1])), textcoords='axes fraction', size=15, color='w')
        plt.annotate(C, (C[0]/self.imshape[0], (1 - C[1]/self.imshape[1])), textcoords='axes fraction', size=15, color='w')
        plt.annotate(D, (D[0]/self.imshape[0], (1 - D[1]/self.imshape[1])), textcoords='axes fraction', size=15, color='w')
        plt.scatter(A[0],A[1])
        plt.scatter(B[0],B[1])
        plt.scatter(C[0],C[1])
        plt.scatter(D[0],D[1])
        plt.axis('off')
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
        if (not self.reference_check):
            raise RuntimeError('Please locate the reference region first as define_reference()')
        ry = np.arange(start=-self.imshape[0]/2,stop=self.imshape[0]/2,step=1)
        rx = np.arange(start=-self.imshape[1]/2,stop=self.imshape[1]/2,step=1)
        R_x,R_y = np.meshgrid(rx,ry)
        self.gvec_1_fin = self.gvec_1_ini
        self.gvec_2_fin = self.gvec_2_ini
        self.P_matrix1_fin = self.P_matrix1_ini
        self.P_matrix2_fin = self.P_matrix2_ini
        for _ in range(int(self.ref_iter)):
            G1_x,G1_y = phase_diff(self.P_matrix1_fin)
            G2_x,G2_y = phase_diff(self.P_matrix2_fin)
            g1_r = (G1_x + G1_y)/(2*np.pi)
            g2_r = (G2_x + G2_y)/(2*np.pi)
            del_g1 = np.asarray((np.median(g1_r[self.ref_reg]/R_y[self.ref_reg]),
                                 np.median(g1_r[self.ref_reg]/R_x[self.ref_reg])))
            del_g2 = np.asarray((np.median(g2_r[self.ref_reg]/R_y[self.ref_reg]),
                                 np.median(g2_r[self.ref_reg]/R_x[self.ref_reg])))
            self.gvec_1_fin += del_g1
            self.gvec_2_fin += del_g2
            self.P_matrix1_fin = phase_matrix(self.gvec_1_fin,self.image,self.blur)
            self.P_matrix2_fin = phase_matrix(self.gvec_2_fin,self.image,self.blur)
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
        if (not self.reference_check):
            raise RuntimeError('Please refine the phase and g vectors first as refine_phase()')
        g_matrix = np.zeros((2,2),dtype=np.float64)
        g_matrix[0,:] = np.flip(np.asarray(self.gvec_1_fin))
        g_matrix[1,:] = np.flip(np.asarray(self.gvec_2_fin))
        self.a_matrix = np.linalg.inv(np.transpose(g_matrix))
        P1 = skr.unwrap_phase(self.P_matrix1_fin)
        P2 = skr.unwrap_phase(self.P_matrix1_fin)
        rolled_p = np.asarray((np.reshape(P1,-1),np.reshape(P2,-1)))
        u_matrix = np.matmul(self.a_matrix,rolled_p)
        u_x = np.reshape(u_matrix[0,:],P1.shape)
        u_y = np.reshape(u_matrix[1,:],P2.shape)
        self.e_xx,e_xy = phase_diff(u_x)
        e_yx,self.e_yy = phase_diff(u_y)
        self.e_th = 0.5*(e_xy - e_yx)
        self.e_dg = 0.5*(e_xy + e_yx)
        self.e_yy -= np.median(self.e_yy[self.ref_reg])
        self.e_dg -= np.median(self.e_dg[self.ref_reg])
        self.e_th -= np.median(self.e_th[self.ref_reg])
        self.e_xx -= np.median(self.e_xx[self.ref_reg])
        return self.e_xx, self.e_yy, self.e_th, self.e_dg