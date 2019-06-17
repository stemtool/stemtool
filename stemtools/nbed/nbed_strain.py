import numpy as np
from scipy import ndimage as scnd
from scipy import optimize as sio
import numba
from ..util import image_utils as iu
from ..proc import sobel_canny as sc
from ..util import gauss_utils as gt

@numba.jit(cache=True)
def angle_fun(angle,image_orig):
    rotated_image = scnd.rotate(image_orig,angle,order=5,reshape=False)
    rotsum = (-1)*(np.sum(rotated_image,1))
    rotmin = np.amin(rotsum)
    return rotmin

@numba.jit(cache=True)
def rotation_finder(image_orig):
    x0 = 90
    x = sio.minimize(angle_fun,x0,args=(image_orig))
    min_x = x.x
    return min_x

@numba.jit(parallel=True)
def rotate_and_center_ROI(data4D_ROI,rotangle,xcenter,ycenter):
    """
    Rotation Corrector
    
    Parameters
    ----------
    data4D_ROI: ndarray 
                Region of interest of the 4D-STEM dataset in
                the form of ROI pixels (scanning), CBED_Y, CBED_x
    rotangle:   float
                angle in counter-clockwise direction to 
                rotate individual CBED patterns
    xcenter:    float
                X pixel co-ordinate of center of mean pattern
    ycenter:    float
                Y pixel co-ordinate of center of mean pattern
                     
    Returns
    -------
    corrected_ROI: ndarray
                   Each CBED pattern from the region of interest
                   first centered and then rotated along the center
     
    
    Notes
    -----
    We start by centering each 4D-STEM CBED pattern 
    and then rotating the patterns with respect to the
    pattern center
                 
    :Authors:
    Debangshu Mukherjee <mukherjeed@ornl.gov>
    """
    data_size = np.asarray(np.shape(data4D_ROI))
    corrected_ROI = np.zeros_like(data4D_ROI)
    for ii in numba.prange(data4D_ROI.shape[0]):
        cbed_pattern = data4D_ROI[ii,:,:]
        moved_cbed = np.abs(iu.move_by_phase(cbed_pattern,(-xcenter + (0.5 * data_size[-1])),(-ycenter + (0.5 * data_size[-2]))))
        rotated_cbed = scnd.rotate(moved_cbed,rotangle,order=5,reshape=False)
        corrected_ROI[ii,:,:] = rotated_cbed
    return corrected_ROI

@numba.jit(parallel=True)
def correlate_4D(data4D,corr_pattern,hybridizer=0.5):
    """
    Hybrid cross-correlate a 4D-STEM dataset's individual patterns
    with a known correlation pattern.
    
    Parameters
    ----------
    data4D: ndarray
            4D-STEM dataset to be correlated where
            the first two dimensions are in Fourier
            space and the last two dimensions are 
            in real space.
    corr_pattern: ndarray
                  Two-dimensional array of the pattern
                  that is to be correlated with each 
                  individual CBED pattern.
    hybridizer: float
                Hybrid Cross Correlation parameter
                Default is 0.5
                
    Returns
    -------
    corr4D: ndarray
            Complex valued 4D dataset where the 
            first two dimensions are the correlated
            patterns.
            
    :Authors:
    Debangshu Mukherjee <mukherjeed@ornl.gov>
    """
    data_size = (np.asarray(data4D.shape)).astype(int)
    corr4D = (np.zeros((data_size[0],data_size[1],data_size[2],data_size[3]))).astype('complex128')
    for jj in numba.prange(data_size[3]):
        for ii in range(data_size[2]):
            cbed = data4D[:,:,ii,jj]
            cc_cbed = iu.normalized_correlation(cbed,corr_pattern,hybridizer)
            corr4D[:,:,ii,jj] = cc_cbed
    return corr4D

@numba.jit(parallel=True)
def correlate_with_disk(data4D,radius,hybridizer=0.5):
    """
    Hybrid cross-correlate a 4D-STEM dataset's 
    individual patterns with a circle whose radius 
    is user defined.
    
    Parameters
    ----------
    data4D: ndarray
            4D-STEM dataset to be correlated where
            the first two dimensions are in Fourier
            space and the last two dimensions are 
            in real space.
    radius: float
            Radius of the circle for correlation
    hybridizer: float
                Hybrid Cross Correlation parameter
                Default is 0.5
                
    Returns
    -------
    corr4D: ndarray
            Complex valued 4D dataset where the 
            first two dimensions are the correlated
            patterns.
    
    See Also
    --------
    correlate4D
    make_circle
            
    :Authors:
    Debangshu Mukherjee <mukherjeed@ornl.gov>
    """
    data_size = (np.asarray(data4D.shape)).astype(int)
    central_disk = make_circle(data_size,(data_size[1] / 2), (data_size[0] / 2), radius)
    corr4D = (np.zeros((data_size[0],data_size[1],data_size[2],data_size[3]))).astype('complex128')
    for jj in numba.prange(data_size[3]):
        for ii in range(data_size[2]):
            cbed = data4D[:,:,ii,jj]
            cc_cbed = iu.normalized_correlation(cbed,central_disk,hybridizer)
            corr4D[:,:,ii,jj] = cc_cbed
    return corr4D

@numba.jit(cache=True)
def gaussian_fit_corr(corr_data,radius_circ,center_distance):
    corr_centers = np.zeros((3,3,2))
    data_size = (np.asarray(corr_data.shape)).astype(int)
    yV, xV = np.mgrid[0:data_size[0],0:data_size[1]]
    for jj in range(3):
        for ii in range(3):
            pp = ii - 1
            qq = jj - 1
            center_x = (data_size[1] / 2) + (pp*center_distance)
            center_y = (data_size[0] / 2) + (qq*center_distance)
            first_circ = (iu.make_circle(data_size,center_x,center_y,radius_circ)).astype(bool)
            xx = xV[first_circ]
            yy = yV[first_circ]
            zz = corr_data[first_circ]
            center_x = xx[zz == np.amax(zz)]
            center_y = yy[zz == np.amax(zz)]
            gauss_ii_jj = gt.fit_gaussian2D_mask(corr_data,center_x,center_y,radius_circ,'circular')
            corr_centers[ii,jj,0] = gauss_ii_jj[0]
            corr_centers[ii,jj,1] = gauss_ii_jj[1]
    return corr_centers

@numba.jit(cache=True)
def get_ROI(image,top_left,top_right,bottom_left,bottom_right):
    p,q = np.shape(image)
    xV, yV = np.meshgrid(np.arange(p),np.arange(q))
    top_slope = (top_right[1] - top_left[1])/(top_right[0] - top_left[0])
    top_intercept = top_right[1] - (top_slope*top_right[0])
    sub_top = (yV > (top_intercept + (top_slope*xV)))
    left_slope = (bottom_left[1] - top_left[1])/(bottom_left[0] - top_left[0])
    left_intercept = top_left[1] - (left_slope*top_left[0])
    sub_left = (yV < (left_intercept + (left_slope*xV)))
    right_slope = (bottom_right[1] - top_right[1])/(bottom_right[0] - top_right[0])
    right_intercept = top_right[1] - (right_slope*top_right[0])
    sub_right = (yV > (right_intercept + (right_slope*xV)))
    bottom_slope = (bottom_right[1] - bottom_left[1])/(bottom_right[0] - bottom_left[0])
    bottom_intercept = bottom_right[1] - (bottom_slope*bottom_right[0])
    sub_bottom = (yV < (bottom_intercept + (bottom_slope*xV)))
    sub = np.multiply(np.multiply(sub_top,sub_bottom),np.multiply(sub_left,sub_right))
    xR = np.ravel(xV[sub])
    yR = np.ravel(yV[sub])
    return sub, xR, yR

@numba.jit(cache=True)
def correlate_4D_ROI_test(data4D_ROI,corr_pattern,hybridizer=0.5):
    corr4D_raw = np.zeros_like(data4D_ROI,dtype=np.double) #raw data
    corr4D_log = np.zeros_like(data4D_ROI,dtype=np.double) #log of the data
    corr4D_sob = np.zeros_like(data4D_ROI,dtype=np.double) #with the sobel filtered data
    corr4D_lsb = np.zeros_like(data4D_ROI,dtype=np.double) #with the log of the sobel filtered data
    
    corr_pattern_raw = corr_pattern
    corr_pattern_log = np.log10(1 + corr_pattern_raw - np.amin(corr_pattern_raw))
    corr_pattern_sob, _ = sc.sobel_filter(corr_pattern_raw)
    corr_pattern_lsb = np.log10(1 + corr_pattern_sob - np.amin(corr_pattern_sob))
    for ii in numba.prange(data4D_ROI.shape[0]):
        cbed_raw = data4D_ROI[ii,:,:]
        cbed_log = np.log10(1 + cbed_raw - np.amin(cbed_raw))
        cbed_sob, _ = sc.sobel_filter(cbed_raw)
        cbed_lsb = np.log10(1 + cbed_sob - np.amin(cbed_sob))
        
        cc_cbed_raw = np.abs(iu.normalized_correlation(cbed_raw,corr_pattern_raw,hybridizer))
        cc_cbed_log = np.abs(iu.normalized_correlation(cbed_log,corr_pattern_log,hybridizer))
        cc_cbed_sob = np.abs(iu.normalized_correlation(cbed_sob,corr_pattern_sob,hybridizer))
        cc_cbed_lsb = np.abs(iu.normalized_correlation(cbed_lsb,corr_pattern_lsb,hybridizer))
        
        corr4D_raw[ii,:,:] = cc_cbed_raw
        corr4D_log[ii,:,:] = cc_cbed_log
        corr4D_sob[ii,:,:] = cc_cbed_sob
        corr4D_lsb[ii,:,:] = cc_cbed_lsb
    return corr4D_raw, corr4D_log, corr4D_sob, corr4D_lsb

@numba.jit(parallel=True)
def gaussian_fit_4D_ROI(corr_data_ROI,radius_circ,center_distance):
    data_size = np.asarray(np.shape(corr_data_ROI))
    fitted_diff_points = np.zeros((data_size[0],3,3,2))
    for ii in numba.prange(data_size[0]):
        corr_cbed = np.abs(corr_data_ROI[ii,:,:])
        corr_points = gaussian_fit_corr(corr_cbed,radius_circ,center_distance)
        fitted_diff_points[ii,:,:,:] = corr_points
    return fitted_diff_points

@numba.jit(parallel=True)
def strain_in_ROI(fitted_cbed,p_matrix,mean_centers,points):
    no_of_disks = int(0.5*(np.size(mean_centers)))
    b_mean = np.reshape(mean_centers, (no_of_disks,2))
    b_mean = b_mean - b_mean[4,:]
    data_size = (np.shape(fitted_cbed))[0]
    e_xx_ROI = np.zeros(data_size)
    e_yy_ROI = np.zeros(data_size)
    e_xy_ROI = np.zeros(data_size)
    e_th_ROI = np.zeros(data_size)
    identity_matrix =  np.asarray(((1,0),
                                   (0,1)))
    l_mean, _,_,_ = np.linalg.lstsq(p_matrix[points,:],b_mean[points,:],rcond=None)
    l_mean_inv = np.linalg.inv(l_mean)
    
    for ii in numba.prange(data_size):
        corr_points = fitted_cbed[ii,:,:,:]
        b_cbed = np.reshape(corr_points, (no_of_disks,2))
        b_cbed = b_cbed - b_cbed[4,:]
        l_cbed, _,_,_ = np.linalg.lstsq(p_matrix[points,:],b_cbed[points,:],rcond=None)
        t_cbed = np.matmul(l_cbed,l_mean_inv)
        s_cbed = t_cbed - identity_matrix
        e_xx = -s_cbed[0,0]
        e_yy = -s_cbed[1,1]
        e_xy = -(s_cbed[0,1] + s_cbed[1,0])
        e_th = (s_cbed[0,1] - s_cbed[1,0])
        e_xx_ROI[ii] = e_xx
        e_yy_ROI[ii] = e_yy
        e_xy_ROI[ii] = e_xy
        e_th_ROI[ii] = e_th
    
    return e_xx_ROI, e_xy_ROI, e_yy_ROI, e_th_ROI

@numba.jit(cache=True)
def ROI_to_map(strain_ROI,image,ROI_range,med_value=0.25):
    y_ROI = ROI_range[0]
    x_ROI = ROI_range[1]
    z_ROI = iu.image_normalizer(image[y_ROI,x_ROI])
    ROI_box = np.zeros_like(image)
    ROI_box[y_ROI[z_ROI > med_value*np.median(z_ROI)],x_ROI[z_ROI > med_value*np.median(z_ROI)]] = 1
    strain_map = np.zeros_like(image)
    strain_map[y_ROI,x_ROI] = strain_ROI
    strain_map = np.multiply(strain_map,ROI_box)
    return strain_map