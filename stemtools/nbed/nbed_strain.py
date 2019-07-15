import numpy as np
from scipy import ndimage as scnd
from scipy import optimize as sio
from scipy import signal as scisig
import matplotlib.pyplot as plt
from ..util import image_utils as iu
from ..proc import sobel_canny as sc
from ..util import gauss_utils as gt

def angle_fun(angle,image_orig,axis=0,):
    """
    Rotation Sum Finder
    
    Parameters
    ----------
    angle:      float 
                Angle to rotate 
    image_orig: (2,2) shape ndarray
                Input Image
    axis:       int
                Axis along which to perform sum
                     
    Returns
    -------
    rotmin: float
            Sum of the rotated image multiplied by -1 along 
            the axis specified
                 
    :Authors:
    Debangshu Mukherjee <mukherjeed@ornl.gov>
    """
    rotated_image = scnd.rotate(image_orig,angle,order=5,reshape=False)
    rotsum = (-1)*(np.sum(rotated_image,1))
    rotmin = np.amin(rotsum)
    return rotmin

def rotation_finder(image_orig,axis=0):
    """
    Angle Finder
    
    Parameters
    ----------
    image_orig: (2,2) shape ndarray
                Input Image
    axis:       int
                Axis along which to perform sum
                     
    Returns
    -------
    min_x: float
           Angle by which if the image is rotated
           by, the sum of the image along the axis
           specified is maximum
                 
    :Authors:
    Debangshu Mukherjee <mukherjeed@ornl.gov>
    """
    x0 = 90
    x = sio.minimize(angle_fun,x0,args=(image_orig))
    min_x = x.x
    return min_x

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
    for ii in range(data4D_ROI.shape[0]):
        cbed_pattern = data4D_ROI[ii,:,:]
        moved_cbed = np.abs(iu.move_by_phase(cbed_pattern,(-xcenter + (0.5 * data_size[-1])),(-ycenter + (0.5 * data_size[-2]))))
        rotated_cbed = scnd.rotate(moved_cbed,rotangle,order=5,reshape=False)
        corrected_ROI[ii,:,:] = rotated_cbed
    return corrected_ROI

def data4Dto2D(data4D):
    """
    Convert 4D data to 2D data
    
    Parameters
    ----------
    data4D: ndarray of shape (4,4)
            the first two dimensions are Fourier
            space, while the next two dimensions
            are real space
                     
    Returns
    -------
    data2D: ndarray of shape (2,2)
            Raveled 2D data where the
            first two dimensions are positions
            while the next two dimensions are spectra
                 
    :Authors:
    Debangshu Mukherjee <mukherjeed@ornl.gov>
    """
    data2D = np.transpose(data4D,(2,3,0,1))
    data_shape = data2D.shape
    data2D.shape = (data_shape[0]*data_shape[1],data_shape[2]*data_shape[3])
    return data2D

def bin4D(data4D,bin_factor):
    """
    Bin 4D data in spectral dimensions
    
    Parameters
    ----------
    data4D:     ndarray of shape (4,4)
                the first two dimensions are Fourier
                space, while the next two dimensions
                are real space
    bin_factor: int
                Value by which to bin data
                     
    Returns
    -------
    binned_data: ndarray of shape (4,4)
                 Data binned in the spectral dimensions
    
    Notes
    -----
    The data is binned in the last two spectral dimensions
    using scipy signal decimate
                 
    :Authors:
    Debangshu Mukherjee <mukherjeed@ornl.gov>
    """
    mean_data = np.mean(np.mean(data4D,-1),-1)
    mean_binned = scisig.decimate(scisig.decimate(mean_data,bin_factor,axis=0),bin_factor,axis=1)
    binned_data = np.zeros((mean_binned.shape[0],mean_binned.shape[1],data4D.shape[2],data4D.shape[3]),dtype=np.float)
    for ii in range(data4D.shape[2]):
        for jj in range(data4D.shape[3]):
            ronchi = data4D[:,:,ii,jj]
            binned_ronchi = scisig.decimate(scisig.decimate(ronchi,bin_factor,axis=0),bin_factor,axis=1)
            binned_data[:,:,ii,jj] = binned_ronchi
    return binned_data

def test_aperture(pattern,center,radius,showfig=True):
    """
    Test an aperture position for Virtual DF image
    
    Parameters
    ----------
    pattern: ndarray of shape (2,2)
             Diffraction pattern, preferably the
             mean diffraction pattern for testing out
             the aperture location
    center:  ndarray of shape (1,2)
             Center of the circular aperture
    radius:  float
             Radius of the circular aperture
    showfig: bool
             If showfig is True, then the image is
             displayed with the aperture overlaid
                     
    Returns
    -------
    aperture: ndarray of shape (2,2)
              A matrix of the same size of the input image
              with zeros everywhere and ones where the aperture
              is supposed to be
    
    Notes
    -----
    Use the showfig option to visually test out the aperture 
    location with varying parameters
                 
    :Authors:
    Debangshu Mukherjee <mukherjeed@ornl.gov>
    """
    center = np.asarray(center)
    yy,xx = np.mgrid[0:pattern.shape[0],0:pattern.shape[1]]
    yy = yy - center[1]
    xx = xx - center[0]
    rr = ((yy ** 2) + (xx ** 2)) ** 0.5
    aperture = np.asarray(rr<=radius, dtype=np.double)
    if showfig:
        plt.figure(figsize=(15,15))
        plt.imshow(iu.image_normalizer(pattern)+aperture,cmap='Spectral')
        plt.scatter(center[0],center[1],c='w', s=25)
    return aperture

def aperture_image(data4D,center,radius):
    """
    Generate Virtual DF image for a given aperture
    
    Parameters
    ----------
    data4D: ndarray of shape (4,4)
            the first two dimensions are Fourier
            space, while the next two dimensions
            are real space
    center: ndarray of shape (1,2)
            Center of the circular aperture
    radius: float
            Radius of the circular aperture
    
    Returns
    -------
    df_image: ndarray of shape (2,2)
              Generated virtual dark field image
              from the aperture and 4D data
    
    Notes
    -----
    We generate the aperture first, and then make copies
    of the aperture to generate a 4D dataset of the same 
    size as the 4D data. Then we do an element wise 
    multiplication of this aperture 4D data with the 4D data
    and then sum it along the two Fourier directions.
                 
    :Authors:
    Debangshu Mukherjee <mukherjeed@ornl.gov>
    """
    center = np.array(center)
    yy,xx = np.mgrid[0:data4D.shape[0],0:data4D.shape[1]]
    yy = yy - center[1]
    xx = xx - center[0]
    rr = ((yy ** 2) + (xx ** 2)) ** 0.5
    aperture = np.asarray(rr<=radius, dtype=data4D.dtype)
    apt_copy = np.empty((data4D.shape[2],data4D.shape[3]) + aperture.shape,dtype=data4D.dtype)
    apt_copy[:] = aperture
    apt_copy = np.transpose(apt_copy,(2,3,0,1))
    apt_mult = apt_copy * data4D
    df_image = np.sum(np.sum(apt_mult,axis=0),axis=0)
    return df_image

def ROI_from_image(image,med_val,style='over',showfig=True):
    if style == 'over':
        ROI = np.asarray(image > (med_val*np.median(image)),dtype=np.double)
    else:
        ROI = np.asarray(image < (med_val*np.median(image)),dtype=np.double)
    if showfig:
        plt.figure(figsize=(15, 15))
        plt.imshow(ROI+iu.image_normalizer(image),cmap='viridis')
        plt.title('ROI overlaid')
    ROI = ROI.astype(bool)
    return ROI

def get_disk_fit(corr_image,disk_size,disk_list,pos_list):
    fitted_disk_list = np.zeros(np.shape(disk_list),dtype=np.float)
    disk_locations = np.zeros(np.shape(disk_list),dtype=np.float)
    for ii in range(np.shape(disk_list)[0]):
        posx = disk_list[ii,0]
        posy = disk_list[ii,1]
        par = gt.fit_gaussian2D_mask(corr_image,posx,posy,disk_size)
        fitted_disk_list[ii,0] = par[0]
        fitted_disk_list[ii,1] = par[1]
    disk_locations[:,0] = (-1)*fitted_disk_list[:,0]
    disk_locations[:,1] = fitted_disk_list[:,1]
    center = disk_locations[np.logical_and((pos_list[:,0] == 0),(pos_list[:,1] == 0)),:]
    disk_locations[:,0:2] = disk_locations[:,0:2] - center
    lcbed,_,_,_ = np.linalg.lstsq(pos_list,disk_locations,rcond=None)
    center[0,0] = (-1)*center[0,0]
    return fitted_disk_list,center,lcbed

def strain_in_ROI(data4D_ROI,center_disk,disk_list,pos_list,reference_axes):
    # Calculate needed values
    no_of_disks = data4D_ROI.shape[-1]
    disk_size = (np.sum(center_disk)/np.pi) ** 0.5
    i_matrix = (np.eye(2)).astype(np.float)
    sobel_center_disk,_ = sc.sobel(center_disk)
    # Initialize matrices
    e_xx_ROI = np.zeros(no_of_disks,dtype=np.float)
    e_xy_ROI = np.zeros(no_of_disks,dtype=np.float)
    e_yy_ROI = np.zeros(no_of_disks,dtype=np.float)
    #Calculate for mean CBED if no reference
    #axes present
    if np.size(reference_axes) < 2:
        mean_cbed = np.mean(data4D_ROI,axis=-1)
        sobel_lm_cbed,_ = sc.sobel(iu.image_logarizer(mean_cbed))
        lsc_mean = iu.cross_corr(sobel_lm_cbed,sobel_center_disk,hybridizer=0.1)
        _,_,mean_axes = get_disk_fit(lsc_mean,disk_size,disk_list,pos_list)
        inverse_axes = np.linalg.inv(mean_axes)
    else:
        inverse_axes = np.linalg.inv(reference_axes)
    for ii in range(no_of_disks):
        pattern = data4D_ROI[:,:,ii]
        sobel_log_pattern,_ = sc.sobel(iu.image_logarizer(pattern))
        lsc_pattern = iu.cross_corr(sobel_log_pattern,sobel_center_disk,hybridizer=0.1)
        _,_,pattern_axes = get_disk_fit(lsc_pattern,disk_size,disk_list,pos_list)
        t_pattern = np.matmul(pattern_axes,inverse_axes)
        s_pattern = t_pattern - i_matrix
        e_xx_ROI[ii] = -s_pattern[0,0]
        e_xy_ROI[ii] = -(s_pattern[0,1] + s_pattern[1,0])
        e_yy_ROI[ii] = -s_pattern[1,1]
    return e_xx_ROI,e_xy_ROI,e_yy_ROI