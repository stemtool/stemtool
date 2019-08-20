import numpy as np
import numba
import warnings
from scipy import ndimage as scnd
from scipy import optimize as sio
from scipy import signal as scisig
import matplotlib.colors as mplc
import matplotlib.pyplot as plt
from ..util import image_utils as iu
from ..proc import sobel_canny as sc
from ..util import gauss_utils as gt
import warnings

def angle_fun(angle,
              image_orig,
              axis=0,):
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

def rotation_finder(image_orig,
                    axis=0):
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

def rotate_and_center_ROI(data4D_ROI,
                          rotangle,
                          xcenter,
                          ycenter):
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

@numba.jit
def resizer(data,
            N):
    """
    Downsample 1D array
    
    Parameters
    ----------
    data: ndarray
    N:    int
          New size of array
                     
    Returns
    -------
    res: ndarray of shape N
         Data resampled
    
    Notes
    -----
    The data is resampled. Since this is a Numba
    function, compile it once (you will get errors)
    by calling %timeit
                 
    :Authors:
    Debangshu Mukherjee <mukherjeed@ornl.gov>
    """
    warnings.filterwarnings('ignore')
    M = data.size
    data = (data).astype(np.float64)
    res=np.zeros(int(N),dtype=np.float64)
    carry=0
    m=0
    for n in range(int(N)):
        data_sum = carry
        while m*N - n*M < M :
            data_sum += data[m]
            m += 1
        carry = (m-(n+1)*M/N)*data[m-1]
        data_sum -= carry
        res[n] = data_sum*N/M
    return res

@numba.jit
def resizer2D(data,
              sampling):
    """
    Downsample 2D array
    
    Parameters
    ----------
    data:     ndarray
              (2,2) shape
    sampling: tuple
              Downsampling factor in each axisa
                     
    Returns
    -------
    resampled: ndarray
              Downsampled by the sampling factor
              in each axis
    
    Notes
    -----
    The data is a 2D wrapper over the resizer function
    
    See Also
    --------
    resizer
                 
    :Authors:
    Debangshu Mukherjee <mukherjeed@ornl.gov>
    """
    warnings.filterwarnings('ignore')
    sampling = np.asarray(sampling)
    data_shape = np.asarray(np.shape(data))
    sampled_shape = (np.round(data_shape/sampling)).astype(int)
    resampled_x = np.zeros((data_shape[0],sampled_shape[1]),dtype=np.float64)
    resampled = np.zeros(sampled_shape,dtype=np.float64)
    for yy in range(int(data_shape[0])):
        resampled_x[yy,:] = resizer(data[yy,:],sampled_shape[1])
    for xx in range(int(sampled_shape[1])):
        resampled[:,xx] = resizer(resampled_x[:,xx],sampled_shape[0])
    return resampled

@numba.jit
def bin4D(data4D,
          bin_factor):
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
    using resizer2D function.
    
    See Also
    --------
    resizer
    resizer2D
                 
    :Authors:
    Debangshu Mukherjee <mukherjeed@ornl.gov>
    """
    warnings.filterwarnings('ignore')
    mean_data = np.mean(data4D,axis=(-1,-2),dtype=np.float64)
    mean_binned = resizer2D(mean_data,(bin_factor,bin_factor))
    binned_data = np.zeros((mean_binned.shape[0],mean_binned.shape[1],data4D.shape[2],data4D.shape[3]),dtype=data4D.dtype)
    for ii in range(data4D.shape[2]):
        for jj in range(data4D.shape[3]):
            binned_data[:,:,ii,jj] = resizer2D(data4D[:,:,ii,jj],(bin_factor,bin_factor))
    return binned_data

def test_aperture(pattern,
                  center,
                  radius,
                  showfig=True):
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

def aperture_image(data4D,
                   center,
                   radius):
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

def ROI_from_image(image,
                   med_val,
                   style='over',
                   showfig=True):
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

@numba.jit
def colored_mcr(conc_data,
                data_shape):
    no_spectra = np.shape(conc_data)[1]
    color_hues = np.arange(no_spectra,dtype=np.float64)/no_spectra
    norm_conc = (conc_data - np.amin(conc_data)) / (np.amax(conc_data) - np.amin(conc_data))
    saturation_matrix = np.ones(data_shape,dtype=np.float64)
    hsv_calc = np.zeros((data_shape[0],data_shape[1],3),dtype=np.float64)
    rgb_calc = np.zeros((data_shape[0],data_shape[1],3),dtype=np.float64)
    hsv_calc[:,:,1] = saturation_matrix
    for ii in range(no_spectra):
        conc_image = (np.reshape(norm_conc[:,ii],data_shape)).astype(np.float64)
        hsv_calc[:,:,0] = saturation_matrix * color_hues[ii]
        hsv_calc[:,:,2] = conc_image
        rgb_calc = rgb_calc + mplc.hsv_to_rgb(hsv_calc)
    rgb_image = rgb_calc/np.amax(rgb_calc)
    return rgb_image

@numba.jit
def fit_nbed_disks(corr_image,
                   disk_size,
                   positions,
                   diff_spots,
                   nan_cutoff=0):
    """
    Disk Fitting algorithm for a single NBED pattern
    
    Parameters
    ----------
    corr_image: ndarray of shape (2,2)
                The cross-correlated image of the NBED that 
                will be fitted
    disk_size:  float
                Size of each NBED disks in pixels
    positions:  ndarray of shape (n,2)
                X and Y positions where n is the number of positions.
                These are the initial guesses that will be refined
    diff_spots: ndarray of shape (n,2)
                a and b Miller indices corresponding to the
                disk positions
    nan_cutoff: float
                Optional parameter that is used for thresholding disk
                fits. If the intensity ratio is below the threshold 
                the position will not be fit. Default value is 0
    
    Returns
    -------
    fitted_disk_list: ndarray of shape (n,2)
                      Sub-pixel precision Gaussian fitted disk
                      locations. If nan_cutoff is greater than zero, then
                      only the positions that are greater than the threshold 
                      are returned.
    center_position:  ndarray of shape (1,2)
                      Location of the central (0,0) disk
    fit_deviation:    ndarray of shape (1,2)
                      Standard deviation of the X and Y disk fits given as pixel 
                      ratios
    lcbed:            ndarray of shape (2,2)
                      Matrix defining the Miller indices axes
    
    Notes
    -----
    Every disk position is fitted with a 2D Gaussian by cutting off a circle
    of the size of disk_size around the initial poistions. If nan-cutoff is above 
    zero then only the locations inside this cutoff where the maximum pixel intensity 
    is (1+nan_cutoff) times the median pixel intensity will be fitted. Use this 
    parameter carefully, because in some cases this may result in no disks being fitted
    and the program throwing weird errors at you. 
                 
    :Authors:
    Debangshu Mukherjee <mukherjeed@ornl.gov>
    """
    warnings.filterwarnings('ignore')
    no_pos = int(np.shape(positions)[0])
    diff_spots = np.asarray(diff_spots,dtype=np.float64)
    fitted_disk_list = np.zeros_like(positions)
    yy,xx = np.mgrid[0:(corr_image.shape[0]),0:(corr_image.shape[1])]
    nancount = 0
    for ii in range(no_pos):
        posx = positions[ii,0]
        posy = positions[ii,1]
        reg = ((yy - posy) ** 2) + ((xx - posx) ** 2) <= (disk_size ** 2)
        peak_ratio = np.amax(corr_image[reg])/np.median(corr_image[reg])
        if peak_ratio < (1+nan_cutoff):
            fitted_disk_list[ii,:] = np.nan
            nancount = nancount + 1
        else:
            par = gt.fit_gaussian2D_mask(corr_image,posx,posy,disk_size)
            fitted_disk_list[ii,0] = par[0]
            fitted_disk_list[ii,1] = par[1]
    nancount = int(nancount)
    if nancount == no_pos:
        center_position = np.nan*np.ones((1,2))
        fit_deviation = np.nan
        lcbed = np.nan
    else:
        diff_spots = (diff_spots[~np.isnan(fitted_disk_list)]).reshape((no_pos - nancount),2)
        fitted_disk_list = (fitted_disk_list[~np.isnan(fitted_disk_list)]).reshape((no_pos - nancount),2)
        disk_locations = np.copy(fitted_disk_list)
        disk_locations[:,1] = (-1)*disk_locations[:,1]
        center = disk_locations[np.logical_and((diff_spots[:,0] == 0),(diff_spots[:,1] == 0)),:]
        cx = center[0,0]
        cy = center[0,1]
        if (nancount/no_pos) < 0.5: 
            disk_locations[:,0:2] = disk_locations[:,0:2] - np.asarray((cx,cy),dtype=np.float64)
            lcbed,_,_,_ = np.linalg.lstsq(diff_spots,disk_locations,rcond=None)
            calc_points = np.matmul(diff_spots,lcbed)
            stdx = np.std(np.divide(disk_locations[np.where(calc_points[:,0] != 0),0],calc_points[np.where(calc_points[:,0] != 0),0]))
            stdy = np.std(np.divide(disk_locations[np.where(calc_points[:,1] != 0),1],calc_points[np.where(calc_points[:,1] != 0),1]))
            cy = (-1)*cy
            center_position = np.asarray((cx,cy),dtype=np.float64)
            fit_deviation = np.asarray((stdx,stdy),dtype=np.float64)
        else:
            cy = (-1)*cy
            center_position = np.asarray((cx,cy),dtype=np.float64)
            fit_deviation = np.nan
            lcbed = np.nan
    return fitted_disk_list,center_position,fit_deviation,lcbed

@numba.jit
def strain_in_ROI(data4D,
                  ROI,
                  center_disk,
                  disk_list,
                  pos_list,
                  reference_axes=0,
                  med_factor=10,
                  gauss_val=3,
                  hybrid_cc=0.1,
                  nan_cutoff=0.5):
    """
    Get strain from a region of interest
    
    Parameters
    ----------
    data4D:         ndarray
                    This is a 4D dataset where the first two dimensions
                    are the diffraction dimensions and the next two 
                    dimensions are the scan dimensions
    ROI:            ndarray of dtype bool
                    Region of interest
    center_disk:    ndarray
                    The blank diffraction disk template where
                    it is 1 inside the circle and 0 outside
    disk_list:      ndarray of shape (n,2)
                    X and Y positions where n is the number of positions.
                    These are the initial guesses that will be refined
    pos_list:       ndarray of shape (n,2)
                    a and b Miller indices corresponding to the
                    disk positions
    reference_axes: ndarray
                    The unit cell axes from the reference region. Strain is
                    calculated by comapring the axes at a scan position with 
                    the reference axes values. If it is 0, then the average 
                    NBED axes will be calculated and will be used as the 
                    reference axes.
    med_factor:     float
                    Due to detector noise, some stray pixels may often be brighter 
                    than the background. This is used for damping any such pixels.
                    Default is 30
    gauss_val:      float
                    The standard deviation of the Gaussian filter applied to the
                    logarithm of the CBED pattern. Default is 3
    hybrid_cc:      float
                    Hybridization parameter to be used for cross-correlation.
                    Default is 0.1
    nan_cutoff:     float
                    Optional parameter that is used for thresholding disk
                    fits. If the intensity ratio is below the threshold 
                    the position will not be fit. Default value is 0.5    
    
    Returns
    -------
    e_xx_map: ndarray
              Strain in the xx direction in the region of interest
    e_xy_map: ndarray
              Strain in the xy direction in the region of interest
    e_th_map: ndarray
              Angular strain in the region of interest
    e_yy_map: ndarray
              Strain in the yy direction in the region of interest
    fit_std: ndarray
             x and y deviations in axes fitting for the scan points
    
    Notes
    -----
    At every scan position, the diffraction disk is filtered by first taking
    the log of the CBED pattern, and then by applying a Gaussian filter. 
    Following this the Sobel of the filtered dataset is calculated. 
    The intensity of the Sobel, Gaussian and Log filtered CBED data is then
    inspected for outlier pixels. If pixel intensities are higher or lower than
    a threshold of the median pixel intensity, they are replaced by the threshold
    value. This is then hybrid cross-correlated with the Sobel magnitude of the 
    template disk. If the pattern axes return a numerical value, then the strain
    is calculated for that scan position, else it is NaN
                 
    :Authors:
    Debangshu Mukherjee <mukherjeed@ornl.gov>
    """
    warnings.filterwarnings('ignore')
    # Calculate needed values
    scan_y,scan_x = np.mgrid[0:data4D.shape[2],0:data4D.shape[3]]
    data4D_ROI = data4D[:,:,scan_y[ROI],scan_x[ROI]]
    no_of_disks = data4D_ROI.shape[-1]
    disk_size = (np.sum(iu.image_normalizer(center_disk))/np.pi) ** 0.5
    i_matrix = (np.eye(2)).astype(np.float64)
    sobel_center_disk,_ = sc.sobel(center_disk)
    # Initialize matrices
    e_xx_ROI = np.nan*(np.ones(no_of_disks,dtype=np.float64))
    e_xy_ROI = np.nan*(np.ones(no_of_disks,dtype=np.float64))
    e_th_ROI = np.nan*(np.ones(no_of_disks,dtype=np.float64))
    e_yy_ROI = np.nan*(np.ones(no_of_disks,dtype=np.float64))
    fit_std = np.nan*(np.ones((no_of_disks,2),dtype=np.float64))
    e_xx_map = np.nan*np.ones_like(scan_y)
    e_xy_map = np.nan*np.ones_like(scan_y)
    e_th_map = np.nan*np.ones_like(scan_y)
    e_yy_map = np.nan*np.ones_like(scan_y)
    #Calculate for mean CBED if no reference
    #axes present
    if np.size(reference_axes) < 2:
        mean_cbed = np.mean(data4D_ROI,axis=-1)
        sobel_lm_cbed,_ = sc.sobel(iu.image_logarizer(mean_cbed))
        sobel_lm_cbed[sobel_lm_cbed > med_factor*np.median(sobel_lm_cbed)] = np.median(sobel_lm_cbed)
        lsc_mean = iu.cross_corr(sobel_lm_cbed,sobel_center_disk,hybridizer=hybrid_cc)
        _,_,_,mean_axes = fit_nbed_disks(lsc_mean,disk_size,disk_list,pos_list)
        inverse_axes = np.linalg.inv(mean_axes)
    else:
        inverse_axes = np.linalg.inv(reference_axes)
    for ii in range(int(no_of_disks)):
        pattern = data4D_ROI[:,:,ii]
        sobel_log_pattern,_ = sc.sobel(scnd.gaussian_filter(iu.image_logarizer(pattern),gauss_val))
        sobel_log_pattern[sobel_log_pattern > med_factor*np.median(sobel_log_pattern)] = np.median(sobel_log_pattern)*med_factor
        sobel_log_pattern[sobel_log_pattern < np.median(sobel_log_pattern)/med_factor] = np.median(sobel_log_pattern)/med_factor
        lsc_pattern = iu.cross_corr(sobel_log_pattern,sobel_center_disk,hybridizer=hybrid_cc)
        _,_,std,pattern_axes = fit_nbed_disks(lsc_pattern,disk_size,disk_list,pos_list,nan_cutoff)
        if ~(np.isnan(np.ravel(pattern_axes))[0]):
            fit_std[ii,:] = std
            t_pattern = np.matmul(pattern_axes,inverse_axes)
            s_pattern = t_pattern - i_matrix
            e_xx_ROI[ii] = -s_pattern[0,0]
            e_xy_ROI[ii] = -(s_pattern[0,1] + s_pattern[1,0])
            e_th_ROI[ii] = s_pattern[0,1] - s_pattern[1,0]
            e_yy_ROI[ii] = -s_pattern[1,1]
    e_xx_map[ROI] = e_xx_ROI
    e_xx_map[np.isnan(e_xx_map)] = 0
    e_xx_map = scnd.gaussian_filter(e_xx_map,1)
    e_xy_map[ROI] = e_xx_ROI
    e_xy_map[np.isnan(e_xy_map)] = 0
    e_xy_map = scnd.gaussian_filter(e_xy_map,1)
    e_th_map[ROI] = e_th_ROI
    e_th_map[np.isnan(e_th_map)] = 0
    e_th_map = scnd.gaussian_filter(e_th_map,1)
    e_yy_map[ROI] = e_yy_ROI
    e_yy_map[np.isnan(e_yy_map)] = 0
    e_yy_map = scnd.gaussian_filter(e_yy_map,1)
    return e_xx_map,e_xy_map,e_th_map,e_yy_map,fit_std

@numba.jit
def strain_log(data4D_ROI,
               center_disk,
               disk_list,
               pos_list,
               reference_axes=0,
               med_factor=10):
    warnings.filterwarnings('ignore')
    # Calculate needed values
    no_of_disks = data4D_ROI.shape[-1]
    disk_size = (np.sum(center_disk)/np.pi) ** 0.5
    i_matrix = (np.eye(2)).astype(np.float64)
    # Initialize matrices
    e_xx_log = np.zeros(no_of_disks,dtype=np.float64)
    e_xy_log = np.zeros(no_of_disks,dtype=np.float64)
    e_th_log = np.zeros(no_of_disks,dtype=np.float64)
    e_yy_log = np.zeros(no_of_disks,dtype=np.float64)
    #Calculate for mean CBED if no reference
    #axes present
    if np.size(reference_axes) < 2:
        mean_cbed = np.mean(data4D_ROI,axis=-1)
        log_cbed,_ = iu.image_logarizer(mean_cbed)
        log_cc_mean = iu.cross_corr(log_cbed,center_disk,hybridizer=0.1)
        _,_,mean_axes = fit_nbed_disks(log_cc_mean,disk_size,disk_list,pos_list)
        inverse_axes = np.linalg.inv(mean_axes)
    else:
        inverse_axes = np.linalg.inv(reference_axes)
    for ii in range(int(no_of_disks)):
        pattern = data4D_ROI[:,:,ii]
        log_pattern,_ = iu.image_logarizer(pattern)
        log_cc_pattern = iu.cross_corr(log_pattern,center_disk,hybridizer=0.1)
        _,_,pattern_axes = fit_nbed_disks(log_cc_pattern,disk_size,disk_list,pos_list)
        t_pattern = np.matmul(pattern_axes,inverse_axes)
        s_pattern = t_pattern - i_matrix
        e_xx_log[ii] = -s_pattern[0,0]
        e_xy_log[ii] = -(s_pattern[0,1] + s_pattern[1,0])
        e_th_log[ii] = s_pattern[0,1] - s_pattern[1,0]
        e_yy_log[ii] = -s_pattern[1,1]
    return e_xx_log,e_xy_log,e_th_log,e_yy_log

@numba.jit
def strain_oldstyle(data4D_ROI,
                    center_disk,
                    disk_list,
                    pos_list,
                    reference_axes=0):
    warnings.filterwarnings('ignore')
    # Calculate needed values
    no_of_disks = data4D_ROI.shape[-1]
    disk_size = (np.sum(center_disk)/np.pi) ** 0.5
    i_matrix = (np.eye(2)).astype(np.float64)
    # Initialize matrices
    e_xx_ROI = np.zeros(no_of_disks,dtype=np.float64)
    e_xy_ROI = np.zeros(no_of_disks,dtype=np.float64)
    e_th_ROI = np.zeros(no_of_disks,dtype=np.float64)
    e_yy_ROI = np.zeros(no_of_disks,dtype=np.float64)
    #Calculate for mean CBED if no reference
    #axes present
    if np.size(reference_axes) < 2:
        mean_cbed = np.mean(data4D_ROI,axis=-1)
        cc_mean = iu.cross_corr(mean_cbed,center_disk,hybridizer=0.1)
        _,_,mean_axes = fit_nbed_disks(cc_mean,disk_size,disk_list,pos_list)
        inverse_axes = np.linalg.inv(mean_axes)
    else:
        inverse_axes = np.linalg.inv(reference_axes)
    for ii in range(int(no_of_disks)):
        pattern = data4D_ROI[:,:,ii]
        cc_pattern = iu.cross_corr(pattern,center_disk,hybridizer=0.1)
        _,_,pattern_axes = fit_nbed_disks(cc_pattern,disk_size,disk_list,pos_list)
        t_pattern = np.matmul(pattern_axes,inverse_axes)
        s_pattern = t_pattern - i_matrix
        e_xx_ROI[ii] = -s_pattern[0,0]
        e_xy_ROI[ii] = -(s_pattern[0,1] + s_pattern[1,0])
        e_th_ROI[ii] = s_pattern[0,1] - s_pattern[1,0]
        e_yy_ROI[ii] = -s_pattern[1,1]
    return e_xx_ROI,e_xy_ROI,e_th_ROI,e_yy_ROI

def ROI_strain_map(strain_ROI,
                   ROI):
    """
    Convert the strain in the ROI array to a strain map
                 
    :Authors:
    Debangshu Mukherjee <mukherjeed@ornl.gov>
    """
    strain_map = np.zeros_like(ROI,dtype=np.float64)
    strain_map[ROI] = (strain_ROI).astype(np.float64)
    return strain_map

def log_sobel(data4D):
    data_lsb = np.zeros_like(data4D,dtype=np.float64)
    for jj in range(data4D.shape[3]):
        for ii in range(data4D.shape[2]):
            data_lsb[:,:,ii,jj],_ = sc.sobel(iu.image_logarizer(data4D[:,:,ii,jj]))
    return data_lsb

def spectra_finder(data4D,yvals,xvals):
    spectra_data = np.ravel(np.mean(data4D[:,:,yvals[0]:yvals[1],xvals[0]:xvals[1]],axis=(-1,-2),dtype=np.float64))
    data_im = np.sum(data4D,axis=(0,1))
    data_im = (data_im - np.amin(data_im))/(np.amax(data_im) - np.amin(data_im))
    overlay = np.zeros_like(data_im)
    overlay[yvals[0]:yvals[1],xvals[0]:xvals[1]] = 1
    return spectra_data,0.5*(data_im+overlay)