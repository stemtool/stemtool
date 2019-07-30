from scipy import ndimage as scnd
from scipy import optimize as sio
import numpy as np
import numba
import warnings
from ..util import image_utils as iu
from ..proc import sobel_canny as sc
from ..util import gauss_utils as gt

@numba.jit
def fit_nbed_disks(corr_image,disk_size,positions,diff_spots):
    warnings.filterwarnings('ignore')
    positions = np.asarray(positions,dtype=np.float64)
    diff_spots = np.asarray(diff_spots,dtype=np.float64)
    fitted_disk_list = np.zeros_like(positions)
    disk_locations = np.zeros_like(positions)
    for ii in range(int(np.shape(positions)[0])):
        posx = positions[ii,0]
        posy = positions[ii,1]
        par = gt.fit_gaussian2D_mask(corr_image,posx,posy,disk_size)
        fitted_disk_list[ii,0] = par[0]
        fitted_disk_list[ii,1] = par[1]
    disk_locations = np.copy(fitted_disk_list)
    disk_locations[:,1] = 0 - disk_locations[:,1]
    center = disk_locations[np.logical_and((diff_spots[:,0] == 0),(diff_spots[:,1] == 0)),:]
    center = np.asarray(center)
    disk_locations[:,0:2] = disk_locations[:,0:2] - center[0:2]
    lcbed,_,_,_ = np.linalg.lstsq(diff_spots,disk_locations,rcond=None)
    center[0,1] = (-1)*center[0,1]
    return fitted_disk_list,center,lcbed

@numba.jit
def strain_and_disk(data4D,disk_size,pixel_list_xy,disk_list,med_factor=15):
    warnings.filterwarnings('ignore')
    # Calculate needed values
    scan_size = np.asarray(data4D.shape)[2:4]
    sy,sx = np.mgrid[0:scan_size[0],0:scan_size[1]]
    scan_positions = (np.asarray((np.ravel(sy),np.ravel(sx)))).astype(int)
    cbed_size = np.asarray(data4D.shape)[0:2]
    center_disk = (iu.make_circle(cbed_size,cbed_size[1]/2,cbed_size[0]/2,disk_size)).astype(np.float64)
    i_matrix = (np.eye(2)).astype(np.float64)
    sobel_center_disk,_ = sc.sobel(center_disk)
    # Initialize matrices
    e_xx = np.zeros(scan_size,dtype=np.float64)
    e_xy = np.zeros(scan_size,dtype=np.float64)
    e_th = np.zeros(scan_size,dtype=np.float64)
    e_yy = np.zeros(scan_size,dtype=np.float64)
    dpc_x = np.zeros(scan_size,dtype=np.float64)
    dpc_y = np.zeros(scan_size,dtype=np.float64)
    #Calculate for mean CBED if no reference
    mean_cbed = np.mean(np.mean(data4D,axis=-1),axis=-1)
    mean_ls_cbed,_ = sc.sobel(iu.image_logarizer(mean_cbed))
    mean_ls_cbed[mean_ls_cbed > med_factor*np.median(mean_ls_cbed)] = np.median(mean_ls_cbed)
    mean_lsc = iu.cross_corr(mean_ls_cbed,sobel_center_disk,hybridizer=0.1)
    _,mean_center,mean_axes = fit_nbed_disks(mean_lsc,disk_size,pixel_list_xy,disk_list)
    inverse_axes = np.linalg.inv(mean_axes)
    for pp in range(np.size(sy)):
        ii = scan_positions[0,pp]
        jj = scan_positions[1,pp]
        pattern = data4D[:,:,ii,jj]
        pattern_ls,_ = sc.sobel(iu.image_logarizer(pattern))
        pattern_ls[pattern_ls > med_factor*np.median(pattern_ls)] = np.median(pattern_ls)
        pattern_lsc = iu.cross_corr(pattern_ls,sobel_center_disk,hybridizer=0.1)
        _,pattern_center,pattern_axes = fit_nbed_disks(pattern_lsc,disk_size,pixel_list_xy,disk_list)
        t_pattern = np.matmul(pattern_axes,inverse_axes)
        s_pattern = t_pattern - i_matrix
        e_xx[ii,jj] = -s_pattern[0,0]
        e_xy[ii,jj] = -(s_pattern[0,1] + s_pattern[1,0])
        e_th[ii,jj] = -(s_pattern[0,1] - s_pattern[1,0])
        e_yy[ii,jj] = -s_pattern[1,1]
        dpc_x[ii,jj] = pattern_center[0,0] - mean_center[0,0]
        dpc_y[ii,jj] = pattern_center[0,1] - mean_center[0,1]
    return e_xx,e_xy,e_th,e_yy,dpc_x,dpc_y