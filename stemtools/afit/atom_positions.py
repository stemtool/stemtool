from skimage import feature as skf
import matplotlib.pyplot as plt
import numpy as np
import numba
from scipy import ndimage as scnd
from scipy import optimize as spo
import pyfftw
import warnings
from ..util import gauss_utils as gt

def peaks_vis(data_image,
              distance=1,
              threshold=0.1,
              imsize=(20,20)):
    """
    Find atom maxima pixels in images
    
    Parameters
    ----------
    data_image: ndarray
                Original atomic resolution image
    distance:   float
                Average distance between neighboring peaks
    threshold:  float
                The cutoff intensity value below which a peak 
                will not be detected
    imsize:     ndarray
                Size of the display image
    
    Returns
    -------
    peaks: ndarray
           List of peak positions as y, x
    
    Notes
    -----
    This is a wrapper around the skimage peak finding
    module which finds the peaks with a given threshold
    value and an inter-peak separation. The function
    additionally plots the peak positions on the original
    image thus users can modify the input values of
    threshold and distance to ensure the right peaks are
    selected.
    
    :Authors:
    Debangshu Mukherjee <mukherjeed@ornl.gov>
    """
    data_image = data_image - np.amin(data_image)
    data_image = data_image / np.amax(data_image)
    data_image[data_image < (0.25*threshold)] = 0
    peaks = skf.peak_local_max(data_image,min_distance=distance,threshold_abs=threshold)
    plt.figure(figsize=imsize)
    plt.imshow(data_image)
    plt.scatter(peaks[:,1],peaks[:,0],c='g', s=15)
    return peaks

@numba.jit
def refine_atoms(image_data,
                 positions,
                 distance):
    warnings.filterwarnings('ignore')
    no_of_points = positions.shape[0]
    refined_pos = (np.zeros((no_of_points,6))).astype(float)
    for ii in range(no_of_points):
        pos_x = (positions[ii,1]).astype(float)
        pos_y = (positions[ii,0]).astype(float)
        fitted_diff = gt.fit_gaussian2D_mask(1+image_data,pos_x,pos_y,distance)
        refined_pos[ii,1] = fitted_diff[0]
        refined_pos[ii,0] = fitted_diff[1]
        refined_pos[ii,2:6] = fitted_diff[2:6]
        refined_pos[ii,-1] = fitted_diff[-1] - 1
    return refined_pos

def fourier_mask(original_image,
                 center,
                 radius,
                 threshold=0.2):
    pyfftw.interfaces.cache.enable()
    image_fourier = pyfftw.interfaces.scipy_fftpack.fftshift(pyfftw.interfaces.scipy_fftpack.fft2(original_image))
    pos_x = center[0]
    pos_y = center[1]
    blurred_image = scnd.filters.gaussian_filter(np.abs(image_fourier),3)
    fitted_diff = gt.fit_gaussian2D_mask(blurred_image,pos_x,pos_y,radius)
    new_x = fitted_diff[0]
    new_y = fitted_diff[1]
    new_center = np.asarray((new_x,new_y))
    size_image = np.asarray(np.shape(image_fourier),dtype=int)
    yV, xV = np.mgrid[0:size_image[0], 0:size_image[1]]
    sub = ((((yV - new_y) ** 2) + ((xV - new_x) ** 2)) ** 0.5) < radius
    circle = (np.asarray(sub)).astype('float')
    filtered_circ = scnd.filters.gaussian_filter(circle,3)
    masked_image = np.multiply(image_fourier,filtered_circ)
    SAED_image = pyfftw.interfaces.scipy_fftpack.ifft2(masked_image)
    mag_SAED = np.abs(SAED_image)
    mag_SAED = (mag_SAED - np.amin(mag_SAED))/(np.amax(mag_SAED) - np.amin(mag_SAED))
    mag_SAED[mag_SAED < threshold] = 0
    mag_SAED[mag_SAED > threshold] = 1
    filtered_SAED = scnd.filters.gaussian_filter(mag_SAED,3)
    fourier_selected_image = np.multiply(original_image,filtered_SAED)
    return fourier_selected_image, SAED_image, new_center, filtered_SAED

def find_diffraction_spots(image,
                           circ_c,
                           circ_y,
                           circ_x):
    """
    Find the diffraction spots visually.
    
    Parameters
    ----------
    image:  ndarray
            Original image
    circ_c: ndarray
            Position of the central beam in
            the Fourier pattern
    circ_y: ndarray
            Position of the y beam in
            the Fourier pattern
    circ_x: ndarray
            Position of the x beam in
            the Fourier pattern
    
    
    Notes
    -----
    Put circles in red(central), y(blue) and x(green) 
    on the diffraction pattern to approximately know
    the positions.
    
    :Authors:
    Debangshu Mukherjee <mukherjeed@ornl.gov>
    """
    pyfftw.interfaces.cache.enable()
    image_ft = pyfftw.interfaces.scipy_fftpack.fftshift(pyfftw.interfaces.scipy_fftpack.fft2(image))
    log_abs_ft = scnd.filters.gaussian_filter(np.log10(np.abs(image_ft)),3)
    f, ax = plt.subplots(figsize=(20, 20))
    circ_c_im = plt.Circle(circ_c, 15,color="red",alpha=0.33)
    circ_y_im = plt.Circle(circ_y, 15,color="blue",alpha=0.33)
    circ_x_im = plt.Circle(circ_x, 15,color="green",alpha=0.33)
    ax.imshow(log_abs_ft,cmap='gray')
    ax.add_artist(circ_c_im)
    ax.add_artist(circ_y_im)
    ax.add_artist(circ_x_im)
    plt.show()

def find_coords(image,
                fourier_center,
                fourier_y,
                fourier_x, 
                y_axis, 
                x_axis):
    """
    Convert the fourier positions to image axes.
    Do not use numba to accelerate as LLVM IR
    throws an error from the the if statements
    
    Parameters
    ----------
    image:  ndarray
            Original image
    four_c: ndarray
            Position of the central beam in
            the Fourier pattern
    four_y: ndarray
            Position of the y beam in
            the Fourier pattern
    four_x: ndarray
            Position of the x beam in
            the Fourier pattern
    
    Returns
    -------
    coords: ndarray
            Axes co-ordinates in the real image,
            as [y1 x1
                y2 x2]
    
    Notes
    -----
    Use the fourier coordinates to define the axes 
    co-ordinates in real space, which will be used 
    to assign each atom position to a axes position
    
    :Authors:
    Debangshu Mukherjee <mukherjeed@ornl.gov>
    """
    image_size = image.shape
    qx = (np.arange((-image_size[1] / 2),(image_size[1] / 2), 1)) / image_size[1]
    qy = (np.arange((-image_size[0] / 2),(image_size[0] / 2), 1)) / image_size[0]
    dx = np.mean(np.diff(qx))
    dy = np.mean(np.diff(qy))
    fourier_calib = ((dy,dx))
    y_axis[y_axis==0] = 1
    x_axis[x_axis==0] = 1
    distance_y = fourier_y - fourier_center
    distance_y = np.divide(distance_y,y_axis[0:2])
    distance_x = fourier_x - fourier_center
    distance_x = np.divide(distance_x,x_axis[0:2])
    fourier_length_y = distance_y * fourier_calib
    fourier_length_x = distance_x * fourier_calib
    angle_y = np.degrees(np.arccos(fourier_length_y[0]/np.linalg.norm(fourier_length_y)))
    real_y_size = 1/np.linalg.norm(fourier_length_y)
    angle_x = np.degrees(np.arccos(fourier_length_x[0]/np.linalg.norm(fourier_length_x)))
    real_x_size = 1/np.linalg.norm(fourier_length_x)
    real_y = (((real_y_size*np.cos(np.deg2rad(angle_y - 90))),(real_y_size*np.sin(np.deg2rad(angle_y - 90)))))
    real_x = (((real_x_size*np.cos(np.deg2rad(angle_x - 90))),(real_x_size*np.sin(np.deg2rad(angle_x - 90)))))
    coords = np.asarray(((real_y[1],real_y[0]),(real_x[1],real_x[0])))
    if (np.amax(np.abs(coords[0,:])) > np.amax(coords[0,:])):
        coords[0,:] = (-1) * coords[0,:]
    if (np.amax(np.abs(coords[1,:])) > np.amax(coords[1,:])):
        coords[1,:] = (-1) * coords[1,:]
    return coords

def get_origin(image,
               peak_pos,
               coords):
    def origin_function(xyCenter,input_data=(peak_pos,coords)):
        peaks = input_data[0]
        coords = input_data[1]
        atom_coords = np.zeros((peaks.shape[0],6))
        atom_coords[:,0:2] = peaks[:,0:2] - xyCenter[0:2]
        atom_coords[:,2:4] = atom_coords[:,0:2] @ np.linalg.inv(coords)
        atom_coords[:,4:6] = np.round(atom_coords[:,2:4])
        average_deviation = (((np.mean(np.abs(atom_coords[:,3] - atom_coords[:,5]))) ** 2) + 
                             ((np.mean(np.abs(atom_coords[:,2] - atom_coords[:,4]))) ** 2)) ** 0.5
        return average_deviation
    initial_x = image.shape[1]/2
    initial_y = image.shape[0]/2
    initial_guess = np.asarray((initial_y,initial_x))
    lower_bound = np.asarray(((initial_y-initial_y/2),(initial_x-initial_x/2)))
    upper_bound = np.asarray(((initial_y+initial_y/2),(initial_x+initial_x/2)))
    res = spo.minimize(fun=origin_function, x0=initial_guess,bounds=(lower_bound,upper_bound))
    origin = res.x
    return origin

def get_coords(image,
               peak_pos,
               origin,
               current_coords):
    ang_1 = np.degrees(np.arctan2(current_coords[0,1],current_coords[0,0]))
    mag_1 = np.linalg.norm((current_coords[0,1],current_coords[0,0]))
    ang_2 = np.degrees(np.arctan2(current_coords[1,1],current_coords[1,0]))
    mag_2 = np.linalg.norm((current_coords[1,1],current_coords[1,0]))
    def coords_function(coord_vals,input_data=(peak_pos,origin,ang_1,ang_2)):
        mag_t = coord_vals[0]
        mag_b = coord_vals[1]
        peaks = input_data[0]
        rigin = input_data[1]
        ang_t = input_data[2]
        ang_b = input_data[3]
        xy_coords = np.asarray(((mag_t*np.cos(np.deg2rad(ang_t)),mag_t*np.sin(np.deg2rad(ang_t))),
                                (mag_b*np.cos(np.deg2rad(ang_b)),mag_b*np.sin(np.deg2rad(ang_b)))))
        atom_coords = np.zeros((peaks.shape[0],6))
        atom_coords[:,0:2] = peaks[:,0:2] - rigin[0:2]
        atom_coords[:,2:4] = atom_coords[:,0:2] @ np.linalg.inv(xy_coords)
        atom_coords[:,4:6] = np.round(atom_coords[:,2:4])
        average_deviation = (((np.mean(np.abs(atom_coords[:,3] - atom_coords[:,5]))) ** 2) + 
                             ((np.mean(np.abs(atom_coords[:,2] - atom_coords[:,4]))) ** 2)) ** 0.5
        return average_deviation
    initial_guess = np.asarray((mag_1,mag_2))
    lower_bound = initial_guess - (0.25*initial_guess)
    upper_bound = initial_guess + (0.25*initial_guess)
    res = spo.minimize(fun=coords_function, x0=initial_guess,bounds=(lower_bound,upper_bound))
    mag = res.x
    new_coords = np.asarray(((mag[0]*np.cos(np.deg2rad(ang_1)),mag[0]*np.sin(np.deg2rad(ang_1))),
                             (mag[1]*np.cos(np.deg2rad(ang_2)),mag[1]*np.sin(np.deg2rad(ang_2)))))
    return new_coords

def coords_of_atoms(peaks,
                    coords,
                    origin):
    """
    Convert atom positions to coordinates
    
    Parameters
    ----------
    peaks:  ndarray
            List of Gaussian fitted peaks
    coords: ndarray
            Co-ordinates of the axes
    
    Returns
    -------
    atom_coords: ndarray
                 Peak positions as the atom coordinates
    
    Notes
    -----
    One atom is chosen as the origin and the co-ordinates
    of all the atoms are calculated with respect to the origin
    
    :Authors:
    Debangshu Mukherjee <mukherjeed@ornl.gov>
    """
    atom_coords = np.zeros((peaks.shape[0],8))
    atom_coords[:,0:2] = peaks[:,0:2] - origin[0:2]
    atom_coords[:,2:4] = atom_coords[:,0:2] @ np.linalg.inv(coords)
    atom_coords[:,0:2] = peaks[:,0:2]
    atom_coords[:,4:6] = np.round(atom_coords[:,2:4])
    atom_coords[:,6:8] = atom_coords[:,4:6] @ coords
    atom_coords[:,6:8] = atom_coords[:,6:8] + origin[0:2]
    return atom_coords

@numba.jit
def three_neighbors(peak_list,
                    coords,
                    delta=0.25):
    warnings.filterwarnings('ignore')
    no_atoms = peak_list.shape[0]
    atoms_neighbors = np.zeros((no_atoms,8))
    atoms_distances = np.zeros((no_atoms,4))
    for ii in range(no_atoms):
        atom_pos = peak_list[ii,0:2]
        neigh_yy = atom_pos + coords[0,:]
        neigh_xx = atom_pos + coords[1,:]
        neigh_xy = atom_pos + coords[0,:] + coords[1,:]
        parnb_yy = ((peak_list[:,0] - neigh_yy[0]) ** 2) + ((peak_list[:,1] - neigh_yy[1]) ** 2)
        neigh_yy = (peak_list[parnb_yy==np.amin(parnb_yy),0:2])[0]
        ndist_yy = np.linalg.norm(neigh_yy - atom_pos)
        parnb_xx = ((peak_list[:,0] - neigh_xx[0]) ** 2) + ((peak_list[:,1] - neigh_xx[1]) ** 2)
        neigh_xx = (peak_list[parnb_xx==np.amin(parnb_xx),0:2])[0]
        ndist_xx = np.linalg.norm(neigh_xx - atom_pos)
        parnb_xy = ((peak_list[:,0] - neigh_xy[0]) ** 2) + ((peak_list[:,1] - neigh_xy[1]) ** 2)
        neigh_xy = (peak_list[parnb_xy==np.amin(parnb_xy),0:2])[0]
        ndist_xy = np.linalg.norm(neigh_xy - atom_pos)
        atoms_neighbors[ii,:] = np.ravel(np.asarray((atom_pos,neigh_yy,neigh_xx,neigh_xy)))
        atoms_distances[ii,:] = np.ravel(np.asarray((0,ndist_yy,ndist_xx,ndist_xy)))
    yy_dist = np.linalg.norm(coords[0,:])
    yy_list = np.asarray(((yy_dist*(1-delta)),(yy_dist*(1+delta))))
    xx_dist = np.linalg.norm(coords[1,:])
    xx_list = np.asarray(((xx_dist*(1-delta)),(xx_dist*(1+delta))))
    xy_dist = np.linalg.norm(coords[0,:] + coords[1,:])
    xy_list = np.asarray(((xy_dist*(1-delta)),(xy_dist*(1+delta))))
    pp = atoms_distances[:,1]
    pp[pp > yy_list[1]] = 0
    pp[pp < yy_list[0]] = 0
    pp[pp==0] = np.nan
    atoms_distances[:,1] = pp
    pp = atoms_distances[:,2]
    pp[pp > xx_list[1]] = 0
    pp[pp < xx_list[0]] = 0
    pp[pp==0] = np.nan
    atoms_distances[:,2] = pp
    pp = atoms_distances[:,3]
    pp[pp > xy_list[1]] = 0
    pp[pp < xy_list[0]] = 0
    pp[pp==0] = np.nan
    atoms_distances[:,3] = pp
    atoms_neighbors = atoms_neighbors[~np.isnan(atoms_distances).any(axis=1)]
    atoms_distances = atoms_distances[~np.isnan(atoms_distances).any(axis=1)]
    return atoms_neighbors, atoms_distances

@numba.jit
def relative_strain(n_list,
                    coords):
    warnings.filterwarnings('ignore')
    identity = np.asarray(((1,0),
                           (0,1)))
    axis_pos = np.asarray(((0, 0), 
                           (1, 0), 
                           (0, 1), 
                           (1, 1)))
    no_atoms = (np.shape(n_list))[0]
    coords_inv = np.linalg.inv(coords)
    cell_center = np.zeros((no_atoms,2))
    e_xx = np.zeros(no_atoms)
    e_xy = np.zeros(no_atoms)
    e_yy = np.zeros(no_atoms)
    e_th = np.zeros(no_atoms)
    for ii in range(no_atoms):
        cc = np.zeros((4,2))
        cc[0,:] = n_list[ii,0:2] - n_list[ii,0:2]
        cc[1,:] = n_list[ii,2:4] - n_list[ii,0:2]
        cc[2,:] = n_list[ii,4:6] - n_list[ii,0:2]
        cc[3,:] = n_list[ii,6:8] - n_list[ii,0:2]
        l_cc, _,_,_ = np.linalg.lstsq(axis_pos,cc,rcond=None)
        t_cc = np.matmul(l_cc,coords_inv) - identity
        e_yy[ii] = t_cc[0,0]
        e_xx[ii] = t_cc[1,1]
        e_xy[ii] = 0.5*(t_cc[0,1] + t_cc[1,0])
        e_th[ii] = 0.5*(t_cc[0,1] - t_cc[1,0])
        cell_center[ii,0] = 0.25*(n_list[ii,0] + n_list[ii,2] + n_list[ii,4] + n_list[ii,6])
        cell_center[ii,1] = 0.25*(n_list[ii,1] + n_list[ii,3] + n_list[ii,5] + n_list[ii,7])
    return cell_center, e_xx, e_xy, e_yy, e_th