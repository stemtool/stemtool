import numpy as np
import numba
from scipy import optimize as spo
from scipy import ndimage as scnd
import image_utils as iu

@numba.jit(parallel=True,cache=True)
def gaussian_2D_function(xy, x0, y0, theta, sigma_x, sigma_y, amplitude):
    x = xy[0] - x0
    y = xy[1] - y0
    term_1 = (((np.cos(theta))**2)/(2*(sigma_x**2))) + (((np.sin(theta))**2)/(2*(sigma_y**2)))
    term_2 = ((np.sin(2*theta))/(2*(sigma_x**2))) - ((np.sin(2*theta))/(2*(sigma_y**2)))
    term_3 = (((np.sin(theta))**2)/(2*(sigma_x**2))) + (((np.cos(theta))**2)/(2*(sigma_y**2)))
    expo_1 = term_1 * np.multiply(x,x)
    expo_2 = term_2 * np.multiply(x,y)
    expo_3 = term_3 * np.multiply(y,y)
    gauss2D = amplitude * np.exp((-1)*(expo_1 + expo_2 + expo_3))
    return np.ravel(gauss2D)

@numba.jit(parallel=True,cache=True)
def gauss2D(image_size, x0, y0, theta, sigma_x, sigma_y, amplitude):
    xr, yr = np.meshgrid(np.arange(image_size[1]),np.arange(image_size[0]))
    x = xr - x0
    y = yr - y0
    term_1 = (((np.cos(theta))**2)/(2*(sigma_x**2))) + (((np.sin(theta))**2)/(2*(sigma_y**2)))
    term_2 = ((np.sin(2*theta))/(2*(sigma_x**2))) - ((np.sin(2*theta))/(2*(sigma_y**2)))
    term_3 = (((np.sin(theta))**2)/(2*(sigma_x**2))) + (((np.cos(theta))**2)/(2*(sigma_y**2)))
    expo_1 = term_1 * np.multiply(x,x)
    expo_2 = term_2 * np.multiply(x,y)
    expo_3 = term_3 * np.multiply(y,y)
    gauss2D = amplitude * np.exp((-1)*(expo_1 + expo_2 + expo_3))
    return gauss2D

@numba.jit(parallel=True,cache=True)
def initialize_gauss(xx, yy, zz, center_type='COM'):
    if (center_type == 'maxima'):
        x_com = xx[zz == np.amax(zz)]
        y_com = yy[zz == np.amax(zz)]
    elif (center_type == 'COM'):
        total = zz.sum()
        x_com = ((xx*zz).sum())/total
        y_com = ((yy*zz).sum())/total
    zz_norm = iu.image_normalizer(zz)
    x_fwhm = xx[zz_norm > 0.5]
    x_fwhm = np.abs(x_fwhm - x_com)
    y_fwhm = yy[zz_norm > 0.5]
    y_fwhm = np.abs(y_fwhm - y_com)
    sigma_x = np.amax(x_fwhm)/(2*((2*np.log(2)) ** 0.5))
    sigma_y = np.amax(y_fwhm)/(2*((2*np.log(2)) ** 0.5))
    height = np.amax(zz)
    return x_com, y_com, 0, sigma_x, sigma_y, height

@numba.jit(parallel=True,cache=True)
def process_circul_mask(image_data,mask_x,mask_y,mask_radius):
    p,q = np.shape(image_data)
    yV, xV = np.mgrid[0:p, 0:q]
    sub = ((((yV - mask_y) ** 2) + ((xV - mask_x) ** 2)) ** 0.5) < mask_radius
    xValues = np.asarray(xV[sub]).astype('float')
    yValues = np.asarray(yV[sub]).astype('float')
    zValues = np.asarray(image_data[sub]).astype('float')
    return xValues, yValues, zValues

@numba.jit(parallel=True,cache=True)
def process_square_mask(image_data,mask_x,mask_y,mask_size):
    p,q = np.shape(image_data)
    yV, xV = np.mgrid[0:p, 0:q]
    sub = np.logical_and((np.abs(yV - mask_y) < mask_size),(np.abs(xV - mask_x) < mask_size))
    xValues = np.asarray(xV[sub]).astype('float')
    yValues = np.asarray(yV[sub]).astype('float')
    zValues = np.asarray(image_data[sub]).astype('float')
    return xValues, yValues, zValues

@numba.jit(parallel=True,cache=True)
def fit_gaussian2D_mask(image_data,mask_x,mask_y,mask_radius,mask_type='circular',center_type='COM'):
    if (mask_type == 'circular'):
        x_pos, y_pos, masked_image = process_circul_mask(image_data,mask_x,mask_y,mask_radius)
    else:
        x_pos, y_pos, masked_image = process_square_mask(image_data,mask_x,mask_y,mask_radius)
    mi_min = np.amin(masked_image)
    mi_max = np.amax(masked_image)
    calc_image = (masked_image - mi_min)/(mi_max - mi_min)
    initial_guess = initialize_gauss(x_pos,y_pos,calc_image,center_type)
    lower_bound = ((initial_guess[0]-mask_radius),(initial_guess[1]-mask_radius),
                   -180,0,0,((-2.5)*initial_guess[5]))
    upper_bound = ((initial_guess[0]+mask_radius),(initial_guess[1]+mask_radius),
                   180,(2.5*mask_radius),(2.5*mask_radius),(2.5*initial_guess[5]))
    xy = (x_pos,y_pos)
    popt, _ = spo.curve_fit(gaussian_2D_function, xy, calc_image, initial_guess,
                                   bounds=(lower_bound,upper_bound),
                                   ftol=0.01, xtol=0.01)
    popt[-1] = (popt[-1]*(mi_max - mi_min)) + mi_min
    return popt

@numba.jit(parallel=True,cache=True)
def create_circmask(image,center,radius):
    pos_x = center[0]
    pos_y = center[1]
    blurred_image = scnd.filters.gaussian_filter(np.abs(image),3)
    fitted_diff = fit_gaussian2D_mask(blurred_image,pos_x,pos_y,radius)
    new_x = fitted_diff[0]
    new_y = fitted_diff[1]
    new_center = np.asarray((new_y,new_x))
    size_image = np.asarray(np.shape(image),dtype=int)
    yV, xV = np.mgrid[0:size_image[0], 0:size_image[1]]
    sub = ((((yV - new_y) ** 2) + ((xV - new_x) ** 2)) ** 0.5) < radius
    circle = (np.asarray(sub)).astype('float')
    masked_image = np.multiply(image,circle)
    return masked_image, new_center, circle