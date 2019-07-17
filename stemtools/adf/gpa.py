import numpy as np
import cmath
import pyfftw
from scipy import ndimage as scnd
from scipy import optimize as spo
from ..util import gauss_utils as gt

def phase_diff(angle_image):
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

def disk_phase(image_orig, g_vec):
    image_size = (np.asarray(image_orig.shape)).astype(int)
    fourier_vec_y = np.linspace((-image_size[0] / 2), ((image_size[0] / 2) - 1), image_size[0])
    fourier_vec_y = fourier_vec_y / image_size[0]
    fourier_cal_y = np.mean(np.diff(fourier_vec_y))
    fourier_vec_x = np.linspace((-image_size[1] / 2), ((image_size[1] / 2) - 1), image_size[1])
    fourier_vec_x = fourier_vec_x / image_size[1]
    fourier_cal_x = np.mean(np.diff(fourier_vec_x))
    [fourier_mesh_y, fourier_mesh_x] = np.meshgrid(fourier_vec_x, fourier_vec_y)
    phase_matrix = 2 * np.pi * ((fourier_mesh_x*g_vec[1]) + (fourier_mesh_y*g_vec[0]))
    return phase_matrix

def get_g_vector(image,g_disk,center_disk):
    image_size = np.shape(image)
    fourier_vec_y = np.linspace((-image_size[0] / 2), ((image_size[0] / 2) - 1), image_size[0])
    fourier_vec_y = fourier_vec_y / image_size[0]
    fourier_cal_y = np.mean(np.diff(fourier_vec_y))
    fourier_vec_x = np.linspace((-image_size[1] / 2), ((image_size[1] / 2) - 1), image_size[1])
    fourier_vec_x = fourier_vec_x / image_size[1]
    fourier_cal_x = np.mean(np.diff(fourier_vec_x))
    fourier_cal = np.asarray((fourier_cal_y,fourier_cal_x))
    g_vec = np.multiply(fourier_cal,(g_disk - center_disk))
    return g_vec

def get_a_matrix(g_vector_1,g_vector_2):
    g_matrix = np.asarray((g_vector_1,g_vector_2))
    a_matrix = np.linalg.inv(np.transpose(g_matrix))
    return a_matrix

def phase_subtract(matrix_1,matrix_2):
    complex_1 = np.exp(1j*matrix_1)
    complex_2 = np.exp(1j*matrix_2)
    complex_div = np.divide(complex_1,complex_2)
    subtracted_matrix = np.angle(complex_div)
    return subtracted_matrix

def get_phase_matrix(image,disk_posn,disk_cent):
    pyfftw.interfaces.cache.enable()
    P_disk = gt.gauss2D(image.shape, disk_posn[1], disk_posn[0], 0, 1, 1, 1)
    P_cent = gt.gauss2D(image.shape, disk_cent[1], disk_cent[0], 0, 1, 1, 1)
    F_disk = pyfftw.interfaces.scipy_fftpack.fftshift(pyfftw.interfaces.scipy_fftpack.fft2(P_disk))
    F_cent = pyfftw.interfaces.scipy_fftpack.fftshift(pyfftw.interfaces.scipy_fftpack.fft2(P_cent))
    phase_matrix = phase_subtract(np.angle(F_disk),np.angle(F_cent))
    return phase_matrix

def saed_mask(original_image,center,radius,threshold=0.2):
    pyfftw.interfaces.cache.enable()
    image_fourier = pyfftw.interfaces.scipy_fftpack.fftshift(pyfftw.interfaces.scipy_fftpack.fft2(original_image))
    size_image = np.asarray(np.shape(image_fourier),dtype=int)
    yV, xV = np.mgrid[0:size_image[0], 0:size_image[1]]
    sub = ((((yV - center[0]) ** 2) + ((xV - center[1]) ** 2)) ** 0.5) < radius
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
    return fourier_selected_image, filtered_SAED

def strain_gpa(diff1x,diff1y,diff2x,diff2y,a_matrix):
    sm = np.transpose(np.asarray([[diff1x,diff1y],[diff2x,diff2y]]),axes=(2,3,0,1))
    sm = (np.matmul(sm,a_matrix))/(((-2)*np.pi))
    yy = sm[:,:,0,0]
    xy = 0.5*(sm[:,:,0,1] + sm[:,:,1,0])
    th = 0.5*(sm[:,:,0,1] - sm[:,:,1,0])
    xx = sm[:,:,1,1]
    return yy,xx,xy,th