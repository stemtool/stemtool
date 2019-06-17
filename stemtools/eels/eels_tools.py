import numpy as np
import numba
import pywt
from scipy import optimize as spo
import matplotlib.pyplot as plt
import matplotlib as mpl

@numba.jit(cache=True)
def cleanEELS(data,threshold):
    wave = pywt.Wavelet('sym4')
    max_level = pywt.dwt_max_level(len(data), wave.dec_len)
    coeffs = pywt.wavedec(data, 'sym4', level=max_level)
    coeffs2 = coeffs
    threshold = 0.1
    for ii in numba.prange(1, len(coeffs)):
        coeffs2[ii] = pywt.threshold(coeffs[ii], threshold*np.amax(coeffs[ii]))
    data2 = pywt.waverec(coeffs2, 'sym4')
    return data2

@numba.jit(cache=True)
def cleanEELS_3D(data3D,threshold):
    data_shape = np.asarray(np.shape(data3D)).astype(int)
    cleaned_3D = np.zeros(data_shape)
    for ii in numba.prange(data_shape[2]):
        for jj in range(data_shape[1]):
            cleaned_3D[:,jj,ii] = cleanEELS(data3D[:,jj,ii],threshold)
    return cleaned_3D

def func_powerlaw(x, m, c):
    return ((x**m) * c)

def powerlaw_fit(xdata,ydata,xrange):
    start_val = np.int((xrange[0] - np.amin(xdata))/(np.median(np.diff(xdata))))
    stop_val = np.int((xrange[1] - np.amin(xdata))/(np.median(np.diff(xdata))))
    popt, _ = spo.curve_fit(func_powerlaw, xdata[start_val:stop_val], ydata[start_val:stop_val])
    fitted = func_powerlaw(xdata,popt[0],popt[1])
    return fitted, popt

def powerlaw_plot(xdata,ydata,xrange):
    plt.figure(figsize=(20,10))
    font = {'family' : 'sans-serif',
            'weight' : 'regular',
            'size'   : 22}
    mpl.rc('font', **font)
    mpl.rcParams['axes.linewidth'] = 2
    fitted_data, popt = powerlaw_fit(xdata,ydata,xrange)
    subtracted_data = ydata - fitted_data
    yrange = func_powerlaw(xrange,popt[0],popt[1])
    plt.plot(xdata,ydata,'c',label='Original Data',linewidth=3)
    plt.plot(xdata,fitted_data,'m',label='Power Law Fit',linewidth=3)
    plt.plot(xdata,subtracted_data,'g',label='Remnant',linewidth=3)
    plt.scatter(xrange,yrange,c='b', s=120,label='Fit Region')
    plt.legend(loc='upper right')
    plt.xlabel('Energy Loss (eV)')
    plt.ylabel('Intensity (A.U.)')
    plt.xlim(np.amin(eV_Vals),np.amax(eV_Vals))
    plt.show()

def region_intensity(xdata,ydata,xrange,peak_range,showdata=True):
    fitted_data, popt = powerlaw_fit(xdata,ydata,xrange)
    subtracted_data = ydata - fitted_data
    start_val = np.int((peak_range[0] - np.amin(xdata))/(np.median(np.diff(xdata))))
    stop_val = np.int((peak_range[1] - np.amin(xdata))/(np.median(np.diff(xdata))))
    peak_ratio = np.sum(subtracted_data[start_val:stop_val])/np.sum(fitted_data)
    return peak_ratio