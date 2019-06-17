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
def cleanEELS_3D(data3D,threshold=0):
    data_shape = np.asarray(np.shape(data3D)).astype(int)
    cleaned_3D = np.zeros(data_shape)
    if (threshold > 0):
        for ii in numba.prange(data_shape[2]):
            for jj in range(data_shape[1]):
                cleaned_3D[:,jj,ii] = cleanEELS(data3D[:,jj,ii],threshold)
    else:
        cleaned_3D = data3D
    return cleaned_3D

def func_powerlaw(x, m, c):
    return ((x**m) * c)

def powerlaw_fit(xdata,ydata,xrange):
    start_val = np.int((xrange[0] - np.amin(xdata))/(np.median(np.diff(xdata))))
    stop_val = np.int((xrange[1] - np.amin(xdata))/(np.median(np.diff(xdata))))
    popt, _ = spo.curve_fit(func_powerlaw, xdata[start_val:stop_val], ydata[start_val:stop_val],maxfev=10000)
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
    plt.xlim(np.amin(xdata),np.amax(xdata))
    plt.show()

def region_intensity(xdata,ydata,xrange,peak_range,showdata=True):
    fitted_data, popt = powerlaw_fit(xdata,ydata,xrange)
    subtracted_data = ydata - fitted_data
    start_val = np.int((peak_range[0] - np.amin(xdata))/(np.median(np.diff(xdata))))
    stop_val = np.int((peak_range[1] - np.amin(xdata))/(np.median(np.diff(xdata))))
    data_floor = np.amin(subtracted_data[start_val:stop_val])
    peak_ratio = np.sum(subtracted_data[start_val:stop_val] - data_floor)/np.sum(fitted_data - data_floor)
    peak_sum = np.sum(subtracted_data[start_val:stop_val] - data_floor)
    yrange = np.zeros_like(peak_range)
    yrange[0] = subtracted_data[start_val]
    yrange[1] = subtracted_data[stop_val]
    if showdata:
        plt.figure(figsize=(20,10))
        plt.plot(xdata,ydata,'c',label='Original Data',linewidth=3)
        plt.plot(xdata,subtracted_data,'g',label='After background subtraction',linewidth=3)
        plt.scatter(peak_range,yrange,c='r', s=120,label='Sum Region')
        plt.legend(loc='upper right')
        plt.xlabel('Energy Loss (eV)')
        plt.ylabel('Intensity (A.U.)')
        plt.title('Sum from region = {}'.format(peak_ratio))
        plt.xlim(np.amin(xdata),np.amax(xdata))
    return peak_ratio, peak_sum

@numba.jit(parallel=True)
def eels_3D(eels_dict,fit_range,peak_range,clean_val=0):
    eels_array = eels_dict['data']
    if (clean_val > 0):
        eels_clean = cleanEELS_3D(eels_array,clean_val)
    else:
        eels_clean = eels_array
    xdata = (np.arange(eels_clean.shape[0]) - eels_dict['pixelOrigin'][0])*eels_dict['pixelSize'][0]
    rel_peak_val = np.zeros((eels_clean.shape[-2],eels_clean.shape[-1]), dtype=np.float64)
    abs_peak_val = np.zeros((eels_clean.shape[-2],eels_clean.shape[-1]), dtype=np.float64)
    for jj in numba.prange(eels_clean.shape[-1]):
        for ii in range(eels_clean.shape[-2]):
            pv,ps = region_intensity(xdata,eels_clean[:,ii,jj],fit_range,peak_range,showdata=False)
            rel_peak_val[ii,jj] = pv
            abs_peak_val[ii,jj] = ps
    return rel_peak_val, abs_peak_val