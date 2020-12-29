import numpy as np
import pywt
import numba
import scipy.optimize as spo
import scipy.signal as scisig
import matplotlib.pyplot as plt
import matplotlib as mpl


def cleanEELS_wavelet(data, threshold):
    wave = pywt.Wavelet("sym4")
    max_level = pywt.dwt_max_level(len(data), wave.dec_len)
    coeffs = pywt.wavedec(data, "sym4", level=max_level)
    coeffs2 = coeffs
    threshold = 0.1
    for ii in numba.prange(1, len(coeffs)):
        coeffs2[ii] = pywt.threshold(coeffs[ii], threshold * np.amax(coeffs[ii]))
    data2 = pywt.waverec(coeffs2, "sym4")
    return data2


def cleanEELS_3D(data3D, method, threshold=0):
    data_shape = np.asarray(np.shape(data3D)).astype(int)
    cleaned_3D = np.zeros(data_shape)
    if method == "wavelet":
        if threshold > 0:
            for ii in range(data_shape[2]):
                for jj in range(data_shape[1]):
                    cleaned_3D[:, jj, ii] = cleanEELS_wavelet(
                        data3D[:, jj, ii], threshold
                    )
        else:
            cleaned_3D = data3D
    if method == "median":
        if threshold > 0:
            for ii in range(data_shape[2]):
                for jj in range(data_shape[1]):
                    cleaned_3D[:, jj, ii] = scisig.medfilt(data3D[:, jj, ii], threshold)
        else:
            cleaned_3D = data3D
    return cleaned_3D


def powerlaw_fit(xdata, ydata, xrange):
    """
    Power Law Fiiting of EELS spectral data

    Parameters
    ----------
    xdata:  ndarray
            energy values in electron-volts
    ydata:  ndarray
            intensity values in A.U.
    xrange: ndarray
            Starting and stopping energy values
            in electron volts

    Returns
    -------
    fitted: ndarray
            Background from the region of xdata
    power:  float
            The power term
    const:  float
            Constant of multiplication

    Notes
    -----
    We first find the array start and stop points
    to which the power law will be fitted to. Once
    done, we take the logarithm of both the intensity
    values and the energy loss values, taking care to
    to only take the log of non-negative intensity
    values to prevent imaginary numbers from occuring.
    We then do a linear polynomial fit in numpy, and
    return the power law fitted data, power and the
    multiplicative constant. Since the fitting is done
    in log-log space, we have to take the exponential
    of the intercept to get the multiplicative constant.

    :Authors:
    Jordan Hachtel <hachtelja@ornl.gov>

    """
    start_val = np.int((xrange[0] - np.amin(xdata)) / (np.median(np.diff(xdata))))
    stop_val = np.int((xrange[1] - np.amin(xdata)) / (np.median(np.diff(xdata))))
    xlog = np.log(xdata[start_val:stop_val][np.where(ydata[start_val:stop_val] > 0)])
    ylog = np.log(ydata[start_val:stop_val][np.where(ydata[start_val:stop_val] > 0)])
    power, const = np.polyfit(xlog, ylog, 1)
    const = np.exp(const)
    fitted = const * (xdata ** power)
    return fitted, power, const


def powerlaw_plot(
    xdata,
    ydata,
    xrange,
    figtitle,
    showdata=True,
    font={"family": "sans-serif", "weight": "bold", "size": 25},
):
    mpl.rc("font", **font)
    mpl.rcParams["axes.linewidth"] = 4
    xrange = np.asarray(xrange)
    if xrange[0] < np.amin(xdata):
        xrange[0] = np.amin(xdata)
    if xrange[1] > np.amax(xdata):
        xrange[1] = np.amax(xdata)
    fitted_data, power, const = powerlaw_fit(xdata, ydata, xrange)
    subtracted_data = ydata - fitted_data
    yrange = const * (xrange ** power)
    zero_line = np.zeros(np.shape(xdata))
    if showdata:
        plt.figure(figsize=(32, 8))
        plt.plot(xdata, ydata, "c", label="Original Data", linewidth=3)
        plt.plot(xdata, fitted_data, "m", label="Power Law Fit", linewidth=3)
        plt.plot(xdata, subtracted_data, "g", label="Remnant", linewidth=3)
        plt.plot(xdata, zero_line, "r", label="Zero Line", linewidth=3)
        plt.scatter(xrange, yrange, c="b", s=200, label="Fit Region")
        plt.legend(loc="upper right", frameon=False)
        plt.xlabel("Energy Loss (eV)", **font)
        plt.ylabel("Intensity (A.U.)", **font)
        plt.xlim(np.amin(xdata), np.amax(xdata))
        plt.ylim(np.amin(ydata) - 1000, np.amax(ydata) + 1000)
        plt.savefig(figtitle, dpi=400)
    return fitted_data


def region_intensity(xdata, ydata, xrange, peak_range, showdata=True):
    fitted_data, _, _ = powerlaw_fit(xdata, ydata, xrange)
    subtracted_data = ydata - fitted_data
    start_val = np.int((peak_range[0] - np.amin(xdata)) / (np.median(np.diff(xdata))))
    stop_val = np.int((peak_range[1] - np.amin(xdata)) / (np.median(np.diff(xdata))))
    data_floor = np.amin(subtracted_data[start_val:stop_val])
    peak_sum = np.sum(subtracted_data[start_val:stop_val] - data_floor)
    yrange = np.zeros_like(peak_range)
    yrange[0] = subtracted_data[start_val]
    yrange[1] = subtracted_data[stop_val]
    zero_line = np.zeros(np.shape(xdata))
    if showdata:
        plt.figure(figsize=(20, 10))
        plt.plot(xdata, ydata, "c", label="Original Data", linewidth=3)
        plt.plot(
            xdata,
            subtracted_data,
            "g",
            label="After background subtraction",
            linewidth=3,
        )
        plt.plot(xdata, zero_line, "b", label="Zero Line", linewidth=2)
        plt.scatter(peak_range, yrange, c="r", s=200, label="Sum Region")
        plt.legend(loc="upper right")
        plt.xlabel("Energy Loss (eV)")
        plt.ylabel("Intensity (A.U.)")
        plt.title("Sum from region = {}".format(peak_sum))
        plt.xlim(np.amin(xdata), np.amax(xdata))
        plt.ylim(np.amin(ydata) - 1000, np.amax(ydata) + 1000)
    return peak_sum


@numba.jit
def eels_3D(eels_dict, fit_range, peak_range, LBA_radius=3):
    fit_range = np.asarray(fit_range)
    peak_range = np.asarray(peak_range)
    no_elements = len(peak_range)
    eels_array = eels_dict["data"]
    elemental_subtracted = np.zeros(
        (eels_array.shape[0], eels_array.shape[1], eels_array.shape[2], no_elements),
        dtype=np.float64,
    )
    yy, xx = np.mgrid[0 : eels_array.shape[1], 0 : eels_array.shape[2]]
    xdata = (np.arange(eels_array.shape[0]) - eels_dict["pixelOrigin"][0]) * eels_dict[
        "pixelSize"
    ][0]
    peak_values = np.zeros(
        (eels_array.shape[-2], eels_array.shape[-1], no_elements), dtype=np.float64
    )
    for ii in range(eels_array.shape[-2]):
        for jj in range(eels_array.shape[-1]):
            for qq in range(no_elements):
                eels_data = eels_array[:, ii, jj]
                fit_points = fit_range[qq, :]
                peak_point = peak_range[qq, :]
                lbi = ((yy - ii) ** 2) + ((xx - jj) ** 2) <= LBA_radius ** 2
                eels_lbi = np.mean(eels_array[:, lbi], axis=-1)
                bg, _, _ = powerlaw_fit(xdata, eels_lbi, fit_points)
                subtracted_data = eels_data - bg
                elemental_subtracted[:, ii, jj, qq] = subtracted_data
                start_val = np.int(
                    (peak_point[0] - np.amin(xdata)) / (np.median(np.diff(xdata)))
                )
                stop_val = np.int(
                    (peak_point[1] - np.amin(xdata)) / (np.median(np.diff(xdata)))
                )
                peak_sum = np.sum(subtracted_data[start_val:stop_val])
                peak_values[ii, jj, qq] = peak_sum
    return peak_values, elemental_subtracted


def lcpl(xx, c1, p1, c2, p2):
    yy = (c1 * (xx ** p1)) + (c2 * (xx ** p2))
    return yy


@numba.jit
def eels_3D_LCPL(eels_dict, fit_range, peak_range, LBA_radius=3, percentile=5):
    fit_range = np.asarray(fit_range)
    peak_range = np.asarray(peak_range)
    no_elements = len(peak_range)
    eels_array = eels_dict["data"]
    elemental_subtracted = np.zeros(
        (eels_array.shape[0], eels_array.shape[1], eels_array.shape[2], no_elements),
        dtype=np.float32,
    )
    yy, xx = np.mgrid[0 : eels_array.shape[1], 0 : eels_array.shape[2]]
    xdata = (np.arange(eels_array.shape[0]) - eels_dict["pixelOrigin"][0]) * eels_dict[
        "pixelSize"
    ][0]
    peak_values = np.zeros(
        (eels_array.shape[-2], eels_array.shape[-1], no_elements), dtype=np.float32
    )
    power_values = np.zeros(
        (eels_array.shape[-2], eels_array.shape[-1], no_elements), dtype=np.float32
    )
    const_values = np.zeros(
        (eels_array.shape[-2], eels_array.shape[-1], no_elements), dtype=np.float32
    )

    for ii in range(eels_array.shape[-2]):
        for jj in range(eels_array.shape[-1]):
            for kk in range(no_elements):
                eels_data = eels_array[:, ii, jj]
                fit_points = fit_range[kk, :]
                peak_point = peak_range[kk, :]
                _, power, const = powerlaw_fit(xdata, eels_data, fit_points)
                power_values[ii, jj, kk] = power
                const_values[ii, jj, kk] = const

    percentile1 = np.zeros(no_elements, dtype=np.float32)
    percentile2 = np.zeros(no_elements, dtype=np.float32)
    lower_bound = np.zeros((4, no_elements), dtype=np.float32)
    upper_bound = np.zeros((4, no_elements), dtype=np.float32)

    for ll in range(no_elements):
        percentile1[ll] = np.percentile(np.ravel(power_values[:, :, ll]), percentile)
        percentile2[ll] = np.percentile(
            np.ravel(power_values[:, :, ll]), 100 - percentile
        )
        lower_bound[:, ll] = (
            0.5 * np.amin(const_values[:, :, ll]),
            1.001 * percentile1[ll],
            0.5 * np.amin(const_values[:, :, ll]),
            1.001 * percentile2[ll],
        )
        upper_bound[:, ll] = (
            2 * np.amax(const_values[:, :, ll]),
            0.999 * percentile1[ll],
            2 * np.amax(const_values[:, :, ll]),
            0.999 * percentile2[ll],
        )

    for pp in range(eels_array.shape[-2]):
        for qq in range(eels_array.shape[-1]):
            for rr in range(no_elements):
                eels_data = eels_array[:, pp, qq]
                fit_points = fit_range[rr, :]
                peak_point = peak_range[rr, :]
                star_val = np.int(
                    (fit_points[0] - np.amin(xdata)) / (np.median(np.diff(xdata)))
                )
                stop_val = np.int(
                    (fit_points[1] - np.amin(xdata)) / (np.median(np.diff(xdata)))
                )

                lbi = (((yy - ii) ** 2) + ((xx - jj) ** 2)) <= (LBA_radius ** 2)
                eels_lbi = np.mean(eels_array[:, lbi], axis=-1)
                popt, _ = spo.curve_fit(
                    lcpl,
                    xdata[star_val:stop_val],
                    eels_lbi[star_val:stop_val],
                    bounds=(lower_bound[:, rr], upper_bound[:, rr]),
                    ftol=0.0001,
                    xtol=0.0001,
                )
                background = lcpl(xdata, popt[0], popt[1], popt[2], popt[3])
                subtracted_data = eels_data - background
                elemental_subtracted[:, pp, qq, rr] = subtracted_data
                star_sum = np.int(
                    (peak_point[0] - np.amin(xdata)) / (np.median(np.diff(xdata)))
                )
                stop_sum = np.int(
                    (peak_point[1] - np.amin(xdata)) / (np.median(np.diff(xdata)))
                )
                peak_values[pp, qq, rr] = np.sum(subtracted_data[star_sum:stop_sum])

    return peak_values, elemental_subtracted
