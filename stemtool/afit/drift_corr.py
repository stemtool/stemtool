import matplotlib.offsetbox as mploff
import numpy as np
import numba
import pyfftw.interfaces as pfi
import stemtool as st
import matplotlib.pyplot as plt
import matplotlib.gridspec as mpgs


@numba.jit(parallel=True, cache=True)
def numba_shift_stack(image_stack, row_stack, col_stack, stack_pos, sampling=500):
    """
    Cross-Correlate stack of images

    Parameters
    ----------
    image_stack: ndarray
                 Stack of images collected in rapid succession,
                 where the the first array position refers to the
                 image collected. Thus the nth image in the stack
                 is image_stack[n-1,:,:]
    row_stack:   ndarray
                 Stack of correlated row positions
    col_stack:   ndarray
                 Stack of correlated column positions
    stack_pos:   ndarray
                 Row and column position list
    sampling:    int, optional
                 Fraction of the pixel to calculate upsampled
                 cross-correlation for. Default is 500

    Notes
    -----
    For a rapidly collected image stack, each image in the stack is
    cross-correlated with all the other images of the stack, to generate
    a skew matrix of row shifts and column shifts, calculated with sub
    pixel precision.

    See Also
    --------
    util.dftregistration

    References
    ----------
    [1]_, Savitzky, B.H., El Baggari, I., Clement, C.B., Waite, E., Goodge, B.H.,
          Baek, D.J., Sheckelton, J.P., Pasco, C., Nair, H., Schreiber, N.J. and
          Hoffman, J., 2018. Image registration of low signal-to-noise cryo-STEM data.
          Ultramicroscopy, 191, pp.56-65.

    Examples
    --------
    Since this is a `numba` function, to initialize the JIT we need
    to call the function with a small dataset first. Running it once also
    allows `pyFFTW` to figure out the fastest FFT route.

    >>> numba_shift_stack(image_stack,row_stack,col_stack,stack_pos[0:10,:])

    Once the JIT is initialized run the function as:

    >>> numba_shift_stack(image_stack,row_stack,col_stack,stack_pos)
    """
    pfi.cache.enable()
    for pp in range(len(stack_pos)):
        ii = stack_pos[pp, 0]
        jj = stack_pos[pp, 1]
        row_stack[ii, jj], col_stack[ii, jj], _, _, _ = st.util.dftregistration(
            pfi.numpy_fft.fft2(image_stack[ii, :, :]),
            pfi.numpy_fft.fft2(image_stack[jj, :, :]),
            sampling,
        )


@numba.jit(parallel=True, cache=True)
def numba_stack_corr(image_stack, moved_stack, rowshifts, colshifts):
    """
    Get corrected image stack

    Parameters
    ----------
    image_stack: ndarray
                 Stack of images collected in rapid succession,
                 where the the first array position refers to the
                 image collected. Thus the nth image in the stack
                 is image_stack[n-1,:,:]
    moved_stack: ndarray
                 Stack of moved images
    rowshifts:   ndarray
                 The size is nXn where n is the n of images in
                 the image_stack
    colshifts:   ndarray
                 The size is nXn where n is the n of images in
                 the image_stack

    Notes
    -----
    The mean of the shift stacks for every image position are the
    amount by which each image is to be shifted. We calculate the
    mean and move each image by that amount in the stack and then
    sum them up.

    See Also
    --------
    util.move_by_phase

    References
    ----------
    .. [2] Savitzky, B.H., El Baggari, I., Clement, C.B., Waite, E., Goodge, B.H.,
       Baek, D.J., Sheckelton, J.P., Pasco, C., Nair, H., Schreiber, N.J. and
       Hoffman, J., 2018. Image registration of low signal-to-noise cryo-STEM data.
       Ultramicroscopy, 191, pp.56-65.

    Examples
    --------
    Since this is a `numba` function, to initialize the JIT we need
    to call the function with a small dataset first

    >>> corrected_stack(image_stack,moved_stack,rowshifts,colshifts)

    Once the JIT is initialized run the function as:

    >>> corr_stack = corrected_stack(image_stack,rowshifts,colshifts)

    """
    row_mean = np.median(rowshifts, axis=0)
    col_mean = np.median(colshifts, axis=0)
    for ii in range(len(row_mean)):
        moved_stack[ii, :, :] = np.abs(
            st.util.move_by_phase(image_stack[ii, :, :], col_mean[ii], row_mean[ii])
        )


class multi_image_drift(object):
    """
    Correct for scan drift through cross-correlating a
    rapidly acquired image stack

    Parameters
    ----------
    image_stack: ndarray
                 Stack of images collected in rapid succession,
                 where the the first array position refers to the
                 image collected. Thus the nth image in the stack
                 is image_stack[n-1,:,:]
    sampling:    int, optional
                 Fraction of the pixel to calculate upsampled
                 cross-correlation for. Default is 500

    References
    ----------
    .. [1] Savitzky, B.H., El Baggari, I., Clement, C.B., Waite, E., Goodge, B.H.,
       Baek, D.J., Sheckelton, J.P., Pasco, C., Nair, H., Schreiber, N.J. and
       Hoffman, J., 2018. Image registration of low signal-to-noise cryo-STEM data.
       Ultramicroscopy, 191, pp.56-65.

    Examples
    --------
    Run the function as:

    >>> cc = drift_corrector(image_stack)
    >>> cc.get_shift_stack()
    >>> corrected = cc.corrected_stack()

    """

    def __init__(self, image_stack, sampling=500):
        if sampling < 1:
            raise RuntimeError("Sampling factor should be a positive integer")
        self.image_stack = image_stack
        self.stack_shape = image_stack.shape[0]
        no_im = image_stack.shape[0]
        self.no_im = no_im
        self.sampling = sampling
        self.row_stack = np.empty((no_im, no_im))
        self.col_stack = np.empty((no_im, no_im))
        self.max_shift = 0
        self.corr_image = np.empty(
            (image_stack.shape[1], image_stack.shape[2]), dtype=image_stack.dtype
        )
        self.moved_stack = np.empty_like(image_stack, dtype=image_stack.dtype)
        self.stack_check = False

    def get_shape_stack(self):
        """
        Cross-Correlate stack of images

        Notes
        -----
        For a rapidly collected image stack, each image in the stack is
        cross-correlated with all the other images of the stack, to generate
        a skew matrix of row shifts and column shifts, calculated with sub
        pixel precision.
        """
        pfi.cache.enable()
        rows, cols = np.mgrid[0 : self.no_im, 0 : self.no_im]
        pos_stack = np.asarray((np.ravel(rows), np.ravel(cols))).transpose()

        # Initialize JIT
        numba_shift_stack(
            self.image_stack,
            self.row_stack,
            self.col_stack,
            pos_stack[0:10, :],
            self.sampling,
        )

        # Run JITted function
        numba_shift_stack(
            self.image_stack, self.row_stack, self.col_stack, pos_stack, self.sampling
        )

        self.max_shift = np.amax(
            np.asarray((np.amax(self.row_stack), np.amax(self.col_stack)))
        )
        self.stack_check = True

    def corrected_stack(self):
        """
        Get corrected image stack

        Returns
        -------
        corr_stack: ndarray
                    Corrected image from the image stack

        Notes
        -----
        The mean of the shift stacks for every image position are the
        amount by which each image is to be shifted. We calculate the
        mean and move each image by that amount in the stack and then
        sum them up.
        """
        if not self.stack_check:
            raise RuntimeError(
                "Please get the images correlated first as get_shape_stack()"
            )
        image_stack = np.copy(self.image_stack[0:3, :, :])
        moved_stack = np.copy(self.moved_stack[0:3, :, :])
        row_stack = np.copy(self.row_stack[0:3, 0:3])
        col_stack = np.copy(self.col_stack[0:3, 0:3])
        # Initialize JIT
        numba_stack_corr(image_stack, moved_stack, row_stack, col_stack)

        # Run JITted code
        numba_stack_corr(
            self.image_stack, self.moved_stack, self.row_stack, self.col_stack
        )
        self.corr_image = np.sum(self.moved_stack, axis=0) / self.no_im
        return self.corr_image

    def plot_shifts(self, imwidth=15):
        """
        Notes
        -----
        Plot the relative shifts of images with each other
        """
        if not self.stack_check:
            raise RuntimeError(
                "Please get the images correlated first as get_shape_stack()"
            )
        fontsize = int(imwidth)
        sc_font = {"weight": "bold", "size": fontsize}
        imsize = (int(imwidth), int(imwidth * 0.6))
        plt.figure(figsize=imsize)
        vm = self.max_shift

        gs = mpgs.GridSpec(12, 20)
        ax1 = plt.subplot(gs[0:10, 0:9])
        ax2 = plt.subplot(gs[0:10, 11:20])
        ax3 = plt.subplot(gs[10:12, :])

        ax1.imshow(self.row_stack, vmin=-vm, vmax=vm, cmap="RdBu_r")
        ax1.set_xlabel("Stack Number", **sc_font)
        ax1.set_ylabel("Stack Number", **sc_font)
        ax1.xaxis.set_tick_params(
            width=0.1 * imwidth, length=0.4 * imwidth, direction="in", pad=10
        )
        ax1.yaxis.set_tick_params(
            width=0.1 * imwidth, length=0.4 * imwidth, direction="in", pad=10
        )
        at = mploff.AnchoredText(
            "Shift along X direction",
            prop=dict(size=fontsize),
            frameon=True,
            loc="upper left",
        )
        at.patch.set_boxstyle("round, pad= 0., rounding_size= 0.2")
        ax1.add_artist(at)

        ax2.imshow(self.col_stack, vmin=-vm, vmax=vm, cmap="RdBu_r")
        ax2.set_xlabel("Stack Number", **sc_font)
        ax2.set_ylabel("Stack Number", **sc_font)
        ax2.xaxis.set_tick_params(
            width=0.1 * imwidth, length=0.4 * imwidth, direction="in", pad=10
        )
        ax2.yaxis.set_tick_params(
            width=0.1 * imwidth, length=0.4 * imwidth, direction="in", pad=10
        )
        at = mploff.AnchoredText(
            "Shift along Y direction",
            prop=dict(size=fontsize),
            frameon=True,
            loc="upper left",
        )
        at.patch.set_boxstyle("round, pad= 0., rounding_size= 0.2")
        ax2.add_artist(at)

        sb = np.zeros((10, 1000), dtype=np.float)
        for ii in range(10):
            sb[ii, :] = np.linspace(-vm, vm, 1000)
        ax3.imshow(sb, cmap="RdBu_r")
        ax3.yaxis.set_visible(False)
        no_labels = 9
        x1 = np.linspace(0, 1000, no_labels)
        ax3.set_xticks(x1)
        ax3.set_xticklabels(np.round(np.linspace(-vm, vm, no_labels), 4))
        for axis in ["top", "bottom", "left", "right"]:
            ax3.spines[axis].set_linewidth(2)
            ax3.spines[axis].set_color("black")
        ax3.xaxis.set_tick_params(width=2, length=6, direction="out", pad=10)
        ax3.set_title("Relative Shift (pixels)", **sc_font)

        plt.autoscale()
