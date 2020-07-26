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
    for pp in numba.prange(len(stack_pos)):
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
    for ii in numba.prange(len(row_mean)):
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

    def plot_shifts(self):
        """
        Notes
        -----
        Plot the relative shifts of images with each other
        """
        if not self.stack_check:
            raise RuntimeError(
                "Please get the images correlated first as get_shape_stack()"
            )
        vm = self.max_shift
        fig = plt.figure(figsize=(20, 10))
        gs = mpgs.GridSpec(1, 2)
        ax1 = plt.subplot(gs[0, 0])
        ax2 = plt.subplot(gs[0, 1])

        im = ax1.imshow(self.row_stack, vmin=-vm, vmax=vm, cmap="RdBu_r")
        ax1.set_xlabel("Stack Number")
        ax1.set_ylabel("Stack Number")
        ax1.set_title(label="Shift along X direction", loc="left")

        im = ax2.imshow(self.col_stack, vmin=-vm, vmax=vm, cmap="RdBu_r")
        ax2.set_xlabel("Stack Number")
        ax2.set_ylabel("Stack Number")
        ax2.set_title(label="Shift along Y direction", loc="left")

        p1 = ax1.get_position().get_points().flatten()
        p2 = ax2.get_position().get_points().flatten()
        ax_cbar = fig.add_axes([p1[0], 0.25, p2[2] - 0.12, 0.02])
        cbar = plt.colorbar(im, cax=ax_cbar, orientation="horizontal")
        cbar.set_label("Relative Shift (pixels)")
