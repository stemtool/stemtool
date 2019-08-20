import numpy as np

def find_max_index(image):
    """
    Find maxima in image
    
    Parameters
    ----------
    image: ndarray
           Input image
    
    Returns
    -------
    ymax: int
          y-index position of maxima
    xmax: int
          x-index position of maxima
    
    Notes
    -----
    Finds the image maxima, and then locates the y 
    and x indices corresponding to the maxima
    
    :Authors:
    Debangshu Mukherjee <mukherjeed@ornl.gov>
    """
    yy,xx = np.mgrid[0:image.shape[0],0:image.shape[1]]
    ymax = (yy[image==np.amax(image)])[0]
    xmax = (xx[image==np.amax(image)])[0]
    return ymax,xmax

def fourier_pad(imFT,
                outsize):
    """
    Pad Fourier images
    
    Parameters
    ----------
    imFT:    ndarray
             Input complex array with DC in [1,1]
    
    outsize: ndarray with (2,1) shape
             ny, nx of output size
    
    Returns
    -------
    imout: ndarray
           Output complex image with DC in [1,1]
    
    Notes
    -----
    Pads or crops the Fourier transform to the desired ouput size. Taking 
    care that the zero frequency is put in the correct place for the output
    for subsequent FT or IFT. Can be used for Fourier transform based
    interpolation, i.e. dirichlet kernel interpolation. 
    
    :Authors:
    Manuel Guizar - June 02, 2014
    Debangshu Mukherjee <mukherjeed@ornl.gov>
    """
    n_in = np.asarray(imFT.shape)
    nout = np.asarray(outsize)
    imFT = np.fft.fftshift(imFT)
    center_in = (np.floor(n_in/2) + 1).astype(int)
    imFTout = np.zeros((outsize))
    center_out = (np.floor(nout/2) + 1).astype(int)
    cc = center_out - center_in
    n_in = n_in.astype(int)
    nout = nout.astype(int)
    imFTout[np.amax((cc[0],0)):np.amin((cc[0]+n_in[0],nout[0])),
            np.amax((cc[1],0)):np.amin((cc[1]+n_in[1],nout[1]))] = imFT[np.amax((-cc[0],0)):np.amin((-cc[0]+nout[0],n_in[0])),
                                                                        np.amax((-cc[1],0)):np.amin((-cc[1]+nout[1],n_in[1]))]
    imout = (np.fft.ifftshift(imFTout) * np.prod(nout))/np.prod(n_in)
    return imout

def dftups(input_image,
           nor=0,
           noc=0,
           usfac=1,
           roff=0,
           coff=0):
    """
    Upsampled discrete Fourier transform
    
    Parameters
    ----------
    input_image: ndarray
                 Input image
    usfac:       int
                 Upsampling Factor
    (nor,noc):   Number of pixels in the output upsampled DFT, in
                 units of upsampled pixels (default = size(in))
    roff, coff:  Row and column offsets, allow to shift the output array to
                 a region of interest on the DFT (default = 0)
    
    
    Returns
    -------
    out_fft: ndarray
             Upsampled Fourier transform
    
    Notes
    -----
    Recieves DC in upper left corner, image center must be in [0,0] 
    This code is intended to provide the same result as if the following
    operations were performed
    - Embed the array "input_image" in an array that is usfac times larger in each
    dimension. ifftshift to bring the center of the image to (1,1).
    - Take the FFT of the larger array
    - Extract an [nor, noc] region of the result. Starting with the 
    [roff+1 coff+1] element.
    It achieves this result by computing the DFT in the output array without
    the need to zeropad. Much faster and memory efficient than the
    zero-padded FFT approach if [nor noc] are much smaller than [nr*usfac nc*usfac]
    
    :Authors:
    Manuel Guizar - Dec 13, 2007
    Modified from dftus, by J.R. Fienup July 31, 2006
    Debangshu Mukherjee <mukherjeed@ornl.gov>
    """
    nr,nc=np.shape(input_image)
    # Set defaults
    if noc==0:
        noc = nc
    if nor==0:
        nor = nr
    nc_arr = (np.fft.ifftshift(np.arange(nc)) - np.floor(nc/2)).reshape((int(nc),1))
    noc_arr = (np.arange(noc) - coff).reshape((int(noc),1))
    nor_arr = (np.arange(nor) - roff).reshape((int(nor),1))
    nr_arr = (np.fft.ifftshift(np.arange(nr)) - np.floor(nr/2)).reshape((int(nr),1))
    kernc = (np.exp((-1j*2*np.pi/(nc*usfac))*np.matmul(nc_arr,noc_arr.T)))
    kernr = (np.exp((-1j*2*np.pi/(nr*usfac))*np.matmul(nor_arr,nr_arr.T)))
    out_fft = np.matmul(np.matmul(kernr,input_image),kernc)
    return out_fft

def dftregistration(buf1ft,
                    buf2ft,
                    usfac=1):
    """
    Upsampled FFT registration between two images
    
    Parameters
    ----------
    buf1ft: ndarray  
            Fourier transform of reference image, 
            DC in (1,1)   [DO NOT FFTSHIFT]
    buf2ft: ndarray
            Fourier transform of image to register, 
            DC in (1,1) [DO NOT FFTSHIFT]
    usfac:  int
            Upsampling factor (integer). Images will be registered to 
            within 1/usfac of a pixel. For example usfac = 20 means the
            images will be registered within 1/20 of a pixel. (default = 1)
    
    Returns
    -------
    row_shift:      float
                    Pixel shift in cartesian y direction
    col_shift:      float
                    Pixel shift in cartesian x direction
    error:          float
                    Translation invariant normalized RMS error between f and g
    phase_diff:     float
                    Global phase difference between the two images (should be
                    zero if images are non-negative).
    registered_fft: ndarray
                    Fourier transform of registered version of buf2ft,
                    the global phase difference is compensated for.
    
    Notes
    -----
    Efficient subpixel image registration by crosscorrelation. This code
    gives the same precision as the FFT upsampled cross correlation in a
    small fraction of the computation time and with reduced memory 
    requirements. It obtains an initial estimate of the crosscorrelation peak
    by an FFT and then refines the shift estimation by upsampling the DFT
    only in a small neighborhood of that estimate by means of a 
    matrix-multiply DFT. With this procedure all the image points are used to
    compute the upsampled crosscorrelation.
    
    References
    ----------
    Manuel Guizar-Sicairos, Samuel T. Thurman, and James R. Fienup, 
    "Efficient subpixel image registration algorithms," Opt. Lett. 33, 
    156-158 (2008).
    
    Copyright
    ----------
    Copyright (c) 2016, Manuel Guizar Sicairos, James R. Fienup, University of Rochester
    All rights reserved.
    
    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in
      the documentation and/or other materials provided with the distribution
    * Neither the name of the University of Rochester nor the names
      of its contributors may be used to endorse or promote products derived
      from this software without specific prior written permission.
    
    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
    ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
    LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
    CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
    SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
    INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
    CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
    ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
    POSSIBILITY OF SUCH DAMAGE.
    """
    nr,nc = np.shape(buf2ft)
    Nr = np.fft.ifftshift(np.linspace((-np.fix(nr/2)),(np.ceil(nr/2)-1),nr))
    Nc = np.fft.ifftshift(np.linspace((-np.fix(nc/2)),(np.ceil(nc/2)-1),nc))
    if (usfac == 0):
        # Simple computation of error and phase difference without registration
        CCmax = np.sum(np.multiply(buf1ft,np.conj(buf2ft)))
        row_shift = 0
        col_shift = 0
    elif (usfac == 1):
        # Single pixel registration
        CC = np.fft.ifft2(np.multiply(buf1ft,np.conj(buf2ft)))
        CCabs = np.abs(CC)
        row_shift,col_shift = find_max_index(CCabs)
        CCmax = CC[row_shift,col_shift]*nr*nc
        # Now change shifts so that they represent relative shifts and not indices
        row_shift = Nr[row_shift]
        col_shift = Nc[col_shift]
    elif (usfac > 1):
        # Start with usfac == 2
        ft_mult = np.multiply(buf1ft,np.conj(buf2ft))
        CC = np.fft.ifft2(fourier_pad(ft_mult,(2*nr,2*nc)))
        CCabs = np.abs(CC)
        row_shift, col_shift = find_max_index(CCabs)
        CCmax = CC[row_shift,col_shift]*nr*nc
        # Now change shifts so that they represent relative shifts and not indices
        Nr2 = np.fft.ifftshift(np.linspace((-np.fix(nr)),(np.ceil(nr)-1),(2*nr)))
        Nc2 = np.fft.ifftshift(np.linspace((-np.fix(nc)),(np.ceil(nc)-1),(2*nc)))
        row_shift = Nr2[row_shift]/2
        col_shift = Nc2[col_shift]/2
        #If upsampling > 2, then refine estimate with matrix multiply DFT
        if (usfac > 2):
            # DFT computation
            # Initial shift estimate in upsampled grid
            row_shift = np.round(row_shift*usfac)/usfac
            col_shift = np.round(col_shift*usfac)/usfac 
            dftshift = np.fix(np.ceil(usfac*1.5)/2)
            dftrow = dftshift-(row_shift*usfac)
            dftcol = dftshift-(col_shift*usfac)
            # Center of output array at dftshift+1
            # Matrix multiply DFT around the current shift estimate
            CC = np.conj(dftups(ft_mult,np.ceil(usfac*1.5),np.ceil(usfac*1.5),usfac,dftrow,dftcol))
            # Locate maximum and map back to original pixel grid 
            CCabs = np.abs(CC)
            rloc, cloc = find_max_index(CCabs)
            CCmax = CC[rloc,cloc]
            rloc = rloc - dftshift
            cloc = cloc - dftshift
            row_shift = row_shift + rloc/usfac
            col_shift = col_shift + cloc/usfac    
        # If its only one row or column the shift along that dimension has no
        # effect. Set to zero.
        if (nr == 1):
            row_shift = 0
        if (nc == 1):
            col_shift = 0
    rg00 = np.sum(np.abs(buf1ft) ** 2)
    rf00 = np.sum(np.abs(buf2ft) ** 2)
    error = (np.abs(1.0 - ((np.abs(CCmax) ** 2)/(rg00*rf00)))) ** 0.5
    phase_diff = np.angle(CCmax)
    # Compute registered version of buf2ft
    if (usfac > 0):
        Nc_grid,Nr_grid = np.meshgrid(Nc,Nr)
        Nr_grid = Nr_grid/nr
        Nc_grid = Nc_grid/nc
        registered_fft = np.multiply(buf2ft,np.exp(1j*2*np.pi*(-1)*((row_shift*Nr_grid) + (col_shift*Nc_grid))))
        registered_fft = registered_fft*np.exp(1j*phase_diff)
    elif (usfac==0):
        registered_fft = buf2ft*np.exp(1j*phase_diff)
    return row_shift,col_shift,phase_diff,error,registered_fft