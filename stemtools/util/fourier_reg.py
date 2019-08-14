import numpy as np

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
    input_size = np.asarray(imFT.shape)
    output_size = np.asarray(outsize)
    imFT = np.fft.fftshift(imFT)
    center_in = np.floor((np.asarray(imFT.shape))/2) + 1
    imFTout = np.zeros(outsize)
    center_out = np.floor((np.asarray(imFTout.shape))/2) + 1
    cenout_cen = center_out - center_in
    imFTout[np.amax(cenout_cen[0],0):np.amin(cenout_cen[0]+input_size[0],output_size[0]),
            np.amax(cenout_cen[1],0):np.amin(cenout_cen[1]+input_size[1],output_size[1])] = 
    imFT[np.amax(-cenout_cen[0],0):np.amin(-cenout_cen[0]+output_size[0],input_size[0]),
         np.amax(-cenout_cen[1],0):np.amin(-cenout_cen[1]+output_size[1],input_size[1])]
    imout = (np.fft.ifftshift(imFTout) * np.prod(output_size))/np.prod(input_size)
    return imout

def dftups(input_image,nor=0,noc=0,usfac=1,roff=0,coff=0):
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
    Recieves DC in upper left corner, image center must be in (1,1) 
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
    % Set defaults
    if noc==0:
        noc = nc
    if nor==0:
        nor = nr
    nc_arr = (np.fft.ifftshift(np.arange(nc)) - np.floor(nc/2)).reshape((nc,1))
    noc_arr = (np.arange(noc) - coff).reshape((noc,1))
    nor_arr = (np.arange(nor) - roff).reshape((nor,1))
    nr_arr = (np.fft.ifftshift(np.arange(nr)) - np.floor(nr/2)).reshape((nr,1))
    kernc = np.exp((-1j*2*np.pi/(nc*usfac))*np.matmul(nc_arr.T,noc_arr))
    kernr = np.exp((-1j*2*np.pi/(nr*usfac))*np.matmul(nor_arr.T,nr_arr))
    out_fft = kernr*input_image*kernc
    return out_fft