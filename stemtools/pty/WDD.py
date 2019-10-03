import numpy as np
import numba
from scipy import ndimage as scnd
from ..util import image_utils as iu
from ..beam import gen_probe as gp

@numba.jit
def sample_4D(original_4D,sampling_ratio=2):
    data_size = (np.asarray(original_4D.shape)).astype(int)
    processed_4D = (np.zeros((data_size[2],data_size[3],data_size[2],data_size[3])))
    for jj in range(data_size[3]):
        for ii in range(data_size[2]):
            ronchigram = original_4D[:,:,ii,jj]
            ronchi_size = (np.asarray(ronchigram.shape)).astype(int)
            resized_ronchigram = st.util.resizer2D((ronchigram + 1),(1/sampling_ratio)) - 1
            resized_shape = (np.asarray(resized_ronchigram.shape)).astype(int)
            cut_shape = (np.asarray(resized_ronchigram.shape)).astype(int)
            BeforePadSize = ((0.5 * ((data_size[2],data_size[3]) - cut_shape)) - 0.25).astype(int)
            padCorrect = (data_size[2],data_size[3]) - (cut_shape + (2*BeforePadSize))
            AfterPadSize = BeforePadSize + padCorrect
            FullPadSize = ((BeforePadSize[0],AfterPadSize[0]),(BeforePadSize[1],AfterPadSize[1]))
            padValue = np.amin(resized_ronchigram)
            padded_ronchi = np.pad(resized_ronchigram, FullPadSize, 'constant', constant_values=(padValue, padValue))
            processed_4D[:,:,ii,jj] = padded_ronchi
    return processed_4D

@numba.jit
def psi_multiply(data_iff, beam_wig):
    data_size = (np.asarray(data_iff.shape)).astype(int)
    multiplied_data = (np.zeros((data_size[0],data_size[1],data_size[2],data_size[3]))).astype('complex')
    for jj in range(data_size[3]):
        for ii in range(data_size[2]):
            iff = data_iff[:,:,ii,jj]
            wig = beam_wig[:,:,ii,jj]
            psi = np.multiply(iff,np.conj(wig))
            multiplied_data[:,:,ii,jj] = psi
    return multiplied_data

@numba.jit
def sparse4D(numer4D,denom4D,bit_depth):
    data_size = (np.asarray(numer4D.shape)).astype(int)
    sparse_divided = (np.zeros((data_size[0],data_size[1],data_size[2],data_size[3]))).astype('complex')
    for jj in range(data_size[3]):
        for ii in range(data_size[2]):
            numer = numer4D[:,:,ii,jj]
            denom = denom4D[:,:,ii,jj]
            psi = sparse_division(numer,denom,bit_depth)
            sparse_divided[:,:,ii,jj] = psi
    return sparse_divided

@numba.jit(parallel=True)
def fft_wigner_probe(aperture_mrad,
                     voltage,
                     (image_x,image_y),
                     calibration_pm,
                     intensity_param):
    tb = gp.make_probe(aperture_mrad,voltage,image_x,image_y,calibration_pm)
    fourier_beam = tb/intensity_param
    wigner_beam = np.zeros((image_x,image_y,image_x,image_y)).astype(complex)
    for rows_x in range(image_x):
        for rows_y in range(image_y):
            xpos = rows_x - (image_x/2)
            ypos = rows_y - (image_y/2)
            moved_fourier_beam = scnd.interpolation.shift(fourier_beam,(-xpos,-ypos))
            convolved_beam = np.multiply(np.conj(fourier_beam),moved_fourier_beam)
            wigner_beam[:,:,rows_x,rows_y] = convolved_beam
    return wigner_beam

def SSB(data4D,
        aperture_mrad,
        voltage,
        (image_x,image_y),
        calibration_pm):
    electron_beam = gp.make_probe(aperture_mrad,voltage,(image_x,image_y),calibration_pm)
    diffractogram_intensity = np.sum(np.mean(data4D,axis=(2,3)))
    mainbeam_intensity = (np.abs(electron_beam) ** 2).sum()
    intensity_changer = (mainbeam_intensity/diffractogram_intensity) ** 0.5
    wigner_beam = fft_wigner_probe(aperture_mrad,voltage,(image_x,image_y),calibration_pm,intensity_changer)
    dataFT = np.fft.fftshift((np.fft.fft2(data4D,axes=(2,3))),axes=(2,3))
    dataIFT = np.fft.ifftshift((np.fft.ifft2(dataFT,axes=(0,1))),axes=(0,1))
    inverse_wigner = np.fft.ifftshift((np.fft.ifft2(wigner_beam,axes=(0,1))),axes=(0,1))
    Psi_Mult = psi_multiply(dataIFT,np.conj(inverse_wigner))
    Psi_Wigner = sparse4D(Psi_Mult,(np.abs(inverse_wigner)) ** 2,16)
    wig_shape = np.asarray(np.shape(Psi_Wigner))
    single_side_band = np.fft.fft2(np.multiply((Psi_Wigner[1 + int(wig_shape[0]/2),1 + int(wig_shape[1]/2),:,:]),test_beam))
    return single_side_band