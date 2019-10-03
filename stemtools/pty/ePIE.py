import numpy as np
import numba
from scipy import ndimage as scnd
from ..proc import sobel_canny as sc
from ..util import gauss_utils as gt
from ..util import image_utils as iu

@numba.jit
def resize_rotate(original_4D,
                  final_size,
                  rotangle,
                  sampler=2,
                  masking=True):
    data_size = (np.asarray(original_4D.shape)).astype(int)
    processed_4D = (np.zeros((data_size[0],data_size[1],final_size[0],final_size[1])))
    _,_,original_radius = iu.fit_circle(np.mean(original_4D,axis=(0,1)))
    new_radius = original_radius*sampler
    circle_mask = iu.make_circle(final_size,final_size[1]/2,final_size[0]/2,new_radius*1.25)
    for jj in range(data_size[1]):
        for ii in range(data_size[0]):
            ronchigram = original_4D[ii,jj,:,:]
            ronchi_size = (np.asarray(ronchigram.shape)).astype(int)
            resized_ronchigram = iu.resizer2D((ronchigram + 1),(1/sampler)) - 1
            resized_rotated_ronchigram = scnd.rotate(resized_ronchigram,rotangle)
            resized_shape = (np.asarray(resized_rotated_ronchigram.shape)).astype(int)
            pad_size = np.round((np.asarray(final_size) - resized_shape)/2)
            before_pad_size = np.copy(pad_size)
            after_pad_size = np.copy(pad_size)
            if (2*pad_size[0] + resized_shape[0]) < final_size[0]:
                after_pad_size[0] = pad_size[0] + 1
            if (2*pad_size[1] + resized_shape[1]) < final_size[1]:
                after_pad_size[1] = pad_size[1] + 1
            before_pad_size = (before_pad_size).astype(int)
            after_pad_size = (after_pad_size).astype(int)
            FullPadSize = ((before_pad_size[0],after_pad_size[0]),(before_pad_size[1],after_pad_size[1]))
            padded_ronchi = np.pad(resized_rotated_ronchigram, FullPadSize, 'constant', constant_values=(0, 0))
            processed_4D[ii,jj,:,:] = padded_ronchi
    if masking:
        processed_4D = np.multiply(processed_4D,circle_mask)
    return processed_4D