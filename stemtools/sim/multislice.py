import numpy as np
import numba
from scipy import special as s2
import PIL

@numba.jit(cache=True)
def atomic_potential(atom_no,pixel_size,sampling=16,potential_extent=4,datafile='Kirkland_Potentials.npy'):
    """
    Calculate the projected potential of a single atom
    
    Parameters
    ----------
    atom_no:          int
                      Atomic number of the atom whose potential is being calculated.
    pixel_size:       float
                      Real space pixel size 
    datafile:         string
                      Load the location of the npy file of the Kirkland scattering factors
    sampling:         int, float
                      Supersampling factor for increased accuracy. Matters more with big
                      pixel sizes. The default value is 16. 
    potential_extent: float
                      Distance in angstroms from atom center to which the projected 
                      potential is calculated. The default value is 4 angstroms.
                
    Returns
    -------
    potential: ndarray
               Projected potential matrix
                
    Notes
    -----
    We calculate the projected screened potential of an 
    atom using the Kirkland formula. Keep in mind however 
    that this potential is for independent atoms only!
    No charge distribution between atoms occure here.
    
    References
    ----------
    Kirkland EJ. Advanced computing in electron microscopy. 
    Springer Science & Business Media; 2010 Aug 12.
                 
    :Authors:
    Debangshu Mukherjee <mukherjeed@ornl.gov>
    
    """
    a0 = 0.5292
    ek = 14.4
    term1 = 4*(np.pi ** 2)*a0*ek
    term2 = 2*(np.pi ** 2)*a0*ek
    kirkland = np.load(datafile)
    xsub = np.arange(-potential_extent,potential_extent,(pixel_size/sampling))
    ysub = np.arange(-potential_extent,potential_extent,(pixel_size/sampling))
    kirk_fun = kirkland[atom_no-1,:]
    ya,xa = np.meshgrid(ysub,xsub)
    r2 = np.power(xa,2) + np.power(ya,2)
    r = np.power(r2,0.5)
    part1 = np.zeros_like(r)
    part2 = np.zeros_like(r)
    sspot = np.zeros_like(r)
    part1 = term1 * (np.multiply(kirk_fun[0],s2.kv(0,(np.multiply((2*np.pi*np.power(kirk_fun[1],0.5)),r)))) +
                     np.multiply(kirk_fun[2],s2.kv(0,(np.multiply((2*np.pi*np.power(kirk_fun[3],0.5)),r)))) +
                     np.multiply(kirk_fun[4],s2.kv(0,(np.multiply((2*np.pi*np.power(kirk_fun[5],0.5)),r)))))
    part2 = term2 * ((kirk_fun[6]/kirk_fun[7])*np.exp(-((np.pi ** 2)/kirk_fun[7])*r2) +
                     (kirk_fun[8]/kirk_fun[9])*np.exp(-((np.pi ** 2)/kirk_fun[9])*r2) +
                     (kirk_fun[10]/kirk_fun[11])*np.exp(-((np.pi ** 2)/kirk_fun[11])*r2))
    sspot = part1 + part2
    finalsize = (np.asarray(sspot.shape)/sampling).astype(int)
    sspot_im = PIL.Image.fromarray(sspot)
    potential = np.array(sspot_im.resize(finalsize,resample=PIL.Image.LANCZOS))
    return potential