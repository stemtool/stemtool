from scipy import ndimage as scnd
from scipy import optimize as sio
import numpy as np
import stemtool as st

def cart2pol(xx,
             yy):
    rho = ((xx ** 2) + (yy ** 2)) ** 0.5
    phi = np.arctan2(yy, xx)
    return rho, phi

def pol2cart(rho,
             phi):
    x = np.multiply(rho,np.cos(phi))
    y = np.multiply(rho,np.sin(phi))
    return x, y

def angle_fun(angle,
              rho_dpc,
              phi_dpc):
    angle = angle*((np.pi)/180)
    new_phi = phi_dpc + angle
    x_dpc,y_dpc = pol2cart(rho_dpc,new_phi)
    charge = np.gradient(x_dpc)[1] + np.gradient(y_dpc)[0]
    angle_sum = np.sum(np.abs(charge))
    return angle_sum

def optimize_angle(rho_dpc,
                   phi_dpc):
    x0 = 90
    x = sio.minimize(angle_fun,x0,args=(rho_dpc,phi_dpc))
    min_x = x.x
    sol1 = min_x - 90
    sol2 = min_x + 90
    return sol1,sol2

def data_rotator(cbed_pattern,
                 rotangle,
                 xcenter,
                 ycenter,
                 data_radius):
    data_size = np.shape(cbed_pattern)
    yV, xV = np.mgrid[0:data_size[0], 0:data_size[1]]
    mask = ((((yV - ycenter) ** 2) + ((xV - xcenter) ** 2)) ** 0.5) > (1.04*data_radius)
    cbed_min = np.amin(scnd.median_filter(cbed_pattern, 15))
    moved_cbed = np.abs(st.util.move_by_phase(cbed_pattern,(xcenter - (0.5 * data_size[1])),(ycenter - (0.5 * data_size[0]))))
    rotated_cbed = scnd.rotate(moved_cbed,rotangle,order=5,reshape=False)
    rotated_cbed[mask] = cbed_min
    return rotated_cbed

def calculate_dpc(data4D):
    diff_y, diff_x = np.mgrid[0:data4D.shape[2],0:data4D.shape[3]]
    avg_CBED = np.mean(data4D,axis=(0,1),dtype=np.float64)
    center_x,center_y,cbed_radius = st.util.fit_circle(np.log10(avg_CBED))
    com_y = (np.sum(np.multiply(data4D,diff_y),axis=(2,3)))/np.sum(data4D,axis=(2,3))
    com_x = (np.sum(np.multiply(data4D,diff_x),axis=(2,3)))/np.sum(data4D,axis=(2,3))
    shift_y = com_y - center_y
    shift_x = com_x - center_x
    dpc_rho, dpc_phi = cart2pol(shift_x,shift_y)
    return dpc_rho, dpc_phi