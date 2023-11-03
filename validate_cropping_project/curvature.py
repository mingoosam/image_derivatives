import numpy as np
import logging

def rot_mat_linear(derivatives):
    fx = derivatives['fx']
    fy = derivatives['fy']
    logging.debug("rot_mat_linear() | fx,fy = {},{}".format(fx,fy))

    if (fx**2 + fy**2) != 0:
        temp = fx**2 + fy**2
        temp_2 = np.sqrt(temp + 1e-9)
        scale = 1 / temp_2 
        #scale = np.div(1,temp_2)
        logging.debug("rot_mat_linear() | scale = {}".format(scale))

        #This is here just so we can see the shape of what we return
        #rot_mat_flat = scale * [fy,fx,-fx,fy]
        fy_scale = scale*fy
        fx_scale = scale*fx
        fx_neg = (-1)*scale*fx
        
        #return scale*fy, scale*fx, (-1)*scale*fx, scale*fy
        return fy_scale, fx_scale, fx_neg, fy_scale
    else: 
        logging.warning("rot_mat_linear() | Encountered edge case!!!")
        p1 = fy
        p2 = 1**fx
        p3 = -(1)**fx
        p4 = fy
        #return fy, 1**fx, -1**fx, fy
        return p1, p2, p3, p4
    
def mag_grad(derivatives):
    fx = derivatives['fx']
    fy = derivatives['fy']
    logging.debug("mag_grad() | fx,fy = {},{}".format(fx,fy))
    temp = fx**2 + fy**2
    #magnitude = torch.sqrt(fx**2 + fy**2)
    magnitude = np.sqrt(temp + 1e-9)
    logging.debug("mag_grad() | magnitude : {}".format(magnitude))
    return magnitude

def rot_mat_quad(derivatives):
    fxx = derivatives['fxx']
    fyy = derivatives['fyy']
    fxy = derivatives['fxy']
    if fxx+fyy+fxy != 0:
        temp = (fxx-fyy)**2 + 4*(fxy**2)
        #vp0 = fxx - fyy - torch.sqrt((fxx-fyy)**2 + 4*(fxy**2))
        vp0 = fxx - fyy - np.sqrt(temp + 1e-9)
        vp1 = 2*fxy
        
        temp2 = vp0**2 + vp1**2
        denominator1 = np.sqrt(temp + 1e-9) + 1e-9
        #denominator1 = torch.sqrt(vp0**2 + vp1**2) + 1e-9
        vp0_adj = vp0 / denominator1
        vp1_adj = vp1 / denominator1
        
        temp = (fxx-fyy)**2 + 4*(fxy**2)        
        #vq0 = fxx - fyy + torch.sqrt((fxx-fyy)**2 + 4*(fxy**2))
        vq0 = fxx - fyy + np.sqrt(temp + 1e-9)
        vq1 = 2*fxy
        #print(f"vq0 = {vq0}, {vq0.dtype}")
        #print(f"vq1 = {vq1}, {vq0.dtype}")

        temp = vq0**2 + vq1**2
        denominator2 = np.sqrt(temp + 1e-9) + 1e-9

        vq0_adj = vq0 / denominator2
        vq1_adj = vq1 / denominator2

        rquad = np.stack((vp0_adj, vp1_adj, vq0_adj, vq1_adj), axis=0)
        rquad = rquad.reshape(2,2)
        rquad = rquad.transpose(0,1)
    else:
        logging.warning("rot_mat_quad() | Encountered edge case!!!")
        rquad = np.stack((fxx, -1**fyy, 1**fxy, fxx), axis=0)
        rquad = rquad.reshape(2,2)
        rquad = rquad.transpose(0,1)
    return rquad

def fpp(derivatives):
    fxx = derivatives['fxx']
    fyy = derivatives['fyy']
    fxy = derivatives['fxy']
    temp = (fxx - fyy)**2 + 4*(fxy**2)
    fpp = 0.5*(fxx + fyy - np.sqrt(temp + 1e-9)) 
    #fpp = 0.5*(fxx + fyy - torch.sqrt((fxx-fyy)**2 + 4*(fxy**2))) 
    return fpp

def fqq(derivatives):
    fxx = derivatives['fxx']
    fyy = derivatives['fyy']
    fxy = derivatives['fxy']
    temp = (fxx-fyy)**2 + 4*(fxy**2)
    fqq = 0.5*(fxx + fyy + np.sqrt(temp + 1e-9))
    #fqq = 0.5*(fxx + fyy + torch.sqrt((fxx-fyy)**2 + 4*(fxy**2)))
    return fqq

def rot_angle(rot_matrix):
    rot_matrix = rot_matrix
    angle = np.arctan2(rot_matrix[1], rot_matrix[0])
    return angle

