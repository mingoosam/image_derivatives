import matplotlib.pyplot as plt
import pickle
import yaml
from matplotlib.ticker import FuncFormatter
import numpy as np
import matplotlib.patches as patches
from IPython import embed
import interpolate
import curvature
import os

def plot_mse(mse_lists, pixel_vals, title):
    
    arr = np.array(mse_lists)
    arr_t = arr.transpose()

    mse_fx = arr_t[0, :]
    mse_fy = arr_t[1, :]
    mse_fxx = arr_t[2, :]
    mse_fyy = arr_t[3, :]
    mse_fxy = arr_t[4, :]
    
    fig, ax = plt.subplots()

    x = pixel_vals
    ax.plot(x, mse_fx, label='fx')
    ax.plot(x, mse_fy, label='fy')
    ax.plot(x, mse_fxx, label='fxx')
    ax.plot(x, mse_fyy, label='fyy')
    ax.plot(x, mse_fxy, label='fxy')
    
    ax.set_xlabel('number of pixels (cropped value)')
    ax.set_ylabel('mean squared error - cropped vs. full')
    ax.set_title(f'MSE of derivatives for cropped vs. full image \n {title}')
    
    ax.legend()

    plt.show()    

    fig.savefig(f"images2/mse_{title}")

def format_curv(curv):
    format_curv = {'rot_mat_linear': [],
                   'mag_grad': None,
                   'rot_mat_quad': [],
                   'fpp': None,
                   'fqq': None,
                   }

    for name, value in curv.items():
        if name == 'rot_mat_linear':
            temp = ['{:.3e}'.format(x) for x in value]
            format_curv['rot_mat_linear'] = ', '.join(temp)
        elif name == 'mag_grad':
            format_curv['mag_grad'] ='{:.3e}'.format(value) 
        elif name == 'rot_mat_quad':
            temp_arr = np.reshape(value, 4)
            format_arr = []
            for val in temp_arr:
                format_arr.append(np.format_float_scientific(val, precision=2, exp_digits=2))
            format_arr = ', '.join(format_arr)
            format_curv['rot_mat_quad'] = format_arr
        elif name == 'fpp':
            format_curv['fpp'] = np.format_float_scientific(value, precision=2, exp_digits=2)
        elif name == 'fqq':
            format_curv['fqq'] = np.format_float_scientific(value, precision=2, exp_digits=2)

    return format_curv

#def make_plots(X_full, Y_full, Z_full, X_cropped, Y_cropped, Z_cropped, full_der, crop_der, full_curv, crop_curv, title=None):
def make_plots(params, image, cropped, full_der, crop_der, full_curv, crop_curv, title):

    fig, ax = plt.subplots(1,2, figsize=(8,8))
    plt.suptitle(title)

    vmin= image.min()
    vmax= image.max()
     
    im = ax[0].pcolormesh(X_full, Y_full, image, vmin=vmin, vmax=vmax)
    ax[0].set_xticks(np.linspace(X_full.min(), X_full.max(), 5))
    ax[0].set_yticks(np.linspace(Y_full.min(), Y_full.max(), 5))
    ax[0].set_xlim([X_full.min(), X_full.max()])
    ax[0].set_ylim([Y_full.min(), Y_full.max()])
    ax[0].set_title("Full Image")
    ax[0].set_xlabel(r"$m$")
    ax[0].set_ylabel(r"$m$")
    ax[0].ticklabel_format(style='scientific')
    
    im2 = ax[1].pcolormesh(X_cropped, Y_cropped, Z_cropped, vmin=vmin, vmax=vmax)
    ax[1].set_xticks(np.linspace(X_full.min(), X_full.max(), 5))
    ax[1].set_yticks(np.linspace(Y_full.min(), Y_full.max(), 5))
    ax[1].set_xlim([X_full.min(), X_full.max()])
    ax[1].set_ylim([Y_full.min(), Y_full.max()])
    ax[1].set_title("Cropped Image")
    ax[1].set_xlabel(r"$m$")
    ax[1].set_ylabel(r"$m$")
    
    cropped_rectangle1 = patches.Rectangle(
        (X_cropped.min(), Y_cropped.min()), 
        X_cropped.max() - X_cropped.min(),   
        Y_cropped.max() - Y_cropped.min(),   
        fill=False,                          
        linestyle='dotted',                  
        edgecolor='black',                   
        linewidth=2,                         
    )

    ax[0].add_patch(cropped_rectangle1)
    
    cropped_rectangle2 = patches.Rectangle(
        (X_cropped.min(), Y_cropped.min()),  
        X_cropped.max() - X_cropped.min(),   
        Y_cropped.max() - Y_cropped.min(),   
        fill=False,                          
        linestyle='dotted',                  
        edgecolor='black',                   
        linewidth=2,                         
    )
    ax[1].add_patch(cropped_rectangle2)

    full = []
    for name, value in full_der.items():
        full.append('{:.3e}'.format(value))
    full = ', '.join(full)

    crop = []
    for name, value in crop_der.items():
        crop.append('{:.3e}'.format(value))
    crop = ', '.join(crop)
   
    fig.text(0.1, 0.9, f"        Full image ders: {full}")
    fig.text(0.1, 0.85, f"Cropped image ders: {crop}") 


    format_full_curv = format_curv(full_curv)
    format_crop_curv = format_curv(crop_curv)
    #fig.text(0.1, 0.75, ""
    fig.text(0.1, 0.65, f"Full curv:\n{format_full_curv['rot_mat_linear']}\n{format_full_curv['mag_grad']}\n{format_full_curv['rot_mat_quad']}\n{format_full_curv['fpp']}\n{format_full_curv['fqq']}")
    fig.text(0.55, 0.65, f"Crop curv:\n{format_crop_curv['rot_mat_linear']}\n{format_crop_curv['mag_grad']}\n{format_crop_curv['rot_mat_quad']}\n{format_crop_curv['fpp']}\n{format_crop_curv['fqq']}")

    plt.tight_layout(rect=[0, 0, 1, 0.65])
    cbar = plt.colorbar(im2, ax=ax[:], shrink=0.2, location='bottom')
    #plt.show()
    
    if title is not None:
        fig.savefig(f"images2/{title}.png")

# go from (1, 166, 166) to derivatives

# derivatives are calculated with fx--> and fy ^
# e.g. [2, 2, 2]                          and [0, 1, 2]
#      [1, 1, 1]                              [0, 1, 2]
#      [0, 0, 0]   has fx = 0 and fy = 1      [0, 1, 2]    has fx = 1 and fy = 0


def get_center(arr):
    if len(arr.shape) == 2: 
        rows, cols = arr.shape
    
        center_row = rows // 2
        center_col = cols // 2

        center = arr[center_row, center_col]

    elif len(arr.shape) == 1:
        length = arr.shape[0]
        center_index = length // 2
        center = arr[center_index]

    return center

def MSE(full_der, crop_der):

    full_der, crop_der = np.array(full_der), np.array(crop_der)
    return np.square(np.subtract(full_der, crop_der)).mean()

def get_avg(arr):
    
    return np.average(arr)
    
def derivatives(grid, center):
    embed()
    grid_flip = np.flip(grid,axis=0) # flip row if NOT using meshgrid
    #grid_flip = grid # this is for meshgrid
    #fy, fx = np.gradient(grid_flip)
    fy, fx = np.gradient(grid)

    fxx = np.diff(grid_flip[1,:], 2)
    fyy = np.diff(grid_flip[:,1], 2) 

    fxy = np.gradient(np.gradient(grid,axis=0),axis=1)
   
    if center == True: 
        
        fx = get_center(fx)
        fy = get_center(fy)
        fxx = get_center(fxx)
        fyy = get_center(fyy)
        fxy = get_center(fxy)
    
    else:
        fx = get_avg(fx)
        fy = get_avg(fy)
        fxx = get_avg(fxx)
        fyy = get_avg(fyy)
        fxy = get_avg(fxy)

    derivatives = {
                   'fx' : fx,
                   'fy' : fy,
                   'fxx': fxx,
                   'fyy': fyy,
                   'fxy': fxy,
                  }
    
    return derivatives

def get_cropped_im(params, image):

    start_row = (params['X_grid'].shape[0] - params['len_pix_full']) // 2
    end_row = start_row + params['len_pix_crop']
    
    start_col = (params['X_grid'].shape[0] - params['len_pix_full']) // 2
    end_col = start_col + params['len_pix_crop']
  
    cropped = image[start_row:end_row, start_col:end_col]
    
    return cropped

#def test_images(X_full, Y_full, Z_full, title=None):
def test_images(params, image, title=None):
    
    cropped = get_cropped_im(params, image)

    full_der = derivatives(image, center=False) # if we set center=True, we'll get the center pixel rather than an average of all the pixels. They are really pretty similar, at least for the toy problems. 
    crop_der = derivatives(cropped, center=False) 

    rot_mat_linear = curvature.rot_mat_linear(full_der)
    mag_grad = curvature.mag_grad(full_der)
    rot_mat_quad = curvature.rot_mat_quad(full_der)
    fpp = curvature.fpp(full_der)
    fqq = curvature.fqq(full_der)

    full_curv = {'rot_mat_linear' : rot_mat_linear,
                 'mag_grad'       : mag_grad,
                 'rot_mat_quad'   : rot_mat_quad,
                 'fpp'            : fpp,
                 'fqq'            : fqq,
                }
    
    rot_mat_linear = curvature.rot_mat_linear(crop_der)
    mag_grad = curvature.mag_grad(crop_der)
    rot_mat_quad = curvature.rot_mat_quad(crop_der)
    fpp = curvature.fpp(crop_der)
    fqq = curvature.fqq(crop_der)

    crop_curv = {'rot_mat_linear' : rot_mat_linear,
                 'mag_grad'       : mag_grad,
                 'rot_mat_quad'   : rot_mat_quad,
                 'fpp'            : fpp,
                 'fqq'            : fqq,
                }

    title = title
    make_plots(image, cropped, full_der, crop_der, full_curv, crop_curv, title)
    
    mse_fx = MSE(full_der['fx'], crop_der['fx'])
    mse_fy = MSE(full_der['fy'], crop_der['fy'])
    mse_fxx = MSE(full_der['fxx'], crop_der['fxx'])
    mse_fyy = MSE(full_der['fyy'], crop_der['fyy'])
    mse_fxy = MSE(full_der['fxy'], crop_der['fxy'])

    return [mse_fx, mse_fy, mse_fxx, mse_fyy, mse_fxy] 

def get_images(params, idx):

    sample = os.path.join(params['data_path'], files[idx])
    sample = pickle.load(open(sample,"rb"))

    images = sample['all_near_fields']['near_fields_1550'].squeeze()
    y_images = images[1]

    y_magnitude = y_images[0]
    y_phase = y_images[1]     

    return y_magnitude, y_phase

def update_len_crop():
    pass

if __name__=="__main__":

    # idx just lets us grab a single sample (.pkl file) from our data subset
    idx = 0
    params = yaml.load(open("config.yaml"), Loader=yaml.FullLoader)

    # get access to the data subset
    files = os.listdir(params['data_path'])
    files = [file for file in files if os.path.isfile(os.path.join(params['data_path'], file))] 
    
    # grab the sample
    mag, phase = get_images(params, idx)

    # get number of pixels directly from the image - we're choosing the crop value here - we could do this smarter maybe
    params['len_pix_full'] = mag.shape[0] 
    params['len_pix_crop'] = mag.shape[0] - 10
    
    # get the grid coordinates for the plot based on the actual micron length of our image and the number of pixels. 
    x_full = np.linspace(-params['len_full']/2, params['len_full']/2, params['len_pix_full'])
    y_full = np.linspace(-params['len_full']/2, params['len_full']/2, params['len_pix_full'])
    
    params['X_grid'], params['Y_grid'] = np.meshgrid(x_full, y_full)
    
    test_images(params, mag) 
    embed()
    
""" 
    x_full = np.linspace(-len_full/2, len_full/2, len_pix_full)
    y_full = np.linspace(-len_full/2, len_full/2, len_pix_full)
    
    X_full, Y_full = np.meshgrid(x_full, y_full)

    Z_full = X_full
    title = "mag_increasing_with_x"
    mse_lists = []
    pixel_vals = []
    for i in range(3, 166, 10):
        len_pix_crop = i
        mse_list = test_images(X_full, Y_full, Z_full)
        mse_lists.append(mse_list)
        pixel_vals.append(len_pix_crop)
        
    plot_mse(mse_lists, pixel_vals, title)

    Z_full = Y_full
    mse_lists = []
    pixel_vals = []
    for i in range(3, 166, 10):
        len_pix_crop = i
        mse_list = test_images(X_full, Y_full, Z_full)
        mse_lists.append(mse_list)
        pixel_vals.append(len_pix_crop)

    plot_mse(mse_lists, pixel_vals, "mag_increasing_with_y")
    
    Z_full = X_full + Y_full
    mse_lists = []
    pixel_vals = []
    for i in range(3, 166, 10):
        len_pix_crop = i
        mse_list = test_images(X_full, Y_full, Z_full)
        mse_lists.append(mse_list)
        pixel_vals.append(len_pix_crop)


    plot_mse(mse_lists, pixel_vals, "mag_increasing_diag_upright")

    Z_full = np.random.uniform(X_full.min(), X_full.max(), size=(166,166))
    mse_lists = []
    pixel_vals = []
    for i in range(3, 166, 10):
        len_pix_crop = i
        mse_list = test_images(X_full, Y_full, Z_full)
        mse_lists.append(mse_list)
        pixel_vals.append(len_pix_crop)

    plot_mse(mse_lists, pixel_vals, "random")

    distance = np.sqrt(X_full**2 + Y_full**2)
    Z_full = X_full.min() + distance * (X_full.max() - X_full.min())
    mse_lists = []
    pixel_vals = []
    for i in range(3, 166, 10):
        len_pix_crop = i
        mse_list = test_images(X_full, Y_full, Z_full)
        mse_lists.append(mse_list)
        pixel_vals.append(len_pix_crop)

    plot_mse(mse_lists, pixel_vals, "bowl")

    distance = np.sqrt(X_full**2 + Y_full**2)
    Z_full = X_full.max() + distance * (X_full.min() - X_full.max())
    mse_lists = []
    pixel_vals = []
    for i in range(3, 166, 10):
        len_pix_crop = i
        mse_list = test_images(X_full, Y_full, Z_full)
        mse_lists.append(mse_list)
        pixel_vals.append(len_pix_crop)

    plot_mse(mse_lists, pixel_vals, "inverted_bowl")

    a = 1
    b = 2

    Z_full = a * X_full**2 - b * Y_full**2
    mse_lists = []
    pixel_vals = []
    for i in range(3, 166, 10):
        len_pix_crop = i
        mse_list = test_images(X_full, Y_full, Z_full)
        mse_lists.append(mse_list)
        pixel_vals.append(len_pix_crop)

    plot_mse(mse_lists, pixel_vals, "saddle")
"""
