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
 
    #fig, ax = plt.subplots()
    fig, ax = plt.subplots(1, 5, figsize=(15,5))
    plt.suptitle(f'MSE of derivatives for cropped vs. analytical \n {title}')

    x = pixel_vals
    ax[0].plot(x, mse_fx, label='fx')
    ax[1].plot(x, mse_fy, label='fy')
    ax[2].plot(x, mse_fxx, label='fxx')
    ax[3].plot(x, mse_fyy, label='fyy')
    ax[4].plot(x, mse_fxy, label='fxy')
    
    for axs in ax:
        axs.set_xlabel('number of pixels (cropped value)')
        axs.set_ylabel('mean squared error')
        axs.legend()
        #axs.set_title(f'MSE of derivatives for cropped vs. analytical \n {title}')
    
    #ax.legend()
    fig.tight_layout()
    plt.show()    
    
    if title is not None:
        fig.savefig(f"images/mse_{title}.png")

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

# can pass analytical_der and analytical_curv to compare cropped images to the ders and curv
# calculated from radii.
def image_plots(params, image, cropped, full_der, crop_der, full_curv, crop_curv, title=None):

    fig, ax = plt.subplots(1,2, figsize=(8,8))
    plt.suptitle(title)

    vmin= image.min()
    vmax= image.max()
   
    im = ax[0].pcolormesh(params['X_grid'], params['Y_grid'], image, vmin=vmin, vmax=vmax)
    ax[0].set_xticks(np.linspace(params['X_grid'].min(), params['X_grid'].max(), 5))
    ax[0].set_yticks(np.linspace(params['Y_grid'].min(), params['Y_grid'].max(), 5))
    ax[0].set_xlim([params['X_grid'].min(), params['X_grid'].max()])
    ax[0].set_ylim([params['Y_grid'].min(), params['Y_grid'].max()])
    ax[0].set_title("Full Image")
    ax[0].set_xlabel(r"$m$")
    ax[0].set_ylabel(r"$m$")
    ax[0].ticklabel_format(style='scientific')
   
    im2 = ax[1].pcolormesh(params['X_grid_crop'], params['Y_grid_crop'], cropped, vmin=vmin, vmax=vmax)
    ax[1].set_xticks(np.linspace(params['X_grid'].min(), params['X_grid'].max(), 5))
    ax[1].set_yticks(np.linspace(params['Y_grid'].min(), params['Y_grid'].max(), 5))
    ax[1].set_xlim([params['X_grid'].min(), params['X_grid'].max()])
    ax[1].set_ylim([params['Y_grid'].min(), params['Y_grid'].max()])
    ax[1].set_title("Cropped Image")
    ax[1].set_xlabel(r"$m$")
    ax[1].set_ylabel(r"$m$")
    
    cropped_rectangle1 = patches.Rectangle(
        (params['X_grid_crop'].min(), params['Y_grid_crop'].min()), 
        params['X_grid_crop'].max() - params['X_grid_crop'].min(),   
        params['Y_grid_crop'].max() - params['Y_grid_crop'].min(),   
        fill=False,                          
        linestyle='dotted',                  
        edgecolor='black',                   
        linewidth=2,                         
    )

    ax[0].add_patch(cropped_rectangle1)
    
    cropped_rectangle2 = patches.Rectangle(
        (params['X_grid_crop'].min(), params['Y_grid_crop'].min()), 
        params['X_grid_crop'].max() - params['X_grid_crop'].min(),   
        params['Y_grid_crop'].max() - params['Y_grid_crop'].min(),   
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
   
    fig.text(0.1, 0.9, f"  Analytical image ders: {full}")
    fig.text(0.1, 0.85, f"Cropped image ders: {crop}") 

    format_full_curv = format_curv(full_curv)
    format_crop_curv = format_curv(crop_curv)
    
    fig.text(0.1, 0.65, f"An   curv:\n{format_full_curv['rot_mat_linear']}\n{format_full_curv['mag_grad']}\n{format_full_curv['rot_mat_quad']}\n{format_full_curv['fpp']}\n{format_full_curv['fqq']}")
    fig.text(0.55, 0.65, f"Crop curv:\n{format_crop_curv['rot_mat_linear']}\n{format_crop_curv['mag_grad']}\n{format_crop_curv['rot_mat_quad']}\n{format_crop_curv['fpp']}\n{format_crop_curv['fqq']}")

    plt.tight_layout(rect=[0, 0, 1, 0.65])
    cbar = plt.colorbar(im2, ax=ax[:], shrink=0.2, location='bottom')
    plt.show()
    if title is not None:
        fig.savefig(f"images2/{title}.png")

# go from (1, 166, 166) to derivatives

# derivatives are calculated with fx--> and fy ^
# e.g. [2, 2, 2]                          and [0, 1, 2]
#      [1, 1, 1]                              [0, 1, 2]
#      [0, 0, 0]   has fx = 0 and fy = 1      [0, 1, 2]    has fx = 1 and fy = 0


def agg_der(arr, params): # choice=0 to return value of center pixel, choice=1 to return average
    
    choice = params['choice']

    if choice==0:
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

    elif choice==1:
        return np.average(arr)

def MSE(full_der, crop_der):

    full_der, crop_der = np.array(full_der), np.array(crop_der)
    return np.square(np.subtract(full_der, crop_der)).mean()

def derivatives(grid, params):
    
    #grid_flip = np.flip(grid,axis=0) # flip row if NOT using meshgrid
    #grid_flip = grid # this is for meshgrid
    #fy, fx = np.gradient(grid_flip)

    fy, fx = np.gradient(grid)

    #fxx = np.diff(grid_flip[1,:], 2)
    #fyy = np.diff(grid_flip[:,1], 2) 

    fxx = np.diff(grid[1,:], 2)
    fyy = np.diff(grid[:,1], 2)

    fxy = np.gradient(np.gradient(grid,axis=0),axis=1)
   
    fx = agg_der(fx, params)
    fy = agg_der(fy, params)
    fxx = agg_der(fxx, params)
    fyy = agg_der(fyy, params)
    fxy = agg_der(fxy, params)
    
    derivatives = {
                   'fx' : fx,
                   'fy' : fy,
                   'fxx': fxx,
                   'fyy': fyy,
                   'fxy': fxy,
                  }
    return derivatives


def get_cropped_im(params, image):

    start_row = (params['full_pix'] - params['crop_pix']) // 2
    end_row = start_row + params['crop_pix']
    
    start_col = (params['full_pix'] - params['crop_pix']) // 2
    end_col = start_col + params['crop_pix']
    
    params['start_row'], params['end_row'] = start_row, end_row 
    params['start_col'], params['end_col'] = start_col, end_col 
 
    cropped = image[start_row:end_row, start_col:end_col]
    
    x_crop = np.linspace(-params['crop_size']/2, params['crop_size']/2, params['crop_pix'])
    y_crop = np.linspace(-params['crop_size']/2, params['crop_size']/2, params['crop_pix'])
    params['X_grid_crop'], params['Y_grid_crop'] = np.meshgrid(x_crop, y_crop)
    
    return cropped

def get_curv(image, cropped, idx, choice, buffer=False):

    full_der = derivatives(image, params)
    
    crop_der = derivatives(cropped, params)
    if buffer == False:
        analytical_der = der_from_pickle(params, idx)
    else:
        analytical_der = der_from_pickle(params, idx, buffer=True)

    full_curv = {'rot_mat_linear' : curvature.rot_mat_linear(full_der),
                 'mag_grad'       : curvature.mag_grad(full_der),
                 'rot_mat_quad'   : curvature.rot_mat_quad(full_der),
                 'fpp'            : curvature.fpp(full_der),
                 'fqq'            : curvature.fqq(full_der),
                }

    analytical_curv = {'rot_mat_linear' : curvature.rot_mat_linear(analytical_der),
                 'mag_grad'       : curvature.mag_grad(analytical_der),
                 'rot_mat_quad'   : curvature.rot_mat_quad(analytical_der),
                 'fpp'            : curvature.fpp(analytical_der),
                 'fqq'            : curvature.fqq(analytical_der),
                }
    
    crop_curv = {'rot_mat_linear' : curvature.rot_mat_linear(crop_der),
                 'mag_grad'       : curvature.mag_grad(crop_der),
                 'rot_mat_quad'   : curvature.rot_mat_quad(crop_der),
                 'fpp'            : curvature.fpp(crop_der),
                 'fqq'            : curvature.fqq(crop_der),
                }

    if choice==0:
        return full_der, full_curv
    if choice==1:
        return analytical_der, analytical_curv
    if choice==2:
        return crop_der, crop_curv

def get_mse(params, image, idx, title=None, plot_im=False):
    
    cropped = get_cropped_im(params, image)

    full_der, full_curv = get_curv(image, cropped, idx, choice=0)
    analytical_der, analytical_curv = get_curv(image, cropped, idx, choice=1)
    crop_der, crop_curv = get_curv(image, cropped, idx, choice=2)
    
    mse_fx = MSE(analytical_der['fx'], crop_der['fx'])
    mse_fy = MSE(analytical_der['fy'], crop_der['fy'])
    mse_fxx = MSE(analytical_der['fxx'], crop_der['fxx'])
    mse_fyy = MSE(analytical_der['fyy'], crop_der['fyy'])
    mse_fxy = MSE(analytical_der['fxy'], crop_der['fxy'])
    
    if plot_im == True:
        image_plots(params, image, cropped, analytical_der, crop_der, analytical_curv, crop_curv)

    return [mse_fx, mse_fy, mse_fxx, mse_fyy, mse_fxy] 

def get_image(params, idx):

    sample = os.path.join(params['data_path'], files[idx])
    sample = pickle.load(open(sample,"rb"))

    images = sample['all_near_fields']['near_fields_1550'].squeeze()
    y_images = images[1]

    y_magnitude = y_images[0]
    y_phase = y_images[1]     

    return y_magnitude, y_phase

def der_from_pickle(params, idx, buffer=False):
    
    files = os.listdir(params['data_path'])
    files = [file for file in files if os.path.isfile(os.path.join(params['data_path'], file))] 
    
    if buffer == False:
        sample = os.path.join(params['data_path'], files[idx])
    else:
        sample = os.path.join(params['data_path'],"000000.pkl")

    sample = pickle.load(open(sample, "rb"))
    
    ders = sample['derivatives']
   
    ders = ders.numpy().flatten() 
    derivatives = {
                   'fx' : ders[0],
                   'fy' : ders[1],
                   'fxx': ders[2],
                   'fyy': ders[3],
                   'fxy': ders[4],
                  }
    
    return derivatives

def update_params(params):

    x_full = np.linspace(-params['full_size']/2, params['full_size']/2, params['full_pix'])
    y_full = np.linspace(-params['full_size']/2, params['full_size']/2, params['full_pix'])
    params['X_grid'], params['Y_grid'] = np.meshgrid(x_full, y_full)
    params['crop_size'] = params['crop_pix'] * ( params['full_size'] / params['full_pix'] ) 

    return params

def plot_raw_vals(params, idx, files):

    crop_vals = list(range(3, 167, 10))
    crop_vals.append(166)                  
    
    mag, phase = get_image(params, idx)
    image = phase
   
    fx_list_an, fx_list_im = [], [] 
    fy_list_an, fy_list_im = [], []
    fxx_list_an, fxx_list_im = [], []
    fyy_list_an, fyy_list_im = [], []
    fxy_list_an, fxy_list_im = [], []

    for i in crop_vals:

        params['full_pix'] = image.shape[0]
        params['crop_pix'] = i    
        
        params = update_params(params)

        cropped = get_cropped_im(params, image)

        crop_der, crop_curv = get_curv(image, cropped, idx, choice=2)       
        analytical_der, analytical_curv = get_curv(image, cropped, idx, choice=1)

        fx_list_an.append(analytical_der['fx'])
        fx_list_im.append(crop_der['fx'])
        
        fy_list_an.append(analytical_der['fy'])
        fy_list_im.append(crop_der['fy'])

        fxx_list_an.append(analytical_der['fxx'])
        fxx_list_im.append(crop_der['fxx'])
    
        fyy_list_an.append(analytical_der['fyy'])
        fyy_list_im.append(crop_der['fyy'])

        fxy_list_an.append(analytical_der['fxy'])
        fxy_list_im.append(crop_der['fxy'])

    fig, ax = plt.subplots(1,5,figsize=(15,5))
    plt.suptitle(f"Image_{idx}")
    x = list(range(len(fx_list_an)))
    
    ax[0].scatter(x, fx_list_an, label='analytical')
    ax[0].scatter(x, fx_list_im, label='image')
    ax[0].set_title("fx")
    ax[1].scatter(x, fy_list_an, label='analytical')
    ax[1].scatter(x, fy_list_im, label='image')
    ax[1].set_title("fy")
    ax[2].scatter(x, fxx_list_an, label='analytical')
    ax[2].scatter(x, fxx_list_im, label='image')
    ax[2].set_title("fxx")
    ax[3].scatter(x, fyy_list_an, label='analytical')
    ax[3].scatter(x, fyy_list_im, label='image')
    ax[3].set_title("fyy")
    ax[4].scatter(x, fxy_list_an, label='analytical')
    ax[4].scatter(x, fxy_list_im, label='image')
    ax[4].set_title("fxy")

    for axs in ax:
        axs.legend()
        axs.set_xlabel("crop value")
        axs.set_ylabel("der value")
    plt.tight_layout()
    plt.show() 

# if you look at mins and maxes, you can confirm that the range of phase images is [-pi, pi]
def phase_min_max(params, files):
    
    phase_images = []
    mins = []
    maxes = []
    for idx in range(len(files)):
        mag, phase = get_image(params, idx)

        params['full_pix'] = phase.shape[0]
        params['crop_pix'] = phase.shape[0]

        params = update_params(params)

        im = get_cropped_im(params, phase)

        phase_images.append(im)
        mins.append(im.min())
        maxes.append(im.max())

    phases = []
    phase_mins = []
    phase_maxes = []
    for idx in range(len(files)):

        sample = os.path.join(params['data_path'], files[idx])
        sample = pickle.load(open(sample, "rb"))
        
        phase_list = sample['phases'].flatten()
        phases.append(phase_list)
        phase_mins.append(phase_list.min())
        phase_maxes.append(phase_list.max())


def der_from_radii(params, radii):
    
    phases = np.asarray(interpolate.radii_to_phase(radii))
    phases = phases.reshape((3,3))
    der = derivatives(phases, params)
    
    curv = {'rot_mat_linear'      : curvature.rot_mat_linear(der),
                 'mag_grad'       : curvature.mag_grad(der),
                 'rot_mat_quad'   : curvature.rot_mat_quad(der),
                 'fpp'            : curvature.fpp(der),
                 'fqq'            : curvature.fqq(der),
                }

    return der, curv

def buffer_study(params):

    #metadata_filename = os.path.join(params['buffer_path'], "gaussian_metadata_with_buffer_5.000.pkl")
    #x, y, z, w = pickle.load(open(metadata_filename, "rb"))
    #fig, ax = plt.subplots(figsize=(5,5))
    #ax.pcolormesh(x, y, cropped, cmap="hsv")
    
    #sample = os.path.join(params['data_path'], "000000.pkl")
    #sample = pickle.load(open(sample, "rb"))
    #phases = sample['phases'].flatten()

    #folder = "uniform_min_rad"
    folder = "inc_with_y"

    phase_im = pickle.load(open(os.path.join(params['buffer_path'],folder,"eps_slice.pkl"),"rb"))
    phase_im = np.angle(phase_im)
    params['full_pix'] = phase_im.shape[0]
    params['crop_pix'] = 166
    params = update_params(params)

    # we need to reset the phase_im to a cropped 166,166 - removing the buffer.
    phase_im = get_cropped_im(params, phase_im)
    # now we reset the full_pix value to the "new" full value -- it's gonna be 166.
    params['full_pix'] = phase_im.shape[0]

    #params['crop_pix'] = 166
    #params = update_params(params)
    cropped = get_cropped_im(params, phase_im) 

    # get analytical derivatives

    # this works if we have a data sample we can grab from. not for toy probs
    #an_der, an_curv = get_curv(phase_im, cropped, idx=None, choice=1, buffer=True) 
    radii = [0.075 for _ in range(9)]
    an_der, an_curv = der_from_radii(params, radii) 
 
    #embed()
    # get image derivatives
    im_der, im_curv = get_curv(phase_im, cropped, idx=None, choice=2, buffer=True)
    # get MSE

    embed()
    
if __name__=="__main__":

    params = yaml.load(open("config.yaml"), Loader=yaml.FullLoader)

    buffer_study(params)
    # get access to the data subset
#    files = os.listdir(params['data_path'])
#    files = [file for file in files if os.path.isfile(os.path.join(params['data_path'], file))] 
       
#    phase_min_max(params, files) 
 #   for idx in range(len(files)):
 #       plot_raw_vals(params, idx, files)

#    mag, phase = get_image(params, idx=0)
#
#    params['full_pix'] = mag.shape[0]
#    params['crop_pix'] = params['full_pix']
#
#    params = update_params(params)
#
#    im = get_cropped_im(params, mag)
#    
#    mse_list = get_mse(params, phase, idx=0, plot_im=True)   
#
#    embed()
    # use this for loop to look at all the images and their ders/
    # curv values.
#    for i in range(len(files)):  # here, 'i' is the index of the image (we're looping through images)
#
#        idx = i 
#        # grab the sample
#        mag, phase = get_image(params, idx)
#
#        # get number of pixels directly from the image
#        params['full_pix'] = mag.shape[0] 
#        params['crop_pix'] = int(params['full_pix'] / 3)
#
#        # other params get set here, including crop params
#        params = update_params(params)
#
#        im = get_cropped_im(params, mag)
#        
#        run_test(params, mag) 

    # use this for loop to get a mse plot for a single image.
#    crop_vals = list(range(3, 167, 10))
#    crop_vals.append(166)
#
#    idx = 0 # this is the image we're going to look at
#    mse_lists = []
#    pixel_vals = []
#    for i in crop_vals: # here, 'i' is the number of pixels in the cropped image. we're only looking at one image.
#                                                                                                                           
#        mag, phase = get_image(params, idx) # idx is which image we're getting
#        params['full_pix'] = mag.shape[0]
#        params['crop_pix'] = i
#                                                                                                                               
#        params = update_params(params)        
#        
#        im = get_cropped_im(params, phase)
#        
#        mse_list = get_mse(params, phase, idx, plot_im=False)
#        mse_lists.append(mse_list)
#        pixel_vals.append(params['crop_pix'])
#    plot_mse(mse_lists, pixel_vals, title=None)        
    
   # get mse plots for all the images 
#    print(f"{len(files)} to look at") 
#    for j in range(len(files)): 
#        idx = j # this is the image we're going to look at
#        mse_lists = []
#        pixel_vals = []
#        for i in crop_vals:
# 
#            mag, phase = get_image(params, idx) # idx is which image we're getting
#            params['full_pix'] = mag.shape[0]
#            params['crop_pix'] = i
#
#            params = update_params(params)        
#            
#            im = get_cropped_im(params, phase)
#            
#            mse_list = get_mse(params, phase, idx, plot_im=False)
#            mse_lists.append(mse_list)
#            pixel_vals.append(params['crop_pix'])
#
#        plot_mse(mse_lists, pixel_vals, title=f"Image_phase_{idx}")


  
