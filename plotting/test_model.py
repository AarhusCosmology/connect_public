"""
This plotting script can be used to test a models performance on the
stored test data. The syntax is 'python test_model.py <path to model>',
and up to three models can be given. Only the first model will be used
to produce cmb spectra, but all models given will be included in the
error plot. The error plot will be saved within the first model specified
as '<path to model>/plots/error.pdf'.

Author: Andreas Nygaard (2022)

"""

import os
import sys
import warnings
import pickle as pkl

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import matplotlib

import PlanckLogLinearScale

# index of test data to use for cmb spectra
n=0

# color of axes and all text in the error plot
color_of_axis_and_text = 'k'

# use latex?
latex = False




################################################################

name = sys.argv[1]
model = tf.keras.models.load_model(name, compile=False)
with open(name+'/test_data.pkl', 'rb') as f:
    test_data = pkl.load(f)



try:
    model_params = np.array(test_data[0])
    output_data     = np.array(test_data[1])
except:
    test_data = tuple(zip(*test_data))
    model_params = np.array(test_data[0])
    output_data     = np.array(test_data[1])

try:    
    with open(name+'/output_info.pkl', 'rb') as f:
        output_info = pkl.load(f)
    pickle_file = True
    warnings.warn("You are using CONNECT models from an old version (before v23.6.0). Support for this is deprecated and will be removed in a later update.")
    
except:
    pickle_file = False
    output_info = eval(model.get_raw_info().numpy().decode('utf-8'))

if pickle_file:
    try:
        if output_info['normalize']['method'] == 'standardization':
            normalize = 'standardization'
        elif output_info['normalize']['method'] == 'log':
            normalize = 'log'
        elif output_info['normalize']['method'] == 'min-max':
            normalize = 'min-max'
        elif output_info['normalize']['method'] == 'factor':
            normalize = 'factor'
        else:
            normalize = 'factor'
    except:
        normalize = 'standardization'

output_predict  = model.predict(model_params, verbose=0)

    
if pickle_file and normalize == 'standardization':
    mean = output_info['normalize']['mean']
    var  = output_info['normalize']['variance']
    output_predict = output_predict * np.sqrt(var) + mean
    output_data = output_data * np.sqrt(var) + mean
elif pickle_file and normalize == 'min-max':
    x_min = np.array(output_info['normalize']['x_min'])
    x_max = np.array(output_info['normalize']['x_max'])
    output_predict = output_predict * (x_max - x_min) + x_min
    output_data = output_data * (x_max - x_min) + x_min


out_predict = {}
out_data    = {}
for output in output_info['output_Cl']:
    lim0 = output_info['interval']['Cl'][output][0]
    lim1 = output_info['interval']['Cl'][output][1]
    out_data[output]    = output_data[n][lim0:lim1]
    out_predict[output] = output_predict[n][lim0:lim1]
    if pickle_file and normalize == 'log':
        for offset in list(reversed(output_info['normalize']['Cl'][output])):
            out_predict[output]=np.exp(out_predict[output]) - offset
            out_data[output]=np.exp(out_data[output]) - offset


if 'output_derived' in output_info.keys():
    for output in output_info['output_derived']:
        if output != '100*theta_s':
            idx = output_info['interval']['derived'][output]
            out_data[output]    = output_data[n][idx]
            out_predict[output] = output_predict[n][idx]
            if pickle_file and normalize == 'log':
                for offset in list(reversed(output_info['normalize']['derived']['100*theta_s'])):
                    out_predict[output]=np.exp(out_predict[output]) - offset
                    out_data[output]=np.exp(out_data[output]) - offset

for output in output_info['output_Cl']:
    if pickle_file and normalize == 'factor':
        normalize_factor = output_info['normalize']['Cl'][output]
    plt.figure(figsize=(10,7))
    ell        = output_info['ell']
    ll         = np.linspace(2,max(ell)+7,int(max(ell)-1+7))
    Cl_predict = out_predict[output]
    Cl_data    = out_data[output]
    Cl_pre_sp  = CubicSpline(ell,Cl_predict, bc_type = 'natural', extrapolate=True)
    Cl_dat_sp  = CubicSpline(ell,Cl_data,  bc_type = 'natural', extrapolate=True)
    if pickle_file and normalize == 'factor':
        plt.plot(ll, Cl_dat_sp(ll)/normalize_factor,'k-',lw=3,label='CLASS')
    else:
        plt.plot(ll, Cl_dat_sp(ll),'k-',lw=3,label='CLASS')
    if pickle_file and normalize == 'factor':
        plt.plot(ll, Cl_pre_sp(ll)/normalize_factor,'r-',lw=3,label='CONNECT')
    else:
        plt.plot(ll, Cl_pre_sp(ll),'r-',lw=3,label='CONNECT')
    plt.legend()
    plt.xscale('log')
    plt.xlabel(r'$\ell$')
    plt.ylabel(r'$C_{\ell}\times\ell(\ell+1)/2\pi$')
    plt.title(output)

if 'output_derived' in output_info.keys(): 
    for output in output_info['output_derived']:
        if output != '100*theta_s':
            if pickle_file and normalize == 'factor':
                normalize_factor = output_info['normalize']['derived'][output]
            print(output)
            if pickle_file and normalize == 'factor':
                print('CLASS:',out_data[output]/normalize_factor)
            else:
                print('CLASS:',out_data[output])
            if pickle_file and normalize == 'factor':
                print('CONNECT:',out_predict[output]/normalize_factor)
            else:
                print('CONNECT:',out_predict[output])


def rms(x):
    return np.sqrt(np.mean(x**2))



    
l = np.linspace(2,2500,2499)


def get_error(path,spectrum):
    model = tf.keras.models.load_model(path, compile=False)

    with open(path + '/test_data.pkl', 'rb') as f:
        test_data = pkl.load(f)
    try:
        with open(path + '/output_info.pkl', 'rb') as f:
            output_info = pkl.load(f)
        pickle_file = True
    except:
        pickle_file = False
        output_info = eval(model.get_raw_info().numpy().decode('utf-8'))

    ell = output_info['ell']
    if pickle_file:
        try:
            if output_info['normalize']['method'] == 'standardization':
                normalize = 'standardization'
            elif output_info['normalize']['method'] == 'log':
                normalize = 'log'
            elif output_info['normalize']['method'] == 'min-max':
                normalize = 'min-max'
            elif output_info['normalize']['method'] == 'factor':
                normalize = 'factor'
            else:
                normalize = 'factor'
        except:
            normalize = 'standardization'
            
    try:
        model_params = test_data[0]
        Cls_data     = test_data[1]
    except:
        test_data = tuple(zip(*test_data))
        model_params = np.array(test_data[0])
        Cls_data     = np.array(test_data[1])


    v = tf.constant(model_params)
    Cls_predict = model(v).numpy()

    if pickle_file and normalize == 'standardization':
        mean = output_info['normalize']['mean']
        var  = output_info['normalize']['variance']
        Cls_predict = Cls_predict * np.sqrt(var) + mean
        Cls_data = Cls_data * np.sqrt(var) + mean
    elif pickle_file and normalize == 'min-max':
        x_min = np.array(output_info['normalize']['x_min'])
        x_max = np.array(output_info['normalize']['x_max'])
        Cls_predict = Cls_predict * (x_max - x_min) + x_min
        Cls_data = Cls_data * (x_max - x_min) + x_min
    
    lim0 = output_info['interval']['Cl'][spectrum][0]
    lim1 = output_info['interval']['Cl'][spectrum][1]

    errors = []
    for j, (cls_d, cls_p) in enumerate(zip(Cls_data, Cls_predict)):
        if pickle_file and normalize == 'factor':
            err = ((np.array(cls_d[lim0:lim1])-np.array(cls_p[lim0:lim1]))/output_info['normalize']['Cl'][spectrum])/rms(np.array(cls_d[lim0:lim1]))
        elif pickle_file and normalize == 'log':
            clsp = cls_p[lim0:lim1]
            clsd = cls_d[lim0:lim1]
            for offset in list(reversed(output_info['normalize']['Cl'][spectrum])):
                clsp=np.exp(clsp) - offset
                clsd=np.exp(clsd) - offset
            err = (np.array(clsd)-np.array(clsp))/rms(np.array(clsd))
        else:
            err = (np.array(cls_d[lim0:lim1])-np.array(cls_p[lim0:lim1]))/rms(np.array(cls_d[lim0:lim1]))
        errors.append(abs(err))

    errors = np.array(errors).T
    return errors, ell





height = 2.2
width = 6.0874173228
fontsize = 11/1.2*1.5

if latex:
    latex_preamble = [
        r'\usepackage{lmodern}',
        r'\usepackage{amsmath}',
        r'\usepackage{amsfonts}',
        r'\usepackage{amssymb}',
        r'\usepackage{mathtools}',
    ]
    matplotlib.rcParams.update({
        'text.usetex'        : True,
        'font.family'        : 'serif',
        'font.serif'         : 'cmr10',
        'font.size'          : fontsize,
        'mathtext.fontset'   : 'cm',
        'text.latex.preamble': latex_preamble,
    })


model_paths = []
model_names = []
for arg in sys.argv[1:]:
    model_paths.append(arg)
    model_names.append(arg.split('/')[-1])

percentiles = [0.682,0.954]
alpha_list  = [0.2,    0.1,   0.05]
if len(model_names) == 3:
    c_list      = ['green','red','blue']
elif len(model_names) == 2:
    c_list      = ['red','blue','green']
else:
    c_list      = ['blue','red','green']
fc_array    = [['cyan',   'blue', 'navy'],
               ['orange', 'red',  'crimson'],
               ['lightgreen','forestgreen','darkgreen']]



change=200

PlanckLogLinearScale.new_change(change)

fig, axs = plt.subplots(2,len(output_info['output_Cl']),figsize=(width, height), gridspec_kw={'height_ratios':[1,4]})
fig.subplots_adjust(wspace=0)
for i in range(len(output_info['output_Cl'])):
    axs[0,i].axis('off')


for k, spectrum in enumerate(output_info['output_Cl']):
    y_max = 0
    y_min = 0
    for i, path in enumerate(model_paths): 
        model_name = model_names[i]
        errors, l_red =  get_error(path, spectrum)

        max_error=[]
        err_lower_array = []
        err_upper_array = []
        for errs in errors:
            err_lower_list = []
            err_upper_list = []
            for p in percentiles:
                err_upper_list.append(np.percentile(errs, 100*p))

            err_upper_array.append(err_upper_list)
            max_error.append(max(errs))
        if max(np.array(err_upper_array).flatten()) > y_max:
            y_max = max(np.array(err_upper_array).flatten())
            
        for j, p in reversed(list(enumerate(sorted(percentiles)))):
            err_u = CubicSpline(l_red,np.array(err_upper_array).T[j])
            
            axs[1,k].fill(np.array([l[0]]+list(l[:change-1])+[l[change-2]]),
                     np.array([0]+list(abs(err_u(l[:change-1])))+[0]),
                     fc=c_list[i],
                     alpha=alpha_list[j],
                     zorder=i)
            axs[1,k].fill(np.array([l[change-2]]+list(l[change-2:])+[l[-1]]),
                     np.array([0]+list(abs(err_u(l[change-2:])))+[0]),
                     fc=c_list[i],
                     alpha=alpha_list[j],
                     zorder=i)

            axs[1,k].axvline(x=change, linestyle="-", color="lightgrey", zorder=-3)
            if j==len(percentiles)-1 and k==0:
                axs[1,k].plot(l,abs(err_u(l)),
                              c=c_list[i],lw=1, zorder=i, alpha=list(reversed(percentiles))[j])
            else:
                axs[1,k].plot(l,abs(err_u(l)),
                              c=c_list[i],lw=1, zorder=i, alpha=list(reversed(percentiles))[j])

    if k==0:
        custom_lines=[matplotlib.lines.Line2D([],[],c=c_list[2]),
                      matplotlib.lines.Line2D([],[],c=c_list[1]),
                      matplotlib.lines.Line2D([],[],c=c_list[0])]
        if len(model_names) == 2:
            custom_lines = custom_lines[1:]
        elif len(model_names) == 1:
            custom_lines = custom_lines[2:]
        axs[1,k].legend(custom_lines,model_names,bbox_to_anchor=(2.64,1.5),ncol=len(model_names),fontsize=fontsize/1.5)

    axs[1,k].set_yscale('log')
    axs[1,k].set_xscale('planck')

    axs[1,k].set_xlim([2,2500])
    if k==1:
        axs[1,k].set_xlabel(r'$\ell$')
    if k == 0:
        if latex:
            axs[1,k].set_ylabel(r'$\dfrac{\left\vert \mathcal{D}_{\ell}^{\textsc{connect}}-\mathcal{D}_{\ell}^{\textsc{class}}\right\vert}{{\rm rms}\left(\mathcal{D}_{\ell}^{\textsc{class}}\right)}$')
        else:
            axs[1,k].set_ylabel(r'$|D_{\ell}^{\rm connect}-D_{\ell}^{\rm class}| /{\rm rms}(D_{\ell}^{\rm class})$')
    else:
        axs[1,k].yaxis.set_ticklabels([])
    axs[1,k].set_ylim([5e-6,5e-2])
    axs[1,k].set_title(f'{spectrum.upper()}', color=color_of_axis_and_text)
    axs[1,k].tick_params(axis='both', which='both', direction='in')
    xticks = [1e+1, 1e+2, 1e+3, 2e+3]
    xticks_minor = [2,3,4,5,6,7,8,9,20,30,40,50,60,70,80,90,200,300,400,500,600,700,800,900,1100,1200,1300,1400,1500,1600,1700,1800,1900,2100,2200,2300,2400,2500] 
    yticklabels=[r'$10^{-2}$',r'$10^{-3}$',r'$10^{-4}$',r'$10^{-5}$']
    xticklabels=[r'$10^{1}$',r'$10^{2}$','1000','2000']
    axs[1,k].set_xticks(xticks)
    axs[1,k].set_xticks(xticks_minor, minor=True)
    axs[1,k].xaxis.set_ticklabels(xticklabels, color=color_of_axis_and_text)
    axs[1,k].set_yticks([1e-2,1e-3,1e-4,1e-5])
    if k==0:
        axs[1,k].yaxis.set_ticklabels(yticklabels, color=color_of_axis_and_text)

factor=1.09
factor_y=0.91
offset=axs[1,0].get_position().x0*(factor - 1) - 0.03
offset_y=axs[0,0].get_position().y1*(factor_y - 1) - 0.06
for ax in axs.flatten():
    box=ax.get_position()
    box.x0 *= factor
    box.x0 += -offset-0.005
    box.x1 *= factor
    box.x1 += -offset-0.005
    box.y0 *= factor_y
    box.y0 += -offset_y + 0.0
    box.y1 *= factor_y
    box.y1 += -offset_y + 0.0
    ax.set_position(box)
    

    ax.xaxis.label.set_color(color_of_axis_and_text)           #setting up X-axis label color
    ax.yaxis.label.set_color(color_of_axis_and_text)           #setting up Y-axis label color

    ax.set_facecolor('w')

#fig.patch.set_alpha(0)
    
if not os.path.isdir(model_paths[0]+'/plots'):
    os.mkdir(model_paths[0]+'/plots')

plt.savefig(model_paths[0]+f'/plots/error.pdf', facecolor=fig.get_facecolor())

plt.show()
