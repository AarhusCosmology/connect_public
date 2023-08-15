import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CloughTocher2DInterpolator
from pynput.keyboard import Key, Listener

profile_dir = sys.argv[1]

def get_profile_lkl(file):
    with open(file,'r') as f:
        f_list = list(f)

    parameters = f_list[0].replace('\n','').split('\t')

    lkl = []
    par_1 = []
    par_2 = []
    for line in f_list[1:]:
        pars = np.float64(line.replace('\n','').split('\t'))
        if pars[2] < 5000.0:
            par_1.append(pars[0])
            par_2.append(pars[1])
            lkl.append(pars[2])

    post = np.exp(-0.5*np.array(lkl))
    return np.array(par_1), np.array(par_2), np.array(lkl)


coords = []


def onclick(event, par_i, par_j, pi, pj):

    global ix, iy
    ix, iy = event.xdata, event.ydata
    global coords
    if ix != None and iy != None:
        coords.append((par_i, par_j, ix, iy))

        
    t = np.linspace(0,2*np.pi)
    x_c = np.cos(t) * 0.03 * (np.max(pi)-np.min(pi)) + ix
    y_c = np.sin(t) * 0.03* (np.max(pj)-np.min(pj)) + iy
       
    plt.close()
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    X = np.linspace(min(pi), max(pi))
    Y = np.linspace(min(pj), max(pj))
    X, Y = np.meshgrid(X, Y)
    interp = CloughTocher2DInterpolator(list(zip(pi, pj)), lkl, fill_value=100000, rescale=True)
    Z = interp(X, Y)
    c = 'r'
    ax.contour(X,Y,Z,levels=[lkl_min_global+11.83/2],zorder=4,colors=c,linewidths=3)
    ax.contour(X,Y,Z,levels=[lkl_min_global+6.18/2],zorder=4,colors=c,linewidths=3)
    ax.contour(X,Y,Z,levels=[lkl_min_global+2.3/2],zorder=4,colors=c,linewidths=3)
    ax.scatter(pi,pj,s=50, c=lkl, cmap='gist_rainbow')
    ax.set_xlabel(par_i)
    ax.set_ylabel(par_j)
    new_x = []
    new_y = []
    for c in coords:
        if c[0] == par_i and c[1] == par_j:
            new_x.append(c[2])
            new_y.append(c[3])
    ax.scatter(new_x,new_y,s=50, c='k')
    ax.plot(x_c, y_c, 'r-', lw=1.5)
    
    cid = fig.canvas.mpl_connect('button_press_event', lambda event: onclick(event, par_i, par_j, pi, pj))
    plt.show()
        
    return coords

with open(os.path.join(profile_dir, 'global_bestfit.txt'), 'r') as f:
    f_list = list(f)
params = f_list[0].strip().split('\t')[1:]
lkl_min_global = np.float64(f_list[1].strip().split('\t'))[0]

for i, par_i in enumerate(params):
    for j, par_j in enumerate(params):
        if j > i:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            filename = os.path.join(profile_dir, f'{par_i}-{par_j}.txt')
            pi, pj, lkl = get_profile_lkl(filename)
            X = np.linspace(min(pi), max(pi))
            Y = np.linspace(min(pj), max(pj))
            X, Y = np.meshgrid(X, Y)
            interp = CloughTocher2DInterpolator(list(zip(pi, pj)), lkl, fill_value=100000, rescale=True)
            Z = interp(X, Y)
            c = 'r'
            ax.contour(X,Y,Z,levels=[lkl_min_global+11.83/2],zorder=4,colors=c,linewidths=3)
            ax.contour(X,Y,Z,levels=[lkl_min_global+6.18/2],zorder=4,colors=c,linewidths=3)
            ax.contour(X,Y,Z,levels=[lkl_min_global+2.3/2],zorder=4,colors=c,linewidths=3)
            
            ax.scatter(pi,pj,s=50, c=lkl, cmap='gist_rainbow')
            ax.set_xlabel(par_i)
            ax.set_ylabel(par_j)

            cid = fig.canvas.mpl_connect('button_press_event', lambda event: onclick(event, par_i, par_j, pi, pj))
            plt.show()


line_numbers = {}
for par_i, par_j, x, y in coords:
    filename = os.path.join(profile_dir, f'{par_i}-{par_j}.txt')
    if filename not in line_numbers:
        line_numbers[filename] = []
    line_numbers[filename].append((x,y))


with open('add_points.txt','w') as f:
    for key in line_numbers:
        for xy in line_numbers[key]:
            line = key
            line += '\t'+str(xy[0])
            line += '\t'+str(xy[1])            
            line += '\n'
            f.write(line)

