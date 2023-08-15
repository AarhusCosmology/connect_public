import os
import sys

import numpy as np
import matplotlib.pyplot as plt

profile_dir = sys.argv[1]

def get_profile_lkl_1d(file):
    with open(file,'r') as f:
        f_list = list(f)

    parameters = f_list[0].replace('\n','').split('\t')

    lkl = []
    par = []
    for line in f_list[1:]:
        pars = np.float64(line.replace('\n','').split('\t'))
        if pars[1] < 5000.0:
            par.append(pars[0])
            lkl.append(pars[1])

    return np.array(par), np.array(lkl)

def get_profile_lkl_2d(file):
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

    return np.array(par_1), np.array(par_2), np.array(lkl)


coords_1d = []
coords_2d = []


def onclick_1d(event, par, p, lkl):
    global ix, iy
    ix, iy = event.xdata, event.ydata

    idx_min = np.argmin(np.sum(np.power((np.array([p,lkl]).T - np.array([ix,iy]))
                                        *np.array([1/(max(p)-min(p)),1/(max(lkl)-min(lkl))]),2),
                               axis=1))
    true_p   = p[idx_min]
    true_lkl = lkl[idx_min]

    global coords_1d
    tup = (par, true_p, true_lkl)
    if tup not in coords_1d:
        coords_1d.append(tup)
    
    t = np.linspace(0,2*np.pi)
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(p, lkl, 'b*',ms=10)
    ax.set_xlabel(par)
    ax.set_ylabel(r"$\log(\mathcal{L})$")
    for c in coords_1d:
        if c[0] == par:
            x_c = np.cos(t) * 0.03 * (max(p)-min(p)) + c[1]
            y_c = np.sin(t) * 0.03 * (max(lkl)-min(lkl)) + c[2]
            ax.plot(x_c, y_c, 'r-', lw=1.5)
    
    cid = fig.canvas.mpl_connect('button_press_event', lambda event: onclick_1d(event, par, p, lkl))
    plt.show()
        
    return coords_1d

def onclick_2d(event, par_i, par_j, pi, pj):
    global ix, iy
    ix, iy = event.xdata, event.ydata

    idx_min = np.argmin(np.sum(np.power((np.array([pi,pj]).T - np.array([ix,iy]))
                                        *np.array([1/(max(pi)-min(pi)),1/(max(pj)-min(pj))]),2),
                               axis=1))
    true_i = pi[idx_min]
    true_j = pj[idx_min]

    global coords_2d
    tup = (par_i, par_j, true_i, true_j)
    if tup not in coords_2d:
        coords_2d.append(tup)
    
    t = np.linspace(0,2*np.pi)
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(pi,pj,s=50, c=lkl, cmap='gist_rainbow')
    ax.set_xlabel(par_i)
    ax.set_ylabel(par_j)
    for c in coords_2d:
        if c[0] == par_i and c[1] == par_j:
            x_c = np.cos(t) * 0.03 * (max(pi)-min(pi)) + c[2]
            y_c = np.sin(t) * 0.03 * (max(pj)-min(pj)) + c[3]
            ax.plot(x_c, y_c, 'r-', lw=1.5)
    
    cid = fig.canvas.mpl_connect('button_press_event', lambda event: onclick_2d(event, par_i, par_j, pi, pj))
    plt.show()
        
    return coords_2d


with open(os.path.join(profile_dir, 'global_bestfit.txt'), 'r') as f:
    f_list = list(f)
params = f_list[0].strip().split('\t')[1:]


for i, par in enumerate(params):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    filename = os.path.join(profile_dir, f'{par}.txt')
    p, lkl = get_profile_lkl_1d(filename)
    ax.plot(p, lkl, 'b*',ms=10)
    ax.set_xlabel(par)
    ax.set_ylabel(r'$\log(\mathcal{L})$')

    cid = fig.canvas.mpl_connect('button_press_event', lambda event: onclick_1d(event, par, p, lkl))
    plt.show()
for i, par_i in enumerate(params):
    for j, par_j in enumerate(params):
        if j > i:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            filename = os.path.join(profile_dir, f'{par_i}-{par_j}.txt')
            pi, pj, lkl = get_profile_lkl_2d(filename)
            ax.scatter(pi,pj,s=50, c=lkl, cmap='gist_rainbow')
            ax.set_xlabel(par_i)
            ax.set_ylabel(par_j)

            cid = fig.canvas.mpl_connect('button_press_event', lambda event: onclick_2d(event, par_i, par_j, pi, pj))
            plt.show()


line_numbers = {}
for par, x, _ in coords_1d:
    filename = os.path.join(profile_dir, f'{par}.txt')
    if filename not in line_numbers:
        line_numbers[filename] = []
    with open(filename,'r') as f:
        f_list = list(f)
    for n, line in enumerate(f_list):
        if n != 0:
            xx = np.float64(line.replace('\n','').split('\t'))[:2]
            if xx[0] == x:
                line_numbers[filename].append(n)

for par_i, par_j, x, y in coords_2d:
    filename = os.path.join(profile_dir, f'{par_i}-{par_j}.txt')
    if filename not in line_numbers:
        line_numbers[filename] = []
    with open(filename,'r') as f:
        f_list = list(f)
    for n, line in enumerate(f_list):
        if n != 0:
            xy = np.float64(line.replace('\n','').split('\t'))[:2]
            if xy[0] == x and xy[1] == y:
                line_numbers[filename].append(n)

                

with open('wrong_points.txt','w') as f:
    for key in line_numbers:
        line = key
        for n in line_numbers[key]:
            line += '\t'+str(n)
        line += '\n'
        f.write(line)

