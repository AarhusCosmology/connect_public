import os

import numpy as np
import tensorflow as tf

from source.profile_likelihoods.get_bins import Cut_histogram


def get_data_from_mcmc_bins_2d(param_names, param_indices, batch_size, chain_folder):
    vals_array = []
    idxs_array = []
    k = 0
    k_array = []
    chains = []
    mp_param_names = []
    for filename in os.listdir(chain_folder):
        if filename.endswith('.txt'):
            with open(os.path.join(chain_folder, filename), 'r') as f:
                chain = []
                for line in f:
                    if line.startswith('#'):
                        chain = []
                    else:
                        content = line.strip().split('\t')
                        pars = [content[0].split(' ')[0]]
                        for p in content[1:]:
                            pars.append(p)
                        chain.append(np.float64(pars))
                chains.append(chain)
        elif filename.endswith('.paramnames'):
            with open(os.path.join(chain_folder, filename), 'r') as f:
                for line in f:
                    mp_param_names.append(line.strip().split(' \t')[0])
    chains = np.concatenate(chains, axis=0)
    for i, par_i in enumerate(param_names):
        for j, par_j in enumerate(param_names):
            if j > i:
                chain_idx_i = mp_param_names.index(par_i)
                chain_idx_j = mp_param_names.index(par_j)
                
                hist, xedges, yedges = np.histogram2d(
                    chains[:, chain_idx_j+1],
                    chains[:, chain_idx_i+1],
                    weights=chains[:, 0],
                    bins=(20, 20),
                    normed=False)
                ch = Cut_histogram(hist, xedges, yedges, one_minus_alpha=1.0)
                data = ch.get_points()
                if par_j == 'omega_b':
                    data[:,0] *= 0.01
                elif par_i == 'omega_b':
                    data[:,1] *= 0.01
                grid_points = tf.reverse(tf.constant(data, dtype=tf.float32), [1])

                idxs_array.append(tf.repeat(tf.constant([[param_indices[i], param_indices[j]]]), grid_points.shape[0], axis=0))
                vals_array.append(tf.reshape(tf.repeat(grid_points, batch_size, axis=0), [len(grid_points), batch_size, idxs_array[-1].shape[1]]))
                k_array.append(tf.repeat(tf.constant([k]), grid_points.shape[0], axis=0).numpy())
                k += 1
    vals_array = tf.concat(vals_array,0)
    idxs_array = tf.concat(idxs_array,0)
    file_number_array = tf.concat(k_array,0)
    return vals_array, idxs_array, file_number_array

def get_data_from_mcmc_bins_1d(param_names, param_indices, batch_size, chain_folder):
    vals_array = []
    idxs_array = []
    k = 0
    k_array = []
    chains = []
    mp_param_names = []
    for filename in os.listdir(chain_folder):
        if filename.endswith('.txt'):
            with open(os.path.join(chain_folder, filename), 'r') as f:
                chain = []
                for line in f:
                    if line.startswith('#'):
                        chain = []
                    else:
                        content = line.strip().split('\t')
                        pars = [content[0].split(' ')[0]]
                        for p in content[1:]:
                            pars.append(p)
                        chain.append(np.float64(pars))
                chains.append(chain)
        elif filename.endswith('.paramnames'):
            with open(os.path.join(chain_folder, filename), 'r') as f:
                for line in f:
                    mp_param_names.append(line.strip().split(' \t')[0])
    chains = np.concatenate(chains, axis=0)
    for i, par in enumerate(param_names):
        chain_idx = mp_param_names.index(par)
        hist, xedges = np.histogram(
            chains[:, chain_idx+1], bins=50,
            weights=chains[:, 0], normed=False, density=False)
        x_centers = 0.5*(xedges[1:]+xedges[:-1])
        data = np.array([x_centers]).T
        if par == 'omega_b':
            data[:,0] *= 0.01
        grid_points = tf.reverse(tf.constant(data, dtype=tf.float32), [1])
    
        idxs_array.append(tf.repeat(tf.constant([[param_indices[i]]]), grid_points.shape[0], axis=0))
        vals_array.append(tf.reshape(tf.repeat(grid_points, batch_size, axis=0), [len(grid_points), batch_size, idxs_array[-1].shape[1]]))
        k_array.append(tf.repeat(tf.constant([k]), grid_points.shape[0], axis=0).numpy())
        k += 1
    vals_array = tf.concat(vals_array,0)
    idxs_array = tf.concat(idxs_array,0)
    file_number_array = tf.concat(k_array,0)
    return vals_array, idxs_array, file_number_array


def get_wrong_points_2d(param_names, param_indices, batch_size, output_dir, points_file, fix_failed_points=True):
    vals_array = []
    idxs_array = []
    k = 0
    k_array = []
    line_numbers = {}
    try:
        with open(points_file, 'r') as f:
            for line in f:
                objs = line.replace('\n','').split('\t')
                line_numbers[objs[0]] = np.int64(objs[1:])
    except:
        pass
    for i, par_i in enumerate(param_names):
        for j, par_j in enumerate(param_names):
            if j > i:
                with open(os.path.join(output_dir,f'{par_i}-{par_j}.txt'),'r') as f:
                    f_list = list(f)
                if fix_failed_points:
                    for n, line in enumerate(f_list[1:]):
                        if np.float64(line.replace('\n','').split('\t'))[2] >= 5000.:
                            ks = [s for s in line_numbers.keys() if par_i in s.split('/')[-1] and par_j in s.split('/')[-1]]
                            if len(ks) == 0 or not n+1 in line_numbers[ks[0]]:
                                vals = tf.constant(np.float64(line.replace('\n','').split('\t'))[:2], dtype=tf.float32)
                                idxs_array.append(tf.constant([[param_indices[i], param_indices[j]]]))
                                vals_array.append(tf.reshape(tf.repeat([vals], batch_size, axis=0), [1, batch_size, idxs_array[-1].shape[1]]))
                                k_array.append(tf.constant([[k,n+1]]))
                for key in line_numbers.keys():
                    if par_i in key.split('/')[-1] and par_j in key.split('/')[-1]:
                        for n in line_numbers[key]:
                            vals = tf.constant(np.float64(f_list[n].replace('\n','').split('\t'))[:2], dtype=tf.float32)
                            idxs_array.append(tf.constant([[param_indices[i], param_indices[j]]]))
                            vals_array.append(tf.reshape(tf.repeat([vals], batch_size, axis=0), [1, batch_size, idxs_array[-1].shape[1]]))
                            k_array.append(tf.constant([[k,n]]))
                k += 1

    try:
        vals_array = tf.concat(vals_array,0)
        idxs_array = tf.concat(idxs_array,0)
        file_number_array = tf.concat(k_array,0)
        return vals_array, idxs_array, file_number_array
    except:
        return tf.zeros((0,batch_size,2),dtype=tf.float32), tf.zeros((0,2),dtype=tf.int32), tf.zeros((0,2),dtype=tf.int32)


def get_wrong_points_1d(param_names, param_indices, batch_size, output_dir, points_file, fix_failed_points=True):
    vals_array = []
    idxs_array = []
    k = 0
    k_array = []
    line_numbers = {}
    try:
        with open(points_file, 'r') as f:
            for line in f:
                for par in param_names:
                    objs = line.replace('\n','').split('\t')
                    if objs[0].split('/')[-1] == par+'.txt':
                        line_numbers[objs[0]] = np.int64(objs[1:])
    except:
        pass
    for i, par in enumerate(param_names):
        with open(os.path.join(output_dir,f'{par}.txt'),'r') as f:
            f_list = list(f)
        if fix_failed_points:
            for n, line in enumerate(f_list[1:]):
                if np.float64(line.replace('\n','').split('\t'))[1] >= 5000.:
                    ks = [s for s in line_numbers.keys() if par in s.split('/')[-1]]
                    if len(ks) == 0 or not n+1 in line_numbers[ks[0]]:
                        vals = tf.constant(np.float64(line.replace('\n','').split('\t'))[:1], dtype=tf.float32)
                        idxs_array.append(tf.constant([[param_indices[i]]]))
                        vals_array.append(tf.reshape(tf.repeat([vals], batch_size, axis=0), [1, batch_size, idxs_array[-1].shape[1]]))
                        k_array.append(tf.constant([[k,n+1]]))
        for key in line_numbers.keys():
            if par in key.split('/')[-1]:
                for n in line_numbers[key]:
                    vals = tf.constant(np.float64(f_list[n].replace('\n','').split('\t'))[:1], dtype=tf.float32)
                    idxs_array.append(tf.constant([[param_indices[i]]]))
                    vals_array.append(tf.reshape(tf.repeat([vals], batch_size, axis=0), [1, batch_size, idxs_array[-1].shape[1]]))
                    k_array.append(tf.constant([[k,n]]))
        k += 1

    try:
        vals_array = tf.concat(vals_array,0)
        idxs_array = tf.concat(idxs_array,0)
        file_number_array = tf.concat(k_array,0)
        return vals_array, idxs_array, file_number_array
    except:
        return tf.zeros((0,batch_size,1),dtype=tf.float32), tf.zeros((0,2),dtype=tf.int32), tf.zeros((0,2),dtype=tf.int32)

def get_additional_points_2d(param_names, param_indices, batch_size, points_file):
    vals_array = []
    idxs_array = []
    k = 0
    k_array = []
    line_numbers = {}
    with open(points_file, 'r') as f:
        for line in f:
            objs = line.replace('\n','').split('\t')
            if objs[0] not in line_numbers:
                line_numbers[objs[0]] = []
            line_numbers[objs[0]].append(np.float64(objs[1:]))
    for i, par_i in enumerate(param_names):
        for j, par_j in enumerate(param_names):
            if j > i:
                for key in line_numbers.keys():
                    if par_i in key.split('/')[-1] and par_j in key.split('/')[-1]:
                        for arr in line_numbers[key]:
                            vals = tf.constant(arr, dtype=tf.float32)
                            idxs_array.append(tf.constant([[param_indices[i], param_indices[j]]]))
                            vals_array.append(tf.reshape(tf.repeat([vals], batch_size, axis=0), [1, batch_size, idxs_array[-1].shape[1]]))
                            k_array.append(tf.constant([k]))
                k += 1

    try:
        vals_array = tf.concat(vals_array,0)
        idxs_array = tf.concat(idxs_array,0)
        file_number_array = tf.concat(k_array,0)
        return vals_array, idxs_array, file_number_array
    except:
        return tf.zeros((0,batch_size,1),dtype=tf.float32), tf.zeros((0,2),dtype=tf.int32), tf.zeros((0,2),dtype=tf.int32)

def get_additional_points_1d(param_names, param_indices, batch_size, points_file):
    vals_array = []
    idxs_array = []
    k = 0
    k_array = []
    line_numbers = {}
    with open(points_file, 'r') as f:
        for line in f:
            for par in param_names:
                objs = line.replace('\n','').split('\t')
                if objs[0].split('/')[-1] == par+'.txt':
                    if objs[0] not in line_numbers:
                        line_numbers[objs[0]] = []
                    line_numbers[objs[0]].append(np.int64(objs[1:]))
    for i, par in enumerate(param_names):
        for key in line_numbers.keys():
            if par in key.split('/')[-1]:
                for arr in line_numbers[key]:
                    vals = tf.constant(arr, dtype=tf.float32)
                    idxs_array.append(tf.constant([[param_indices[i]]]))
                    vals_array.append(tf.reshape(tf.repeat([vals], batch_size, axis=0), [1, batch_size, idxs_array[-1].shape[1]]))
                    k_array.append(tf.constant([k]))
        k += 1

    try:
        vals_array = tf.concat(vals_array,0)
        idxs_array = tf.concat(idxs_array,0)
        file_number_array = tf.concat(k_array,0)
        return vals_array, idxs_array, file_number_array
    except:
        return tf.zeros((0,batch_size,1),dtype=tf.float32), tf.zeros((0,2),dtype=tf.int32), tf.zeros((0,2),dtype=tf.int32)
