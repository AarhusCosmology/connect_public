import numpy as np
import pickle as pkl
import tensorflow as tf

def percentiles_of_histogram(data,
                             percentiles,
                             N_bins=25):
    
    n, bins = np.histogram(data, bins=N_bins)
    area = 0
    for i in range(N_bins):
        area += n[i]*(bins[i+1]-bins[i])

    x_percentiles = np.zeros(shape=(len(percentiles),2))
    A = 0
    x = 0
    for i in range(len(n)):
        if n[i] != 0:
            for l, perc in enumerate(percentiles):
                p_interval = [(1-perc)/2, 1 - (1-perc)/2]
                for m, p in enumerate(p_interval):
                    x = (area*p - A)/n[i] + bins[i]
                    if x < bins[i+1] and x > bins[i]:
                        x_percentiles[l][m] = x
        A += (bins[i+1]-bins[i])*n[i]

    return x_percentiles


def get_error(path):
    model = tf.keras.models.load_model(path, compile=False)

    with open(path + '/test_data.pkl', 'rb') as f:
        test_data = pkl.load(f)

    try:
        model_params = test_data[0]
        Cls_data     = test_data[1]
    except:
        test_data = tuple(zip(*test_data))
        model_params = np.array(test_data[0])
        Cls_data     = np.array(test_data[1])
        
    Cls_predict  = model.predict(model_params)

    l = np.linspace(2,2500,2499)
    l_red = []
    for i,ll in enumerate(l):
        if i%10 == 0:
            l_red.append(ll)
    l_red.append(l[-1])
    l_red = np.array(l_red)

    err_mean  = []
    err_upper = []
    err_lower = []
    err_upper2 = []
    err_lower2 = []
    errors    = []

    for cls_d, cls_p in zip(Cls_data, Cls_predict):
        errors.append(abs(np.array(cls_d)-np.array(cls_p))/np.array(cls_d))

    errors = np.array(errors).T
    return errors
