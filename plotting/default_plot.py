import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('Agg')

predict_params = tf.constant(list(np.concatenate([x for x,y in test_dataset], axis=0)[-10:]))
Cls_p = model.predict(predict_params)

from classy import Class
import matplotlib.pyplot as plt
j=0
for pre_par, Cls in zip(predict_params,Cls_p):
    params = {'output':'tCl,lCl', 'lensing':'yes'}
    for i,name in enumerate(paramnames):
        params[name] = pre_par.numpy()[i]
    cosmo = Class(params)
    cosmo.compute()
    cls = cosmo.lensed_cl()
    l = cls['ell'][2:]
    Cl = cls['tt'][2:]*l*(l+1)/(2*np.pi)
    l_red = []
    for i,ll in enumerate(l):
        if i%10 == 0:
            l_red.append(ll)
    l_red.append(l[-1])
    plt.figure()
    plt.plot(l,Cl,'b-',lw=2,label='Simulated (CLASS)')
    plt.plot(l_red,Cls*1e-11,'r.',label='Emulated (NN)')
    plt.xlim([2,2500])
    plt.xscale('log')
    plt.legend()
    plt.title(str(N_train)+' training sets')
    if COSMIC_VARIANCE:
        plt.savefig(f'test_plots/{N_train}_bs{BATCH_SIZE}_e{EPOCHS}_cosmic_variance_'+str(j)+'.pdf')
    else:
        plt.savefig(f'test_plots/{N_train}_bs{BATCH_SIZE}_e{EPOCHS}_'+str(j)+'.pdf')
    j+=1

plt.figure()
train_acc = history.history['accuracy']
train_loss = history.history['val_loss']
epochs = range(1, len(train_acc) + 1)
plt.plot(epochs, train_acc)
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.savefig('accuracy.pdf')


plt.figure()
plt.plot(epochs, train_loss)
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.savefig('loss.pdf')

if COSMIC_VARIANCE:
    with open(f'Accuracy_{N_train}_bs{BATCH_SIZE}_e{EPOCHS}_cosmic_variance.pkl','wb') as f:
        pickle.dump(train_acc, f)

    with open(f'Val_Loss_{N_train}_bs{BATCH_SIZE}_e{EPOCHS}_cosmic_variance.pkl','wb') as f:
        pickle.dump(train_loss, f)
else:
    with open(f'Accuracy_{N_train}_bs{BATCH_SIZE}_e{EPOCHS}.pkl','wb') as f:
        pickle.dump(train_acc, f)

    with open(f'Val_Loss_{N_train}_bs{BATCH_SIZE}_e{EPOCHS}.pkl','wb') as f:
        pickle.dump(train_loss, f)
