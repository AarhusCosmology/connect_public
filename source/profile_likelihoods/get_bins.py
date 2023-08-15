import numpy as np
import matplotlib.pyplot as plt

class Cut_histogram():
    def __init__(self, hist, xedges, yedges, one_minus_alpha=0.997):
        self.h = hist
        self.one_minus_alpha = one_minus_alpha
        self.x = 0.5*(xedges[1:]+xedges[:-1])
        self.y = 0.5*(yedges[1:]+yedges[:-1])

    def get_points(self):
        vals = np.flip(np.sort(self.h.flatten()))
        idx  = np.flip(np.argsort(self.h.flatten()))

        N_tot = np.sum(vals)

        acc_idx = idx[np.where(np.cumsum(vals) < self.one_minus_alpha*N_tot)]

        idxs = []
        for ai in acc_idx:
            i_y = ai % len(self.y)
            i_x = int((ai-i_y) / len(self.y))
            idxs.append((i_x,i_y))
    
        ia, ib = zip(*idxs)
        self.a = self.x[np.array(ia)]
        self.b = self.y[np.array(ib)]
        return np.array([self.a,self.b]).T
        
    def plot_points(self):
        X,Y = np.meshgrid(self.x,self.y)
        plt.plot(X, Y, 'k*')
        plt.plot(self.a, self.b, 'r*')

        plt.show()

