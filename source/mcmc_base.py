import os
import subprocess as sp

import numpy as np

from .tools import get_node_with_most_cpus

class MCMC_base_class():
    # A new mcmc sampler class that inherits from this class must contain the following methods:
    #
    #    * get_Rminus1_of_chains(all_chains, iteration) returning (bool) kill_iteration, (str) Rm1_line
    #    * save_accepted_points(indices_accepted, iteration)
    #    * check_version()
    #    * compare_iterations(iteration) returning (bool) kill_iteration
    #    * get_number_of_accepted_steps(iteration) returning (int) N_accepted
    #    * run_mcmc_sampling(model, iteration)
    #    * filter_chains(iteration) returning (int) N_points_to_keep
    #    * import_points_from_chains(iteration) returning (ndarray) data
    #    * backup_full_chains(iteration)


    def __init__(self, param, CONNECT_PATH):
        self.param = param
        self.CONNECT_PATH = CONNECT_PATH
        slurm_bool = int(sp.run('if [ -z $SLURM_NPROCS ]; then echo 0; else echo 1; fi', shell=True, stdout=sp.PIPE).stdout.decode('utf-8'))
        self.mcmc_node = None
        self.temperature = 0.0
        if slurm_bool:
            self.mcmc_node = get_node_with_most_cpus()
        os.environ["OMPI_MCA_rmaps_base_oversubscribe"] = "1"



    def filter_steps(self, N_left, N_max_points):
        N_to_use = [0]*len(N_left)
        if np.sum(N_left) > N_max_points:
            N_per_chain = int(np.floor(N_max_points/len(N_left)))
            if not N_per_chain == 0:
                for i in range(len(N_left)):
                    if N_left[i] >= N_per_chain:
                        N_left[i] = N_left[i]-N_per_chain
                        N_to_use[i] = N_per_chain
                    else:
                        N_to_use[i] = N_left[i]
                        N_left[i] = 0
            N_missing = N_max_points - np.sum(N_to_use)
            if N_missing > 0:
                stop = False
            else:
                stop = True
            while not stop:
                for i in range(len(N_left)):
                    if N_left[i] > 0:
                        N_to_use[i] += 1
                        N_left[i] -= 1
                        N_missing -= 1
                        if not N_missing > 0:
                            stop = True
                            break
            return N_to_use

        else:
            return N_left


    def Rm1(self, chains):
        covs = []
        Ns = []
        means = []
        for c in chains:
            Np = c.shape[1]
            N = int(len(c[:,0]))
            Ns.append(N)
            means.append(np.sum(c,axis=0)/N)
            covs.append(np.cov(c,rowvar=False))
        covs=np.array(covs)
        means=np.array(means)
        Ns=np.array(Ns)
        mean_of_covs = np.average(covs, weights=Ns, axis=0)
        cov_of_means = np.atleast_2d(np.cov(means.T))
        d = np.sqrt(np.diag(cov_of_means))
        if Np > 1:
            corr_of_means = (cov_of_means / d).T / d
            norm_mean_of_covs = (mean_of_covs / d).T / d
            L = np.linalg.cholesky(norm_mean_of_covs)
            Linv = np.linalg.inv(L)
            eigvals = np.linalg.eigvalsh(Linv.dot(corr_of_means).dot(Linv.T))
            Rminus1 = max(np.abs(eigvals))
        else:
            means = means.T[0]
            total_N = np.sum(Ns)
            total_mean = 0
            for m, n in zip(means,Ns):
                total_mean += m*n
            total_mean /= total_N
            within = 0
            between = 0
            for n, v, m in zip(Ns, covs, means):
                within += n*v
                between += n*(m-total_mean)**2
                within /= total_N
                between /= (total_N-1)
            Rminus1 = between/within

        return Rminus1


    def discard_oversampled_points(self,
                                   iteration         # Current iteration number
                               ):

        if iteration-1 == 0:
            file_old_data = f"data/{self.param.jobname}/N-{self.param.N}/model_params.txt"
        else:
            file_old_data = f"data/{self.param.jobname}/number_{iteration-1}/model_params.txt"
        points1, points2 = self.import_points_to_compare(file_old_data, iteration)

        num_points_in_vicinity = 10

        ranges = []
        for par1, par2 in zip(points1.T, points2.T):
            ranges.append([min(min(par1),min(par2)),max(max(par1),max(par2))])
        ranges = np.array(ranges)
        range_dim = np.diff(ranges).T[0]


        indices_accepted = []
        for i,p in enumerate(points2):
            distances1 = np.sqrt(np.sum(((points1-p)/range_dim)**2,axis=1))
            min_dist = min(distances1)
            points2_i = np.delete(points2,i,0)
            distances2 = np.sqrt(np.sum(((points2_i-p)/range_dim)**2,axis=1))
            j = np.argmin(distances1)
            p1_min = points1[j]
            points1_i = np.delete(points1,np.argmin(distances1),0)
            distances1_2 = np.sqrt(np.sum(((points1_i-p1_min)/range_dim)**2,axis=1))
            cp_list1 = distances1_2[np.argsort(distances1_2)[:num_points_in_vicinity]]
            av_dist1 = np.sum(cp_list1)/num_points_in_vicinity
            std_dist1 = np.std(cp_list1)
            cp_list2 = distances2[np.argsort(distances2)[:num_points_in_vicinity]]
            av_dist2 = np.sum(cp_list2)/num_points_in_vicinity
            std_dist2 = np.std(cp_list2)
            if min_dist > av_dist1 + 2*std_dist1:
                indices_accepted.append(i)
            elif av_dist2 + 0*std_dist2 < av_dist1 - 2*std_dist1:
                indices_accepted.append(i)

        self.backup_full_chains(iteration)

        self.save_accepted_points(indices_accepted, iteration)

        return len(indices_accepted)


    def import_points_to_compare(self,
                                 file_old_data,    # model_params.txt file with all the previous data
                                 iteration         # Current iteration number
                             ):

        with open(file_old_data, 'r') as f:
            lines=list(f)
        points1 = []
        for line in lines[1:]:
            points1.append(np.float32(line.replace('\n', '').split('\t')))
        points1 = np.array(points1)

        points2 = self.import_points_from_chains(iteration)

        return points1, points2


    def Gelman_Rubin_log_ini(self):

        with open(f'data/{self.param.jobname}/Gelman-Rubin.txt', 'w') as f:
            f.write('# R-1 values for each parameter in every iteration\n')
            header = '# iterations'
            for par in self.param.parameters:
                header += '\t'+par
            header += '\tCombined\n'
            f.write(header)


    def Gelman_Rubin_log(self,
                         iteration,       # Iteration number                     
                         all_chains=None, # chains object 
                     ):

        Rm1_line = f'{iteration-1}-{iteration}'
        kill_iteration, Rm1_line_add = self.get_Rminus1_of_chains(all_chains, iteration)
        Rm1_line += Rm1_line_add

        with open(f'data/{self.param.jobname}/Gelman-Rubin.txt', 'a') as f:
            f.write(Rm1_line)

        return kill_iteration


    def get_number_of_data_points(self,
                                  iteration   # Iteration number
                              ):

        if iteration == 0:
            folder = f"N-{self.param.N}"
        else:
            folder = f"number_{iteration}"
        with open(os.path.join(self.CONNECT_PATH, f'data/{self.param.jobname}/{folder}/model_params.txt'), 'r') as f:
            return len(list(f)) - 1
