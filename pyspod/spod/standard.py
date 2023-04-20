'''Derived module from spod_base.py for standard SPOD.'''

# Import standard Python packages
import os
import sys
import time
import math

import numpy as np
from numpy import linalg as la
import scipy.io.matlab as siom

# Import custom Python packages
from pyspod.spod.base import Base
import pyspod.utils.parallel as utils_par
try:
    from mpi4py import MPI
except:
    pass

class Standard(Base):
    '''
    Class that implements a distributed batch version of the
    Spectral Proper Orthogonal Decomposition algorithm to the input data.

    The computation is performed on the `data` passed
    to the `fit` method of the `Standard` class, derived
    from the `Base` class.
    '''
    def fit(self, data_list, variables = None):
        '''
        Class-specific method to fit the data matrix using the SPOD
        batch algorithm.

        :param list data_list: list containing data matrices for which
            to compute the SPOD.
        '''
        start0 = time.time()
        start = time.time()

        ## initialize data and variables
        self._pr0(f' ')
        self._pr0(f'Initialize data ...')
        self._initialize(data_list, variables)
        self._pr0(f'Time to initialize: {time.time() - start} s.')

        start = time.time()

        self._pr0(f' ')
        self._pr0(f'Calculating temporal DFT (parallel)')
        self._pr0(f'------------------------------------')

        # check if blocks are already saved in memory
        blocks_present = False
        if self._reuse_blocks:
            blocks_present = self._are_blocks_present(
                self._n_blocks, self._n_freq, self._blocks_folder, self._comm)

        # loop over number of blocks and generate Fourier realizations,
        # if blocks are not saved in storage

        ## check if blocks already computed or not
        if blocks_present:
            # load blocks if present
            Q_hats = {}
            size_qhat = [*self._xshape, self._n_blocks]
            for f in range(self._n_freq):
                Q_hats[f] = np.empty(size_qhat, dtype=self._complex)

            for i_blk in range(0, self._n_blocks):
                print(f'Loading block {i_blk}/{self._n_blocks}')
                for i_freq in range(0, self._n_freq):
                    file = f'fft_block{i_blk:08d}_freq{i_freq:08d}.npy'
                    path = os.path.join(self._blocks_folder, file)
                    Q_hats[i_freq][...,i_blk] = np.load(path)
            for f in range(self._n_freq):
                Q_hats[f] = utils_par.distribute_dimension(
                    data=Q_hats[f], max_axis=self._max_axis, comm=self._comm)
                qhat = Q_hats[f]
                shape = [qhat[...,0].size, qhat.shape[-1]]
                Q_hats[f] = np.reshape(Q_hats[f], shape)
            del self.data
        else:
            # loop over number of blocks and generate Fourier realizations
            if isinstance(self.data, dict):
                last_key = list(self.data)[-1]
                last_val = self.data[last_key]["v"]
                xvsize = last_val[0,...].size
            else:
                xvsize = self.data[0,...].size

            Q_hats = {}
            for i_blk in range(0,self._n_blocks):
                st = time.time()

                # compute block
                qhat = np.empty([self._n_freq, xvsize], dtype=self._complex)
                qhat[:], offset = self._compute_blocks(i_blk)
                Q_hats[i_blk] = {}
                for f in range(self._n_freq):
                    Q_hats[i_blk][f] = qhat[f,:].copy()

                # save FFT blocks in storage memory
                if self._savefft == True:
                    Q_blk_hat = qhat
                    for i_freq in range(0, self._n_freq):
                        Q_blk_hat_fr = Q_blk_hat[i_freq,:]
                        file = f'fft_block{i_blk:08d}_freq{i_freq:08d}.npy'
                        path = os.path.join(self._blocks_folder, file)
                        shape = [*self._xshape]
                        if self._comm: shape[self._max_axis] = -1
                        Q_blk_hat_fr.shape = shape
                        utils_par.npy_save(
                            self._comm, path, Q_blk_hat_fr,
                            axis=self._max_axis)

                # print info file
                self._pr0(f'block {(i_blk+1)}/{(self._n_blocks)}'
                          f' ({(offset)}:{(self._n_dft+offset)});  '
                          f'Elapsed time: {time.time() - st} s.')

            del self.data

            st = time.time()
            # move from Q_hats[i_blk][f] to Q_hats[f]
            Q_hats = self._flip_qhat(Q_hats)
            self._pr0(f'- Time spent transposing Q_hats dictionaries: {time.time() - st} s.')

        self._pr0(f'------------------------------------')
        self._pr0(f'Time to compute DFT: {time.time() - start} s.')
        if self._comm: self._comm.Barrier()
        start = time.time()

        # Loop over all frequencies and calculate SPOD
        self._pr0(f' ')
        self._pr0(f'Calculating SPOD (parallel)')
        self._pr0(f'------------------------------------')
        self._eigs = np.zeros([self._n_freq,self._n_blocks],
            dtype=self._complex)

        ## compute standard spod
        self._compute_standard_spod(Q_hats)

        # store and save results
        self._store_and_save()
        self._pr0(f'------------------------------------')
        self._pr0(f' ')
        self._pr0(f'Results saved in folder {self._savedir_sim}')
        self._pr0(f'Time to compute SPOD: {time.time() - start} s.')
        self._pr0(f'------------------------------------')
        self._pr0(f' ')
        self._pr0(f'Total time: {time.time() - start0} s.')
        if self._comm: self._comm.Barrier()
        return self


    def _compute_blocks(self, i_blk):
        '''Compute FFT blocks.'''
        # get time index for present block
        offset = min(i_blk * (self._n_dft - self._n_overlap) \
            + self._n_dft, self._nt) - self._n_dft

        Q_blk = self._get_block(offset, offset+self._n_dft)

        # Subtract longtime or provided mean
        Q_blk -= self._t_mean

        # if block mean is to be subtracted,
        # do it now that all data is collected
        if self._mean_type.lower() == 'blockwise':
            Q_blk -= np.mean(Q_blk, axis=0)

        # normalize by pointwise variance
        if self._normalize_data:
            den = self._n_dft - 1
            Q_var = np.sum((Q_blk - np.mean(Q_blk, axis=0))**2, axis=0) / den
            # address division-by-0 problem with NaNs
            Q_var[Q_var < 4 * np.finfo(float).eps] = 1
            Q_blk /= Q_var

        Q_blk *= self._window
        Q_blk = self._set_dtype(Q_blk)

        if self._isrealx and not self._fullspectrum:
            Q_blk_hat = (self._win_weight / self._n_dft) * np.fft.rfft(Q_blk, axis=0)
        else:
            Q_blk_hat = (self._win_weight / self._n_dft) * np.fft.fft(Q_blk, axis=0)[0:self._n_freq,:]
        return Q_blk_hat, offset

    def _compute_standard_spod(self, Q_hats):
        '''Compute standard SPOD.'''

        comm = self._comm
        # compute inner product in frequency space, for given frequency
        st = time.time()
        M = [None]*self._n_freq
        for f in range(0,self._n_freq):
            Q_hat_f = np.squeeze(Q_hats[f])#np.squeeze(Q_hat[f,:,:])#.astype(complex)
            M[f] = Q_hat_f.conj().T @ (Q_hat_f * self._weights) / self._n_blocks
        del Q_hat_f
        M = np.stack(M)
        M = utils_par.allreduce(data=M, comm=self._comm)
        self._pr0(f'- M computation: {time.time() - st} s.')
        st = time.time()

        ## compute eigenvalues and eigenvectors
        L, V = la.eig(M)
        L = np.real_if_close(L, tol=1000000)
        del M


        # reorder eigenvalues and eigenvectors
        ## double non-zero freq and non-Nyquist
        for f, Lf in enumerate(L):
            idx = np.argsort(Lf)[::-1]
            L[f,:] = L[f,idx]
            vf = V[f,...]
            vf = vf[:,idx]
            V[f] = vf
        self._pr0(f'- Eig computation: {time.time() - st} s.')
        st = time.time()

        # compute spatial modes for given frequency
        L_diag = np.sqrt(self._n_blocks) * np.sqrt(L)
        L_diag_inv = 1. / L_diag

        if not self._savefreq_disk2:
            for f in range(0,self._n_freq):
                s0 = time.time()
                ## compute
                phi = np.matmul(Q_hats[f], V[f,...] * L_diag_inv[f,None,:])
                phi = phi[...,0:self._n_modes_save]
                del Q_hats[f]

                sstime = time.time()
                ## save modes
                if self._savefreq_disk:
                    filename = f'freq_idx_{f:08d}.npy'
                    p_modes = os.path.join(self._modes_dir, filename)

                    shape = [*self._xshape,self._nv,self._n_modes_save]

                    if comm:
                        shape[self._max_axis] = -1

                    phi.shape = shape
                    utils_par.npy_save(self._comm, p_modes, phi, axis=self._max_axis)

                self._pr0(
                    f'freq: {f+1}/{self._n_freq};  (f = {self._freq[f]:.5f});  '
                    f'Elapsed time: {(time.time() - s0):.5f} s.')


        ####################################
        ####################################
        ####################################
        else:               # savefreq_disk2
        ####################################
        ####################################
        ####################################
            assert self._reader._flattened, "savefreq_disk2 currently only works with flattened data"
            rank = comm.rank
            ftype = MPI.C_FLOAT_COMPLEX if self._complex==np.complex64 else MPI.C_DOUBLE_COMPLEX

            cum_cctime = 0
            cum_sstime = 0

            phi_dict = {}
            for f in range(0,self._n_freq):
                s0 = time.time()
                phi_dict[f] = {}
                phi = np.matmul(Q_hats[f], V[f,...] * L_diag_inv[f,None,:])[:,:self._n_modes_save]
                Q_hats[f] = None
                cum_cctime += time.time() - s0

                s1 = time.time()
                for m in range(0,self._n_modes_save):
                    phi_dict[f][m] = phi[:,m].copy() # make sure modes beyond n_modes_save can be deallocated
                del phi
                cum_sstime += time.time() - s1

                self._pr0(
                    f'freq: {f+1}/{self._n_freq};  (f = {self._freq[f]:.5f});  '
                    f'Elapsed time: {(time.time() - s0):.5f} s.')

            del V
            del Q_hats

            sstime = time.time()

            # get max phi shape
            phi0_max = comm.allreduce(phi_dict[0][0].shape[0], op=MPI.MAX)
            phi_dtype = phi_dict[0][0].dtype
            mpi_dtype = ftype.Create_contiguous(phi0_max).Commit()
            local_elements = np.array(phi_dict[0][0].shape[0])
            recvcounts = np.zeros(comm.size, dtype=np.int64)
            comm.Allgather(local_elements, recvcounts)

            total_files = self._n_freq * self._n_modes_save

            for ipass in range(0,math.ceil(total_files/comm.size)):
                write_s = ipass * comm.size
                write_e = min((ipass+1) * comm.size, total_files)
                write = None

                data = np.zeros(phi0_max*comm.size, dtype=phi_dtype)

                s_msgs = {}
                reqs_r = []
                reqs_s = []

                for i in range(write_s, write_e):
                    f = i // self._n_modes_save
                    m = i % self._n_modes_save
                    writer = i % comm.size

                    s_msgs[i] = [np.zeros(phi0_max, dtype=phi_dtype), mpi_dtype]
                    s_msgs[i][0][0:phi_dict[f][m].shape[0]] = phi_dict[f][m][:] # phi0_max-shaped and 0-padded
                    del phi_dict[f][m]
                    reqs_s.append(comm.Isend(s_msgs[i], dest=writer))

                    if rank == writer:
                        write = (f,m)
                        for irank in range(comm.size):
                            reqs_r.append(comm.Irecv([data[phi0_max*irank:],mpi_dtype],source=irank))

                MPI.Request.Waitall(reqs_s)
                s_msgs = {}

                if write:
                    f, m = write
                    xtime = time.time()
                    MPI.Request.Waitall(reqs_r)
                    self._pr0(f'  Waitall({len(reqs_r)}) {time.time()-xtime} seconds')

                    for irank in range(comm.size):
                        start = irank*phi0_max
                        end = start+recvcounts[irank]
                        start_nopad = np.sum(recvcounts[:irank])
                        end_nopad = np.sum(recvcounts[:irank+1])
                        data[start_nopad:end_nopad,...] = data[start:end,...]

                    # write to disk
                    data = data[:np.sum(recvcounts)].reshape(self._xshape+(self._nv,))
                    filename = f'freq_idx_f{f:08d}_m{m:08d}.npy'
                    print(f'rank {rank} saving {filename}')
                    p_modes = os.path.join(self._modes_dir, filename)
                    np.save(p_modes, data)

            mpi_dtype.Free()

            cum_sstime += time.time() - sstime
            self._pr0(f'- Modes computation {cum_cctime} s. Saving: {cum_sstime} s.')

        ## correct Fourier for one-sided spectrum
        if self._isrealx:
            L[1:-1,:] = 2 * L[1:-1,:]

        # get eigenvalues and confidence intervals
        self._eigs = np.abs(L)

        fac_lower = 2 * self._n_blocks / self._xi2_lower
        fac_upper = 2 * self._n_blocks / self._xi2_upper
        self._eigs_c[...,0] = self._eigs * fac_lower
        self._eigs_c[...,1] = self._eigs * fac_upper

    def _get_block(self, start, end):
        if isinstance(self.data, dict):
            last_key = list(self.data)[-1]
            last_val = self.data[last_key]["v"]
            Q_blk = np.empty((self._n_dft,)+last_val.shape[1:],dtype=last_val.dtype)

            cnt = 0
            for k,v in self.data.items():
                v_s = v["s"]
                v_e = v["e"]

                read_here_s = max(v_s, start)
                read_here_e = min(v_e, end)
                read_here_cnt = read_here_e - read_here_s

                # print(f'LBL key {k} contains {v_s}:{v_e} while i need {start}:{end}')
                if read_here_cnt > 0:
                    # print(f'LBL key {k} contains {v_s}:{v_e} while i need {start}:{end} - will read {read_here_s}:{read_here_e} to {cnt}:{cnt+read_here_cnt}')
                    vals = v["v"]
                    Q_blk[cnt:cnt+read_here_cnt,...] = vals[read_here_s-v_s:read_here_e-v_s,...]
                    cnt += read_here_cnt
                    start += read_here_cnt

            # delete blocks that are no longer needed
            keys_to_del = []
            for k,v in self.data.items():
                v_s = v["s"]
                v_e = v["e"]
                if start > v_e:
                    keys_to_del.append(k)

            for k in keys_to_del:
                del self.data[k]

            Q_blk = Q_blk.reshape(self._n_dft, last_val[0,...].size)
            return Q_blk
        else:
            Q_blk = self.data[start:end,...].copy()
            Q_blk = Q_blk.reshape(self._n_dft, self.data[0,...].size)
            return Q_blk

    def _flip_qhat(self, Q_hats):
        last_blk = list(Q_hats)[-1]
        last_frq = Q_hats[last_blk]
        last_val = last_frq[list(last_frq)[-1]]
        xvsize = last_val.size

        Q_hat_f = {}

        for f in range(0,self._n_freq):
            Q_hat_f[f] = np.zeros((xvsize, self._n_blocks),dtype=last_val.dtype)
            for b,v in Q_hats.items():
                Q_hat_f[f][:,b] = v[f][:]
            for b,_ in Q_hats.items():
                del Q_hats[b][f]
        return Q_hat_f
