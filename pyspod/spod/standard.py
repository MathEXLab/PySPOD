'''Derived module from spod_base.py for standard SPOD.'''

# Import standard Python packages
import os
import sys
import time
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
import copy

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
        size_Q_hat = [self._n_freq, self.data[0,...].size, self._n_blocks]
        Q_hat = None
        ## check if blocks already computed or not
        if blocks_present:
            # load blocks if present
            size_Q_hat = [self._n_freq, *self._xshape, self._n_blocks]
            Q_hat = np.empty(size_Q_hat, dtype=self._complex)
            for i_blk in range(0, self._n_blocks):
                print(f'Loading block {i_blk}/{self._n_blocks}')
                for i_freq in range(0, self._n_freq):
                    file = f'fft_block{i_blk:08d}_freq{i_freq:08d}.npy'
                    path = os.path.join(self._blocks_folder, file)
                    Q_hat[i_freq,...,i_blk] = np.load(path)
            Q_hat = utils_par.distribute_dimension(
                data=Q_hat, max_axis=self._max_axis+1, comm=self._comm)
            shape = [Q_hat.shape[0], Q_hat[0,...,0].size, Q_hat.shape[-1]]
            Q_hat = np.reshape(Q_hat, shape)
        else:
            # loop over number of blocks and generate Fourier realizations
            size_Q_hat = [self._n_freq, self.data[0,...].size, self._n_blocks]
            Q_hat = np.empty(size_Q_hat, dtype=self._complex)
            for i_blk in range(0,self._n_blocks):
                st = time.time()

                # compute block
                Q_hat[:,:,i_blk], offset = self._compute_blocks(i_blk)

                # save FFT blocks in storage memory
                if self._savefft == True:
                    Q_blk_hat = Q_hat[:,:,i_blk]
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
        self._compute_standard_spod(Q_hat)

        # store and save results
        self._store_and_save()
        self._pr0(f'------------------------------------')
        self._pr0(f' ')
        self._pr0(f'Results saved in folder {self._savedir_sim}')
        self._pr0(f'Time to compute SPOD: {time.time() - start} s.')
        if self._comm: self._comm.Barrier()
        return self


    def _compute_blocks(self, i_blk):
        '''Compute FFT blocks.'''
        # get time index for present block
        offset = min(i_blk * (self._n_dft - self._n_overlap) \
            + self._n_dft, self._nt) - self._n_dft

        # Get data
        Q_blk = self.data[offset:self._n_dft+offset,...]
        Q_blk = Q_blk.reshape(self._n_dft, self.data[0,...].size)

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
            Q_var[Q_var < 4 * np.finfo(float).eps] = 1;
            Q_blk /= Q_var

        Q_blk *= self._window
        Q_blk = self._set_dtype(Q_blk)
        Q_blk_hat = (self._win_weight / self._n_dft) * np.fft.fft(Q_blk, axis=0)
        Q_blk_hat = Q_blk_hat[0:self._n_freq,:]
        return Q_blk_hat, offset

    def _compute_standard_spod(self, Q_hat):
        '''Compute standard SPOD.'''
        # compute inner product in frequency space, for given frequency
        st = time.time()
        M = [None]*self._n_freq
        for f in range(0,self._n_freq):
            Q_hat_f = np.squeeze(Q_hat[f,:,:])#.astype(complex)
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

        self._saved_freqs = {}

        if self._savefreq_disk2:
            local_elements = None
            recvcounts = None
            datas = {}
            reqs = []


        for f in range(0,self._n_freq):
            s0 = time.time()
            ## compute
            phi = np.matmul(Q_hat[f,...], V[f,...] * L_diag_inv[f,None,:])
            phi = phi[...,0:self._n_modes_save]

            # print(f'{Q_hat[f,...].shape} x {V[f,...].shape} * {L_diag_inv[f,None,:].shape} = {phi.shape}')

            ## save modes
            if self._savefreq_disk:
                filename = f'freq_idx_{f:08d}.npy'
                p_modes = os.path.join(self._modes_dir, filename)
                shape = [*self._xshape,self._nv,self._n_modes_save]
                if self._comm:
                    shape[self._max_axis] = -1
                phi.shape = shape
                utils_par.npy_save(self._comm, p_modes, phi, axis=self._max_axis)

            # store freqs/modes on proc (f mod rank == 0)
            if self._savefreq_mem:
                rank = self._comm.rank
                target_proc = f % self._comm.size
                ftype = MPI.C_FLOAT_COMPLEX
                local_elements = np.prod(phi.shape)
                recvcounts = np.zeros(self._comm.size, dtype=np.int64)
                self._comm.Gather(local_elements, recvcounts, root=target_proc)

                if rank == target_proc:
                    data = np.zeros(np.sum(recvcounts), dtype=phi.dtype)

                # TODO: phi is not contiguous if self._n_modes_save < number_of_blocks/modes
                s_msg = [phi, ftype]
                r_msg = [data, (recvcounts, None), ftype] if rank == target_proc else None

                self._comm.Gatherv(sendbuf=s_msg, recvbuf=r_msg, root=target_proc)

                if rank == target_proc:
                    full_freq = np.zeros((self._xshape)+(self._nv,)+(self._n_modes_save,),dtype=phi.dtype)
                    for proc in range(self._comm.size):
                        start = np.sum(recvcounts[:proc])
                        end = np.sum(recvcounts[:proc+1])

                        x_prod_not_max = np.prod(self._xshape)
                        x_prod_not_max /= self._xshape[self._max_axis]
                        max_axis_from = int(start//(self._nv*self._n_modes_save*x_prod_not_max))
                        max_axis_to   = int(end//(self._nv*self._n_modes_save*x_prod_not_max))

                        idx_full_freq = [np.s_[:]] * (self._xdim + 2)
                        idx_full_freq[self._max_axis] = slice(max_axis_from, max_axis_to)

                        full_freq[tuple(idx_full_freq)] = np.reshape(data[start:end],full_freq[tuple(idx_full_freq)].shape)
                        self._saved_freqs[f] = full_freq

            if self._savefreq_disk2:
                rank = self._comm.rank
                target_proc = f % self._comm.size
                ftype = MPI.C_FLOAT_COMPLEX
                if f == 0:
                    local_elements = np.prod(phi.shape)
                    recvcounts = np.zeros(self._comm.size, dtype=np.int64)
                    self._comm.Allgather(local_elements, recvcounts)

                    for ff in range(0,self._n_freq):
                        if ff%self._comm.size == rank:
                            datas[ff] = np.zeros(np.sum(recvcounts), dtype=phi.dtype)

                s_msg = [copy.deepcopy(phi), ftype]
                r_msg = [datas[f], (recvcounts, None), ftype] if rank == target_proc else None

                req = self._comm.Igatherv(sendbuf=s_msg, recvbuf=r_msg, root=target_proc)

                reqs.append(req)

                if len(reqs) > 64:
                    xxtime = time.time()
                    if rank == 0:
                        print(f'waiting for {len(reqs)} requests')
                    MPI.Request.Waitall(reqs)
                    reqs = []
                    if rank == 0:
                        print(f'  partial waitall {time.time()-xxtime} seconds')

            self._pr0(
                f'freq: {f+1}/{self._n_freq};  (f = {self._freq[f]:.5f});  '
                f'Elapsed time: {(time.time() - s0):.5f} s.')

        if self._savefreq_disk2:
            xst = time.time()
            MPI.Request.Waitall(reqs)
            reqs=[]
            self._pr0(f'waitall time: {time.time() - xst} s.')

            xst = time.time()
            for f in range(0,self._n_freq):
                if rank == f % self._comm.size:
                    full_freq = np.zeros((self._xshape)+(self._nv,)+(self._n_modes_save,),dtype=phi.dtype)
                    for proc in range(self._comm.size):
                        start = np.sum(recvcounts[:proc])
                        end = np.sum(recvcounts[:proc+1])

                        x_prod_not_max = np.prod(self._xshape)
                        x_prod_not_max /= self._xshape[self._max_axis]
                        max_axis_from = int(start//(self._nv*self._n_modes_save*x_prod_not_max))
                        max_axis_to   = int(end//(self._nv*self._n_modes_save*x_prod_not_max))

                        idx_full_freq = [np.s_[:]] * (self._xdim + 2)
                        idx_full_freq[self._max_axis] = slice(max_axis_from, max_axis_to)

                        full_freq[tuple(idx_full_freq)] = np.reshape(datas[f][start:end],full_freq[tuple(idx_full_freq)].shape)
                        self._saved_freqs[f] = full_freq

            self._pr0(f'transposes: {time.time() - xst} s.')

            xst = time.time()
            for f in self._saved_freqs.keys():
                filename = f'freq_idx_{f:08d}.npy'
                print(f'rank {rank} saving {filename}')
                p_modes = os.path.join(self._modes_dir, filename)
                np.save(p_modes, self._saved_freqs[f])
            self._comm.Barrier()
            self._pr0(f'saving: {time.time() - xst} s.')


        self._pr0(f'- Modes computation and saving: {time.time() - st} s.')

        ## correct Fourier for one-sided spectrum
        if self._isrealx:
            L[1:-1,:] = 2 * L[1:-1,:]

        # get eigenvalues and confidence intervals
        self._eigs = np.abs(L)

        fac_lower = 2 * self._n_blocks / self._xi2_lower
        fac_upper = 2 * self._n_blocks / self._xi2_upper
        self._eigs_c[...,0] = self._eigs * fac_lower
        self._eigs_c[...,1] = self._eigs * fac_upper
