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

        # TODO: can we use rfft and when?
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

        if self._savefreq_disk2:
            local_elements = None
            recvcounts = None
            s_msgs = {}
            reqs = []
            reqs_r = []
            reqs_s = []
            saved_freq = -1
            phi0_max = None
            nreqs = 1024
            ftype = MPI.C_FLOAT_COMPLEX if self._complex==np.complex64 else MPI.C_DOUBLE_COMPLEX
            data = None
            use_padding = True
            mpi_dtype = None


        for f in range(0,self._n_freq):
            s0 = time.time()
            ## compute
            phi = np.matmul(Q_hats[f], V[f,...] * L_diag_inv[f,None,:])
            phi = phi[...,0:self._n_modes_save]
            del Q_hats[f]

            ## save modes
            if self._savefreq_disk:
                filename = f'freq_idx_{f:08d}.npy'
                p_modes = os.path.join(self._modes_dir, filename)

                shape = [*self._xshape,self._nv,self._n_modes_save]

                if comm:
                    shape[self._max_axis] = -1

                phi.shape = shape
                utils_par.npy_save(self._comm, p_modes, phi, axis=self._max_axis)

            if self._savefreq_disk2:
                rank = comm.rank
                target_proc = f % comm.size

                # get counts once
                if f == 0:
                    # FIXME: no padding branch is broken. also check the reader
                    if False and (MPI.VERSION >= 4 or comm.size*phi0_max*phi.shape[1] < np.iinfo(np.int32).max):
                        use_padding = False

                    # get max phi shape
                    phi0_max = comm.allreduce(phi.shape[0], op=MPI.MAX)
                    mpi_dtype = ftype.Create_contiguous(phi0_max*phi.shape[1]).Commit()

                    local_elements = np.prod(phi.shape)
                    recvcounts = np.zeros(comm.size, dtype=np.int64)
                    comm.Allgather(local_elements, recvcounts)

                if not use_padding:
                    # utils_par.pr0(f'\t\t Using Igatherv with float/double datatype (MPI-4 available or number of elements < INT32_MAX)', comm)
                    if rank == target_proc:
                        data = np.zeros(np.sum(recvcounts), dtype=phi.dtype)
                        saved_freq = f

                    s_msgs[f] = [phi.copy(), ftype]
                    r_msg = [data, (recvcounts, None), ftype] if rank == target_proc else None

                    req = comm.Igatherv(sendbuf=s_msgs[f], recvbuf=r_msg, root=target_proc)
                    reqs.append(req)

                else:
                    # utils_par.pr0(f'\t\t Using Igather with a custom datatype (MPI-4 not available and number of elements >= INT32_MAX)', comm)
                    if rank == target_proc:
                        data = np.zeros(phi0_max*phi.shape[1]*comm.size, dtype=phi.dtype)
                        saved_freq = f

                    s_msgs[f] = [np.zeros((phi0_max,phi.shape[1]), dtype=phi.dtype), mpi_dtype]
                    s_msgs[f][0][0:phi.shape[0],:] = phi[:,:] # phi0_max-shaped and 0-padded
                    # r_msg = [data, (np.array([1]*comm.size), None), mpi_dtype] if rank == target_proc else None

                    # req = comm.Igatherv(sendbuf=s_msgs[f], recvbuf=r_msg, root=target_proc)
                    # reqs.append(req)

                    if rank == target_proc:
                        for irank in range(comm.size):
                            reqs_r.append(comm.Irecv([data[phi0_max*phi.shape[1]*irank:],mpi_dtype],source=irank))
                    reqs_s.append(comm.Isend(s_msgs[f],dest=target_proc))

                if len(reqs_s) == comm.size or f == self._n_freq-1:
                    xtime = time.time()
                    self._pr0(f'waiting for {len(reqs_r)} requests')
                    MPI.Request.Waitall(reqs_r)
                    self._pr0(f'  Waitall({len(reqs_r)}) {time.time()-xtime} seconds')
                    reqs_r = []

                    xtime = time.time()
                    if saved_freq != -1:
                        if self._reader._flattened:
                            full_freq = np.zeros((np.prod(self._xshape),self._nv,self._n_modes_save),dtype=phi.dtype)
                        else:
                            full_freq = np.zeros(self._xshape+(self._nv,self._n_modes_save),dtype=phi.dtype)

                        for proc in range(comm.size):
                            if use_padding:
                                start = proc*phi0_max*phi.shape[1]
                                end = start+recvcounts[proc]
                            else:
                                start = np.sum(recvcounts[:proc])
                                end = np.sum(recvcounts[:proc+1])

                            if self._reader._flattened:
                                idx_full_freq = [np.s_[:]] * 3
                                xfrom = int(np.sum(recvcounts[:proc])/self._nv/self._n_modes_save)
                                xto = int(np.sum(recvcounts[:proc+1])/self._nv/self._n_modes_save)
                                idx_full_freq[0] = slice(xfrom, xto)
                                # print(f'proc {proc} xfrom {xfrom} xto {xto} start {start} end {end} {full_freq.shape = :}')
                            else:
                                x_prod_not_max = np.prod(self._xshape)
                                x_prod_not_max /= self._xshape[self._max_axis]
                                max_axis_from = int(start//(self._nv*self._n_modes_save*x_prod_not_max))
                                max_axis_to   = int(end//(self._nv*self._n_modes_save*x_prod_not_max))

                                idx_full_freq = [np.s_[:]] * (self._xdim + 2)
                                idx_full_freq[self._max_axis] = slice(max_axis_from, max_axis_to)

                            full_freq[tuple(idx_full_freq)] = np.reshape(data[start:end],full_freq[tuple(idx_full_freq)].shape)

                        # write to disk
                        if self._reader._flattened:
                            full_freq = full_freq.reshape(self._xshape+(self._nv,)+(self._n_modes_save,))

                        filename = f'freq_idx_{saved_freq:08d}.npy'
                        print(f'rank {rank} saving {filename}')
                        p_modes = os.path.join(self._modes_dir, filename)
                        np.save(p_modes, full_freq)
                        saved_freq = -1
                        data = None

                    MPI.Request.Waitall(reqs_s)
                    reqs_s = []
                    s_msgs = {}

                    # comm.Barrier()
                    self._pr0(f'saving: {time.time() - xtime} s.')

            self._pr0(
                f'freq: {f+1}/{self._n_freq};  (f = {self._freq[f]:.5f});  '
                f'Elapsed time: {(time.time() - s0):.5f} s.')

        if self._savefreq_disk2:
            mpi_dtype.Free()

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