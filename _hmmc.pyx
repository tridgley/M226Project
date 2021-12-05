# cython: language_level=3, boundscheck=False, wraparound=False

from numpy.math cimport expl, logl, log1pl, isinf, fabsl, INFINITY
import numpy as np

ctypedef double dtype_t
ctypedef long double dtype_f
ctypedef int dtype_d

cdef inline int _argmax(dtype_t[:] X) nogil:
    cdef dtype_t X_max = -INFINITY
    cdef int pos = 0
    cdef int i
    for i in range(X.shape[0]):
        if X[i] > X_max:
            X_max = X[i]
            pos = i
    return pos


cdef inline int _argmin(dtype_t[:] X) nogil:
    cdef dtype_t X_min = INFINITY
    cdef int pos = 0
    cdef int i
    for i in range(X.shape[0]):
        if X[i] < X_min:
            X_min = X[i]
            pos = i
    return pos


cdef inline dtype_t _max(dtype_t[:] X) nogil:
    return X[_argmax(X)]


cdef inline dtype_t _min(dtype_t[:] X) nogil:
    return X[_argmin(X)]


cdef inline dtype_t _logsumexp(dtype_t[:] X) nogil:
    cdef dtype_t X_max = _max(X)
    if isinf(X_max):
        return -INFINITY

    cdef dtype_t acc = 0
    for i in range(X.shape[0]):
        acc += expl(X[i] - X_max)

    return logl(acc) + X_max


cdef inline dtype_t _logaddexp(dtype_t a, dtype_t b) nogil:
    if isinf(a) and a < 0:
        return b
    elif isinf(b) and b < 0:
        return a
    else:
        return max(a, b) + log1pl(expl(-fabsl(a - b)))


def forward_log(dtype_f[:] log_startprob,
                dtype_f[:, :] log_transmat,
                dtype_f[:, :] framelogprob,
                int kSeg):
    """
    Compute the forward/alpha lattice using logarithms:
        P(O_1, O_2, ..., O_t, q_t=S_i | model)
    """
    cdef ssize_t ns = framelogprob.shape[0], nc = framelogprob.shape[1]
    cdef dtype_t[:, ::1] fwdlattice = np.zeros((ns, nc))
    cdef dtype_t[:, ::1] seglattice = np.zeros((kSeg+1, nc))
    cdef dtype_t[:, ::1] countlattice = np.ones((ns, (kSeg+1)*nc))
    cdef dtype_t[:, ::1] tmplattice = np.zeros((kSeg+1, nc))
    cdef ssize_t t, i, j, k, s
    cdef dtype_t[:, ::1] tmp_buf = np.zeros((kSeg+1, nc))
    cdef dtype_t[::1] tmp = np.zeros(nc)

    with nogil:
        for i in range(nc):
            fwdlattice[0, i] = log_startprob[i] + framelogprob[0, i]
            tmp_buf[0, i] = -INFINITY
            for s in range(kSeg+1):
                # Store the fw probs for each # of segments in its own row (1, 2, ..., kSeg)
                # The array is 1D for summing easily using the logsumexp() provided
                if s==1:
                    seglattice[s, i] = fwdlattice[0, i]
                    tmplattice[s, i] = fwdlattice[0, i]
                    countlattice[0, s*nc + i] = fwdlattice[0, i]
                else:
                    seglattice[s, i] = -INFINITY
                    tmplattice[s, i] = -INFINITY
                    countlattice[0, s*nc + i] = 10

        for t in range(1, ns):
            for j in range(nc):
                # j is the state at step t
                for s in range(1, kSeg+1):
                    # s is the current 1-indexed # of segments that we are summing over
                    for i in range(nc):
                        # i is the state at step t-1
                        if i == j and seglattice[s, i] != -INFINITY:
                            tmp_buf[s, i] = seglattice[s, i] + log_transmat[i, j]
                        elif i != j and seglattice[s-1, i] != -INFINITY:
                            tmp_buf[s, i] = seglattice[s-1, i] + log_transmat[i, j]
                        else:
                            tmp_buf[s, i] = -INFINITY
                    # seglattice is calculated after tmp is full (after each i*k iterations)
                    tmplattice[s, j] = _logsumexp(tmp_buf[s, :]) + framelogprob[t, j]
                # fwdlattice is now calculated after each s (summing over the segments 1...kSeg)
                fwdlattice[t, j] = _logsumexp(tmplattice[:, j])

            for s in range(1, kSeg+1):
                for i in range(nc):
                    seglattice[s, i] = tmplattice[s, i]
            for s in range(1, kSeg+1):
                for i in range(nc):
                    if s==0:
                        countlattice[t, s*nc+i] = 10
                    else:
                        countlattice[t, s*nc+i] = seglattice[s, i]

    return np.asarray(fwdlattice), np.asarray(countlattice)


def forward_scaling(dtype_t[:] startprob,
                    dtype_t[:, :] transmat,
                    dtype_t[:, :] frameprob,
                    dtype_t min_scaling=1e-300):
    """
    Compute the fwdlattice/alpha lattice using scaling_factors:
        P(O_1, O_2, ..., O_t, q_t=S_i | model)
    """
    cdef ssize_t ns = frameprob.shape[0], nc = frameprob.shape[1]
    cdef dtype_t[:, ::1] fwdlattice = np.zeros((ns, nc))
    cdef dtype_t[::1] scaling_factors = np.zeros(ns)
    cdef ssize_t t, i, j

    with nogil:

        # Compute intial column of fwdlattice
        for i in range(nc):
            fwdlattice[0, i] = startprob[i] * frameprob[0, i]
        for i in range(nc):
            scaling_factors[0] += fwdlattice[0, i]
        if scaling_factors[0] < min_scaling:
            raise ValueError("Forward pass failed with underflow, "
                             "consider using implementation='log' instead")
        else:
            scaling_factors[0] = 1.0 / scaling_factors[0]
        for i in range(nc):
            fwdlattice[0, i] *= scaling_factors[0]

        # Compute rest of Alpha
        for t in range(1, ns):
            for j in range(nc):
                for i in range(nc):
                    fwdlattice[t, j] += fwdlattice[t-1, i] * transmat[i, j]
                fwdlattice[t, j] *= frameprob[t, j]
            for i in range(nc):
                scaling_factors[t] += fwdlattice[t, i]
            if scaling_factors[t] < min_scaling:
                raise ValueError("Forward pass failed with underflow, "
                                 "consider using implementation='log' instead")
            else:
                scaling_factors[t] = 1.0 / scaling_factors[t]
            for i in range(nc):
                fwdlattice[t, i] *= scaling_factors[t]

    return np.asarray(fwdlattice), np.asarray(scaling_factors)


def backward(dtype_f[:] log_startprob,
                 dtype_f[:, :] log_transmat,
                 dtype_f[:, :] framelogprob,
                 int kSeg):
    """
    Compute the backward/beta lattice using logarithms:
        P(O_t+1, O_t+2, ..., O_t, q_t=S_i | model)
    """
    cdef ssize_t ns = framelogprob.shape[0], nc = framelogprob.shape[1]
    cdef dtype_t[:, ::1] bwdlattice = np.zeros((ns, nc))
    cdef dtype_t[:, ::1] seglattice = np.zeros((kSeg+1, nc))
    cdef dtype_t[:, ::1] tmplattice = np.zeros((kSeg+1, nc))
    cdef dtype_t[:, ::1] countlattice = np.ones((ns, (kSeg+1)*nc))
    cdef ssize_t t, i, j, k, s
    cdef dtype_t[:, ::1] tmp_buf = np.zeros((kSeg+1, nc))
    cdef dtype_t[::1] tmp = np.zeros(nc)

    with nogil:
        for i in range(nc):
            bwdlattice[ns-1, i] = 0
            tmp_buf[0, i] = -INFINITY
            for s in range(kSeg+1):
                # Store the fw probs for each # of segments in its own row (1, 2, ..., kSeg)
                # The array is 1D for summing easily using the logsumexp() provided
                if s==1:
                    seglattice[s, i] = bwdlattice[ns-1, i]
                    tmplattice[s, i] = bwdlattice[ns-1, i]
                    countlattice[0, s*nc + i] = bwdlattice[ns-1, i]
                else:
                    seglattice[s, i] = -INFINITY
                    tmplattice[s, i] = -INFINITY
                    countlattice[0, s*nc + i] = 10

        for t in range(ns-2, -1, -1):
            for i in range(nc):
                # i is the state at t
                for s in range(1, kSeg+1):
                    # s is the current 1-indexed # of segments that we are summing over
                    for j in range(nc):
                        # j is the state at t+1
                        # Check if there is a state transition and how many counts:
                        if i == j and seglattice[s, j] != -INFINITY:
                            tmp_buf[s, j] = seglattice[s, j] + log_transmat[i, j]
                        elif i != j and seglattice[s-1, j] != -INFINITY:
                            tmp_buf[s, j] = seglattice[s-1, j] + log_transmat[i, j]
                        else:
                            tmp_buf[s, j] = -INFINITY
                    # seglattice is calculated after tmp is full (after each i*k iterations)
                    tmplattice[s, i] = _logsumexp(tmp_buf[s, :]) + framelogprob[t, i]
                # fwdlattice is now calculated after each s (summing over the segments 1...kSeg)
                bwdlattice[t, i] = _logsumexp(tmplattice[:, i])

            for s in range(1, kSeg+1):
                for j in range(nc):
                    seglattice[s, j] = tmplattice[s, j]
            for s in range(1, kSeg+1):
                for j in range(nc):
                    if s==0:
                        countlattice[t, s*nc+j] = 10
                    else:
                        countlattice[t, s*nc+j] = seglattice[s, j]

    return np.asarray(bwdlattice), np.asarray(countlattice)


def backward_scaling(dtype_t[:] startprob,
                     dtype_t[:, :] transmat,
                     dtype_t[:, :] frameprob,
                     dtype_t[:] scaling_factors):
    """
    Compute the backward/beta lattice using scaling_factors:
        P(O_t+1, O_t+2, ..., O_t, q_t=S_i | model)
    """
    cdef ssize_t ns = frameprob.shape[0], nc = frameprob.shape[1]
    cdef dtype_t[:, ::1] bwdlattice = np.zeros((ns, nc))
    cdef ssize_t t, i, j
    with nogil:
        bwdlattice[:] = 0
        bwdlattice[ns-1, :] = scaling_factors[ns-1]
        for t in range(ns-2, -1, -1):
            for j in range(nc):
                for i in range(nc):
                    bwdlattice[t, j] += (transmat[j, i]
                                         * frameprob[t+1, i]
                                         * bwdlattice[t+1, i])
                bwdlattice[t, j] *= scaling_factors[t]
    return np.asarray(bwdlattice)


def compute_log_xi_sum(dtype_t[:, :] fwdlattice,
                       dtype_f[:, :] log_transmat,
                       dtype_t[:, :] bwdlattice,
                       dtype_f[:, :] framelogprob,
                       dtype_t[:, :] fwd_segments,
                       dtype_t[:, :] bw_segments,
                       int kSeg):
    cdef ssize_t ns = framelogprob.shape[0], nc = framelogprob.shape[1]
    cdef dtype_t[:, ::1] log_xi_sum = np.full((nc, nc), -INFINITY)
    cdef int t, i, j, k, s
    cdef dtype_t log_xi, logprob = _logsumexp(fwdlattice[ns-1])
    with nogil:
        for t in range(ns-1):
            for i in range(nc):
                for s in range(1, kSeg+1):
                    #logprob = _logsumexp(fwd_segments[ns-1, s*nc:s*nc+nc])
                    for j in range(nc):
                        if (i == j and fwd_segments[t, s*nc + i] <= 0 and bw_segments[t+1, s*nc + j] <= 0):
                            log_xi = (fwd_segments[t, s*nc + i]
                                      + log_transmat[i, j]
                                      + framelogprob[t+1, j]
                                      + bw_segments[t+1, s*nc + j]
                                      - logprob)
                        elif (i != j and fwd_segments[t, (s-1)*nc + i] <= 0 and bw_segments[t+1, (s-1)*nc + j] <= 0):
                            log_xi = (fwd_segments[t, (s-1)*nc + i]
                                      + log_transmat[i, j]
                                      + framelogprob[t+1, j]
                                      + bw_segments[t+1, (s-1)*nc + j]
                                      - logprob)
                        else:
                            log_xi = -INFINITY
                        log_xi_sum[i, j] = _logaddexp(log_xi_sum[i, j], log_xi)

    return np.asarray(log_xi_sum)


def compute_scaling_xi_sum(dtype_t[:, :] fwdlattice,
                           dtype_t[:, :] transmat,
                           dtype_t[:, :] bwdlattice,
                           dtype_t[:, :] frameprob):
    cdef ssize_t ns = frameprob.shape[0], nc = frameprob.shape[1]
    cdef dtype_t[:, ::1] xi_sum = np.zeros((nc, nc))
    cdef int t, i, j
    with nogil:
        for t in range(ns-1):
            for i in range(nc):
                for j in range(nc):
                    xi_sum[i, j] += (fwdlattice[t, i]
                                     * transmat[i, j]
                                     * frameprob[t+1, j]
                                     * bwdlattice[t+1, j])
    return np.asarray(xi_sum)


def viterbi(dtype_f[:] log_startprob,
            dtype_f[:, :] log_transmat,
            dtype_f[:, :] framelogprob,
            int kSeg):

    cdef ssize_t ns = framelogprob.shape[0], nc = framelogprob.shape[1]
    cdef int i, j, t, k, prev
    cdef dtype_t logprob
    cdef int[::1] state_sequence = np.empty(ns, dtype=np.int32)
    cdef dtype_t[:, ::1] viterbi_lattice = np.zeros((ns, nc))
    cdef dtype_t[:, ::1] viterbi_lattice2 = np.zeros((ns, nc))
    cdef dtype_t[:, ::1] viterbi_lattice3 = np.zeros((ns, nc))
    cdef dtype_t[::1] tmp_buf = np.empty(nc)
    cdef dtype_t[:, ::1] stateswitchlattice = np.zeros((ns, nc))
    cdef int[::1] best_xj_index = np.array([0, 1, 2, 3, 4], dtype=np.int32)
    cdef int[::1] best_xj_count = np.ones(nc, dtype=np.int32)
    cdef int[::1] new_count = np.ones(nc, dtype=np.int32)

    with nogil:
        for i in range(nc):
            viterbi_lattice[0, i] = log_startprob[i] + framelogprob[0, i]

        # Induction
        for t in range(1, ns):
            for i in range(nc):
                # i is the state at t
                for j in range(nc):
                    # j is the state at t-1
                    if (i==j and best_xj_count[j] <= kSeg) or (i!=j and best_xj_count[j] < kSeg):
                        tmp_buf[j] = log_transmat[j, i] + viterbi_lattice[t-1, j]
                    else:
                        tmp_buf[j] = -INFINITY                        

                    #tmp_buf[j] = log_transmat[j, i] + viterbi_lattice[t-1, j]

                # Take the max over valid transitions and update segment variables
                viterbi_lattice[t, i] = _max(tmp_buf) + framelogprob[t, i]
                best_xj_index[i] = _argmax(tmp_buf)
                if best_xj_index[i] != i:
                    # best_xj_count[i] = best_xj_count[best_xj_index[i]] + 1
                    new_count[i] = best_xj_count[best_xj_index[i]] + 1
                stateswitchlattice[t-1, i] = best_xj_index[i]
            for k in range(nc):
                # Now update the counts at the end of time step t for next time step t+1
                best_xj_count[k] = new_count[k]

        # Observation traceback
        state_sequence[ns-1] = prev = _argmax(viterbi_lattice[ns-1])
        logprob = viterbi_lattice[ns-1, prev]
        
        for t in range(ns-2, -1, -1):
            for i in range(nc):
                # prev is at time step t+1 and i is at time step t
                if i == stateswitchlattice[t, prev]:
                    tmp_buf[i] = viterbi_lattice[t, i] + log_transmat[i, prev]
                else:
                    tmp_buf[i] = -INFINITY
            state_sequence[t] = prev = _argmax(tmp_buf)
    return np.asarray(state_sequence), logprob


def viterbi_path_fix(dtype_f[:] log_startprob,
            dtype_f[:, :] log_transmat,
            dtype_f[:, :] framelogprob,
            int kSeg):

    cdef ssize_t ns = framelogprob.shape[0], nc = framelogprob.shape[1]
    cdef int i, j, t, s, prev
    cdef dtype_t logprob
    cdef int[::1] state_sequence = np.empty(ns, dtype=np.int32)
    cdef dtype_t[:, ::1] viterbi_lattice = np.zeros((ns, kSeg*nc))
    cdef dtype_t[:, ::1] seg_lattice = np.zeros((ns, kSeg*nc))
    cdef dtype_t[::1] tmp_buf = np.empty(kSeg*nc)
    cdef dtype_t[::1] bt_buf = np.empty(kSeg*nc)

    with nogil:
        for i in range(nc):
            for s in range(kSeg):
                if s==0:
                    viterbi_lattice[0, s*nc+i] = log_startprob[i] + framelogprob[0, i]
                else:
                    viterbi_lattice[0, s*nc+i] = -INFINITY
                seg_lattice[0, s*nc+i] = i

        # Induction
        for t in range(1, ns):
            for i in range(nc):
                # i is the state at t
                for s in range(kSeg):
                    # iterate over paths with s segments
                    for j in range(nc):
                        # j is the state at t-1
                        if (i==j and viterbi_lattice[t-1, s*nc + j] != -INFINITY):
                            tmp_buf[s*nc + j] = log_transmat[j, i] + viterbi_lattice[t-1, s*nc+j]
                        elif s and (i!=j and viterbi_lattice[t-1, (s-1)*nc + j] != -INFINITY):
                            tmp_buf[s*nc + j] = log_transmat[j, i] + viterbi_lattice[t-1, (s-1)*nc+j]
                        else:
                            tmp_buf[s*nc + j] = -INFINITY

                    # Take the max over valid transitions and update segment variables
                    if s==0:
                        viterbi_lattice[t, s*nc + i] = _max(tmp_buf[:nc]) + framelogprob[t, i]
                        best = _argmax(tmp_buf[:nc])
                        seg_lattice[t, s*nc + i] = best
                    else:
                        viterbi_lattice[t, s*nc + i] = _max(tmp_buf[s*nc - nc : s*nc + nc]) + framelogprob[t, i]
                        best = (s-1)*nc + _argmax(tmp_buf[s*nc - nc : s*nc + nc])
                        seg_lattice[t, s*nc + i] = best

        # Observation traceback
        state_sequence[ns-1] = prev = _argmax(viterbi_lattice[ns-1])
        logprob = viterbi_lattice[ns-1, prev]
        
        for t in range(ns-2, -1, -1):
            for i in range(kSeg*nc):
                # prev is at time step t+1 and i is at time step t
                if i == seg_lattice[t, prev]:
                    bt_buf[i] = viterbi_lattice[t, i] + log_transmat[i % kSeg, prev % kSeg]
                else:
                    bt_buf[i] = -INFINITY

            state_sequence[t] = prev = _argmax(bt_buf)

    return np.asarray(state_sequence), logprob
