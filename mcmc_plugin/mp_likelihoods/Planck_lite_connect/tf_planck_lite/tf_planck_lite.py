#!/usr/bin/env python

# author: Alessio Spurio Mancini and Andreas Nygaard

import os

import numpy as np
from scipy.io import FortranFile

import tensorflow as tf
import tf_keras

from Splines import Spline


pi = tf.constant(np.pi, dtype=tf.float32)
log_two_pi = tf.constant(np.log(2.0 * np.pi), dtype=tf.float32)


class tf_planck2018_lite:
    """
    TensorFlow implementation of the approximate Planck 2018 likelihoods.

    The class can independently include:

        - high-l TTTEEE Plik-lite
        - low-l TT lognormal approximation
        - low-l EE lognormal approximation

    All enabled likelihood components share a single CONNECT network
    evaluation and the same spline-interpolated TT/TE/EE spectra.

    Parameters
    ----------
    tf_planck2018_lite_path : str
        Path to the tf_planck_lite directory.

    spectra : str
        Currently only 'TTTEEE' is supported.

    units_factor : float
        Conversion factor from dimensionless spectra to microK^2.
        Default is (2.7255e6)^2.

    connect_ttteee_model : str
        Path to the CONNECT TTTEEE network.

    use_highl : bool
        Include the high-l Plik-lite TTTEEE likelihood.

    use_lowl_TT : bool
        Include the low-l TT two-bin lognormal likelihood.

    use_lowl_EE : bool
        Include the low-l EE three-bin lognormal likelihood.

    Notes
    -----
    The final entry of the parameter vector is assumed to be A_planck
    (called `cal` internally).

    Calibration is applied consistently to all enabled likelihoods as

        theory -> theory / cal^2

    before comparison with Planck data.
    """

    # ==================================================================
    # INITIALIZATION
    # ==================================================================

    def __init__(
        self,
        tf_planck2018_lite_path='./',
        spectra='TTTEEE',
        units_factor=(2.7255e6)**2,
        connect_ttteee_model=None,
        fiducial=None,
        use_highl=True,
        use_lowl_TT=True,
        use_lowl_EE=True
    ):

        # --------------------------------------------------------------
        # General configuration
        # --------------------------------------------------------------

        self.data_directory = os.path.join(
            tf_planck2018_lite_path,
            'data'
        )

        self.spectra = spectra

        self.units_factor = tf.constant(
            units_factor,
            dtype=tf.float32
        )

        self.use_highl = use_highl
        self.use_lowl_TT = use_lowl_TT
        self.use_lowl_EE = use_lowl_EE

        if not (
            self.use_highl
            or self.use_lowl_TT
            or self.use_lowl_EE
        ):
            raise ValueError(
                "At least one likelihood must be enabled."
            )

        # --------------------------------------------------------------
        # CONNECT emulator
        # --------------------------------------------------------------

        if connect_ttteee_model is None:
            raise ValueError(
                'A CONNECT TTTEEE emulator must be supplied.'
            )

        self.model_name = connect_ttteee_model

        self.model = tf_keras.models.load_model(
            self.model_name,
            compile=False
        )

        self.output_info = eval(
            self.model
            .get_raw_info()
            .numpy()
            .decode('utf-8')
        )

        self.cosmo_params = self.output_info['input_names']

        # --------------------------------------------------------------
        # Spectrum grids
        # --------------------------------------------------------------

        # Full target grid:
        #
        # index 0 -> ell = 2
        # ...
        # index 2506 -> ell = 2508

        self.ell = tf.linspace(
            2.0,
            2508.0,
            2507
        )

        self.ell_computed = tf.constant(
            self.output_info['ell'],
            dtype=tf.float32
        )

        self.N = self.ell_computed.shape[0] - 1

        # One spline object is sufficient because TT, TE, and EE
        # use the same input/output ell grids.

        self.spline = Spline(
            self.ell_computed,
            self.ell
        )

        # Indices of the CONNECT outputs corresponding to TT/TE/EE.

        self.lim0_tt = self.output_info[
            'interval'
        ]['Cl']['tt'][0]

        self.lim1_tt = self.output_info[
            'interval'
        ]['Cl']['tt'][1]

        self.lim0_te = self.output_info[
            'interval'
        ]['Cl']['te'][0]

        self.lim1_te = self.output_info[
            'interval'
        ]['Cl']['te'][1]

        self.lim0_ee = self.output_info[
            'interval'
        ]['Cl']['ee'][0]

        self.lim1_ee = self.output_info[
            'interval'
        ]['Cl']['ee'][1]

        if fiducial is not None:
            v = tf.constant(
                [fiducial],
                dtype=tf.float32
            )

            self.output_predict = self.model(v)

        # --------------------------------------------------------------
        # Spectrum support
        # --------------------------------------------------------------

        if self.spectra != 'TTTEEE':
            raise ValueError(
                "Currently only spectra='TTTEEE' is supported."
            )

        self.use_tt = True
        self.use_te = True
        self.use_ee = True

        # --------------------------------------------------------------
        # Load selected likelihoods
        # --------------------------------------------------------------

        if self.use_highl:
            self._initialize_highl()

        if self.use_lowl_TT or self.use_lowl_EE:
            self._initialize_lowl()


    # ==================================================================
    # HIGH-L INITIALIZATION
    # ==================================================================

    def _initialize_highl(self):

        self.plmin_TT = 30
        self.plmin = 30
        self.plmax = 2508

        self.data_dir = os.path.join(
            self.data_directory,
            'planck2018_plik_lite'
        )

        version = 22

        self.nbintt_hi = 215
        self.nbinte = 199
        self.nbinee = 199

        self.nbintt = self.nbintt_hi

        self.nbin_hi = (
            self.nbintt_hi
            + self.nbinte
            + self.nbinee
        )

        self.nbin_tot = (
            self.nbintt
            + self.nbinte
            + self.nbinee
        )

        # --------------------------------------------------------------
        # Files
        # --------------------------------------------------------------

        self.like_file = os.path.join(
            self.data_dir,
            'cl_cmb_plik_v'
            + str(version)
            + '.dat'
        )

        self.cov_file = os.path.join(
            self.data_dir,
            'c_matrix_plik_v'
            + str(version)
            + '.dat'
        )

        self.blmin_file = os.path.join(
            self.data_dir,
            'blmin.dat'
        )

        self.blmax_file = os.path.join(
            self.data_dir,
            'blmax.dat'
        )

        self.binw_file = os.path.join(
            self.data_dir,
            'bweight.dat'
        )

        # --------------------------------------------------------------
        # Observed Planck bandpowers
        # --------------------------------------------------------------

        (
            self.bval,
            X_data,
            self.X_sig
        ) = np.genfromtxt(
            self.like_file,
            unpack=True
        )

        self.X_data = tf.constant(
            X_data.astype(np.float32)[None, :],
            dtype=tf.float32
        )

        # --------------------------------------------------------------
        # Binning definitions
        # --------------------------------------------------------------

        self.blmin = np.loadtxt(
            self.blmin_file
        ).astype(int)

        self.blmax = np.loadtxt(
            self.blmax_file
        ).astype(int)

        self.bin_w = np.loadtxt(
            self.binw_file
        ).astype(np.float32)

        self.blmin_TT = self.blmin
        self.blmax_TT = self.blmax
        self.bin_w_TT = self.bin_w

        # --------------------------------------------------------------
        # Fisher matrix
        # --------------------------------------------------------------

        self.fisher = self.get_inverse_covmat()

        # --------------------------------------------------------------
        # Planck window ranges
        # --------------------------------------------------------------

        begintt = (
            self.blmin_TT
            + self.plmin_TT
            - 2
        )

        endtt = (
            self.blmax_TT
            + self.plmin_TT
            + 1
            - 2
        )

        beginte = (
            self.blmin
            + self.plmin
            - 2
        )

        endte = (
            self.blmax
            + self.plmin
            + 1
            - 2
        )

        beginee = (
            self.blmin
            + self.plmin
            - 2
        )

        endee = (
            self.blmax
            + self.plmin
            + 1
            - 2
        )

        beginwtt = self.blmin_TT
        endwtt = self.blmax_TT + 1

        beginwte = self.blmin
        endwte = self.blmax + 1

        beginwee = self.blmin
        endwee = self.blmax + 1

        # --------------------------------------------------------------
        # TT windows
        # --------------------------------------------------------------

        indicestt = []
        windowtt = []
        indices_reptt = []

        for i in range(self.nbintt):

            ell_indices = np.arange(
                begintt[i],
                endtt[i]
            )

            indicestt.append(
                ell_indices
            )

            windowtt.append(
                self.bin_w_TT[
                    beginwtt[i]:endwtt[i]
                ]
            )

            indices_reptt.append(
                np.repeat(
                    i,
                    len(ell_indices)
                )
            )

        indices_reptt = np.concatenate(
            indices_reptt
        ).astype(np.int32)

        indicestt = np.concatenate(
            indicestt
        ).astype(np.int32)

        windowtt = np.concatenate(
            windowtt
        ).astype(np.float32)

        # --------------------------------------------------------------
        # TE windows
        # --------------------------------------------------------------

        indiceste = []
        windowte = []
        indices_repte = []

        for i in range(self.nbinte):

            ell_indices = np.arange(
                beginte[i],
                endte[i]
            )

            indices_repte.append(
                np.repeat(
                    self.nbintt + i,
                    len(ell_indices)
                )
            )

            # Offset into concatenated [TT | TE | EE] C_ell vector.

            indiceste.append(
                (self.plmax - 1)
                + ell_indices
            )

            windowte.append(
                self.bin_w[
                    beginwte[i]:endwte[i]
                ]
            )

        indices_repte = np.concatenate(
            indices_repte
        ).astype(np.int32)

        indiceste = np.concatenate(
            indiceste
        ).astype(np.int32)

        windowte = np.concatenate(
            windowte
        ).astype(np.float32)

        # --------------------------------------------------------------
        # EE windows
        # --------------------------------------------------------------

        indicesee = []
        windowee = []
        indices_repee = []

        for i in range(self.nbinee):

            ell_indices = np.arange(
                beginee[i],
                endee[i]
            )

            indices_repee.append(
                np.repeat(
                    self.nbintt
                    + self.nbinte
                    + i,
                    len(ell_indices)
                )
            )

            indicesee.append(
                2 * (self.plmax - 1)
                + ell_indices
            )

            windowee.append(
                self.bin_w[
                    beginwee[i]:endwee[i]
                ]
            )

        indices_repee = np.concatenate(
            indices_repee
        ).astype(np.int32)

        indicesee = np.concatenate(
            indicesee
        ).astype(np.int32)

        windowee = np.concatenate(
            windowee
        ).astype(np.float32)

        # --------------------------------------------------------------
        # Combined flattened representation
        # --------------------------------------------------------------

        indices_rep = np.concatenate([
            indices_reptt,
            indices_repte,
            indices_repee
        ]).astype(np.int32)

        indices = np.concatenate([
            indicestt,
            indiceste,
            indicesee
        ]).astype(np.int32)

        window = np.concatenate([
            windowtt,
            windowte,
            windowee
        ]).astype(np.float32)

        self.indices_rep = tf.constant(
            indices_rep,
            dtype=tf.int32
        )

        self.indices = tf.constant(
            indices,
            dtype=tf.int32
        )

        self.window_ttteee = tf.constant(
            window,
            dtype=tf.float32
        )

        # --------------------------------------------------------------
        # XLA-friendly padded bin representation
        # --------------------------------------------------------------

        bin_lengths = np.bincount(
            indices_rep,
            minlength=self.nbin_tot
        )

        max_bin_width = int(
            np.max(bin_lengths)
        )

        bin_indices = np.zeros(
            (
                self.nbin_tot,
                max_bin_width
            ),
            dtype=np.int32
        )

        bin_weights = np.zeros(
            (
                self.nbin_tot,
                max_bin_width
            ),
            dtype=np.float32
        )

        for b in range(self.nbin_tot):

            mask = (
                indices_rep == b
            )

            idx_b = indices[mask]
            weight_b = window[mask]

            n = len(idx_b)

            bin_indices[
                b,
                :n
            ] = idx_b

            bin_weights[
                b,
                :n
            ] = weight_b

        self.bin_indices = tf.constant(
            bin_indices,
            dtype=tf.int32
        )

        self.bin_weights = tf.constant(
            bin_weights,
            dtype=tf.float32
        )

        self.max_bin_width = max_bin_width


    # ==================================================================
    # LOW-L INITIALIZATION
    # ==================================================================

    def _initialize_lowl(self):

        self.lowl_data_dir = os.path.join(
            self.data_directory,
            'lowl'
        )

        # Low-l likelihood uses ell = 2,...,29.
        #
        # Since self.ell starts at ell=2, these are simply the first
        # 28 entries of the full spline result.

        self.lowl_n_ell = 28

        self.lowl_ell = tf.range(
            2,
            30,
            dtype=tf.float32
        )

        # --------------------------------------------------------------
        # Low-l TT
        # --------------------------------------------------------------

        if self.use_lowl_TT:

            self.nbins_TT = 2

            (
                lmin_TT,
                lmax_TT,
                mu_TT,
                sig_TT
            ) = np.loadtxt(
                os.path.join(
                    self.lowl_data_dir,
                    'lognormal_fit_'
                    + str(self.nbins_TT)
                    + 'bins_TT.txt'
                ),
                unpack=True
            )

            lmin_TT = lmin_TT.astype(np.int32)
            lmax_TT = lmax_TT.astype(np.int32)

            self.mu_LN_TT = tf.constant(
                mu_TT.astype(np.float32),
                dtype=tf.float32
            )

            self.sig_LN_TT = tf.constant(
                sig_TT.astype(np.float32),
                dtype=tf.float32
            )

            # TT has loc=0 in the original likelihood.

            self.loc_LN_TT = tf.zeros(
                self.nbins_TT,
                dtype=tf.float32
            )

            self.lowl_TT_bin_matrix = tf.constant(
                self._make_lowl_bin_matrix(
                    lmin_TT,
                    lmax_TT
                ),
                dtype=tf.float32
            )

        # --------------------------------------------------------------
        # Low-l EE
        # --------------------------------------------------------------

        if self.use_lowl_EE:

            self.nbins_EE = 3

            (
                lmin_EE,
                lmax_EE,
                mu_EE,
                sig_EE,
                loc_EE
            ) = np.loadtxt(
                os.path.join(
                    self.lowl_data_dir,
                    'lognormal_fit_'
                    + str(self.nbins_EE)
                    + 'bins_EE.txt'
                ),
                unpack=True
            )

            lmin_EE = lmin_EE.astype(np.int32)
            lmax_EE = lmax_EE.astype(np.int32)

            self.mu_LN_EE = tf.constant(
                mu_EE.astype(np.float32),
                dtype=tf.float32
            )

            self.sig_LN_EE = tf.constant(
                sig_EE.astype(np.float32),
                dtype=tf.float32
            )

            self.loc_LN_EE = tf.constant(
                loc_EE.astype(np.float32),
                dtype=tf.float32
            )

            self.lowl_EE_bin_matrix = tf.constant(
                self._make_lowl_bin_matrix(
                    lmin_EE,
                    lmax_EE
                ),
                dtype=tf.float32
            )


    def _make_lowl_bin_matrix(
        self,
        lmin_list,
        lmax_list
    ):
        """
        Construct a matrix that reproduces the original low-l binning:

            Dl_bin[i] = mean(Dl[lmin_i:lmax_i])

        for ell = 2,...,29.

        Returns
        -------
        matrix : np.ndarray
            Shape (28, n_bins).
        """

        n_bins = len(lmin_list)

        matrix = np.zeros(
            (
                self.lowl_n_ell,
                n_bins
            ),
            dtype=np.float32
        )

        for i, (
            lmin,
            lmax
        ) in enumerate(
            zip(
                lmin_list,
                lmax_list
            )
        ):

            # ell=2 corresponds to array index 0.

            start = int(lmin - 2)

            # Python upper index is exclusive.

            stop = int(lmax - 2 + 1)

            n_ell = stop - start

            matrix[
                start:stop,
                i
            ] = (
                1.0
                / n_ell
            )

        return matrix


    # ==================================================================
    # HIGH-L INVERSE COVARIANCE
    # ==================================================================

    def get_inverse_covmat(self):
        """
        Read and invert the high-l covariance matrix.
        """

        f = FortranFile(
            self.cov_file,
            'r'
        )

        covmat = f.read_reals(
            dtype=float
        ).reshape(
            (
                self.nbin_hi,
                self.nbin_hi
            )
        )

        for i in range(self.nbin_hi):

            for j in range(
                i,
                self.nbin_hi
            ):

                covmat[i, j] = covmat[j, i]

        if (
            self.use_tt
            and self.use_ee
            and self.use_te
        ):

            bin_no = self.nbin_hi
            cov = covmat

        else:

            raise NotImplementedError(
                "Only TTTEEE is currently implemented."
            )

        cov = tf.constant(
            cov.astype(np.float32),
            dtype=tf.float32
        )

        fisher = tf.linalg.cholesky_solve(
            tf.linalg.cholesky(cov),
            tf.eye(
                bin_no,
                dtype=tf.float32
            )
        )

        return tf.transpose(
            fisher
        )


    # ==================================================================
    # CONNECT + SPLINE
    # ==================================================================

    def _get_theory_Dl(
        self,
        cosmo_params
    ):
        """
        Evaluate CONNECT once and spline TT/TE/EE onto ell=2,...,2508.

        The CONNECT output is treated as D_ell. This is the quantity that
        the previous high-l implementation converted to C_ell through

            C_ell = 2*pi*D_ell / [ell*(ell+1)].

        Returns
        -------
        Dl_tt, Dl_te, Dl_ee : tf.Tensor
            Each has shape (batch_size, 2507).
        """

        output_predict = self.model(
            cosmo_params
        )

        Dl_computed_tt = output_predict[
            :,
            self.lim0_tt:self.lim1_tt
        ]

        Dl_computed_te = output_predict[
            :,
            self.lim0_te:self.lim1_te
        ]

        Dl_computed_ee = output_predict[
            :,
            self.lim0_ee:self.lim1_ee
        ]

        Dl_tt = self.spline.do_spline(
            Dl_computed_tt
        )

        Dl_te = self.spline.do_spline(
            Dl_computed_te
        )

        Dl_ee = self.spline.do_spline(
            Dl_computed_ee
        )

        return (
            Dl_tt,
            Dl_te,
            Dl_ee
        )


    # ==================================================================
    # HIGH-L LIKELIHOOD
    # ==================================================================

    def _highl_loglike(
        self,
        Dl_tt,
        Dl_te,
        Dl_ee,
        cal
    ):
        """
        High-l Planck Plik-lite likelihood.

        Returns
        -------
        loglkl : tf.Tensor
            Shape (batch_size,).
        """

        ell_factor = (
            2.0
            * pi
            / (
                self.ell
                * (
                    self.ell
                    + 1.0
                )
            )
        )

        Cl_tt = (
            Dl_tt
            * ell_factor
        )

        Cl_te = (
            Dl_te
            * ell_factor
        )

        Cl_ee = (
            Dl_ee
            * ell_factor
        )

        Cl = (
            self.units_factor
            * tf.concat(
                [
                    Cl_tt,
                    Cl_te,
                    Cl_ee
                ],
                axis=1
            )
        )

        # --------------------------------------------------------------
        # Planck binning
        # --------------------------------------------------------------

        Cl_selected = tf.gather(
            Cl,
            self.bin_indices,
            axis=1
        )

        Cl_bin = tf.reduce_sum(
            Cl_selected
            * self.bin_weights[
                None,
                :,
                :
            ],
            axis=-1
        )

        # --------------------------------------------------------------
        # A_planck calibration
        # --------------------------------------------------------------

        X_model = (
            Cl_bin
            / tf.square(cal)[:, None]
        )

        # --------------------------------------------------------------
        # Chi-square
        # --------------------------------------------------------------

        diff_vec = (
            self.X_data
            - X_model
        )

        fisher_diff = tf.matmul(
            diff_vec,
            self.fisher
        )

        chi2 = tf.reduce_sum(
            fisher_diff
            * diff_vec,
            axis=1
        )

        return (
            -0.5
            * chi2
        )


    # ==================================================================
    # LOGNORMAL LOW-L LIKELIHOOD
    # ==================================================================

    @staticmethod
    def _lognormal_logpdf(
        x,
        mu,
        sigma,
        loc
    ):
        """
        Logarithm of the lognormal probability density.

        This is evaluated directly in log-space for numerical stability.

        Values x <= loc have zero probability and therefore log p = -inf.
        """

        z = (
            x
            - loc
        )

        # Avoid evaluating log(0) or log(negative value), even in the
        # branch that will later be replaced by -inf.

        tiny = tf.constant(
            np.finfo(np.float32).tiny,
            dtype=tf.float32
        )

        z_safe = tf.maximum(
            z,
            tiny
        )

        logpdf = (
            -tf.math.log(z_safe)
            -tf.math.log(sigma)
            -0.5 * log_two_pi
            -tf.square(
                tf.math.log(z_safe)
                - mu
            )
            / (
                2.0
                * tf.square(sigma)
            )
        )

        minus_inf = tf.constant(
            -np.inf,
            dtype=tf.float32
        )

        return tf.where(
            z > 0.0,
            logpdf,
            minus_inf
        )


    # ==================================================================
    # LOW-L TT
    # ==================================================================

    def _lowl_TT_loglike(
        self,
        Dl_tt,
        cal
    ):
        """
        Planck low-l TT two-bin lognormal likelihood.

        Parameters
        ----------
        Dl_tt : tf.Tensor
            Full TT D_ell spectrum on ell=2,...,2508.

        cal : tf.Tensor
            A_planck calibration parameter.

        Returns
        -------
        loglkl : tf.Tensor
            Shape (batch_size,).
        """

        # First 28 entries correspond to ell=2,...,29.

        Dl_low = Dl_tt[
            :,
            :self.lowl_n_ell
        ]

        # CONNECT spectra are converted to microK^2 here.
        #
        # Apply the same calibration convention as the high-l
        # likelihood:
        #
        #     theory -> theory / A_planck^2

        Dl_low = (
            self.units_factor
            * Dl_low
            / tf.square(cal)[:, None]
        )

        # Reproduce the original arithmetic mean within each bin.

        Dl_bin = tf.matmul(
            Dl_low,
            self.lowl_TT_bin_matrix
        )

        logpdf = self._lognormal_logpdf(
            Dl_bin,
            self.mu_LN_TT[None, :],
            self.sig_LN_TT[None, :],
            self.loc_LN_TT[None, :]
        )

        return tf.reduce_sum(
            logpdf,
            axis=-1
        )


    # ==================================================================
    # LOW-L EE
    # ==================================================================

    def _lowl_EE_loglike(
        self,
        Dl_ee,
        cal
    ):
        """
        Planck low-l EE three-bin lognormal likelihood.

        Parameters
        ----------
        Dl_ee : tf.Tensor
            Full EE D_ell spectrum on ell=2,...,2508.

        cal : tf.Tensor
            A_planck calibration parameter.

        Returns
        -------
        loglkl : tf.Tensor
            Shape (batch_size,).
        """

        Dl_low = Dl_ee[
            :,
            :self.lowl_n_ell
        ]

        Dl_low = (
            self.units_factor
            * Dl_low
            / tf.square(cal)[:, None]
        )

        Dl_bin = tf.matmul(
            Dl_low,
            self.lowl_EE_bin_matrix
        )

        logpdf = self._lognormal_logpdf(
            Dl_bin,
            self.mu_LN_EE[None, :],
            self.sig_LN_EE[None, :],
            self.loc_LN_EE[None, :]
        )

        return tf.reduce_sum(
            logpdf,
            axis=-1
        )


    # ==================================================================
    # OPTIONAL PUBLIC LOW-L FUNCTIONS
    # ==================================================================

    @tf.function
    def get_lowl_TT_loglkl(
        self,
        Dl_tt,
        cal
    ):
        """
        Evaluate only the low-l TT likelihood from a supplied D_ell
        spectrum and A_planck.

        Dl_tt must have shape (batch_size, >=28).
        cal must have shape (batch_size,).
        """

        if not self.use_lowl_TT:
            raise ValueError(
                "Low-l TT was not enabled during initialization."
            )

        return self._lowl_TT_loglike(
            Dl_tt,
            cal
        )[:, None]


    @tf.function
    def get_lowl_EE_loglkl(
        self,
        Dl_ee,
        cal
    ):
        """
        Evaluate only the low-l EE likelihood from a supplied D_ell
        spectrum and A_planck.

        Dl_ee must have shape (batch_size, >=28).
        cal must have shape (batch_size,).
        """

        if not self.use_lowl_EE:
            raise ValueError(
                "Low-l EE was not enabled during initialization."
            )

        return self._lowl_EE_loglike(
            Dl_ee,
            cal
        )[:, None]


    # ==================================================================
    # COMBINED LIKELIHOOD
    # ==================================================================

    @tf.function
    def get_loglkl(
        self,
        parameters
    ):
        """
        Compute the combined selected Planck likelihood.

        Parameters
        ----------
        parameters : tf.Tensor
            Shape

                (batch_size, n_cosmological_parameters + 1)

            where the final parameter is A_planck.

        Returns
        -------
        loglkl : tf.Tensor
            Shape (batch_size, 1).

        Notes
        -----
        CONNECT is evaluated exactly once per call, regardless of how many
        likelihood components are enabled.
        """

        cosmo_params = parameters[
            :,
            :-1
        ]

        cal = parameters[
            :,
            -1
        ]

        # --------------------------------------------------------------
        # ONE CONNECT evaluation + one spline evaluation per spectrum
        # --------------------------------------------------------------

        (
            Dl_tt,
            Dl_te,
            Dl_ee
        ) = self._get_theory_Dl(
            cosmo_params
        )

        loglkl = tf.zeros(
            tf.shape(parameters)[0],
            dtype=tf.float32
        )

        # --------------------------------------------------------------
        # High-l TTTEEE
        # --------------------------------------------------------------

        if self.use_highl:

            loglkl += self._highl_loglike(
                Dl_tt,
                Dl_te,
                Dl_ee,
                cal
            )

        # --------------------------------------------------------------
        # Low-l TT
        # --------------------------------------------------------------

        if self.use_lowl_TT:

            loglkl += self._lowl_TT_loglike(
                Dl_tt,
                cal
            )

        # --------------------------------------------------------------
        # Low-l EE
        # --------------------------------------------------------------

        if self.use_lowl_EE:

            loglkl += self._lowl_EE_loglike(
                Dl_ee,
                cal
            )

        return loglkl[:, None]
