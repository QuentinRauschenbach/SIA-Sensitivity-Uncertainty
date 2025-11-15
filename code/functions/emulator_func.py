# Emulator

import numpy as np
import functions.siametadata as meta
import statsmodels.api as sm
from statsmodels.regression.linear_model import yule_walker
import pandas as pd

import matplotlib.pyplot as plt

import functions.memberhandling as mbhl
import glob
import pickle

#datapath = "C:/Users/quentin/Documents/Uni/Thesis/data/"
#datapath = "/Users/quraus001/Documents/Uni/Thesis/data/"
datapath = "/Users/quraus001/Documents/Uni/SIA-Sensitivity-Uncertainty/data/" # has subfolder "splines"

spline_files = glob.glob(datapath + "splines/splines/*/spline*")
if len(spline_files):
    print(f"WARNING: No spline files found in {datapath}/splines")


def create_ar1_noise(n, noise_std, ar1_corrcoef):
    """
    Generate autoregressive AR(1) Gaussian noise.

    Parameters
    ----------
    n : int
        Number of samples to generate.
    noise_std : float
        Standard deviation of the underlying white noise.
    ar1_corrcoef : float
        Autoregressive correlation coefficient (phi). Must satisfy
        ``-1 < ar1_corrcoef < 1`` for stationarity.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n,)`` containing AR(1) noise.

    Notes
    -----
    The noise is generated using the recurrence

    .. math::
        x_t = \\phi x_{t-1} + \\sqrt{1 - \\phi^2} \\, \epsilon_t,

    where :math:`\epsilon_t` is i.i.d. Gaussian noise with standard deviation
    ``noise_std``.

    Examples
    --------
    >>> x = create_ar1_noise(5, noise_std=1.0, ar1_corrcoef=0.8)
    >>> x.shape
    (5,)
    """
    
    noise = np.random.normal(0, noise_std, n)

    for i in range(1, n):
        noise[i] = ar1_corrcoef * noise[i - 1] + (1 - ar1_corrcoef ** 2) ** 0.5 * noise[i]

    return noise


def cholesky_noise(data, noise_std, T):
    """
    Generate correlated noise using Cholesky decomposition of the empirical
    autocorrelation matrix estimated from input data.

    Parameters
    ----------
    data : array_like
        One-dimensional input time series used to estimate the
        autocorrelation function (ACF).
    noise_std : float
        Standard deviation of the white noise used before correlation is applied.
    T : int
        Length of the output correlated noise sequence. If ``T`` is larger than
        the number of lags in the estimated ACF, the autocorrelation matrix is
        zero-padded to match ``T``.

    Returns
    -------
    numpy.ndarray
        A ``(T, 1)`` array containing Gaussian noise with autocorrelation
        structure matching that estimated from ``data``.

    Examples
    --------
    >>> y = np.random.randn(100)
    >>> correlated = cholesky_noise(y, noise_std=1.0, T=50)
    >>> correlated.shape
    (50, 1)
    """
    acf_result = sm.tsa.acf(data, fft=False, nlags=43)
    np.append(acf_result,[0])
    
    x = acf_result.copy()
    Correlation = np.zeros((len(x),len(x)))
    Correlation[0,:] = x
    for i in range(1,len(x)):
        Correlation[i,:] = np.append(np.flip(x[1:i+1]),x[:-i])

    if T > len(x):
        extended_matrix = np.zeros((T, T))
        extended_matrix[:Correlation.shape[0], :Correlation.shape[1]] = Correlation.copy()
        Correlation = extended_matrix.copy()
    
    # Perform Cholesky decomposition
    lower_triangle = np.linalg.cholesky(Correlation)

    noise = np.random.normal(0, noise_std, size=(T,1))
    
    # Transform noise using the lower triangle matrix
    # to achieve the desired autocorrelation
    correlated_noise = lower_triangle @ noise
    
    return correlated_noise

def find_spline_correction(sigma, phi, datapath, T=43):
    """
    Look up a bias correction from precomputed spline fits saved on disk.

    The function searches for spline pickles at
    ``{datapath}/splines/sigma/spline_sigma_{Sigma:.2f}_T_{T}.pkl`` where
    ``Sigma`` values are parsed from file names. For each available ``Sigma``
    >= ``sigma`` the routine loads the corresponding spline, evaluates it at
    ``phi`` to obtain a bias, and computes ``sigma_hat = Sigma - bias``. The
    stored (sigma_hat, bias) pairs are then used to find the bias corresponding
    to the sigma value closest to the requested ``sigma``. If the requested
    ``sigma`` is larger than all available spline fits, the function prints a
    warning and returns the bias from the largest available Sigma.

    Parameters
    ----------
    sigma : float
        Target sigma value for which a bias correction is requested.
    phi : float or array_like
        The independent variable passed to the saved spline callable. Typically
        a scalar; if an array-like is provided the function will attempt to
        handle it but returned dtype depends on the loaded spline's behavior.
    datapath : str
        Base path containing the ``splines/sigma/`` directory with files
        named like ``spline_sigma_<Sigma>_T_<T>.pkl``.
    T : int, optional
        Lenght of the time series do be bias corrected. The ``T`` value that appears in the spline file names. 
        Default is 43.

    Returns
    -------
    float or numpy.ndarray
        The bias correction evaluated at ``phi`` corresponding to the closest
        available ``sigma_hat`` to the requested ``sigma``. .

    Notes
    -----
    - The function expects spline files with names containing the Sigma value
      formatted to two decimal places (``{Sigma:.2f}``). 

    """

    # Fetch and sort unique sigma values from file names
    files = glob.glob(f"{datapath}splines/sigma/spline_sigma_*_T_{T}.pkl")
    Sigmas = np.array(sorted({float(x.split("_")[-3]) for x in files}))
    Sigmas_filtered = Sigmas[Sigmas >= sigma]

    if len(Sigmas_filtered)==0:
        print("!!! sigma value higher than available spline fits", sigma)
        print("    Taking the highest available value", Sigmas[-1])
        file = glob.glob(f"{datapath}splines/sigma/spline_sigma_{Sigmas[-1]}_T_{T}.pkl")[0]
        with open(file, 'rb') as f:
            loaded_spline = pickle.load(f)
            bias = loaded_spline(phi)
            return bias

    else:
        # Initialize dictionaries
        Bias = {}

        # Compute sigma_hat and bias
        for Sigma in Sigmas_filtered:
            file = glob.glob(f"{datapath}splines/sigma/spline_sigma_{Sigma:.2f}_T_{T}.pkl")[0]
            with open(file, 'rb') as f:
                loaded_spline = pickle.load(f)
                sigma_hat = Sigma - loaded_spline(phi)

                try:
                    Bias[sigma_hat.item()] = loaded_spline(phi.item())
                except:
                    Bias[sigma_hat.item()] = loaded_spline(phi)

        # Convert Sigma_hats keys and values to NumPy arrays for indexing
        Sigma_hats = np.array(list(Bias.keys()))
        Biases     = np.array(list(Bias.values()))

        # Find closest sigma_hat to specified sigma
        closest_sigma = np.abs(Sigma_hats - sigma).argmin()
        bias = Biases[closest_sigma]
        return bias
    
def find_spline_correction_phi(phi, T, datapath):
    """
    Load and evaluate a spline-based bias correction for a given ``phi`` value.

    The function searches for a single spline file with the pattern
    ``{datapath}/splines/phi/spline_phi_T_{T}.pkl``.  
    If exactly one file is found, it is unpickled and evaluated at ``phi``.
    If zero or multiple files are present, a message is printed and the
    function returns ``None``.

    Parameters
    ----------
    phi : float or array_like
        Input value(s) at which the stored spline is evaluated.
    T : int
        Lenght of the time series do be bias corrected. The ``T`` index used in the filename pattern. 
        Must match the value embedded in the saved spline file.
    datapath : str
        Base directory containing ``splines/phi/`` and the spline pickle file.

    Returns
    -------
    float or numpy.ndarray or None
        The bias correction evaluated at ``phi`` if exactly one spline file is
        found. Returns ``None`` if zero or more than one matching file exists.

    """
    files = glob.glob(f"{datapath}splines/phi/spline_phi_T_{T}.pkl")
    if len(files) > 1:
        print("more than one spline file")
        return
    elif len(files) == 0:
        print("no spline file")
        return
    else:
        with open(files[0], 'rb') as f:
            loaded_spline = pickle.load(f)
        bias = loaded_spline(phi)
        
    return bias

# Emulator
def emulator(y_pred, runs, noise_type, 
             T_analysis=None, residuals=None, amplitude=None, ar1_corrcoef=None, y_pred_long=None, 
             sigma_correction=True, sigma_increase_correction = None, phi_correction=False):
    
    """
    Generate an ensemble of noisy emulator time series based on features of the residuals of a time series 
    or using custom set values.

    This function produces `runs` noisy realizations of a smooth timeseries from e.g. a linear regression. 
    Several noise-generation options are supported:
    - `"white"` : i.i.d. Gaussian noise with standard deviation `amplitude`.
    - `"ar1"`   : AR(1) noise generated by `create_ar1_noise`.
    - `"cholesky"` : correlated noise produced by `cholesky_noise` using the
      autocorrelation structure estimated from `residuals`.

    The function can optionally apply empirical bias corrections for the
    estimated AR(1) coefficient (`phi_correction`) and for the noise standard
    deviation (`sigma_correction`) by calling `find_spline_correction_phi` and
    `find_spline_correction` respectively.

    Parameters
    ----------
    y_pred : array_like
        Base predicted time series (1-D) e.g. used when `y_pred_long` is not provided.
        Must have length equal to `T_analysis` if `y_pred_long` is ``None``.
    runs : int
        Number of emulator realizations to generate.
    noise_type : {"white", "ar1", "cholesky"}
        Type of noise to add to the predictions.
    T_analysis : int, optional
        Length of time series to be generated.
        If ``None``, the length is inferred from ``residuals``. Default is ``None``.
    residuals : array_like, optional
        Residual time series from e.g. linear regression used to estimate AR(1) parameters and (for
        `"cholesky"`) the empirical autocorrelation matrix. Required if
        `T_analysis` is ``None`` or if `noise_type == "cholesky"`.
    amplitude : float, optional
        Base standard deviation for the generated noise. If ``None``, it is
        computed as `np.std(residuals)`. Default is ``None``.
    ar1_corrcoef : float, optional
        Initial AR(1) correlation coefficient (phi). If ``None``, it will be
        estimated from `residuals` via `yule_walker(..., order=1)`. Values are
        constrained to be <= 1.
    y_pred_long : array_like, optional
        Alternative long predicted time series. If provided, noise is added to
        this array instead of `y_pred`. Its length defines the emulator output
        length. Default is ``None``.
    sigma_correction : bool, optional
        Whether to apply an empirical correction to `amplitude` using
        `find_spline_correction`. Default is ``True``.
    sigma_increase_correction : float or None, optional
        If provided, after applying the spline `amplitude` correction the
        amplitude is increased by `sigma_increase_correction * amplitude`. 
        This allow to mimic the increase of internal varibility approacing near ice-free conditions.
        Default is ``None`` (no additional increase).
    phi_correction : bool, optional
        Whether to apply an empirical correction to `ar1_corrcoef` using
        `find_spline_correction_phi`. Default is ``False``.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(runs, T_emulator)`` where ``T_emulator`` is
        ``len(y_pred_long)`` if ``y_pred_long`` is provided, otherwise it is
        ``T_analysis``. Each row contains one noisy realization of the
        predicted time series (``y_pred`` or ``y_pred_long`` plus generated noise).

    Raises
    ------
    ValueError
        If both ``T_analysis`` and ``residuals`` are ``None`` (can't infer T),
        or if ``noise_type == "cholesky"`` and ``residuals`` are ``None``.

    Notes
    -----
    - When ``ar1_corrcoef`` is ``None``, the code uses `yule_walker` to obtain
      an estimate and then applies a small-sample correction:
      ``ar1_corrcoef = rho[0] + ((1+4*rho)/T_analysis)``. The coefficient is
      constrained to be at most 1.

    """

    
    # Determine Emulator time series length
    if T_analysis is None:
        if residuals is None:
            raise ValueError("T_analysis and residuals are None [emulator_func.emulator]")
        T_analysis = len(residuals)

    if y_pred_long is None:
        T_emulator = T_analysis
    else:
        T_emulator = len(y_pred_long)

    # Declare Emulator time sereis array
    Emulator_sias = np.zeros((runs, T_emulator))

    # First estimate of standard deviation for the noise
    if amplitude is None:
        amplitude = np.std(residuals)

    # First estimate of AR(1) correlation coefficient for the noise
    if ar1_corrcoef is None:#noise_type=="ar1" and 
        rho, sigma = yule_walker(residuals, order=1)
        ar1_corrcoef = rho[0] + ((1+4*rho)/T_analysis)
        if ar1_corrcoef > 1:
            print("constraining phi to 1")
            ar1_corrcoef = 1

    # Apply empirical corrections
    if phi_correction:
        bias_correction = find_spline_correction_phi(ar1_corrcoef, T_analysis, datapath)
        ar1_corrcoef += bias_correction
        if bias_correction < 0:
            print("variability bias correction <0")
        if ar1_corrcoef > 1:
            print("constraining phi to 1")
            ar1_corrcoef = 1

    if sigma_correction: 
        bias_correction = find_spline_correction(amplitude, ar1_corrcoef, datapath, T=T_analysis)
        if bias_correction < 0:
            print("variability bias correction <0")
        amplitude += bias_correction
        if sigma_increase_correction is not None:
            amplitude += sigma_increase_correction * amplitude  

    # Create noisy time series
    for run in range(runs):
        if noise_type=="white":
            noise = np.random.normal(0, amplitude, T_emulator)
        elif noise_type=="ar1":
            noise = create_ar1_noise(T_emulator, amplitude, ar1_corrcoef)
        elif noise_type=="cholesky":
            if residuals is None:
                raise ValueError("residuals can't be None for cholesky noise [emulator_func.emulator]")
            else:
                noise = cholesky_noise(residuals, amplitude, T_emulator).squeeze()
        else:
            print("!!! Wrong noise name")
            return
        
        # Add noise back to the trend
        if y_pred_long is None:
            emulator_sia = y_pred + noise
        else:
            emulator_sia = y_pred_long + noise
        Emulator_sias[run,:] = emulator_sia

    return Emulator_sias

###

def grab_parameters(member, df_forcing, df_sia, observation_start, observation_end, 
                    forcing="ssp245", sigma_correction=True, phi_correction=False, co2_units=True):
    """
    Extract model/observation parameters (sensitivity, noise amplitude / standard deviation, AR(1) correlation coefficient)
    for a single model/member over a specified observation window.

    This routine slices forcing and SIA time series
    to the requested observation interval, computes metadata (via
    ``meta.get_meta_data``) to obtain the predicted SIA and observational
    sensitivity, and then computes residuals and default noise parameters.
    Optional empirical spline-based corrections for the standard deviation
    (``sigma_correction``) and AR(1) coefficient (``phi_correction``) are
    applied by calling helper functions which read precomputed spline files
    from disk (these helpers require a ``datapath`` variable to be available
    in the calling scope).

    Parameters
    ----------
    member : str
        Column name (member identifier) in ``df_sia`` identifying the SIA series
        to use.
    df_forcing : pandas.DataFrame
        Time-indexed forcing DataFrame (e.g., CO₂ / scenario). Must contain a
        column matching the ``forcing`` argument.
    df_sia : pandas.DataFrame
        Time-indexed SIA DataFrame containing a column ``member``.
    observation_start : pandas-compatible index
        Start of the observation window used to slice both ``df_forcing`` and
        ``df_sia`` (inclusive).
    observation_end : pandas-compatible index
        End of the observation window used to slice both ``df_forcing`` and
        ``df_sia`` (inclusive).
    forcing : str, optional
        Column name in ``df_forcing`` to use (default ``"ssp245"``).
    sigma_correction : bool, optional
        If ``True`` (default), apply an empirical sigma correction via
        ``find_spline_correction`` (this function reads spline files from disk).
    phi_correction : bool, optional
        If ``True``, apply an empirical phi correction via
        ``find_spline_correction_phi`` (this function reads spline files from disk).
    co2_units : bool, optional
        Passed through to ``meta.get_meta_data`` to control CO₂ units handling.
        Default is ``True``.

    Returns
    -------
    tuple
        obs_sens : float
            Observational sensitivity returned by ``meta.get_meta_data``.
        amplitude : float
            Final noise standard deviation (after optional corrections).
        phi : float
            AR(1) coefficient (after optional corrections).
    """
    forcing = df_forcing.loc[observation_start:observation_end][forcing]
    sia     = df_sia.loc[observation_start:observation_end][member]
    obs_sens, y_pred, lin_timing, intercept = meta.get_meta_data(sia, df_forcing, co2_units=co2_units)
    residuals = sia - y_pred

    
    # First estimate of standard deviationa and AR(1) coefficient
    amplitude = np.std(residuals)

    rho, sigma = yule_walker(residuals, order=1)
    ar1_corrcoef = rho[0] + ((1+4*rho)/len(residuals))
    if ar1_corrcoef > 1:
        print("constraining phi to 1")
        ar1_corrcoef = 1

    # Add emipirical corrections
    if phi_correction:
        bias_correction = find_spline_correction_phi(ar1_corrcoef, len(residuals), datapath)
        ar1_corrcoef += bias_correction
        if bias_correction < 0:
            print("variability bias correction <0")
        if ar1_corrcoef > 1:
            print("constraining phi to 1")
            ar1_corrcoef = 1

    if sigma_correction: 
        bias_correction = find_spline_correction(amplitude, ar1_corrcoef, datapath, T=len(residuals))
        if bias_correction < 0:
            print("variability bias correction <0")
        amplitude += bias_correction

    return obs_sens, amplitude, ar1_corrcoef[0]   


#Experiment
def experiment(df_forcing, df_sia, runs, noise_type, observation_start, observation_end, 
               sia_ts=None, true_slope=None, intercept=None, amplitude=None, ar1_corrcoef=None, co2_name="rcp85", 
               sigma_correction=True, phi_correction=False, co2_units=True):
    
    """
    Prepare time series for the emulator, run the emulator, and re-estimate
    sensitivities from the emulated ensemble.

    This routine either (a) uses an observed SIA time series from `df_sia`
    (column `sia_ts`) and derives the predicted SIA, residuals and
    observational sensitivity from `meta.get_meta_data`, or (b) constructs a
    deterministic predicted SIA from the supplied `true_slope` and `intercept`.
    It then calls `emulator(...)` to generate `runs` noisy realizations and
    finally re-estimates the sensitivity for each realization using
    `meta.get_meta_data`.

    Parameters
    ----------
    df_forcing : pandas.DataFrame
        Time-indexed forcing DataFrame. Must contain a column named `co2_name`.
    df_sia : pandas.DataFrame
        Time-indexed SIA DataFrame. Required if `sia_ts` is provided.
    runs : int
        Number of emulator realizations to generate.
    noise_type : {"white", "ar1", "cholesky"}
        Type of noise passed to `emulator`.
    observation_start, observation_end : index-compatible (e.g. int)
        Start and end of the observation window used to slice `df_forcing` and
        `df_sia`. These are passed directly to pandas `.loc[...]`.
    sia_ts : str or None, optional
        Column name in `df_sia` to use as the observed SIA series. If `None`,
        the function expects `true_slope`, `intercept`, `amplitude`, and
        `ar1_corrcoef` to be provided and will construct a deterministic
        `y_pred` instead of deriving it from observations.
    true_slope : float or None, optional
        If provided (and `sia_ts` is None), the true slope (sensitivity) used
        to create `y_pred = forcing * true_slope + intercept`. If `co2_units`
        is True, `true_slope` is scaled by 1e-3 before use.
    intercept : float or None, optional
        Intercept used when constructing `y_pred` from `true_slope`.
    amplitude : float or None
        Passed to `emulator`. If ``None`` the emulator (or `grab` logic) will
        infer amplitude from residuals (when available).
    ar1_corrcoef : float or None
        Initial AR(1) coefficient passed to `emulator`. If ``None`` it will be
        estimated from residuals inside the emulator.
    co2_name : str, optional
        Column name in `df_forcing` to use (default ``"rcp85"``).
    sigma_correction : bool, optional
        Passed to `emulator` (default True) — whether to apply spline-based
        sigma corrections when generating noise.
    phi_correction : bool, optional
        Passed to `emulator` (default False) — whether to apply spline-based
        phi corrections when generating AR(1) noise.
    co2_units : bool, optional
        If True (default), `true_slope` is scaled by 1e-3 before constructing
        `y_pred` and the `meta.get_meta_data(..., co2_units=co2_units)` call is
        made with the same flag.

    Returns
    -------
    tuple
        Sensitivities: list of length `runs`
            Re-estimated sensitivities (one per emulator realization) returned
            by `meta.get_meta_data` when applied to each emulated series.
        obs_sens : float
            Observational sensitivity. If `sia_ts` was provided this is the
            value returned by `meta.get_meta_data` on the observed SIA;
            otherwise it is `true_slope`.
        emulator_sia : numpy.ndarray
            Array of shape ``(runs, T_emulator)`` containing the emulated SIA
            realizations (the output from `emulator`).

    Raises
    ------
    ValueError
        If `sia_ts` is None and one or more of `true_slope`, `intercept`,
        `amplitude`, or `ar1_corrcoef` is None (the function cannot construct
        the deterministic `y_pred` without those).


    Examples
    --------
    >>> # Using an observed SIA series from df_sia
    >>> sens_list, obs_sens, sims = experiment(
    ...     df_forcing=forcing_df, df_sia=sia_df,
    ...     runs=10, noise_type="ar1",
    ...     observation_start=1979, observation_end=2024,
    ...     sia_ts="osisaf", co2_name="co2_cum")
    >>> len(sens_list) == 10
    >>> sims.shape
    (10, len(forcing_df.loc[1979:2024]))
    """

    forcing = df_forcing.loc[observation_start:observation_end][co2_name]

    if sia_ts is None:
        # Check if true_slope, intercept, amplitude , ar1_corrcoef is not None
        if None in [true_slope, intercept, amplitude , ar1_corrcoef]:
            raise ValueError("One of sia_ts, true_slope, intercept, amplitude , ar1_corrcoef is None")
        else:
            residuals = None
            T_analysis = observation_end - observation_start +1
            obs_sens = true_slope
    else:
        sia = df_sia.loc[observation_start:observation_end][sia_ts]
        obs_sens, y_pred, lin_timing, intercept = meta.get_meta_data(sia, forcing, co2_units=co2_units)
        residuals = sia - y_pred
        T_analysis = len(residuals)

    if true_slope is not None:
        if co2_units:
            true_slope = true_slope *1e-3
        y_pred = forcing * true_slope + intercept # was set 6.11 before #deleted 

    emulator_sia = emulator(y_pred, runs, noise_type, T_analysis=T_analysis, residuals=residuals, amplitude=amplitude, ar1_corrcoef=ar1_corrcoef, sigma_correction=sigma_correction, phi_correction=phi_correction)

    Sensitivities = []
    for i in range(runs):
        sens, y_pred, lin_timing, intercept = meta.get_meta_data(emulator_sia[i,:], forcing, co2_units=co2_units)
        Sensitivities.append(sens)

    return Sensitivities, obs_sens, emulator_sia

def experiment_indepth_long(df_sia, df_forcing, runs, noise_type, observation_start, observation_end, sia_ts, true_slope=None, intercept=None, amplitude=None, ar1_corrcoef=None, co2_name="rcp85", sigma_correction=True, prediction_end=None, df_co2_long = None, sigma_increase_correction=None, phi_correction=False, co2_units=True):
    
    forcing = df_forcing.loc[observation_start:observation_end][co2_name]

    if sia_ts is None:
        # Check if true_slope, intercept, amplitude , ar1_corrcoef is not None
        if None in [true_slope, intercept, amplitude , ar1_corrcoef]:
            raise ValueError("One of sia_ts, true_slope, intercept, amplitude, ar1_corrcoef is None")
        else:
            residuals = None
            T_analysis = observation_end - observation_start +1
            obs_sens = true_slope
    else:
        sia = df_sia.loc[observation_start:observation_end][sia_ts]
        obs_sens, y_pred, lin_timing, intercept = meta.get_meta_data(sia, forcing, co2_units=False)
        residuals = sia - y_pred
        T_analysis = len(residuals)

    if prediction_end is not None:

        if df_co2_long is not None:
            co2_scenario = np.array(df_co2_long.loc[observation_end:prediction_end])
            diff = df_forcing.loc[observation_end][co2_name] - df_co2_long.loc[observation_end]
            co2_long = np.append(forcing, co2_scenario + diff)
        else:
            co2_long = np.array(df_forcing[co2_name].loc[observation_start:prediction_end])

        y_pred_long = co2_long * obs_sens + intercept #np.concatenate((y_pred, arr2))
    else:
        y_pred_long = None

    if true_slope is not None:
        if co2_units:
            true_slope = true_slope *1e-3
        y_pred = forcing * true_slope + intercept
        if prediction_end is not None:
            y_pred_long = co2_long * true_slope + intercept


    # emulate sia    
    emulator_sia = emulator(y_pred, runs, noise_type,  T_analysis=T_analysis, residuals=residuals, 
                            amplitude=amplitude, ar1_corrcoef=ar1_corrcoef, y_pred_long=y_pred_long, 
                            sigma_correction=sigma_correction, sigma_increase_correction = sigma_increase_correction, phi_correction=phi_correction)

    # read out emulated sia
    Sensitivities = []
    Ar1_corrcoefs = []
    Sigmas = []
    Lin_timing = []
    EM_timing = []
    EM_timing_year = []

    for i in range(runs):
        sens, y_pred, lin_timing, intercept = meta.get_meta_data(emulator_sia[i,:len(residuals)], forcing, co2_units=co2_units)
        Sensitivities.append(sens)
        residuals = emulator_sia[i,:len(residuals)] - y_pred
        Sigmas.append(np.std(residuals))
        Lin_timing.append(lin_timing)

        ice_free_index = meta.find_icefree_idx(emulator_sia[i,:], amount_icefree=1, threshold_value=1, printing=False)
        if ice_free_index != False:
            EM_timing.append(co2_long[ice_free_index])
            EM_timing_year.append(observation_start + ice_free_index)
        else:
            EM_timing.append(co2_long[-1])
            EM_timing_year.append(prediction_end)

        if noise_type=="ar1":
            rho, sigma = yule_walker(residuals, order=1)
            ar1_corrcoef = rho[0] + ((1+4*rho)/len(residuals))
            Ar1_corrcoefs.append(ar1_corrcoef.item())

    if co2_units:
        obs_sens = obs_sens * 1e3
    return Sensitivities, obs_sens, emulator_sia, Ar1_corrcoefs, Sigmas, Lin_timing, EM_timing_year, EM_timing


def create_emulator_members(num_members, noise, df_co2, true_slope, intercept, amplitude, ar1_corrcoef, 
                            start_year=1979, end_year=2024, dataframe=False, 
                            sigma_correction=False, co2_name="rcp45", phi_correction=False):
    
    """
    Create an ensemble of emulated SIA members based on a deterministic prediction.

    This function generates `num_members` emulator realizations by constructing
    a deterministic `y_pred` from the supplied `true_slope` and `intercept`
    and then adding noise via the `experiment` and `emulator` functions.
    All noise and AR(1) parameters must be explicitly provided because no
    observed SIA time series is used.

    Parameters
    ----------
    num_members : int
        Number of emulator members (ensemble size) to generate.
    noise : {"white", "ar1", "cholesky"}
        Noise type passed to `experiment` / `emulator`.
    df_co2 : pandas.DataFrame
        Time-indexed forcing DataFrame. Must contain a column named `co2_name`.
    true_slope : float
        Slope used to construct deterministic `y_pred = forcing * true_slope + intercept`.
    intercept : float
        Intercept used when constructing `y_pred`.
    amplitude : float
        Standard deviation of the noise added by the emulator.
    ar1_corrcoef : float
        AR(1) correlation coefficient (phi) used by the emulator.
    start_year : int, optional
        First year (inclusive) of the simulation window. Default is 1979.
    end_year : int, optional
        Last year (inclusive) of the simulation window. Default is 2024.
    dataframe : bool, optional
        If True, return a pandas DataFrame of members along with sensitivities.
        If False (default), return only the sensitivities list.
    sigma_correction : bool, optional
        Passed to `experiment` to toggle spline-based sigma correction.
    co2_name : str, optional
        Column name in `df_co2` to use for forcing. Default is "rcp45".
    phi_correction : bool, optional
        Passed to `experiment` to toggle spline-based phi correction.

    Returns
    -------
    If `dataframe` is False:
        list
            Sensitivities re-estimated from each emulator realization (length `num_members`).
    If `dataframe` is True:
        tuple
            (df_dummy, Sensitivities)
            - df_dummy : pandas.DataFrame
                Columns are member names (mb0000, mb0001, ...) and index is
                `np.arange(start_year, end_year+1)`. Each column contains one
                emulated member realization.
            - Sensitivities : list
                Re-estimated sensitivities for each member.
    

    Examples
    --------
    >>> # Generate DataFrame of members
    >>> df_members, sens = create_emulator_members(10, "ar1", df_co2,
    ...                                            true_slope=0.015, intercept=6.11, amplitude=0.4, ar1_corrcoef=0.2
    ...                                            start_year=1980, end_year=2020,
    ...                                            dataframe=True)
    >>> df_members.shape
    (41, 10)
    """

    Sensitivities, background_dummy_sens, emulator_sia = experiment(df_co2, None, num_members, noise, start_year, end_year, true_slope=true_slope, intercept=intercept, amplitude=amplitude, ar1_corrcoef=ar1_corrcoef, sigma_correction=sigma_correction, co2_name=co2_name, phi_correction=phi_correction)

    if dataframe:
        members_dummy = mbhl.create_member_list("mb", zfill=4, N=num_members)

        # Build a dict of new member columns
        member_data = {member: emulator_sia[i, :].copy() for i, member in enumerate(members_dummy)}

        # Combine the original df_dummy with all new member columns at once
        df_dummy = pd.DataFrame(member_data, index=np.arange(start_year,end_year+1))

        return df_dummy, Sensitivities
    else:
        return Sensitivities
    