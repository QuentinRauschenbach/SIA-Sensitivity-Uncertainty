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
    
    noise = np.random.normal(0, noise_std, n)

    for i in range(1, n):
        noise[i] = ar1_corrcoef * noise[i - 1] + (1 - ar1_corrcoef ** 2) ** 0.5 * noise[i]

    return noise


def cholesky_noise(data, noise_std, T):
    acf_result = sm.tsa.acf(data, fft=False, nlags=43)
    np.append(acf_result,[0])
    
    x = acf_result.copy()
    Correlation = np.zeros((len(x),len(x)))
    Correlation[0,:] = x
    for i in range(1,len(x)):
        Correlation[i,:] = np.append(np.flip(x[1:i+1]),x[:-i])

    if T > len(x):
        #print("huhu")
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
            #print(Sigma, T)
            #print(glob.glob(f"{datapath}splines/spline_sigma_*_T_*.pkl"))
            file = glob.glob(f"{datapath}splines/sigma/spline_sigma_{Sigma:.2f}_T_{T}.pkl")[0]
            with open(file, 'rb') as f:
                loaded_spline = pickle.load(f)
                sigma_hat = Sigma - loaded_spline(phi)
                #print(Bias, sigma_hat, loaded_spline(phi), phi)
                #print(type(Bias), type(sigma_hat),type(loaded_spline), type(loaded_spline(phi)), type(phi), )
                #if not type(sigma_hat) is float:
                #    sigma_hat.item()
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
def emulator(y_pred, runs, noise_type, T_analysis=None, residuals=None, amplitude=None, ar1_corrcoef=None, y_pred_long=None, sigma_correction=True, sigma_increase_correction = None, phi_correction=False):
    # Calc Noise
    if amplitude is None:
        amplitude = np.std(residuals)

    if T_analysis is None:
        if residuals is None:
            raise ValueError("T_analysis and residuals are None [emulator_func.emulator]")
        T_analysis = len(residuals)

    if y_pred_long is None:
        T_emulator = T_analysis
    else:
        T_emulator = len(y_pred_long)

    Emulator_sias = np.zeros((runs, T_emulator))

    if ar1_corrcoef is None:#noise_type=="ar1" and 
        rho, sigma = yule_walker(residuals, order=1)
        ar1_corrcoef = rho[0] + ((1+4*rho)/T_analysis)
        if ar1_corrcoef > 1:
            print("constraining phi to 1")
            ar1_corrcoef = 1

    if phi_correction:
        bias_correction = find_spline_correction_phi(ar1_corrcoef, T_analysis, datapath)
        ar1_corrcoef += bias_correction
        if bias_correction < 0:
            print("variability bias correction <0")
        if ar1_corrcoef > 1:
            print("constraining phi to 1")
            ar1_corrcoef = 1

    if sigma_correction: #(noise_type=="ar1") and 
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
        if y_pred_long is None:
            emulator_sia = y_pred + noise
        else:
            emulator_sia = y_pred_long + noise
        Emulator_sias[run,:] = emulator_sia

    return Emulator_sias

###

def grab_parameters(member, df_forcing, df_sia, observation_start, observation_end, amplitude=None, ar1_corrcoef=None, forcing="rcp85", sigma_correction=True, phi_correction=False, co2_units=True):
    
    forcing = df_forcing.loc[observation_start:observation_end][forcing]
    sia     = df_sia.loc[observation_start:observation_end][member]
    obs_sens, y_pred, lin_timing, intercept = meta.get_meta_data(sia, df_forcing, co2_units=co2_units)
    residuals = sia - y_pred

    # Calc Noise
    if amplitude is None:
        amplitude = np.std(residuals)

    if ar1_corrcoef is None:
        rho, sigma = yule_walker(residuals, order=1)
        ar1_corrcoef = rho[0] + ((1+4*rho)/len(residuals))
        if ar1_corrcoef > 1:
            print("constraining phi to 1")
            ar1_corrcoef = 1

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
def experiment(df_forcing, df_sia, runs, noise_type, observation_start, observation_end, sia_ts=None, true_slope=None, intercept=None, amplitude=None, ar1_corrcoef=None, co2_name="rcp85", sigma_correction=True, phi_correction=False, co2_units=True):
    

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


def create_emulator_members(num_members, noise, df_co2, true_slope=None, intercept=None, amplitude=None, ar1_corrcoef=None, start_year=1979, end_year=2024, dataframe=False, sigma_correction=False, co2_name="rcp45", phi_correction=False):

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
    