# calculate meta data like sensitivity, ice-free timing, ...

import numpy as np
from scipy.stats import linregress
    
def find_icefree_idx(sia, amount_icefree=1, threshold_value=1, printing=True):
    """
    Find the index of the first occurrence of ice-free conditions in a sea-ice area (SIA) array.

    This function searches for the index where the sea ice area falls below a specified threshold,
    indicating ice-free conditions. It can be configured to find the nth occurrence of ice-free conditions.

    Parameters:
    -----------
    sia : array-like
        An array of sea ice area values.
    amount_icefree : int, optional
        The nth occurrence of ice-free conditions to find (default is 1, i.e., the first occurrence).
    threshold_value : float, optional
        The threshold value below which the sea ice area is considered ice-free (default is 1).

    Returns:
    --------
    int or False
        The index of the nth ice-free occurrence if found, or False if no ice-free conditions are detected.

    Raises:
    -------
    Exception
        If an error occurs during the execution of the function, it will be caught and printed.

    Notes:
    ------
    - The function assumes that ice-free conditions occur when SIA < threshold_value.
    - If the specified number of ice-free occurrences is not found, the function returns False and prints "never ice free".
    """
    
    try:
        sim_idx = np.where(sia < 1)[0][amount_icefree - threshold_value]
        return sim_idx
    except Exception as e:
        #print(e)
        if printing:
            print("never ice free")
        return False

def get_meta_data(sia:np.ndarray, forcing:np.ndarray, start:float = None, stop:float = None, co2_units:bool = True):

    """
    Perform a linear regression of sea ice area (SIA) against a forcing variable and compute metadata.

    This function computes the slope (sensitivity), predicted values, linear timing, and intercept
    of the regression between SIA and a forcing array. Optionally, a subset of the data can be used
    via `start` and `stop` indices. If `co2` is True, the slope is scaled to m2/t.

    Parameters
    ----------
    sia : np.ndarray
        Array of sea ice area (SIA) values.
    forcing : np.ndarray
        Array of forcing values corresponding to SIA (CO2 or GMST).
    start : float, optional
        Starting index for a subset of the data (default is None, meaning use all data).
    stop : float, optional
        Ending index for a subset of the data (default is None, meaning use all data).
    co2_units : bool, optional
        If True, converts slope to m2/t (default is True).

    Returns
    -------
    sens : float
        Sensitivity of SIA to forcing (slope of regression, optionally scaled to m2/t).
    y_pred : np.ndarray
        Predicted SIA values from the regression over the full forcing array.
    lin_timing : float
        The timing when SIA reaches 1 (derived from regression intercept and slope).
    intercept : float
        Intercept of the linear regression.
    """
    
    # perform linear regression
    if start is None and stop is None:
        x = forcing.copy()
        y = sia.copy()
    else:
        x = forcing[start:stop+1]
        y = sia[start:stop+1]
    result = linregress(x, y)

    # write down results
    sens       = result.slope
    if co2_units:
        sens = sens *1e3 # convert to m2/t
    lin_timing = -(result.intercept-1)/(result.slope)
    intercept  = result.intercept 
    y_pred     = result.intercept + result.slope*forcing
    
    return sens, y_pred, lin_timing, intercept

def get_meta_data_indepth(sia:np.ndarray, forcing:np.ndarray, start:float=None, stop:float=None, co2_units:bool = True):

    """
    Perform an in-depth linear regression of SIA against CO2 and compute regression metadata.

    This function computes the slope (sensitivity), predicted values, linear timing, intercept,
    and standard error of the regression. A subset of the data can be selected via `start` and `stop`.

    Parameters
    ----------
    sia : np.ndarray
        Array of sea ice area (SIA) values.
    forcing : np.ndarray
        Array of CO2 concentration or GMST values corresponding to SIA.
    start : float, optional
        Starting index for a subset of the data (default is None, meaning use all data).
    stop : float, optional
        Ending index for a subset of the data (default is None, meaning use all data).

    Returns
    -------
    sens : float
        Sensitivity of SIA to CO2 (slope of regression, scaled to m/s).
    y_pred : np.ndarray
        Predicted SIA values from the regression over the full CO2 array.
    lin_timing : float
        The timing when SIA reaches 1 (derived from regression intercept and slope).
    intercept : float
        Intercept of the linear regression.
    std_err : float
        Standard error of the slope estimate from the regression.
    correlation :float
        Pearson r of from the regression.
    """
    
    # perform linear regression
    if start is None and stop is None:
        x = forcing.copy()
        y = sia.copy()
    else:
        x = forcing[start:stop+1]
        y = sia[start:stop+1]
    result = linregress(x, y)

    # write down results
    sens        = result.slope
    if co2_units:
        sens = sens *1e3 # convert to m2/t
    lin_timing  = -(result.intercept-1)/(result.slope)
    intercept   = result.intercept 
    y_pred      = result.intercept + result.slope*forcing
    std_err     = result.stderr
    correlation = result.rvalue  # Pearson r
    
    return sens, y_pred, lin_timing, intercept, std_err, correlation

