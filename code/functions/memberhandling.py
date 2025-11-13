import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pylab as pl

import glob 


import cftime

def cftime_to_pd_datetime(cftime_obj):
  """Converts a cftime.DatetimeNoLeap object to a Pandas datetime."""
  return pd.to_datetime(
      f"{cftime_obj.year}-{cftime_obj.month:02d}-{cftime_obj.day:02d} "
      f"{cftime_obj.hour:02d}:{cftime_obj.minute:02d}:{cftime_obj.second:02d}"
  )


def load_model_data(inpath : str, scenario : str ="rcp85", rolling_window : int=30):
    """
    Load model data from NetCDF files into a pandas dataframe, combine ensemble members, and compute statistics.

    This function searches for NetCDF files matching a given scenario in a directory,
    extracts relevant variables (sia_nh, sic, or gmst), concatenates ensemble members,
    and computes ensemble mean, median, and rolling mean over a specified window.

    Parameters
    ----------
    inpath : str
        Path to the directory containing NetCDF files. The function will search for
        files containing the scenario string.
    scenario : str, optional
        Scenario identifier to filter files (default is "rcp85").
    rolling_window : int, optional
        Window size (in years) for calculating the rolling mean of the ensemble mean
        (default is 30).

    Returns
    -------
    pandas.DataFrame or None
        DataFrame indexed by year containing:
        - Individual ensemble member columns
        - "mean": Ensemble mean across members
        - "median": Ensemble median across members
        - "rolling": Rolling mean of the ensemble mean over `rolling_window` years
        
        Returns None if no matching files are found.

    Notes
    -----
    - The function assumes files are named in a way that the ensemble member identifier
      can be extracted from the second-to-last underscore-separated segment.
    """
    
    df_list = []

    # Search for files
    inpath = f"{inpath}*{scenario}"
    files = glob.glob(inpath + "*.nc")   
    if len(files) == 0:
        print("No files found")
        return None                    
    
    # Loop trough files and load data
    files = sorted(files, key = lambda x: x.split("_")[-2])  # sort files by member    
    for file in files[:]:
        ds = xr.open_dataset(file)
        mb = file.split("_")[-2]
        try:
            df = ds.sia_nh.to_dataframe()
            df = df.rename(columns={"sia_nh": mb})
        except:
            try:
                df = ds.sic.to_dataframe()
                df = df.rename(columns={"sic": mb})
            except:
                df = ds.gmst.to_dataframe()
                df = df.rename(columns={"gmst": mb})
        df_list.append(df)
        ds.close()

    # Combine the data from the different files
    df1 = pd.concat(df_list, axis=1)  
    df = df1.copy()

    df["time"] = ds.time.values

    try:
        df['time'] = pd.to_datetime(df['time'])
    except:
        # Assuming 'time_data' is your list or array of cftime.DatetimeNoLeap objects
        df['time'] = [cftime_to_pd_datetime(t) for t in ds.time.values]

    # Extract the year as an integer
    df['year'] = df['time'].dt.year
    # Set 'year' column as index
    df.set_index('year', inplace=True)

    # Drop 'datetime_column'
    df.drop('time', axis=1, inplace=True)
    # Calculate ensemble mean and rolling ensemble temporal mean
    df["mean"] = df.mean(axis=1)
    df["median"] = df.median(axis=1)
    df["rolling"] = df["mean"].rolling(window=rolling_window, center=True).mean()

    return df

def create_member_list(prefix, suffix="", zfill=3, N=100):
    members = list(range(1,N+1))
    for cnt,mb in enumerate(members[:]): 
        mb = prefix + str(mb).zfill(zfill) + suffix
        members[cnt] = mb

    return members

def colors_for_members(members):
    colors = pl.cm.nipy_spectral(np.linspace(0,1,len(members)))
    color_mbs = dict.fromkeys(members)
    for cnt,mb in enumerate(members):
        color_mbs[mb] = colors[cnt]

    return color_mbs

