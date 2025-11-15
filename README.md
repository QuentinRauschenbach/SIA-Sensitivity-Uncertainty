# What is the Observed Sensitivity of Arctic Sea Ice?

This repository contains the code used in the paper:

**Rauschenbach, Q., Wernecke, A., and Notz, D. (2025)**  
*What is the Observed Sensitivity of Arctic Sea Ice?*
(_in review, GRL_)  

## Overview

This project implements a **linear AR(1) emulator** to analyze the response of Arctic summer Sea Ice Area (SIA) to external forcing (anthropogenic CO₂ emissions and global mean surface temperature (GMST)). 
The emulator runs directly from observational records and reproduces the internal variability, memory, and background forcing of the Arctic SIA time series.

## Repository Contents

- `code/` : Python scripts for emulator construction, experiments, and analyses
  - `.ipynb` for data processing & paper figure plotting. Files with the ending `_Figure.ipynb` indicate scripts used for plotting.
  - `functions/` : `.py` scripts containing the emulator and helper functions 
  - _doc strings were created using ChatGPT and checked by hand_
- `data/SIA/` : Subset of UHH-SIA (see below), september only
- `data/CO2/` : Processed data are not shared for licencing reasons (for raw data check the links provided below)
- `data/GMST/` : Processed data are not shared for licencing reasons (for raw data check the links provided below)

## Used Data

### Sea Ice Area (SIA)

- **Original dataset:** Rauschenbach, Q., Dörr, J., Notz, D., Kern, S., 2024, *UHH sea-ice area product, 1850-2023*, University of Hamburg, v2024_fv0.01, https://doi.org/10.25592/uhhfdm.11346
- **Extended version (this repository / Zenodo):** includes data up to 2024, contributed by Sarah Thomae.  
- **License:** CC-BY 4.0  

### Temperature datasets

Processed from multiple sources and rebased to 1951–1980:

| Dataset        | Source / DOI / Link | Notes |
|----------------|------------------|-------|
| GISSTEMPv4     | [NASA GISS](https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv) | |
| BerkeleyEarth  | [ESSD](https://doi.org/10.5194/essd-12-3469-2020) |  |
| NOAA GlobalTemp | [NOAA](https://www.ncei.noaa.gov/products/land-based-station/noaa-global-temp) | Rebasing applied |
| HadCRUT5       | [MetOffice](https://www.metoffice.gov.uk/hadobs/hadcrut5/data/HadCRUT.5.0.2.0/download.html) | Rebasing applied |
| Kadow et al. 2025 | [Zenodo DOI](https://doi.org/10.5281/zenodo.15622091) | AI-infilled HadCRUT4, rebasing applied |

### Anthropogenic CO₂

- **Source:** [Global Carbon Budget (2024)](https://globalcarbonbudget.org/archive/), processed into tonnes CO₂, summing fossil fuel and land-use change contributions.  

### Model Data (CMIP6)

- **Model:** MPI-ESM2-1-LR / MPI-GE CMIP6 
- **Data hosted:** DKRZ Levante system  
- **Accessible via:** ESGF search (Earth System Grid Federation)

## Code Usage

### Emulator

The core Python functions allow you to:

- Construct AR(1) emulators of SIA time series.
- Estimate sea-ice sensitivities for different forcing scenarios.
- Reproduce figures and analyses from the paper.
