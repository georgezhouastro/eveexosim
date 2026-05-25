# EVE Exoplanet Yield Forward Modeling

Forward modeling of planet yields for the EVE mission.

---

### ATTENTION! YOU ALSO NEED some CSVs that are too big for github. Download them from my google drive and put them in the code directory!

https://drive.google.com/drive/folders/1qOMBFDkFTMt4chmUcUW7UFFT7miOuMJA?usp=sharing

## Repository Structure

### Core Scripts
* `forwardmodel.py`: Main execution script that runs the forward modeling simulation.
* `readsnr.py`: Local module containing helper functions to read the planet recovery probability as a function of transit SNR.

### Catalogs & Simulation Data
* `Jan25_masterlist_roquette.csv`: Masterlist of all-sky young stars.
* `roquette_rms.csv`: RMS of the ORION SFR stars from TESS.
* `spoc_rms_age.csv`: RMS of other young stars from TESS (as per Vach et al.).
* `GasDwarfs_EVE_April_8.csv`: Gas dwarf input planet population.
* `EVESNR.csv`: EVE instrument flux vs. SNR curve.
* `targetregions_20260410.csv`: Latest survey target fields.
* `snrcurves/`: Directory containing SNR vs. transit recovery probability curves.
* `tic_query_fields/`: Local cache for TESS Input Catalog (TIC) for EVE target fields.

### Outputs & Environment
* `gasdwarf_sim/`: Directory containing simulation outputs.
* `environment.yml`: Conda environment file required to replicate dependencies.
* `readme.md`: This file.

---

### Step-by-Step Setup

   ```bash
   git clone [https://github.com/georgezhouastro/eveexosim.git](https://github.com/georgezhouastro/eveexosim.git)
   cd eveexosim


   conda env create -f environment.yml
   conda activate eveexosim

   python forwardmodel.py




usage: forwardmodel.py [-h] [--baseline BASELINE] [--fov FOV] [--psf PSF]
                       [--draws DRAWS] [--eve_snr_model EVE_SNR_MODEL]

Forward model simulation for EVE exoplanet yield.

optional arguments:
  -h, --help            show this help message and exit
  --baseline BASELINE   Baseline in days (default: 30 days)
  --fov FOV             Field of view (default: 5 deg)
  --psf PSF             PSF in arcsec (default: 10 arcsec)
  --draws DRAWS         Number of random draws to perform (default: 1x draw)
  --eve_snr_model EVE_SNR_MODEL
                        Path to the instrument flux vs snr (default: EVESNR.csv)

