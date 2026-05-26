import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate

### astro dependencies
import pyphot
import minimint

# Define the filters needed for the EVE forward model
FILTERS = [
    "Gaia_G_DR2Rev", "Gaia_BP_EDR3", 'Gaia_RP_EDR3', 
    'TESS', '2MASS_J', '2MASS_H', '2MASS_Ks'
]

# Attempt to load the interpolator safely
try:
    ISO_INTERPOLATOR = minimint.Interpolator(FILTERS)
    
except (FileNotFoundError, RuntimeError):
    print("Minimint isochrone grids not found locally. Downloading and preparing now...")
    print("This will take 10-30 minutes but only needs to happen once per machine.")
    
    # Download the base MIST tracks and compile the specific requested filters
    minimint.download_and_prepare(filters=FILTERS)
    
    # Initialize again now that the data is prepared
    ISO_INTERPOLATOR = minimint.Interpolator(FILTERS)
    print("Minimint grids successfully prepared and loaded.")


from astroquery.mast import Catalogs

### local modules
import readsnr  

### data files
FILE_PATHS = {
    "eve_snr_model": "EVESNR.csv", #### this is the flux vs snr curve! 
    "spoc_rms": "spoc_rms_age.csv", ### leave the rest of the csv 
    "orion_rms": "roquette_rms.csv",
    "master_list": "Jan25_masterlist_roquette.csv",
    "synthetic_planets": "GasDwarfs_EVE_April_8.csv",
    "output_dir_base": "gasdwarf_sim"
}

R_EARTH = 6.37e6       # meters
M_SUN = 2e30           # kg
R_SUN = 696340000.0    # meters
G_CONST = 6.67e-11     # m^3 kg^-1 s^-2

### define some relations between teff and UV flux
teff_UV = np.array([[6000,3000],[2e-2,7e-5]])
teff_UV = np.polyfit(teff_UV[0],np.log10(teff_UV[1]),1)


def querytic(ra,dec,radius=9):

    radec = str(ra)+" "+str(dec)
    catalog_data = Catalogs.query_object(radec, catalog="Tic",radius=radius/60/60)
    catalog_data = catalog_data.to_pandas()
    catalog_data["TESS"] = catalog_data["Tmag"]

    return catalog_data

def prepareinputfields(df, fov=5.0): 
    targetfields = pd.read_csv("targetregions_20260410.csv")

    # Hard-coded indices for the chosen fields in the excel spreadsheet
    index = (
        np.array([
            2, 3, 15, 16, 23, 34, 9, 10, 11, 13, 29, 24, 18, 37, 39, 43, 46,
            60, 7, 20, 22, 27, 35, 44, 52, 55, 14, 17, 42, 45,
        ])
        - 2
    )

    targetfields = targetfields.iloc[index]
    targetfields = targetfields.reset_index(drop=True)

    nobs = np.zeros(len(df))

    output_dir = "tic_query_fields"
    os.makedirs(output_dir, exist_ok=True)

    tic_catalog_list = []
    fieldlist = np.zeros(len(df))

    for i in range(len(targetfields)):
        ra_target = targetfields["FieldRA"].iloc[i]
        dec_target = targetfields["FieldDEC"].iloc[i]

        mask = (
            abs(df["ra"] - ra_target) * np.cos(df["dec"] * np.pi / 180) < fov / 2.0
        )
        mask *= abs(df["dec"] - dec_target) < fov / 2.0
        nobs[mask] += 1
        fieldlist[mask] = i

        # save tic query based on RA and DEC coordinates
        filename = f"tic_query_RA_{ra_target:.5f}_DEC_{dec_target:.5f}.csv"
        filepath = os.path.join(output_dir, filename)

        if os.path.exists(filepath):
            # Load from local if it exists
            print(f"Loading tic catalog data for RA: {ra_target}, DEC: {dec_target}")
            tic_catalog_field = pd.read_csv(filepath)
        else:
            print(f"Querying tic catalog for RA: {ra_target}, DEC: {dec_target}")
            tic_catalog_field = querytic(ra_target, dec_target, radius=fov)

            # Save the result to cache (assuming querytic returns a pandas DataFrame)
            if isinstance(tic_catalog_field, pd.DataFrame):
                tic_catalog_field.to_csv(filepath, index=False)
            else:
                pd.DataFrame(tic_catalog_field).to_csv(filepath, index=False)

        tic_catalog_list.append(tic_catalog_field)

        df['fieldID'] = fieldlist

    return df[nobs>0],tic_catalog_list

def UVdetectable(star):
    
    UV = 10**np.polyval(teff_UV,star.teff)
    TESS_FLUX = 10**(star.tmag/-2.5)*ZPT['TESS']*4000
    return UV*TESS_FLUX > 5e-13


def determinedilution(sample,n,tic_catalog_field, psf=10):

    fluxratio = 0
    tmag = 8 ## a bright star in our field is 8th mag
    while tmag < sample['tmag'].iloc[n]+8:
        subset = tic_catalog_field['TESS']>tmag
        subset *= tic_catalog_field['TESS']<tmag+1
        subset *= abs(sample.iloc[n]['ra']-tic_catalog_field['ra'])<0.25 ### look at the stellar density with 0.25 deg of the target star
        subset *= abs(sample.iloc[n]['dec']-tic_catalog_field['dec'])<0.25

        ### calculate stellar density 
        density = sum(subset)/(0.25**2)
        expected_n_stars = density*psf**2

        if expected_n_stars < 1:
            if np.random.uniform() < expected_n_stars:
                expected_n_stars = 1
                fluxratio += ((10**(tmag/-2.5)) / (10**(sample['tmag'].iloc[n]/-2.5)))*expected_n_stars

        tmag += 1

    return fluxratio


### photometric zero points
def get_zero_points():
    lib = pyphot.get_library()
    def extract_zp(filter_name):
        return float(str(lib[filter_name].Vega_zero_flux).split(" ")[0])
    
    return {
        'J': extract_zp('2MASS_J'),
        'H': extract_zp('2MASS_H'),
        'K': extract_zp('2MASS_Ks'),
        'TESS': extract_zp('TESS')
    }

ZPT = get_zero_points()


# Initialize SNR Curves
snrcurves_conservative = readsnr.readsnr_conservative()
snrcurves_optimistic = readsnr.readsnr_optimistic()



def calc_tess_rms(tmag):
    """
    TESS photometric error estimate [ppm] modeled via photon noise, sky noise, and floor.
    """
    x = tmag + 0.5
    
    # Model 1: Photon Noise
    log_y1 = 0.20607148 * x + 0.17646274
    # Model 2: Sky Noise
    log_y2 = 0.37592988 * x - 2.02516766
    # Model 3: Floor
    log_y3 = -0.00304001 * x + 1.80153205
    
    # Sum the values in linear space
    half_hr_ppm = np.sqrt((10**log_y1)**2 + (10**log_y2)**2 + (10**log_y3)**2)
    return half_hr_ppm * np.sqrt(15) * 1e-6


def prepare_rms_interpolators(eve_optical, eve_ir, eve_df, spoc_df, orion_df):
    """
    RMS scaling distributions for active young stars
    """

    # TESS young star RMS
    spoc_df = spoc_df[spoc_df['tmag'] > 10].copy()
    resfactor = spoc_df['sigma'] / calc_tess_rms(spoc_df['tmag'])
    rms_dist = pd.DataFrame({
        'res': resfactor, 
        'tmag': spoc_df['tmag'], 
        'age': spoc_df['age']
    }).dropna()

    # TESS Orion specific RMS
    orion_df = orion_df.copy()
    orion_df['rms'] *= np.sqrt(15)
    resfactor_orion = orion_df['rms'] / calc_tess_rms(orion_df['tmag'])
    rms_dist_orion = pd.DataFrame({
        'res': resfactor_orion, 
        'tmag': orion_df['tmag'], 
        'age': orion_df['age']
    }).dropna()


    return eve_optical, eve_ir, rms_dist, rms_dist_orion


def calc_transit_snr(period_days, mstar_solar, rstar_solar, rp_earth, b, baseline_days, sigma, dilution):
    """
    Transit SNR
    """
    mstar = mstar_solar * M_SUN
    rstar = rstar_solar * R_SUN
    rp = rp_earth * R_EARTH
    period = period_days * 24 * 3600  # seconds

    a = (period**2 * G_CONST * mstar / (4 * np.pi**2))**(1/3)
    
    # Transit duration
    tdur = period / np.pi * np.arcsin(np.sqrt((rstar + rp)**2 - (b * rstar)**2) / a)

    ntransits = (baseline_days * 24 * 3600) / period
    npoints = tdur / 120.0  # 120s cadence

    sigmaseg = sigma / np.sqrt(ntransits * npoints)
    delta = (rp / rstar)**2
    snr = delta * (1-dilution) * np.sqrt(1 / sigmaseg**2)
    
    return snr, delta, sigma, npoints, ntransits


def get_snr_curve_key(teff, age):
    """
    readsnr curves based on Teff and Age.
    """
    age_str = "50myr" if age < 50 else "100myr"
    
    if teff < 4000:
        sp_type = "m"
    elif 4000 <= teff < 5200:
        sp_type = "k"
    elif 5200 <= teff < 6000:
        sp_type = "g"
    else:
        sp_type = "f"
        
    return f"snr_{age_str}_{sp_type}"


def draw_planet(baseline, master_raw, synthetic_raw, interpolator_pack, useUV=True, useIR=True, fov=5, psf=5):
    """
    injection and recovery
    """
    eve_opt, eve_ir, rms_dist, rms_dist_orion = interpolator_pack

    # Prepare datasets
    masterlist = master_raw.drop_duplicates('tic')
    synthetic = synthetic_raw.copy()
    synthetic['tic'] = pd.to_numeric(synthetic['tic'])
    synthetic = synthetic.drop_duplicates('tic').sample(frac=0.7) #### draw 0.7 planets per star
    
    df = synthetic.merge(masterlist, on='tic', suffixes=('', '_y'))
    df,tic_catalog_list = prepareinputfields(df,fov=fov) ### trim input catalog to only the fields obs by EVE
    
    results = {
        'T_retrieve': np.zeros(len(df)),
        'pl_period': np.full(len(df), np.nan),
        'pl_radius': np.full(len(df), np.nan),
        'delta': np.full(len(df), np.nan),
        'sigmaOptical': np.full(len(df), np.nan),
        'sigmaIR': np.full(len(df), np.nan),
        'npoints': np.full(len(df), np.nan),
        'ntransits': np.full(len(df), np.nan),
        'draw': np.full(len(df), np.nan)
    }

    for i, row in enumerate(df.itertuples()):
        teff, mstar, rstar = row.teff, row.mstar, row.rstar
        tmag, age = row.tmag, row.age
        p, rp = row.per, row.prad
        fieldID = row.fieldID

        

        # Flux Conversions
        TESS_flux = 10**(tmag / -2.5) * ZPT['TESS'] * 4000
        iso = ISO_INTERPOLATOR(mstar, np.log10(age * 1e6), 0)
        
        TESS_isocor = 10**(iso['TESS'] / -2.5) * ZPT['TESS']
        J_isocor = 10**(iso['2MASS_J'] / -2.5) * ZPT['J']
        H_isocor = 10**(iso['2MASS_H'] / -2.5) * ZPT['H']
        K_isocor = 10**(iso['2MASS_Ks'] / -2.5) * ZPT['K']

        Jflux = TESS_flux * (J_isocor / TESS_isocor)
        Hflux = TESS_flux * (H_isocor / TESS_isocor)
        Kflux = TESS_flux * (K_isocor / TESS_isocor)

        # Baseline sigmas
        sigma_optical = 10**interpolate.splev([np.log10(TESS_flux)], eve_opt)[0]
        sigma_IR = 10**interpolate.splev([np.log10(Jflux + Kflux)], eve_ir)[0]

        # Apply Age and Tmag filters for RMS distribution
        age_mask = (rms_dist["age"] >= 0)
        if age < 30: age_mask &= (rms_dist["age"] < 30)
        elif age < 50: age_mask = (rms_dist["age"] >= 30) & (rms_dist["age"] < 50)
        elif age < 100: age_mask = (rms_dist["age"] >= 50) & (rms_dist["age"] < 100)
        else: age_mask = (rms_dist["age"] >= 100)

        tmag_mask = (rms_dist["tmag"] >= 0)
        if tmag < 10: tmag_mask &= (rms_dist["tmag"] < 10)
        elif tmag < 12: tmag_mask = (rms_dist["tmag"] >= 10) & (rms_dist["tmag"] < 12)

        rms_mask = tmag_mask & age_mask
        if rms_mask.sum() < 10:
            rms_mask = rms_dist["tmag"].notna()  # Fallback

        if age < 15:
            sigma_scaler = np.random.choice(rms_dist_orion['res'].dropna(), size=1)[0]
        else:
            sigma_scaler = np.random.choice(rms_dist['res'][rms_mask].dropna(), size=1)[0]

        sigma_optical *= sigma_scaler
        sigma_IR *= sigma_scaler

        # Transit Probability
        a = (p * 24 * 3600 * np.sqrt(G_CONST * M_SUN * mstar) / (2 * np.pi))**(2/3.0)
        ars = a / (R_SUN * rstar)
        trprob = 1 / ars

        if np.random.uniform(0, 1) < trprob:
            b = np.random.uniform(0, 1)

            ### calculate snr based on dilution of the field
            dilution = determinedilution(df,i,tic_catalog_list[int(fieldID)],psf=psf)

            
            snr_opt, delta, __, npoints, ntransits = calc_transit_snr(
                p, mstar, rstar, rp, b, baseline, sigma_optical, dilution
            )
            snr_ir, __, __, __, __ = calc_transit_snr(
                p, mstar, rstar, rp, b, baseline, sigma_IR, dilution
            )

            if useIR:
                snr = np.sqrt(snr_opt**2 + snr_ir**2)

            else:
                snr = snr_opt

            ### if the star is detectable in the UV, then use optimistic SNR curves
            
            if UVdetectable(row) and useUV:
                snrcurves = snrcurves_optimistic
            else:
                snrcurves = snrcurves_conservative
    

            
            # Determine Recovery Probability
            curve_key = get_snr_curve_key(teff, age)
            snr_select = snrcurves[curve_key]
            
            try:
                indx = np.argmin(np.abs(snr_select['bins'] - snr))
                rec_prob = snr_select['spoc_snr'].iloc[indx]
            except Exception:
                rec_prob = 0

            # Determine if retrieved
            draw = np.random.uniform(0, 1)
            is_retrieved = 1 if draw < rec_prob else 0
            
            if is_retrieved: ## check if 2x transits in baseline
                t0 = np.random.uniform(0,p)
                if t0 + p > baseline:
                    is_retrieved = 0

            results['pl_period'][i] = p
            results['pl_radius'][i] = rp
            results['delta'][i] = delta
            results['sigmaOptical'][i] = sigma_optical
            results['sigmaIR'][i] = sigma_IR
            results['npoints'][i] = npoints
            results['ntransits'][i] = ntransits
            results['draw'][i] = draw
            results['T_retrieve'][i] = is_retrieved

    for key, val_array in results.items():
        df[key] = val_array

    return df


def run_simulation_suite(eve_df,baseline, draws=1,useUV=True,useIR=True, fov=5, psf=10):

    spoc_df = pd.read_csv(FILE_PATHS["spoc_rms"])
    orion_df = pd.read_csv(FILE_PATHS["orion_rms"])
    orion_df['age'] = 10
    
    master_raw = pd.read_csv(FILE_PATHS["master_list"])
    synthetic_raw = pd.read_csv(FILE_PATHS["synthetic_planets"])

    print("Preparing splines and interpolators...")

    # Compute RMS vs Flux for EVE Optical and IR
    eve_optical = interpolate.splrep(np.log10(eve_df['Flux']), np.log10(1/eve_df['SNROP']), k=1)
    eve_ir = interpolate.splrep(np.log10(eve_df['Flux']), np.log10(1/eve_df['SNRIR']), k=1)

    interpolator_pack = prepare_rms_interpolators(eve_optical,eve_ir,eve_df, spoc_df, orion_df)

    base_out = FILE_PATHS["output_dir_base"]
    os.makedirs(base_out, exist_ok=True)

    folder_name = f"{base_out}/gasdwarf{int(baseline)}d"
    os.makedirs(folder_name, exist_ok=True)

    print(f"Running scenario: {baseline} days")
    for i in range(draws):
        print(f"  --> Draw {i+1}/{draws}...")
        final_df = draw_planet(
            baseline=baseline, 
            master_raw=master_raw, 
            synthetic_raw=synthetic_raw, 
            interpolator_pack=interpolator_pack,
            useUV=True,
            useIR=True,
            fov=fov,
            psf=psf
        )
        final_df.to_csv(os.path.join(folder_name, f"{i}.csv"), index=False)
                
    print("Simulations complete!")

    return os.path.join(folder_name, f"{i}.csv")
    
def simulation_summary(filepath):

    df = pd.read_csv(filepath)
    df = df[df['T_retrieve'] == 1] ### return only those with planets

    print("Number of total planets",len(df))
    SE = df['pl_radius'] > 1.25        
    SE *= df['pl_radius'] < 2

    SN = df['pl_radius'] > 2        
    SN *= df['pl_radius'] < 4

    G = df['pl_radius'] > 4        

    print("Total Number of super Earths",sum(SE),"sub-Neptunes",sum(SN),"Giants",sum(G))
    
    df = df[df['age'] < 50]
    print("Number of planets <50 yrs old",len(df))

    SE = df['pl_radius'] > 1.25        
    SE *= df['pl_radius'] < 2

    SN = df['pl_radius'] > 2        
    SN *= df['pl_radius'] < 4

    G = df['pl_radius'] > 4        

    print("<50 Myr super Earths",sum(SE),"sub-Neptunes",sum(SN),"Giants",sum(G))
    


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Forward model simulation for EVE exoplanet yield.")
    
    # Add numerical arguments with defaults
    parser.add_argument("--baseline", type=float, default=30, help="Baseline in days (default: 30 days)")
    parser.add_argument("--fov", type=float, default=5, help="Field of view (default: 5 deg)")
    parser.add_argument("--psf", type=float, default=10, help="PSF in arcsec (default: 10 arcsec)")
    parser.add_argument("--draws", type=int, default=1, help="Number of random draws to perform (default: 1x draw)")
    
    # Add file path argument with FILE_PATHS dict as the default
    parser.add_argument(
        "--eve_snr_model", 
        type=str, 
        default=FILE_PATHS["eve_snr_model"], 
        help="Path to the instrument flux vs snr (default: "+FILE_PATHS["eve_snr_model"]+")"
    )
    
    args = parser.parse_args()

    # Map the parsed arguments to your original variable names (optional, but keeps the rest of your code working)
    BASELINE = args.baseline
    FOV = args.fov
    PSF = args.psf
    DRAWS = args.draws

    #### read in the RMS model
    eve_df = pd.read_csv(args.eve_snr_model, delim_whitespace=True)
    


    ### run the simulation
    sampleresultcsv = run_simulation_suite(eve_df,BASELINE, draws=DRAWS, useUV=True, useIR=True, fov=FOV, psf=PSF)

    ### provide simulation summary
    simulation_summary(sampleresultcsv)
