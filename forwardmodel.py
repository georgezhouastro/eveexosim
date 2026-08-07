import os
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astroquery.mast import Catalogs

os.environ["MINIMINT_DATA_PATH"] = "MINIMINT_DATA"
import minimint

FILTERS = [
    "Gaia_G_DR2Rev", "Gaia_BP_EDR3", 'Gaia_RP_EDR3', 
    'TESS', '2MASS_J', '2MASS_H', '2MASS_Ks'
]

ISO_INTERPOLATOR = minimint.Interpolator(FILTERS)

### local modules
import readsnr  


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


def prepareinputfields(df, fov=5.0, custom_ra=None, custom_dec=None): 

    if custom_ra is not None and custom_dec is not None:
        targetfields = pd.DataFrame({"FieldRA": [custom_ra], "FieldDEC": [custom_dec]})
    else:
        #targetfields = pd.read_csv("targetregions_simplified_20260611.csv")
        targetfields = pd.read_csv("targetregions_24fields_20260709.csv")
        print(targetfields)
        targetfields['FieldRA'] = pd.to_numeric(targetfields['FieldRA'], errors='coerce')
        targetfields['FieldDEC'] = pd.to_numeric(targetfields['FieldDEC'], errors='coerce')
        targetfields = targetfields.dropna(subset=['FieldRA', 'FieldDEC']).reset_index(drop=True)


        #targetfields = targetfields.loc[np.repeat(targetfields.index, 3)].reset_index(drop=True)

    output_dir = "tic_query_fields"
    os.makedirs(output_dir, exist_ok=True)

    tic_catalog_list = []
    # Track the chronological sequence of fields each star is in
    observed_fields = [[] for _ in range(len(df))]

    for i in range(len(targetfields)):
        ra_target = float(targetfields["FieldRA"].iloc[i])
        dec_target = float(targetfields["FieldDEC"].iloc[i])

        mask = (abs(df["ra"] - ra_target) * np.cos(df["dec"] * np.pi / 180) < fov / 2.0)
        mask &= (abs(df["dec"] - dec_target) < fov / 2.0)

        
        mask_indices = np.where(mask)[0]
        for idx in mask_indices:
            observed_fields[idx].append(i)

        filename = f"tic_query_RA_{ra_target:.5f}_DEC_{dec_target:.5f}.csv"
        filepath = os.path.join(output_dir, filename)

        if os.path.exists(filepath):
            print(f"Loading tic catalog data for RA: {ra_target}, DEC: {dec_target}")
            tic_catalog_field = pd.read_csv(filepath)
        else:
            print(f"Querying tic catalog for RA: {ra_target}, DEC: {dec_target}")
            tic_catalog_field = querytic(ra_target, dec_target, radius=fov)
            if isinstance(tic_catalog_field, pd.DataFrame):
                tic_catalog_field.to_csv(filepath, index=False)
            else:
                pd.DataFrame(tic_catalog_field).to_csv(filepath, index=False)

        tic_catalog_list.append(tic_catalog_field)

    df['observed_fields'] = observed_fields
    df['nobs'] = df['observed_fields'].apply(len)

    return df[df['nobs'] > 0].copy(), tic_catalog_list


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
        subset *= abs(sample.iloc[n]['ra']-tic_catalog_field['ra'])<0.25 
        subset *= abs(sample.iloc[n]['dec']-tic_catalog_field['dec'])<0.25

        density = sum(subset)/(0.25**2)
        expected_n_stars = density*psf**2

        if expected_n_stars < 1:
            if np.random.uniform() < expected_n_stars:
                expected_n_stars = 1
                fluxratio += ((10**(tmag/-2.5)) / (10**(sample['tmag'].iloc[n]/-2.5)))*expected_n_stars
        tmag += 1

    return fluxratio


def get_zero_points(cache_file='zero_points.npy'):
    if os.path.exists(cache_file):
        return np.load(cache_file, allow_pickle=True).item()
    
    import pyphot 
    lib = pyphot.get_library()
    
    def extract_zp(filter_name):
        return float(str(lib[filter_name].Vega_zero_flux).split(" ")[0])
    
    zps = {
        'J': extract_zp('2MASS_J'),
        'H': extract_zp('2MASS_H'),
        'K': extract_zp('2MASS_Ks'),
        'TESS': extract_zp('TESS')
    }
    
    np.save(cache_file, zps)
    return zps


ZPT = get_zero_points()

# Initialize SNR Curves
snrcurves_conservative = readsnr.readsnr_conservative()
snrcurves_optimistic = readsnr.readsnr_optimistic()

def calc_tess_rms(tmag):
    x = tmag + 0.5
    log_y1 = 0.20607148 * x + 0.17646274
    log_y2 = 0.37592988 * x - 2.02516766
    log_y3 = -0.00304001 * x + 1.80153205
    half_hr_ppm = np.sqrt((10**log_y1)**2 + (10**log_y2)**2 + (10**log_y3)**2)
    return half_hr_ppm * np.sqrt(15) * 1e-6


def prepare_rms_interpolators(spoc_df, orion_df):
    spoc_df = spoc_df[spoc_df['tmag'] > 10].copy()
    resfactor = spoc_df['sigma'] / calc_tess_rms(spoc_df['tmag'])
    rms_dist = pd.DataFrame({
        'res': resfactor, 
        'tmag': spoc_df['tmag'], 
        'age': spoc_df['age']
    }).dropna()

    orion_df = orion_df.copy()
    orion_df['rms'] *= np.sqrt(15)
    resfactor_orion = orion_df['rms'] / calc_tess_rms(orion_df['tmag'])
    rms_dist_orion = pd.DataFrame({
        'res': resfactor_orion, 
        'tmag': orion_df['tmag'], 
        'age': orion_df['age']
    }).dropna()

    return rms_dist, rms_dist_orion


def calc_transit_snr(period_days, mstar_solar, rstar_solar, rp_earth, b, baseline_days, sigma, dilution):
    mstar = mstar_solar * M_SUN
    rstar = rstar_solar * R_SUN
    rp = rp_earth * R_EARTH
    period = period_days * 24 * 3600  

    a = (period**2 * G_CONST * mstar / (4 * np.pi**2))**(1/3)
    tdur = period / np.pi * np.arcsin(np.sqrt((rstar + rp)**2 - (b * rstar)**2) / a)

    ntransits = (baseline_days * 24 * 3600) / period
    npoints = tdur / 120.0  

    sigmaseg = sigma / np.sqrt(ntransits * npoints)
    delta = (rp / rstar)**2
    snr = delta * (1-dilution) * np.sqrt(1 / sigmaseg**2)
    
    return snr, delta, sigma, npoints, ntransits


def get_snr_curve_key(teff, age):
    age_str = "50myr" if age < 50 else "100myr"
    if teff < 4000: sp_type = "m"
    elif 4000 <= teff < 5200: sp_type = "k"
    elif 5200 <= teff < 6000: sp_type = "g"
    else: sp_type = "f"
    return f"snr_{age_str}_{sp_type}"


def draw_planet(baseline, master_raw, synthetic_raw, interpolator_pack, useUV=True, useIR=True, fov=5, psf=5, custom_ra=None, custom_dec=None, snr_optical_scalar=1.0, snr_ir_scalar=1.0):
    rms_dist, rms_dist_orion = interpolator_pack

    masterlist = master_raw.drop_duplicates('tic')
    synthetic = synthetic_raw.copy()
    synthetic['tic'] = pd.to_numeric(synthetic['tic'])
    synthetic = synthetic.drop_duplicates('tic').sample(frac=0.7) 

    df = synthetic.merge(masterlist, on='tic', suffixes=('', '_y'))
    df, tic_catalog_list = prepareinputfields(df, fov=fov, custom_ra=custom_ra, custom_dec=custom_dec)

    
    if len(df) == 0:
        return df

    max_visits = df['nobs'].max()

    results = {
        'pl_period': np.full(len(df), np.nan),
        'pl_radius': np.full(len(df), np.nan),
        'delta': np.full(len(df), np.nan),
        'sigmaOptical': np.full(len(df), np.nan),
        'sigmaIR': np.full(len(df), np.nan),
        'npoints': np.full(len(df), np.nan),
        'ntransits': np.full(len(df), np.nan),
        'draw': np.full(len(df), np.nan),
        'T_retrieve_final': np.zeros(len(df), dtype=int)
    }
    
    # Pre-allocate visit-dependent arrays
    for v in range(1, max_visits + 1):
        results[f'snr_v{v}'] = np.full(len(df), np.nan)
        results[f'T_retrieve_v{v}'] = np.zeros(len(df), dtype=int)

    for i, row in enumerate(df.itertuples()):
        teff, mstar, rstar = row.teff, row.mstar, row.rstar
        tmag, age = row.tmag, row.age
        p, rp = row.per, row.prad
        obs_fields = row.observed_fields
        nobs = row.nobs

        #if rp > 20:
        #    rp = np.exp(np.random.uniform(np.log(4),np.log(20)))

        TESS_flux = 10**(tmag / -2.5) * ZPT['TESS'] * 4000

        # """
        # The exo formula given flux in erg/s/cm2
        # SNR = a*b*flux / sqrt(b*flux + c)
        # VIs: a = 4.22, b = 1.45e14, c = 49.05 (sensitivity) or 297.7 (dyn. range)
        # NIR: a = 5.16, b = 1.39e14, c = 441.0
        # """
        # eve_opt = [4.22, 1.45e14, 49.05]
        # eve_ir = [5.16, 1.39e14, 441.0]


        if False: ### old 

            """ The updated exo formula given flux in erg/s/cm2
            SNR = a*flux / sqrt(b*flux + c)
            VIs: a = 4.22, b = 1.45e14, c = 49.05 (sensitivity) or 297.7 (dyn. range)
            NIR: a = 5.16, b = 1.39e14, c = 441.0
            """
            eve_opt = [7.76E14, 1.23E14, 1.86E01]
            eve_ir = [1.32E14, 2.20E12, 6.07E00]


            #snr_optical = eve_opt[0] * eve_opt[1] * TESS_flux / np.sqrt(eve_opt[1]*TESS_flux + eve_opt[2])
            snr_optical = eve_opt[0] *  TESS_flux / np.sqrt(eve_opt[1]*TESS_flux + eve_opt[2])
            snr_optical *= snr_optical_scalar

            if snr_optical > 297.7: ### saturation
                snr_optical = 297.7
            if snr_optical < 0.1:### so it doesn't go below 0
                snr_optical = 0.1 

            #snr_IR = eve_ir[0] * eve_ir[1] * TESS_flux / np.sqrt(eve_ir[1]*TESS_flux + eve_ir[2])
            snr_IR = eve_ir[0] *  TESS_flux / np.sqrt(eve_ir[1]*TESS_flux + eve_ir[2])
            snr_IR *= snr_ir_scalar

        if True: ### new
            """
            SNR = a*flux / sqrt(b*flux + c)
            VIs: a = 4.22, b = 1.45e14, c = 49.05 (sensitivity) or 297.7 (dyn. range)
            NIR: a = 5.16, b = 1.39e14, c = 441.0
            """
            eve_opt = [7.76E14, 1.23E14, 1.86E01]
            eve_ir = [1.32E14, 2.20E12, 6.07E00]

            snr_optical = eve_opt[0] * TESS_flux / np.sqrt(eve_opt[1] * TESS_flux + eve_opt[2])
            snr_optical *= snr_optical_scalar
            if snr_optical > 297.7:
                snr_optical = 297.7
            if snr_optical < 0.1:
                snr_optical = 0.1

            snr_IR = eve_ir[0] * TESS_flux / np.sqrt(eve_ir[1] * TESS_flux + eve_ir[2])
            snr_IR *= snr_ir_scalar


        if snr_IR < 0.1:
            snr_IR = 0.1

        sigma_optical = 1/snr_optical
        sigma_IR = 1/snr_IR

        age_mask = (rms_dist["age"] >= 0)
        if age < 30: age_mask &= (rms_dist["age"] < 30)
        elif age < 50: age_mask = (rms_dist["age"] >= 30) & (rms_dist["age"] < 50)
        elif age < 100: age_mask = (rms_dist["age"] >= 50) & (rms_dist["age"] < 100)
        else: age_mask = (rms_dist["age"] >= 100)

        tmag_mask = (rms_dist["tmag"] >= 0)
        if tmag < 10: tmag_mask &= (rms_dist["tmag"] < 10)
        elif tmag < 12: tmag_mask = (rms_dist["tmag"] >= 10) & (rms_dist["tmag"] < 12)

        rms_mask = tmag_mask & age_mask
        if rms_mask.sum() < 10: rms_mask = rms_dist["tmag"].notna() 

        if age < 15:
            sigma_scaler = np.random.choice(rms_dist_orion['res'].dropna().values, size=1)[0]
        else:
            sigma_scaler = np.random.choice(rms_dist['res'][rms_mask].dropna().values, size=1)[0]

        sigma_optical *= sigma_scaler
        sigma_IR *= sigma_scaler

        # Geometrical Transit Probability (Independent of Visits)
        a = (p * 24 * 3600 * np.sqrt(G_CONST * M_SUN * mstar) / (2 * np.pi))**(2/3.0)
        ars = a / (R_SUN * rstar)
        trprob = 1 / ars

        if np.random.uniform(0, 1) < trprob:
            b = np.random.uniform(0, 1)
            t0 = np.random.uniform(0, p)
            
            rec_draw = np.random.uniform(0, 1)
            
            results['draw'][i] = rec_draw
            results['pl_period'][i] = p
            results['pl_radius'][i] = rp

            dilution = determinedilution(df, i, tic_catalog_list[obs_fields[-1]], psf=psf)

            # for each visit
            for v in range(1, nobs + 1):
                effective_baseline = v * baseline
                
                snr_opt, delta, _, npoints, ntransits = calc_transit_snr(
                    p, mstar, rstar, rp, b, effective_baseline, sigma_optical, dilution
                )
                snr_ir, _, _, _, _ = calc_transit_snr(
                    p, mstar, rstar, rp, b, effective_baseline, sigma_IR, dilution
                )

                snr = np.sqrt(snr_opt**2 + snr_ir**2) if useIR else snr_opt

                if UVdetectable(row) and useUV:
                    snrcurves = snrcurves_optimistic
                else:
                    snrcurves = snrcurves_conservative

                curve_key = get_snr_curve_key(teff, age)
                snr_select = snrcurves[curve_key]
                
                try:
                    indx = np.argmin(np.abs(snr_select['bins'] - snr))
                    rec_prob = snr_select['spoc_snr'].iloc[indx]
                except Exception:
                    rec_prob = 0

                is_retrieved = 1 if rec_draw < rec_prob else 0
                
                # Verify that it caught at least minimum required transits using total timeline
                if is_retrieved and (t0 + p > effective_baseline):
                    is_retrieved = 0

                results[f'snr_v{v}'][i] = snr
                results[f'T_retrieve_v{v}'][i] = is_retrieved

                # Save base observables on first valid visit setup to prevent overwriting
                if v == 1:
                    results['delta'][i] = delta
                    results['sigmaOptical'][i] = sigma_optical
                    results['sigmaIR'][i] = sigma_IR

            # Save the parameters scaling with the maximum baseline observed
            results['npoints'][i] = npoints
            results['ntransits'][i] = ntransits
            results['T_retrieve_final'][i] = results[f'T_retrieve_v{nobs}'][i]

    for key, val_array in results.items():
        df[key] = val_array

    return df



def run_simulation_suite(baseline, FILE_PATHS, useUV=True, useIR=True, fov=5, psf=10, custom_ra=None, custom_dec=None, snr_optical_scalar=1.0, snr_ir_scalar=1.0):
    spoc_df = pd.read_csv(FILE_PATHS["spoc_rms"])
    orion_df = pd.read_csv(FILE_PATHS["orion_rms"])
    orion_df['age'] = 10
    
    master_raw = pd.read_csv(FILE_PATHS["master_list"])
    synthetic_raw = pd.read_csv(FILE_PATHS["synthetic_planets"])

    print("Load snr curve")
    
    interpolator_pack = prepare_rms_interpolators(spoc_df, orion_df)

    base_out = FILE_PATHS["output_dir_base"]
    os.makedirs(base_out, exist_ok=True)
    folder_name = f"{base_out}/baseline_{int(baseline)}d"
    os.makedirs(folder_name, exist_ok=True)

    print(f"Running scenario: {baseline} days")
    final_df = draw_planet(
        baseline=baseline, 
        master_raw=master_raw, 
        synthetic_raw=synthetic_raw, 
        interpolator_pack=interpolator_pack,
        useUV=useUV, useIR=useIR, fov=fov, psf=psf,
        custom_ra=custom_ra, custom_dec=custom_dec,
        snr_optical_scalar=snr_optical_scalar, snr_ir_scalar=snr_ir_scalar
    )
    
    # 1. Save standard primary output
    main_output_path = os.path.join(folder_name, f"0.csv")
    final_df.to_csv(main_output_path, index=False)
    
    final_df = final_df[final_df['age'] < 50]

    # 2. Extract and save unique stars within the field
    # Dropping duplicates by TIC ID to get one row per star
    stars_cols = ['tic', 'sigmaOptical', 'sigmaIR', 'mstar', 'rstar', 'teff','age','tmag']
    stars_df = final_df.drop_duplicates('tic')[stars_cols].copy()
    stars_df['baseline'] = baseline # Broadcast the baseline scalar to all rows
    print(os.path.join(folder_name, "stars_0.csv"))
    stars_df.to_csv(os.path.join(folder_name, "stars_0.csv"), index=False)
    
    # 3. Extract and save all simulated planets
    planets_cols = ['tic', 'pl_radius', 'pl_period', 'mstar', 'rstar', 'tmag', 'teff','age']
    planets_df = final_df[final_df['T_retrieve_final']==1][planets_cols].copy()
    # Rename variables to match your requested format
    planets_df.rename(columns={'pl_radius': 'rp', 'pl_period': 'period'}, inplace=True)
    planets_df.to_csv(os.path.join(folder_name, "planets_0.csv"), index=False)
                
    print("Simulations complete! Generated main, stars, and planets datasets.")
    return main_output_path


# def run_simulation_suite(baseline, FILE_PATHS, useUV=True, useIR=True, fov=5, psf=10, custom_ra=None, custom_dec=None, snr_optical_scalar=1.0, snr_ir_scalar=1.0):
#     spoc_df = pd.read_csv(FILE_PATHS["spoc_rms"])
#     orion_df = pd.read_csv(FILE_PATHS["orion_rms"])
#     orion_df['age'] = 10
    
#     master_raw = pd.read_csv(FILE_PATHS["master_list"])
#     synthetic_raw = pd.read_csv(FILE_PATHS["synthetic_planets"])

#     print("Load snr curve")
    
#     interpolator_pack = prepare_rms_interpolators(spoc_df, orion_df)

#     base_out = FILE_PATHS["output_dir_base"]
#     os.makedirs(base_out, exist_ok=True)
#     folder_name = f"{base_out}/baseline_{int(baseline)}d"
#     os.makedirs(folder_name, exist_ok=True)

#     print(f"Running scenario: {baseline} days")
#     final_df = draw_planet(
#         baseline=baseline, 
#         master_raw=master_raw, 
#         synthetic_raw=synthetic_raw, 
#         interpolator_pack=interpolator_pack,
#         useUV=useUV, useIR=useIR, fov=fov, psf=psf,
#         custom_ra=custom_ra, custom_dec=custom_dec,
#         snr_optical_scalar=snr_optical_scalar, snr_ir_scalar=snr_ir_scalar
#     )
#     final_df.to_csv(os.path.join(folder_name, f"0.csv"), index=False)
                
#     print("Simulations complete!")
#     return os.path.join(folder_name, f"0.csv")
    

def simulation_summary(filepath):
    df = pd.read_csv(filepath)
    # Filter on the final eventual retrieval after all distinct visits
    df = df[df['T_retrieve_final'] == 1]

    print("Number of total planets:", len(df))
    SE = (df['pl_radius'] > 1.25) & (df['pl_radius'] < 2)
    SN = (df['pl_radius'] > 2) & (df['pl_radius'] < 4)
    G = df['pl_radius'] > 4        

    print("Total Number of super Earths:", sum(SE), "sub-Neptunes:", sum(SN), "Giants:", sum(G))
    
    df = df[df['age'] < 50]
    print("Number of planets <50 yrs old:", len(df))

    SE = (df['pl_radius'] > 1.25) & (df['pl_radius'] < 2)
    SN = (df['pl_radius'] > 2) & (df['pl_radius'] < 4)
    G = df['pl_radius'] > 4        

    print("<50 Myr super Earths:", sum(SE), "sub-Neptunes:", sum(SN), "Giants:", sum(G))
    

def field_yield_summary(filepath, fov=5.0, custom_ra=None, custom_dec=None):
    df = pd.read_csv(filepath)
    df['observed_fields'] = df['observed_fields'].apply(ast.literal_eval)

    # Filter to only young systems where a planet was *eventually* retrieved 
    df = df[(df['T_retrieve_final'] == 1) & (df['age'] < 50)]
    
    if custom_ra is not None and custom_dec is not None:
        targetfields = pd.DataFrame({"FieldRA": [custom_ra], "FieldDEC": [custom_dec]})
    else:
        try:
            targetfields = pd.read_csv("targetregions_simplified_20260611.csv")
            targetfields['FieldRA'] = pd.to_numeric(targetfields['FieldRA'], errors='coerce')
            targetfields['FieldDEC'] = pd.to_numeric(targetfields['FieldDEC'], errors='coerce')
            targetfields = targetfields.dropna(subset=['FieldRA', 'FieldDEC'])

        except FileNotFoundError:
            print("Error: targetregions_simplified_20260611.csv not found.")
            return

    print("\n--- Young Planet Yield (<50 Myr) by Target Field ---")
    
    seen_visits = {}
    retrieved_tics = set()
    
    for i, row in targetfields.iterrows():
        ra_target = row["FieldRA"]
        dec_target = row["FieldDEC"]
        
        # 1. Identify which eventually-retrieved planets physically exist inside this field ID
        planets_in_field = df[df['observed_fields'].apply(lambda x: i in x)]
        N_planets_in_field = len(planets_in_field)
        
        new_yield = 0
        for idx, planet in planets_in_field.iterrows():
            tic = planet['tic']
            
            # If the planet crossed SNR threshold in an earlier field, skip counting
            if tic in retrieved_tics:
                continue
            
            # 2. Increment the visit index for this specific star
            seen_visits[tic] = seen_visits.get(tic, 0) + 1
            current_visit = seen_visits[tic]
            
            # 3. Check if the planet becomes retrievable on *this specific visit*
            col_name = f'T_retrieve_v{current_visit}'
            if col_name in planet and planet[col_name] == 1:
                retrieved_tics.add(tic)
                new_yield += 1
        
        print(f"Field {i:02d} (RA: {ra_target:6.2f}, DEC: {dec_target:6.2f}) | "
              f"New Planets: {new_yield:3d} | Total observable planets in field: {N_planets_in_field:4d}")


if __name__ == "__main__":


    ### data files (Removed eve_snr_model since it's obsolete)
    FILE_PATHS = {
        "spoc_rms": "spoc_rms_age.csv", 
        "orion_rms": "roquette_rms.csv",
        "master_list": "Jan25_masterlist_roquette.csv",
        "synthetic_planets": "GasDwarfs_EVE_April_8.csv",
        "output_dir_base": "gasdwarf_sim"
    }

    import argparse
    parser = argparse.ArgumentParser(description="Forward model simulation for EVE exoplanet yield.")
    
    parser.add_argument("--baseline", type=float, default=45, help="Baseline in days (default: 45 days)")
    parser.add_argument("--fov", type=float, default=5, help="Field of view (default: 5 deg)")
    parser.add_argument("--psf", type=float, default=10, help="PSF in arcsec (default: 10 arcsec)")
    parser.add_argument("--ra", type=float, default=None, help="Custom RA for a single target field")
    parser.add_argument("--dec", type=float, default=None, help="Custom DEC for a single target field")
    parser.add_argument("--snr_optical_scalar", type=float, default=1.0, help="Multiplier for optical SNR (default: 1.0)")
    parser.add_argument("--snr_ir_scalar", type=float, default=1.0, help="Multiplier for IR SNR (default: 1.0)")
    parser.add_argument(
        "--waterworld", 
        action="store_true", 
        help="Simulate water worlds (defaults to gas dwarf unless flag is selected)"
    )
    
    args = parser.parse_args()


    if args.waterworld:
        ### update file paths if we decide to sim waterdwarfs
        FILE_PATHS["synthetic_planets"] = "WaterWorlds_EVE.csv"
        FILE_PATHS["output_dir_base"] = "waterworld_sim"


    BASELINE = args.baseline
    FOV = args.fov
    PSF = args.psf
    CUSTOM_RA = args.ra
    CUSTOM_DEC = args.dec

    if (CUSTOM_RA is None) != (CUSTOM_DEC is None):
        parser.error("You must provide BOTH --ra and --dec, or NEITHER.")

    sampleresultcsv = run_simulation_suite(
        BASELINE, FILE_PATHS,useUV=True, useIR=True, fov=FOV, psf=PSF, 
        custom_ra=CUSTOM_RA, custom_dec=CUSTOM_DEC,
        snr_optical_scalar=args.snr_optical_scalar, 
        snr_ir_scalar=args.snr_ir_scalar
    )

    simulation_summary(sampleresultcsv)
    field_yield_summary(sampleresultcsv, fov=FOV, custom_ra=CUSTOM_RA, custom_dec=CUSTOM_DEC)
