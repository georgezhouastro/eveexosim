import os,sys
import pandas as pd
from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2

import readsnr
snrcurves_conservative = readsnr.readsnr_conservative() 
snrcurves_optimistic = readsnr.readsnr_optimistic()

rearth = 6.37e6
msun = 2e30
rsun = 696340000.
G = 6.67e-11

teff_UV = np.array([[6000,3000],[2e-2,7e-5]])
teff_UV = np.polyfit(teff_UV[0],np.log10(teff_UV[1]),1)


import pyphot

lib = pyphot.get_library()

J_2MASS_zpt = lib['2MASS_J'].Vega_zero_flux
H_2MASS_zpt = lib['2MASS_H'].Vega_zero_flux
K_2MASS_zpt = lib['2MASS_Ks'].Vega_zero_flux
tess_zpt = lib['TESS'].Vega_zero_flux

J_2MASS_zpt = float(str.split(str(J_2MASS_zpt)," ")[0])
H_2MASS_zpt = float(str.split(str(H_2MASS_zpt)," ")[0])
K_2MASS_zpt = float(str.split(str(K_2MASS_zpt)," ")[0])
tess_zpt = float(str.split(str(tess_zpt)," ")[0])

def UVdetectable(star):
    UV = 10**np.polyval(teff_UV,star['teff'])
    TESS_FLUX = 10**(star['tmag']/-2.5)*tess_zpt*4000
    return UV*TESS_FLUX > 5e-13

def redo_recovery(entry,snr):

    teff = entry['teff']
    age = entry['age']

    if UVdetectable(entry):
        snrcurves = snrcurves_optimistic
    else:
        snrcurves = snrcurves_conservative
    
    if teff < 4000 and age < 50:
        snrselect = snrcurves['snr_50myr_m']
    if teff > 4000 and teff < 5200 and age < 50:
        snrselect = snrcurves['snr_50myr_k']
    if teff > 5200 and teff < 6000 and age < 50:
        snrselect = snrcurves['snr_50myr_g']
    if teff > 6000 and age < 50:
        snrselect = snrcurves['snr_50myr_f']

    if teff < 4000 and age >= 50 and age <= 100:
        snrselect = snrcurves['snr_100myr_m']
    if teff > 4000 and teff < 5200 and age >= 50 and age <= 100:
        snrselect = snrcurves['snr_100myr_k']
    if teff > 5200 and teff < 6000 and age >= 50 and age <= 100:
        snrselect = snrcurves['snr_100myr_g']
    if teff > 6000 and age >= 50 and age <= 100:
        snrselect = snrcurves['snr_100myr_f']

    if teff < 4000 and age > 100:
        snrselect = snrcurves['snr_100myr_m']
    if teff > 4000 and teff < 5200 and age > 100:
        snrselect = snrcurves['snr_100myr_k']
    if teff > 5200 and teff < 6000 and age > 100:
        snrselect = snrcurves['snr_100myr_g']
    if teff > 6000 and age > 100:
        snrselect = snrcurves['snr_100myr_f']

    try:
        indx = np.argmin(np.abs(snrselect['bins']-snr))
        rec_prob = snrselect['spoc_snr'].iloc[indx]
    except UnboundLocalError:
        rec_prob = 0


    return rec_prob



def calc_transit_snr(period,mstar,rstar,rp,b,baseline,sigma):

    mstar *= msun
    rstar *= rsun
    rp *= rearth
    period *= 60*60*24.


    a = (period**2 * G * mstar / (4 * np.pi**2))**(1/3)
    tdur = period / np.pi * np.arcsin(np.sqrt((rstar + rp)**2 - (b * rstar)**2) / a)

    ntransits = baseline*24*60*60/period
    npoints = tdur/120. #cadence

    sigmaseg = sigma / np.sqrt(ntransits * npoints)
    delta = (rp/rstar)**2
    snr = delta * np.sqrt(1/sigmaseg**2)
    
    return snr,delta,sigma,npoints,ntransits,a/rstar

def computesnr(period,radius,star):
    sigmaOptical,sigmaIR,baseline = star['sigmaOptical'],star['sigmaIR'],star['baseline']
    b = np.random.uniform(0,1)
    snr_opt_tuple = calc_transit_snr(period,star['mstar'],star['rstar'],radius,b,baseline,sigmaOptical)
    snr_IR_tuple = calc_transit_snr(period,star['mstar'],star['rstar'],radius,b,baseline,sigmaIR)

    snr = np.sqrt(snr_opt_tuple[0]**2 + snr_IR_tuple[0]**2)


    tr_prob = 1/snr_opt_tuple[-1]
    rec_prob = redo_recovery(star, snr)*tr_prob
    rec_draw = np.random.uniform(0,1)

    is_retrieved = 1 if rec_draw < rec_prob else 0
                
    # Verify that it caught at least minimum required transits using total timeline
    if is_retrieved and (np.random.uniform(0, period) + period > baseline):
        is_retrieved = 0

    return is_retrieved

def gridpoint(periodmin,periodmax,radiusmin,radiusmax,stars_sigma):

    rec_stars = 0
    for star in stars_sigma:
        period = np.exp(np.random.uniform(np.log(periodmin),np.log(periodmax)))
        radius = np.exp(np.random.uniform(np.log(radiusmin),np.log(radiusmax)))

        rec_prob = computesnr(period,radius,star)
        rec_stars += rec_prob

    return rec_stars



if __name__ == "__main__":


    import argparse
    parser = argparse.ArgumentParser(description="Occurrence rate calculations")
    
    parser.add_argument("--planetsfile", type=str, default='gasdwarf_sim/baseline_45d/planets_0.csv', help="list of input planets")
    parser.add_argument("--starsfile", type=str, default='gasdwarf_sim/baseline_45d/stars_0.csv', help="list of input stars")
    args = parser.parse_args()

    
    stars_sigma_df = pd.read_csv(args.starsfile)
    stars_sigma = stars_sigma_df.to_dict('records')
    
    periodaxis = np.linspace(np.log10(1), np.log10(20), 5)
    radiusaxis = np.linspace(np.log10(4), np.log10(50), 5)
    
    n_periods = len(periodaxis) - 1
    n_radii = len(radiusaxis) - 1
    
    expected_planets = np.zeros((n_radii, n_periods))
    
    print("Computing expected planets...")
    for i in range(n_radii):
        for j in range(n_periods):
            p_min = 10**periodaxis[j]
            p_max = 10**periodaxis[j+1]
            r_min = 10**radiusaxis[i]
            r_max = 10**radiusaxis[i+1]
            
            expected_planets[i, j] = gridpoint(p_min, p_max, r_min, r_max, stars_sigma)
            print('computed for period radius',10**periodaxis[j],10**radiusaxis[i])
            
    planet_list_df = pd.read_csv(args.planetsfile)
    
    log_per_obs = np.log10(planet_list_df['period'])
    log_rp_obs = np.log10(planet_list_df['rp'])
    
    observed_planets, xedges, yedges = np.histogram2d(
        log_per_obs, 
        log_rp_obs, 
        bins=[periodaxis, radiusaxis]
    )
    observed_planets = observed_planets.T 
    
    occurrence_rate = np.zeros_like(expected_planets)
    err = np.zeros_like(expected_planets)
    
    mask = expected_planets > 0
    occurrence_rate[mask] = observed_planets[mask] / expected_planets[mask]
    print('total number of expected planets if every star had a planet in each grid point',np.sum(expected_planets))

    alpha = 1 - 0.682689
    
    for i in range(n_radii):
        for j in range(n_periods):
            k = observed_planets[i, j]
            exp = expected_planets[i, j]
            
            if exp == 0:
                continue

            if k == 0:
                err_k = 0
            else:
                err_k = chi2.ppf(alpha / 2, 2 * k) / 2
                
            err[i, j] = (k - err_k) / exp 

    goodgrid = expected_planets > 30
    #goodgrid *= observed_planets > 0

    total_occurrence = np.sum(occurrence_rate[goodgrid])
    total_err = np.sqrt(np.sum(err[goodgrid]**2))

    
    
    print(f"\nTotal Occurrence Rate: {total_occurrence:.4f} +/-{total_err:.4f}")

    occurrence_rate *= 100
    err *= 100
    total_occurrence *= 100
    total_err *= 100

    fig, ax = plt.subplots(figsize=(5, 4))
    
    extent = [periodaxis[0], periodaxis[-1], radiusaxis[0], radiusaxis[-1]]
    occurrence_rate[np.invert(goodgrid)] = np.nan
    
    im = ax.imshow(occurrence_rate, origin='lower', aspect='auto', cmap='Oranges', extent=extent)
    
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Planets per 100 Stars')
    
    csv_data = []
    
    for i in range(n_radii):
        for j in range(n_periods):
            x_center = (periodaxis[j] + periodaxis[j+1]) / 2.0
            y_center = (radiusaxis[i] + radiusaxis[i+1]) / 2.0
            
            rate_val = occurrence_rate[i, j]
            err_val = err[i, j]
            
            # Calculate linear boundaries for the CSV
            p_min = int(round(10**periodaxis[j]))
            p_max = int(round(10**periodaxis[j+1]))
            r_min = int(round(10**radiusaxis[i]))
            r_max = int(round(10**radiusaxis[i+1]))
            
            csv_data.append({
                'Period_min_days': p_min,
                'Period_max_days': p_max,
                'Radius_min_earth': r_min,
                'Radius_max_earth': r_max,
                'Planets_per_100_stars': rate_val,
                'Error': err_val
            })
            
            # Adjusted decimal points to .1f since values are scaled by 100
            cell_text = f"{rate_val:.1f}\n+/-{err_val:.1f}"
            
            ax.text(x_center, y_center, cell_text, ha='center', va='center', color='k', fontsize=8)
            
    ax.set_xticks(periodaxis)
    ax.set_xticklabels([str(int(round(10**p))) for p in periodaxis])
    
    ax.set_yticks(radiusaxis)
    ax.set_yticklabels([str(int(round(10**r))) for r in radiusaxis])
            
    ax.set_xlabel("Period [days]")
    ax.set_ylabel("Radius [Earth Radii]")
    ax.set_title(f"Planet Occurrence Rate\nTotal = {total_occurrence:.1f} +/-{total_err:.1f} planets / 100 stars", fontsize=10)
    
    plt.tight_layout()
    plt.show()

    df_out = pd.DataFrame(csv_data)
    df_out.to_csv("occurrence_rate_table.csv", index=False)




