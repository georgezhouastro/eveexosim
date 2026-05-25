import pandas
snr_50myr_m = pandas.read_csv("snrcurves/50myr_breakdown/0to50_snr_m.csv")
snr_50myr_k = pandas.read_csv("snrcurves/50myr_breakdown/0to50_snr_k.csv")
snr_50myr_g = pandas.read_csv("snrcurves/50myr_breakdown/0to50_snr_g.csv")
snr_50myr_f = pandas.read_csv("snrcurves/50myr_breakdown/0to50_snr_f.csv")

snr_100myr_m = pandas.read_csv("snrcurves/50myr_breakdown/50to100_snr_m.csv")
snr_100myr_k = pandas.read_csv("snrcurves/50myr_breakdown/50to100_snr_k.csv")
snr_100myr_g = pandas.read_csv("snrcurves/50myr_breakdown/50to100_snr_g.csv")
snr_100myr_f = pandas.read_csv("snrcurves/50myr_breakdown/50to100_snr_f.csv")

snr_200myr_m = pandas.read_csv("snrcurves/100to200_snr_m.csv")
snr_200myr_k = pandas.read_csv("snrcurves/100to200_snr_k.csv")
snr_200myr_g = pandas.read_csv("snrcurves/100to200_snr_g.csv")
snr_200myr_f = pandas.read_csv("snrcurves/100to200_snr_f.csv")

### read snr curves
def readsnr_conservative():

    snrcurves = {
        "snr_50myr_m" : snr_50myr_m,
        "snr_50myr_k" : snr_50myr_k,
        "snr_50myr_g" : snr_50myr_g,
        "snr_50myr_f" : snr_50myr_f,
        "snr_100myr_m" : snr_100myr_m,
        "snr_100myr_k" : snr_100myr_k,
        "snr_100myr_g" : snr_100myr_g,
        "snr_100myr_f" : snr_100myr_f,
        "snr_200myr_m" : snr_200myr_m,
        "snr_200myr_k" : snr_200myr_k,
        "snr_200myr_g" : snr_200myr_g,
        "snr_200myr_f" : snr_200myr_f}

    return snrcurves



### read snr curves
def readsnr_optimistic():

    snrcurves = {
        "snr_50myr_m" : snr_100myr_g,
        "snr_50myr_k" : snr_100myr_g,
        "snr_50myr_g" : snr_100myr_g,
        "snr_50myr_f" : snr_100myr_g,
        "snr_100myr_m" : snr_100myr_g,
        "snr_100myr_k" : snr_100myr_g,
        "snr_100myr_g" : snr_100myr_g,
        "snr_100myr_f" : snr_100myr_g,
        "snr_200myr_m" : snr_100myr_g,
        "snr_200myr_k" : snr_100myr_g,
        "snr_200myr_g" : snr_100myr_g,
        "snr_200myr_f" : snr_100myr_g}

    return snrcurves
