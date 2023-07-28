import radis as rd
import os
import os.path as op
import scipy
from brukeropusreader import read_file
from bisect import bisect_left
import csv
from scipy.sparse.linalg import spsolve
from scipy import sparse
import numpy as np
from lmfit.models import ExponentialModel, SkewedGaussianModel

def read_opus_data_from_folder_into_array_for_gui(dir_inv):
    """
    The function gets the usable data-files from the given directory. The usable files are either OPUS-files, csv-files
    or dat-files (not yet implemented as not yet used).

    :param dir_inv: given directory where to find the data-files
    :return: array filled with the gained data-files from the given directory
    """
    directory_data = dir_inv
    files_in_directory_temp = [f for f in os.listdir(directory_data) if op.isfile(op.join(directory_data,f))]
    files_in_directory = []
    for file in files_in_directory_temp:
        if file[-2:] == ".0":
            files_in_directory.append(file)
        elif file[-4:] == ".csv":
            files_in_directory.append(file)
        elif file[-4:] == ".dat":
            files_in_directory.append(file)
    dat_array = []
    for file in files_in_directory:
        if file[-2:] == ".0":
            dat_array.append(read_file(directory_data + file))
        elif file[-4:] == ".csv":
            w_data = []
            t_data = []
            with open(directory_data + "\\" + file) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=",")
                line_count = 0
                for row in csv_reader:
                    w_data.append(float(row[0]))
                    t_data.append(float(row[1]))
            full_data = [np.array(w_data), np.array(t_data)]
            dat_array.append(full_data)
        #elif file[-4:] == ".dat":
        #    print("hello")
    return dat_array

def spectrum_in_air_creator(mol, mf,pres, temp, path_l, wl_min, wl_max, step, dil='air'):
    """
    Function that can be used to simulate a spectrum for a gas. It's not much changed from the original code of radis, but
    it's made like this, so that in the original file, it's slightly cleaner, since some fo the parameters that are
    constant, don't have to be filled in (like the data coming from Hitran for example).

    :param mol: Type of molecule one wants to gain the spectra from.
    :param mf: The mole fraction one wants to simulate.
    :param pres: The pressure at which one wants their gas to be.
    :param temp: The temperature at which one wants their gas to be.
    :param path_l: The path length the light goes through the gas one wants to look at.
    :param wl_min: The minimum wavelength one wants to simulate their spectrum for.
    :param wl_max: The maximum wavelength one wants to simulate their spectrum for.
    :param step: The step size (for the wavelength) one wants to simulate their spectrum for
    :return: Returns the created simulated spectrum.
    """

    spectrum = rd.calc_spectrum(wl_min, wl_max, molecule=mol, isotope="all", pressure=pres, Tgas=temp,
                                wstep=step, path_length=path_l, databank="hitran", mole_fraction=mf, medium="air",
                                warnings={"AccuracyError": "ignore"}, diluent=dil)

    return spectrum

def take_closest(myList, myNumber):
    """
    Assumes myList is sorted. Returns closest value to myNumber.

    If two numbers are equally close, return the smallest number.
    """
    pos = bisect_left(myList, myNumber)
    if pos == 0:
        return myList[0]
    if pos == len(myList):
        return myList[-1]
    before = myList[pos - 1]
    after = myList[pos]
    if after - myNumber < myNumber - before:
        return after
    else:
        return before

def trold_to_trnew(Told, c_factor):
    """
    Function that transforms an old transmission spectra to a new transmission spectra using a factor c. This function
    is used to fasten the fitting process.

    :param Told: The old transmission spectra
    :param c_factor: The factor with which one wants to change their transmission spectra
    :return: The new transmission spectra
    """
    Tnew = 10 ** (-c_factor * np.log10(1 / Told))
    return Tnew

def tr_to_ab(transmission):
    """
    Function which transforms a transmission spectra to an absorption spectra.

    :param transmission: The transmission spectra that needs to be transformed
    :return: The new absorption spectra
    """
    tr = np.array(transmission)
    return np.log10(1/tr)

def ab_to_tr(absorption):
    """
    Function that transforms an absorption spectra to a transmission spectra.

    :param absorption: The absorption spectra that needs to be transformed
    :return: The new transmission spectra
    """
    ab = np.array(absorption)
    return 10**(-ab)

def baseline_correction_calculator(y, baseline_diff, lam, p, niter=10):
    L = len(y)
    D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
    D = lam * D.dot(D.transpose()) # Precompute this term since it does not depend on `w`
    w = np.ones(L)
    W = sparse.spdiags(w, 0, L, L)
    baseline = np.array([])

    for i_bas in range(niter):
        W.setdiag(w) # Do not create a new matrix, just update diagonal values
        Z = W + D
        z = spsolve(Z, w*y)
        w = p * (y > z) + (1-p) * (y < z)
        baseline = z

    list_baseline = []
    for i in range(len(baseline)):
        if np.abs(y[i]-baseline[i]) <= baseline_diff:
            list_baseline.append(y[i])
    baseline_average = np.mean(list_baseline)
    return baseline, baseline_average

def fit_baseline(w, a, a_b, bool):
    def baseline_creator(x, A1, center, sigma, gamma, A2, decay):
        if A1 > 0:
            skewed = (A1/(sigma*np.sqrt(2*np.pi)))*np.exp(-1*(x-center)**2/2/sigma**2)*\
                     (1+scipy.special.erf(gamma*(x-center)/(sigma*np.sqrt(2))))
        else:
            skewed = np.zeros(len(x))

        exp_now = A2*np.exp(-1*x/decay)

        full_baseline = skewed + exp_now
        return full_baseline

    bas, b = baseline_correction_calculator(a, 0.0001, 10000000, 0.5)
    bas_b, b_b = baseline_correction_calculator(a_b, 0.0001, 1000000, 0.5)

    a_background_removed = a-bas_b
    a_zero = a_b - bas_b

    bas_zero, b_zero = baseline_correction_calculator(a_zero, 0.0001, 1000000, 0.5)
    bas_background_removed, b_background_removed = baseline_correction_calculator(
        a_background_removed, 0.0001, 1000000, 0.5)

    bas_line_temp = []
    bas_w = []

    for k in range(len(w)):
        if (np.abs(a_background_removed[k] - bas_background_removed[k])) <= 0.00005:
            bas_w.append(w[k])
            bas_line_temp.append(a_background_removed[k])

    bas_w = np.array(bas_w)
    bas_line_temp = np.array(bas_line_temp)

    gaus = SkewedGaussianModel(prefix="g_")
    g_center_min = 3134
    g_center_max = 3139
    pars = gaus.guess(bas_line_temp, x=bas_w)
    pars["g_center"].set(value=3136, min=g_center_min, max=g_center_max)
    pars["g_sigma"].set(value=155.3, min=162, max=166.5)
    pars["g_gamma"].set(value=2.33, min=1.95, max=2.8)

    exp = ExponentialModel(prefix="e_")
    pars += exp.guess(bas_line_temp, x=bas_w)
    model = gaus + exp
    out = model.fit(bas_line_temp, pars, x=bas_w)

    g_center = out.best_values["g_center"]
    g_amplitude = out.best_values["g_amplitude"]
    g_sigma = out.best_values["g_sigma"]
    g_gamma = out.best_values['g_gamma']
    e_amplitude = out.best_values["e_amplitude"]
    e_decay = out.best_values["e_decay"]

    baseline = baseline_creator(x=w, A1=g_amplitude, center=g_center, sigma=g_sigma, gamma=g_gamma, A2=e_amplitude,
                                 decay = e_decay)

    #plt.plot(w, a_background_removed)
    #plt.plot(w, baseline, alpha=0.7)
    #plt.plot(w, bas_zero, alpha=0.7)
    #plt.show()

    a_new = a-baseline
    bas_line_new, bas_new = baseline_correction_calculator(a_new, 0.0001, 10000000, 0.5)

    error = []
    for k in range(len(w)):
        if bas_line_new[k] <= bas_zero[k]:
            error.append(bas_line_new[k])

    a_new += np.abs(np.mean(error))

    #plt.plot(w, a_new, alpha=0.7)
    #plt.plot(w, a, alpha=0.7)
    #plt.show()

    return baseline
