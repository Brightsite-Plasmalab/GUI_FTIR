import csv                as csv
import copy               as copy
import json               as json
import numpy              as np
import os                 as os
import sys                as sys
import matplotlib.pyplot  as plt
import radis              as radis
from brukeropusreader import read_file

from lmfit import minimize, Parameters, fit_report

# from radis import Spectrum, MergeSlabs, calc_spectrum
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit

import io                 as io #output management (suppress uneven sampling and mole fraction warnings and radis output)
import contextlib         as contextlib
import warnings           as warnings

from ttictoc import tic,toc #Timing and beautification of output. 
from pprint import pprint
import re                 as re

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname( __file__ ))))
import Code_Gui.Gui_General_Code.General_Functions_Library as GFL
import Code_Gui.Gui_General_Code.Gas_Mixtures_Spectra_Library as GMSL

import spectrochempy as spectrochempy

def read_FTIR_measurement(fnam): 
    """
    This function reads several different filetypes and returns the wavelength (cm-1) and transmission. 
    """
    fext = fnam.rsplit('.',1)[1]
    if fext == "tsv":
        fdat = np.genfromtxt(fnam, skip_header=1)
        fdat[:,1] = np.power(10.0, -fdat[:,1])
    elif fext == "0":
        fdat = read_file(fnam)
    elif fext == "csv":
        sys.exit()
    elif fext == "dat":
        sys.exit()
    elif fext == "spa":
        frea = spectrochempy.read_spa(fnam)
        print(frea.coordset.x)
        print(frea.title)
        print(frea.units)
        fdat = np.dstack((frea.coordset.x.data, frea.data[0]))[0]
        fdat[:,1] = np.power(10.0, -fdat[:,1])
    else:
        print("File extension " + fext + " not implemented, exiting")
        sys.exit()
    return fdat

def spectra_molecules_dilute(pars: Parameters, w_meas, t_meas, s, test=False, recalc=True):
    # print(recalc)
    tic() #Spectrum has to be recalculated all the time (rescale_mole_fraction does not recalculate broadening)
    # print({key: pars[key].value for key in pars.keys()})

    if sum([par.vary for nam, par in pars.items() if nam.startswith('c_')]) != 0 and recalc: #Only recalculate is any concentration is varied, and recalc is true. 
        s = radis.calc_spectrum(mole_fraction={nam[2::]: par.value for nam, par in pars.items() if nam.startswith('c_')},
                                       pressure=pars["pressure"].value, Tgas=pars["temperature"].value,
                                       path_length=pars["pathlength"].value, diluent='air', medium='air',
                                       wavenum_min=np.amin(w_meas), wavenum_max=np.amax(w_meas), wstep=pars["calcres"].value if ('calcres'in pars.keys()) else 'auto',
                                       warnings={"AccuracyError": "ignore"})
    # elif not recalc: #It does need to be rescaled if the recalculation is turned off!
        # s.rescale_mole_fraction() -> how to do for multiple
        # print(dir(s))
        # pprint(dir(s))
        # s.print_conditions()
        # print(s.populations)
        # print(s.get_conditions())
        # print(s.get_vars())
        # print(s._q.keys())
        # pprint(s)
        # spop = s.populations
        # print(spop)
        # cdic = {nam[2::]: par.value for nam, par in pars.items() if nam.startswith('c_')}
        # print(cdic)
        # s.rescale_mole_fraction(cdic)
        # s.print_conditions()
        # print(radis.spectrum.operations._get_unique_var(s, 'mole_fraction', True))
        # print(s.get('species'))
        # sys.exit()
        # print(s.utils.PHYSICAL_PARAMS)
        # print(s.utils())
        # print(s._get_items('PHYSICAL_PARAMS'))
        # PHYSICAL_PARAMS = ['species', 'wavenum_max', 'wavenum_min', 'mole_fraction', 'isotope', 'state', 'path_length', 'medium', 'self_absorption', 'slit_function_base', 'pressure', 'wavelength_min', 'wavelength_max', 'Telec', 'Tvib', 'Trot', 'Tgas', 'vib_distribution', 'rot_distribution', 'overpopulation', 'thermal_equilibrium']
        # print(s['CO2'])
        # s.rescale_mole_fraction
        # sys.exit()

    s.apply_slit(pars["slitsize"].value, unit="cm-1", norm_by="area", inplace=True, shape="gaussian")    # Apply experimental slit (broadening coefficient term) to spectra object
    w_temp, t_temp = s.get("transmittance")     # Get necessary list with wavenumbers and transmission
    spec_new = radis.Spectrum({"wavenumber": pars["k0"].value + pars["k1"].value*w_temp, "transmittance": t_temp}, wunit='cm-1',
                              units={"transmittance": ""}, conditions={"path_length": pars["pathlength"].value})     # Re-create spectrum objectw

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spec_new.resample(w_meas, inplace=True, energy_threshold=2.0)     # Resample spectrum object onto experimental spectrum, to be able to compare them

    w_new, t_new = spec_new.get("transmittance")     # Get necessary list with wavenumbers and transmission
    t_new[np.isnan(t_new)] = 1.0     # if any non-numbers exist within the spectra, change these into a no-molecule zone (transmission = 1)
    t_new[t_new==0] = 1E-12

    print("Function call: {0:07.3f} ms".format(toc()*1.0E+3))
    if test:
        return t_new
    else:
        return (-np.log10(t_new) + np.log10(t_meas)) ** 2 #The fit residuals are with respect to the absorbance to have a better recognition that signal happens at high absorbances. 
        # # return (t_meas - t_new) **2
        # return 0.5*(-np.log10(t_new) + np.log10(t_meas)) ** 2 + 0.5*(t_meas - t_new) **2

def spectra_molecules_c(pars: Parameters, w_meas, t_meas, s, test=False):
    """
    This fuction is used in order to fit a calculated spectra to the gained experimental data. This formula can only be
    used when one wants to fit a single molecule. Because of the way scipy.curve_fit() works, this function is repeated
    below, but with more c's added, so more molecules can be fitted at the same time. Also, due to the way
    scipy.curve_fit() works, we make use of a storage-formula.

    :param w: wavenumber range needed to match simulated with experimental data
    :param c1: mole fraction of the molecule one wants to fit
    :param slit_size: Size of the needed slit to match simulated with experimental data
    :param k0: The imported k0, which is the offset from 0
    :param k1: The imported k1, which is a linear change in the x-axis
    :return: return fitted transmittance
    """
    tic()
    spectra = []    
    if pars["temperature"].vary: 
        for mol in s.keys(): #If the temperature is varied the spectrum is freshly calculated
            s[mol] = GFL.spectrum_in_air_creator(mol=mol, mf=pars["c_" + mol].value,
                                                 pres=pars["pressure"].value, temp=pars["temperature"].value,
                                                 path_l=pars["pathlength"].value,
                                                 wl_min=np.amin(w_meas), wl_max=np.amax(w_meas), step=float(meta["calcres"]) if ('calcres'in meta.keys()) else 'auto')
            spectra.append(s[mol])
    else: #Else the mole fractions are simply rescaled. 
        for mol in s.keys():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                spectra.append(s[mol].rescale_mole_fraction(pars['c_' + mol].value))

    with contextlib.redirect_stdout(io.StringIO()) as f:            
        spec_new = radis.MergeSlabs(*spectra, out="transparent")

    spec_new.apply_slit(pars["slitsize"].value, unit="cm-1", norm_by="area", inplace=True, shape="gaussian")    # Apply experimental slit (broadening coefficient term) to spectra object
    w_temp, t_temp = spec_new.get("transmittance")     # Get necessary list with wavenumbers and transmission
    spec_new = radis.Spectrum({"wavenumber": pars["k0"].value + pars["k1"].value*w_temp, "transmittance": t_temp}, wunit='cm-1',
                              units={"transmittance": ""}, conditions={"path_length": pars["pathlength"].value})     # Re-create spectrum objectw

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spec_new.resample(w_meas, inplace=True, energy_threshold=2.0)     # Resample spectrum object onto experimental spectrum, to be able to compare them

    w_new, t_new = spec_new.get("transmittance")     # Get necessary list with wavenumbers and transmission
    t_new[np.isnan(t_new)] = 1.0     # if any non-numbers exist within the spectra, change these into a no-molecule zone (transmission = 1)
    t_new[t_new==0] = 1E-12

    print("Function call: {0:07.3f} ms".format(toc()*1.0E+3))
    if test:
        return t_new
    else:
        return (-np.log10(t_new) + np.log10(t_meas)) ** 2 #The fit residuals are with respect to the absorbance to have a better recognition that signal happens at high absorbances. 
        # # return (t_meas - t_new) **2
        # return 0.5*(-np.log10(t_new) + np.log10(t_meas)) ** 2 + 0.5*(t_meas - t_new) **2

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def make_plot_cols(numseries):
    if numseries <= 3:
        return ['blue', 'red', 'black']
    if numseries == 4:
        return ['blue', 'red', 'orange', 'black']
    if numseries == 5:
        return ['blue', 'deepskyblue', 'red', 'orange', 'black']
    if numseries == 6:
        return ['blue', 'deepskyblue', 'red', 'orange', 'gray', 'black']
    if numseries == 7:
        return ['blueviolet', 'blue', 'deepskyblue', 'red', 'orange', 'gray', 'black']
    if numseries > 7: #Thanks to the internet: http://stackoverflow.com/questions/8931268/using-colormaps-to-set-color-of-line-in-matplotlib
        jet = plt.get_cmap('jet') 
        cNorm  = colors.Normalize(vmin=0, vmax=numseries-1)
        scalarMap = cm.ScalarMappable(norm=cNorm, cmap=jet)
        cvals = []
        for i in range(0, numseries):
            cvals.append(scalarMap.to_rgba(i))
    return cvals

def main(): 
    """
    This script should read the metadata from the json file it is given as argument, fit the spectra, and then save a name_fit.json output for each spectrum.
    To do:
        Make a nice legend with fit statistics in the plot -> or too much?
        Save data of selected values for all fits in an excel file 

    """
    plt.style.use(os.path.join(os.path.dirname( __file__ ), 'matplotlibrc_Niek.txt'))
    radis.config['MISSING_BROAD_COEF'] = 'air' #Replace missing broadening coefficients by air. 

    with open(sys.argv[1]) as json_data:
        metadata = json.load(json_data)
        json_data.close()

    savelabs = []
    for meta in metadata["fit_list"]:
        savelabs += [key for key in meta.keys()]
        fmol = meta["molecules"].replace(' ', '').split(',')
        savelabs += ['c_' + mol for mol in fmol]
    savelabs = sorted(list(set(savelabs)))
    savedata = [] #savelabs
    savedata.append(savelabs)

    for meta in metadata["fit_list"]:
        
        pprint(meta)

        if 'plotname' in meta.keys():
            pnam = '_' + meta["plotname"]
        else:
            pnam = ''

        measdata = read_FTIR_measurement(meta["filename"])
        fitrange = (measdata[:,0] > float(meta["wmin"]))*(measdata[:,0] < float(meta["wmax"]))

        if 'corrbase' in meta.keys() and meta["corrbase"].lower() not in ['0', 'no', 'false']: #Baseline correction requested.
            basetest = (measdata[:,0] > float(meta["wmin"]))*(measdata[:,0] < float(meta["wmax"]))
            specderi = savgol_filter(measdata[:,1], window_length=12, polyorder=6, deriv=1, delta=(measdata[1,0] - measdata[0,0]), cval=1.0)
            basetest *= (np.abs(specderi) < 0.1) #Parts with a large derivative are not part of the baseline. 
            temptest  = np.abs(measdata[:,1] - np.median(measdata[basetest,1])) < 1.0*np.std(measdata[basetest,1]) 
            pfit = np.polyfit(measdata[basetest*temptest,0], measdata[basetest*temptest,1], deg=1) #Fit upper band for initial guess. 
            basetest *= np.abs(measdata[:,1] - (pfit[0]*measdata[:,0] + pfit[1])) < 0.5*np.std(measdata[basetest,1]) 
            pfit = np.polyfit(measdata[basetest*temptest,0], measdata[basetest*temptest,1], deg=1) #Then fit closer band to fitter baseline. 
            # print(pfit)

        if 'tranlimit' in meta.keys():
            fitrange *= (measdata[:,1] > float(meta['tranlimit']))

        if 'plotpars'in meta.keys():
            ppar = meta["plotpars"].replace(' ', '').split(',')
        else:
            ppar = []

        if 'measurement' in ppar or 'meas' in ppar:
            fig, ax = plt.subplots() 
            plt.title(os.path.split(meta["filename"])[1], fontsize=12)
            plt.xlabel('Wavenumber (cm$^{-1}$)', fontsize=20)
            plt.ylabel('Transmittance', fontsize=20)
            plt.plot(measdata[:,0], measdata[:,1], linestyle='-', marker='none', color='gray')                
            # plt.plot(measdata[urang,0], measdata[urang,1], linestyle='-', marker='none', color='gray')                            
            plt.plot(measdata[fitrange,0], measdata[fitrange,1], linestyle='-', marker='none', color='blue')   
            if 'corrbase' in meta.keys() and meta["corrbase"].lower() not in ['0', 'no', 'false']: 
                plt.plot(measdata[:,0], pfit[0]*measdata[:,0] + pfit[1], linestyle='-', marker='none', color='red')        
                plt.plot(measdata[basetest,0], measdata[basetest,1], linestyle='none', marker='o', markersize=1, markeredgecolor='orange', color='orange')        
            plt.xlim(4000.0, 500.0)
            plt.ylim(0.0, 1.05)    
            plt.savefig(meta["filename"].split('.')[0] + '_Meas' + pnam + '.png', transparent='False')
            plt.savefig(meta["filename"].split('.')[0] + '_Meas' + pnam + '.pdf', transparent='False')                
            if "skipplot" in meta.keys() and meta["skipplot"].lower() not in ['0', 'no', 'false']:
                plt.close()
            else:
                plt.show()  

        if 'corrbase' in meta.keys() and meta["corrbase"].lower() not in ['0', 'no', 'false']: #Baseline correction requested. -> cut off 50 % as signal. 
            measdata[:,1] -= (pfit[0]*measdata[:,0] + pfit[1]) - 1.0 #Tranmittance should be 1 on baseline!            

        fmol = meta["molecules"].replace(' ', '').split(',')
        molf = str(meta["molfinit"]).replace(' ', '').split(',')

        if 'fixpars'in meta.keys():
            fixp = meta["fixpars"].replace(' ', '').split(',')
        else:
            fixp = []

        fpar = Parameters()
        fpar.add('k0', value=float(meta["k0"]), min=-5, max=5, vary=not "k0" in fixp)
        fpar.add('k1', value=float(meta["k1"]), vary=not "k1"in fixp)
        fpar.add('slitsize', value=float(meta["slitsize"]), vary=not "slitsize"in fixp)
        fpar.add('temperature', value=float(meta["temperature"]), vary=not "temperature"in fixp)
        fpar.add('pressure', value=float(meta["pressure"]), vary=not "pressure"in fixp)
        fpar.add('pathlength', value=float(meta["pathlength"]), vary=not "pathlength" in fixp)
        if 'calcres' in meta.keys():
            fpar.add('calcres', value=float(meta["calcres"]), vary=not "calcres" in fixp)
        for i, mol in enumerate(fmol):
            fpar.add('c_' + mol, value=np.amax([float(molf[i]), 1.0E-9]), min=1.0E-9, max=2, vary=not 'c_' + mol in fixp) 


        if not 'method'in meta.keys() or meta['method'].lower() in ['old', 'default', 'opt']:
            specdict = {}
            for i, mol in enumerate(fmol):
                specdict[mol] = GFL.spectrum_in_air_creator(mol=mol, mf=np.amax([float(molf[i]), 1.0E-9]), #Set 0 mole fraction to 1 ppb to avoid Radis crashing. 
                                                            pres=float(meta["pressure"]), temp=float(meta["temperature"]),
                                                            path_l=float(meta["pathlength"]),
                                                            wl_min=float(meta["wmin"]), wl_max=float(meta["wmax"]), step=float(meta["calcres"]) if ('calcres'in meta.keys()) else 'auto')
                # specdict[mol].print_conditions()
            # sys.exit()
            tic()
            out = minimize(spectra_molecules_c, fpar, args=(measdata[fitrange,0], measdata[fitrange,1], specdict), method='leastq')#, max_nfev=10)#, max_nfev=1000)
            ffit = spectra_molecules_c(out.params, measdata[:,0], measdata[:,1], specdict, test=True)            

            if meta['method'].lower() in ['opt']:  #Not possible to recalculate broadening coefficients every n steps. -> one cannot rescale the mole fraction of a combined spectrum!
                # print("Prefit with static broadening coefficients: {0:07.3f}s".format(toc()))
                # print({nam[2::]: par.value for nam, par in fpar.items() if nam.startswith('c_')})
                # npar = 
                # print(out.params)
                # print(out.params.items())
                # print(out.params.keys())
                # print(out.params['k0'])
                # print(out.params['k0'].value)
                # print(out.params.items()['k0'])

                # print({key[2::]: out.params[key].value for key in out.params.keys() if key.startswith('c_')})                
                # dict_keys(['k0', 'k1', 'slitsize', 'temperature', 'pressure', 'pathlength', 'calcres', 'c_CH4', 'c_C2H2', 'c_HCN', 'c_C2H4', 'c_C2H6'])

                # for par in out.params: #Overwrite parameters with fit results
                # tempdict[par] = out.params[par].value    
                # sys.exit()
                fitsspec = radis.calc_spectrum(mole_fraction={key[2::]: out.params[key].value for key in out.params.keys() if key.startswith('c_')},
                                               pressure=out.params["pressure"].value, Tgas=out.params["temperature"].value,
                                               path_length=out.params["pathlength"].value, diluent='air', medium='air',
                                               wavenum_min=float(meta["wmin"]), wavenum_max=float(meta["wmax"]), wstep=out.params["calcres"].value if ('calcres' in meta.keys()) else 'auto',
                                               export_populations='vib')
                # tic()
                out = minimize(spectra_molecules_dilute, copy.deepcopy(out.params), args=(measdata[fitrange,0], measdata[fitrange,1], fitsspec), method='leastq')#, max_nfev=10)#, max_nfev=10)#, max_nfev=1000)  kws={'recalc':False}
                ffit = spectra_molecules_dilute(out.params, measdata[:,0], measdata[:,1], fitsspec, test=True)

                # print(out)
                # print(fit_report(out))
                # # print(out.status)
                # print(out.aborted) #True if maxfev is hit. 
                # print(out.params)
                # print(fpar)
                # print(out.nfev)
                
                # # sys.exit()

            
            # sys.exit()                                                      
        elif meta['method'].lower() in ['full']: #This recalculates the broadening coefficients every step of the fit.   
            fitsspec = radis.calc_spectrum(mole_fraction={nam[2::]: par.value for nam, par in fpar.items() if nam.startswith('c_')},
                                           pressure=float(meta["pressure"]), Tgas=float(meta["temperature"]),
                                           path_length=float(meta["pathlength"]), diluent='air', medium='air',
                                           wavenum_min=float(meta["wmin"]), wavenum_max=float(meta["wmax"]), wstep=float(meta["calcres"]) if ('calcres'in meta.keys()) else 'auto',
                                           export_lines=True)
                                           # warnings={"AccuracyError": "ignore"}) #float(meta["calcres"])
            # pprint(dir(fitsspec))
            # fitsspec.print_conditions()
            # # fitsspec.print_perf_profile()
            # print(fitsspec.get_name())
            # print(fitsspec.get_vars())
            # lidf = fitsspec.lines
            # print(lidf)
            #  # hwhm_lorentz  hwhm_gauss -> hwhm_lorentz is pressure broadening, 0.0003 - 0.0006 cm-1
            # sys.exit()
            tic()
            out = minimize(spectra_molecules_dilute, fpar, args=(measdata[fitrange,0], measdata[fitrange,1], fitsspec), method='leastq')#, max_nfev=10)#, max_nfev=1000)
            ffit = spectra_molecules_dilute(out.params, measdata[:,0], measdata[:,1], fitsspec, test=True)

        # print(testdict)
        # print(s2.get_conditions()['diluents'])        
        # sys.exit()

            # fpar.add('c_' + mol, value=np.amax([float(molf[i]), 1.0E-9]), min=1.0E-9, max=2, vary=not 'c_' + mol in fixp) #1 ppb = 0 ppb
            # print(specdict[mol].get_conditions()['diluents'])
        # sys.exit()
        # tic()
        # out = minimize(spectra_molecules_c, fpar, args=(measdata[fitrange,0], measdata[fitrange,1], specdict), method='leastq')#, max_nfev=10)#, max_nfev=1000)
        # out = minimize(spectra_molecules_dilute, fpar, args=(measdata[fitrange,0], measdata[fitrange,1], fitsspec), method='leastq')#, max_nfev=10)#, max_nfev=1000)
        # ffit = spectra_molecules_c(out.params, measdata[:,0], measdata[:,1], specdict, test=True)
        print("Total fitting time: {0:07.3f}s".format(toc()))
        print(fit_report(out))
        # print(out.params['c_CH4'])
        # print(out.params['c_CH4'].value)
        # sys.exit()

        if 'fit' in ppar:      
            fig, ax = plt.subplots(2, sharex=True, height_ratios=np.array([3.0, 1.0])) 
            ax[0].set_title(os.path.split(meta["filename"])[1], fontsize=12)

            ax[0].set(xlabel='', ylabel='Transmittance')            
            ax[1].set(xlabel='Wavenumber (cm$^{-1}$)', ylabel='Meas - fit (%)') 
            ax[0].plot(measdata[:,0], measdata[:,1], linestyle='-', marker='none', color='gray')    
            ax[0].plot(measdata[fitrange,0], measdata[fitrange,1], linestyle='-', marker='none', color='blue', label='Measurement')
            ax[0].plot(measdata[:,0], ffit, linestyle='-', marker='none', color='orange', label='Fit')#, $\sigma$ = ' + '{0:04.2f}'.format(out.params['w_g'].value) + ' cm$^{-1}$')

            hand, labl = ax[0].get_legend_handles_labels()
            ax[0].legend(hand, labl, loc='best', numpoints=1, prop={'size':14}, ncol=1)

            ax[1].plot(measdata[:,0], (measdata[:,1] - ffit)*100.0, linestyle='-', marker='none', color='gray')
            ax[1].plot(measdata[fitrange,0], (measdata[fitrange,1] - ffit[fitrange])*100.0, linestyle='-', marker='none', color='black')
            ax[0].set_xlim(float(meta["wmax"]), float(meta["wmin"]))
            ax[1].set_xlim(float(meta["wmax"]), float(meta["wmin"]))
            ax[0].set_ylim(0.0, 1.05)    
            ax[1].set_ylim(-10.0, +10.0)                
            plt.savefig(meta["filename"].split('.')[0] + '_Fit' + pnam + '.png', transparent='False')
            plt.savefig(meta["filename"].split('.')[0] + '_Fit' + pnam + '.pdf', transparent='False')        
            if "skipplot" in meta.keys() and meta["skipplot"].lower() not in ['0', 'no', 'false']:
                plt.close()
            else:
                plt.show()  

        if 'fitA' in ppar:
            fig, ax = plt.subplots(2, sharex=True, height_ratios=np.array([3.0, 1.0])) 
            ax[0].set_title(os.path.split(meta["filename"])[1], fontsize=12)

            ax[0].set(xlabel='', ylabel='Absorbance')            
            ax[1].set(xlabel='Wavenumber (cm$^{-1}$)', ylabel='Meas - fit (%)') 
            ax[0].plot(measdata[:,0], GFL.tr_to_ab(measdata[:,1]), linestyle='-', linewidth=1, marker='none', color='gray')    
            ax[0].plot(measdata[fitrange,0], GFL.tr_to_ab(measdata[fitrange,1]), linestyle='-', marker='none', color='black', label='Measurement')

            pcol = make_plot_cols(len(fmol)+2)
            for i, mol in enumerate(fmol):
                tempspec = GFL.spectrum_in_air_creator(mol=mol, mf=out.params['c_' + mol].value,
                                                       pres=out.params['pressure'].value, temp=out.params['temperature'].value,
                                                       path_l=out.params['pathlength'].value,
                                                       wl_min=float(meta["wmin"]), wl_max=float(meta["wmax"]), step=out.params['calcres'])                
                tempspec.apply_slit(out.params['slitsize'].value, unit="cm-1", norm_by="area", inplace=True, shape="gaussian")
                w_temp, t_temp = tempspec.get("transmittance")
                ax[0].plot(out.params["k0"].value + out.params["k1"].value*w_temp, GFL.tr_to_ab(t_temp), linestyle='-', marker='none', color=pcol[i], label=re.sub('([0-9]+)', '$_{\g<0>}$', mol))
            ax[0].plot(measdata[:,0], GFL.tr_to_ab(ffit), linestyle='-', marker='none', linewidth=1, color='gray', label='Fit')#, $\sigma$ = ' + '{0:04.2f}'.format(out.params['w_g'].value) + ' cm$^{-1}$')

            hand, labl = ax[0].get_legend_handles_labels()
            ax[0].legend(hand, labl, loc='best', numpoints=1, prop={'size':14}, ncol=1)

            ax[1].plot(measdata[:,0], (GFL.tr_to_ab(measdata[:,1]) - GFL.tr_to_ab(ffit))*100.0, linestyle='-', marker='none', color='gray')
            ax[1].plot(measdata[fitrange,0], (GFL.tr_to_ab(measdata[fitrange,1]) - GFL.tr_to_ab(ffit[fitrange]))*100.0, linestyle='-', marker='none', color='black')
            ax[0].set_xlim(float(meta["wmax"]), float(meta["wmin"]))
            ax[1].set_xlim(float(meta["wmax"]), float(meta["wmin"]))
            ax[0].set_ylim(0.0, ax[0].get_ylim()[1])    
            ax[1].set_ylim(-10.0, +10.0)                
            plt.savefig(meta["filename"].split('.')[0] + '_FitA' + pnam + '.png', transparent='False')
            plt.savefig(meta["filename"].split('.')[0] + '_FitA' + pnam + '.pdf', transparent='False')        
            if "skipplot" in meta.keys() and meta["skipplot"].lower() not in ['0', 'no', 'false']:
                plt.close()
            else:
                plt.show()              

        if 'noise' in ppar:
            tres = measdata[fitrange,1] - ffit[fitrange]
            ares = -np.log10(ffit[fitrange]) + np.log10(measdata[fitrange,1])

            this, tedg = np.histogram(tres, bins=100, range=(-0.003, +0.003), density=True)
            ahis, aedg = np.histogram(ares, bins=100, range=(-0.010, +0.010), density=True)
            tbin, abin = running_mean(tedg, 2), running_mean(aedg, 2)

            def gfit(x, sig):
                return (1.0/(sig*np.sqrt(2.0*np.pi)))*np.exp(-0.5*np.square(x/sig))

            topt, tcov = curve_fit(gfit, tbin, this)
            aopt, acov = curve_fit(gfit, abin, ahis)            

            fig, ax = plt.subplots() 
            plt.title(os.path.split(meta["filename"])[1], fontsize=12)
            plt.xlabel('Normalized residual', fontsize=20)
            plt.ylabel('Relative occurrence', fontsize=20)
            plt.plot(abin/aopt[0], ahis*aopt[0]*np.sqrt(2.0*np.pi), linestyle='-', marker='none', color='blue', label='Absorbance')                
            plt.plot(tbin/aopt[0], this*aopt[0]*np.sqrt(2.0*np.pi), linestyle='-', marker='none', color='red', label='Transmittance')   
            plt.plot(np.linspace(-10.0, 10.0, num=250), np.sqrt(2.0*np.pi)*gfit(np.linspace(-10.0, 10.0, num=250), 1.0), linestyle='-', marker='none', color='black', label='Fit')   
            plt.xlim(-4.0, 4.0)
            plt.ylim(0.0, 1.5)
            plt.legend(loc='best', numpoints=1, prop={'size':14}, ncol=1)                    
            # plt.savefig(meta["filename"].split('.')[0] + '_Meas' + pnam + '.png', transparent='False')
            # plt.savefig(meta["filename"].split('.')[0] + '_Meas' + pnam + '.pdf', transparent='False')  
            if "skipplot" in meta.keys() and meta["skipplot"].lower() not in ['0', 'no', 'false']:
                plt.close()
            else:
                plt.show()                       

        tempdict = {}
        for key in meta: #Copy input parameters for trackability
            tempdict[key] = meta[key]
        for par in out.params: #Overwrite parameters with fit results
            tempdict[par] = out.params[par].value    
        # tempdict["molfinit"] = ','.join(['{0:07.5f}'.format(tempdict["c_" + mol]) for mol in fmol]) #Write fit parameters as initial values so that the output file can be directly used as input for an eventual refinement of the fit. 
        if 'corrbase' in meta.keys() and meta["corrbase"].lower() not in ['0', 'no', 'false']: #Baseline correction requested. -> cut off 50 % as signal.         
            tempdict["basefit"] = ','.join(['{0:07.5f}'.format(val) for val in pfit])

        mind = ['nfev', 'covar', 'nvarys', 'ndata', 'nfree', 'aborted', 'success', 'errorbars', 'ier', 'message', 'method', 'chisqr', 'redchi', 'aic', 'bic', 'params', 'var_names', 'init_vals', 'init_values', 'call_kws']
        for a in dir(out):
            if not a.startswith('_') and a in mind:
                if a == 'params':
                    tempdict[a] = out.params.dumps()
                else:    
                    fout = out.__getattribute__(a)
                    if type(fout) == np.ndarray:
                        tempdict[a] = out.__getattribute__(a).tolist()                        
                    else:
                        tempdict[a] = out.__getattribute__(a)
                
        pprint(tempdict)

        with open(meta["filename"].split('.')[0] + '_Fit' + pnam + '.json', 'w') as fp:
            json.dump(tempdict, fp, indent=4)

        templist = []
        for key in savelabs:
            if key in tempdict.keys():
                templist.append(tempdict[key])
            else:
                templist.append(0.0)
        savedata.append(templist)

    with open(sys.argv[1].split('.')[0] + '.csv', mode='w', newline='') as file:
        csv_writer = csv.writer(file) 
        csv_writer.writerows(savedata)

if __name__ == '__main__':
    main()