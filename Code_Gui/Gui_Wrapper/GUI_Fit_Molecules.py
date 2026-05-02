import csv                as csv
import json               as json
import numpy              as np
import os                 as os
import sys                as sys
import matplotlib.pyplot  as plt

from brukeropusreader import read_file

from lmfit import minimize, Parameters, fit_report
from radis import Spectrum, MergeSlabs

import io                 as io #output management (suppress uneven sampling and mole fraction warnings and radis output)
import contextlib         as contextlib
import warnings           as warnings

from ttictoc import tic,toc #Timing and beautification of output. 
from pprint import pprint

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname( __file__ ))))
import Code_Gui.Gui_General_Code.General_Functions_Library as GFL
import Code_Gui.Gui_General_Code.Gas_Mixtures_Spectra_Library as GMSL

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
        sys.exit()        
    else:
        print("File extension " + fext + " not implemented, exiting")
        sys.exit()
    return fdat

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
                                                 wl_min=np.amin(w_meas), wl_max=np.amax(w_meas), step=pars["calcres"].value)
            spectra.append(s[mol])
    else: #Else the mole fractions are simply rescaled. 
        for mol in s.keys():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                spectra.append(s[mol].rescale_mole_fraction(pars['c_' + mol].value))

    with contextlib.redirect_stdout(io.StringIO()) as f:            
        spec_new = MergeSlabs(*spectra, out="transparent")

    spec_new.apply_slit(pars["slitsize"].value, unit="cm-1", norm_by="area", inplace=True, shape="gaussian")    # Apply experimental slit (broadening coefficient term) to spectra object
    w_temp, t_temp = spec_new.get("transmittance")     # Get necessary list with wavenumbers and transmission
    spec_new = Spectrum({"wavenumber": pars["k0"].value + pars["k1"].value*w_temp, "transmittance": t_temp}, wunit='cm-1',
                         units={"transmittance": ""}, conditions={"path_length": pars["pathlength"].value})     # Re-create spectrum objectw

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spec_new.resample(w_meas, inplace=True, energy_threshold=0.025)     # Resample spectrum object onto experimental spectrum, to be able to compare them

    w_new, t_new = spec_new.get("transmittance")     # Get necessary list with wavenumbers and transmission
    t_new[np.isnan(t_new)] = 1.0     # if any non-numbers exist within the spectra, change these into a no-molecule zone (transmission = 1)
    t_new[t_new==0] = 1E-12

    print("Function call: {0:07.3f} ms".format(toc()*1.0E+3))
    if test:
        return t_new
    else:
        return (-np.log10(t_new) + np.log10(t_meas)) ** 2 #The fit residuals are with respect to the absorbance to have a better recognition that signal happens at high absorbances. 

def main(): 
    """
    This script should read the metadata from the json file it is given as argument, fit the spectra, and then save a name_fit.json output for each spectrum.
    To do:
        Make a nice legend with fit statistics in the plot -> or too much?
        Save data of selected values for all fits in an excel file 

    """
    plt.style.use(os.path.join(os.path.dirname( __file__ ), 'matplotlibrc_Niek.txt'))

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

        measdata = read_FTIR_measurement(meta["filename"])
        fitrange = (measdata[:,0] > float(meta["wmin"]))*(measdata[:,0] < float(meta["wmax"]))

        if 'plotpars'in meta.keys():
            ppar = meta["plotpars"].replace(' ', '').split(',')
        else:
            ppar = []

        if 'measurement' in ppar:
            lrang = (measdata[:,0] < float(meta["wmin"]))
            urang = (measdata[:,0] > float(meta["wmax"]))
            fig, ax = plt.subplots() 
            plt.title(os.path.split(meta["filename"])[1], fontsize=14)
            plt.xlabel('Wavenumber (cm$^{-1}$)', fontsize=20)
            plt.ylabel('Transmittance', fontsize=20)
            plt.plot(measdata[lrang,0], measdata[lrang,1], linestyle='-', marker='none', color='gray')                
            plt.plot(measdata[urang,0], measdata[urang,1], linestyle='-', marker='none', color='gray')                            
            plt.plot(measdata[fitrange,0], measdata[fitrange,1], linestyle='-', marker='none', color='blue')        
            plt.xlim(4000.0, 1000.0)
            plt.ylim(0.0, 1.05)     
            plt.show()

        fmol = meta["molecules"].replace(' ', '').split(',')
        molf = meta["molfinit"].replace(' ', '').split(',')

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
        fpar.add('calcres', value=float(meta["calcres"]), vary=not "calcres" in fixp)

        specdict = {}
        for i, mol in enumerate(fmol):
            specdict[mol] = GFL.spectrum_in_air_creator(mol=mol, mf=float(molf[i]),
                                                        pres=float(meta["pressure"]), temp=float(meta["temperature"]),
                                                        path_l=float(meta["pathlength"]),
                                                        wl_min=float(meta["wmin"]), wl_max=float(meta["wmax"]), step=float(meta["calcres"]))
            fpar.add('c_' + mol, value=float(molf[i]), min=1.0E-12, max=2, vary=not 'c_' + mol in fixp)
        
        tic()
        out = minimize(spectra_molecules_c, fpar, args=(measdata[fitrange,0], measdata[fitrange,1], specdict), method='leastq')#, max_nfev=1000)
        print("Total fitting time: {0:07.3f}s".format(toc()))
        print(fit_report(out))

        ffit = spectra_molecules_c(out.params, measdata[fitrange,0], measdata[fitrange,1], specdict, test=True)

        if 'fit' in ppar:      
            fig, ax = plt.subplots(2, sharex=True, height_ratios=np.array([3.0, 1.0])) 
            ax[0].set_title(os.path.split(meta["filename"])[1], fontsize=14)

            ax[0].set(xlabel='', ylabel='Transmittance')            
            ax[1].set(xlabel='Wavenumber (cm$^{-1}$)', ylabel='Meas - fit (%)')    
            ax[0].plot(measdata[fitrange,0], measdata[fitrange,1], linestyle='-', marker='none', color='blue', label='Measurement')
            ax[0].plot(measdata[fitrange,0], ffit, linestyle='-', marker='none', color='orange', label='Fit')#, $\sigma$ = ' + '{0:04.2f}'.format(out.params['w_g'].value) + ' cm$^{-1}$')

            hand, labl = ax[0].get_legend_handles_labels()
            ax[0].legend(hand, labl, loc='lower right', numpoints=1, prop={'size':14}, ncol=1)

            ax[1].plot(measdata[fitrange,0], (measdata[fitrange,1] - ffit)*100.0, linestyle='-', marker='none', color='black')
            ax[0].set_xlim(float(meta["wmax"]), float(meta["wmin"]))
            ax[1].set_xlim(float(meta["wmax"]), float(meta["wmin"]))
            ax[0].set_ylim(0.0, 1.05)    
            ax[1].set_ylim(-10.0, +10.0)                
            plt.savefig(meta["filename"].split('.')[0] + '_Fit.png', transparent='False')
            plt.savefig(meta["filename"].split('.')[0] + '_Fit.pdf', transparent='False')        
            plt.show()  

        tempdict = {}
        for key in meta: #Copy input parameters for trackability
            tempdict[key] = meta[key]
        for par in out.params: #Overwrite parameters with fit results
            tempdict[par] = out.params[par].value    
        tempdict["molfinit"] = ','.join(['{0:07.5f}'.format(tempdict["c_" + mol]) for mol in fmol]) #Write fit parameters as initial values so that the output file can be directly used as input for an eventual refinement of the fit. 

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

        with open(meta["filename"].split('.')[0] + '_Fit.json', 'w') as fp:
            json.dump(tempdict, fp, indent=4)

        templist = []
        for key in savelabs:
            if key in tempdict.keys():
                templist.append(tempdict[key])
            else:
                templist.append(0.0)
        savedata.append(templist)

    with open(sys.argv[1].replace('.json', '.csv'), mode='w', newline='') as file:
        csv_writer = csv.writer(file) 
        csv_writer.writerows(savedata)

if __name__ == '__main__':
    main()