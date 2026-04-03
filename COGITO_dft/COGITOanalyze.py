#!/usr/bin/env python3

import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# --- This python file will analyze output files from COGITO to access the run quality and if neccesssary, make recommendations about improving run. ---
# -- Each function here analyzes a different file. ---

def analyze_orb_converg_info(dir: str = '',tag: str = '',make_change_plot: bool = True, show_fig: bool=False, group_same_elem: bool = True):
    """
    This function serves to analyze the orbital convergence of the COGITO run using the 'orb_converg_info.json' created
    when save_orb_converg_info=True in COGITO.generate_TBmodel(). The function checks the convergence for errors or warnings
    and recommends adjustments to the COGITO inputs or VASP NBANDS. The function also plots the orbital radius changes for different elements.

    Args:
        dir (str): Directory of the 'orb_converg_info' file
        tag (str): Tag appended to 'orb_converg_info' file. Final path is (dir "orb_converg_info" + tag + ".json").
        make_change_plot (bool): Whether to create, save to (dir+"orb_change"+tag+".png"), and show the orbital radius change plot.
        show_fig (bool): Whether to plt.show() the orbital change figure.
        group_same_elem (bool): Whether to group atoms that are of the same element in the orbital change plot.
                    Set to False to observe differences between the same element in varying coordination.
    """
    with open(dir+"orb_converg_info" +tag+".json", "r") as f:
        orb_info = json.load(f)

    orb_type = np.char.split(orb_info["orb_type"],sep="-")
    num_orbs = len(orb_type)
    atm_types = np.zeros(num_orbs,dtype="U2")
    orb_types = np.zeros(num_orbs,dtype="U2")
    atmnum = np.zeros(num_orbs,dtype=int)
    for orb in range(num_orbs):
        atm_types[orb] = orb_type[orb][0]
        orb_types[orb] = orb_type[orb][1][0] # just the first character
        atmnum[orb] = orb_type[orb][2]
    orb_rad = np.array(orb_info["orb_rad"])
    orb_change = np.array(orb_info["orb_change"])*100
    orb_change_periter = np.array(orb_info["orb_change_periter"])
    radial_change_periter = np.array(orb_info["radial_change_periter"])
    bloch_nrmse_iter = np.array(orb_info["bloch_nrmse_iter"])
    bloch_nme_iter = np.array(orb_info["bloch_nme_iter"])
    gaus_fit_error = np.array(orb_info["gaus_fit_error"])


    # first plot orbital percent change by type
    if make_change_plot:
        colors =['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
        hatch_type = ["","//","xx"]
        orb_switch = {}
        orb_switch['s'] = 0
        orb_switch['p'] = 1
        orb_switch['d'] = 2
        uniq_elems,ind,elems = np.unique(atm_types,return_inverse=True,return_index=True)
        # rearange the uniq_elems so that it is in the same order as the elements in cell defintion
        uniq_elems = uniq_elems[np.argsort(ind)]
        for i,elem in enumerate(uniq_elems):
            elems[atm_types==elem] = i
        offset = 0.2
        width = 0.25
        x_label = []
        if group_same_elem:
            uniq = range(len(uniq_elems))
        else:
            uniq = range(np.amax(atmnum)+1)

        for atm in uniq:
            if group_same_elem:
                keep_bool = elems==atm
            else:
                keep_bool = atmnum==atm
            uniq_orbs = np.unique(orb_types[keep_bool])[::-1]
            avg_chang = []
            color = colors[elems[keep_bool][0]]
            hatch = []
            for o in uniq_orbs:
                avg_chang.append(np.average(orb_change[keep_bool][orb_types[keep_bool]==o]))
                hatch.append(hatch_type[orb_switch[o]])
            x = np.arange(0,len(uniq_orbs)) * width + offset
            rects = plt.bar(x,avg_chang,width-0.01,ec="black",color=color, fill=True,hatch=hatch,linestyle="-", linewidth=2)#,label = uniq_orbs)
            plt.bar_label(rects,padding=3,fmt='{:.1f}%')
            x_label.append(np.average(x))

            offset += len(uniq_orbs)*width + 0.2
        plt.bar([-1,-0.8,-0.6], [0,0,0], width - 0.01, ec="black",color="white", fill=True, hatch=["","////","xxxx"], linestyle="-",linewidth=2 ,label = ["s","p","d"])
        offset += -0.2
        plt.ylim(-np.amax(np.abs(orb_change))-1,np.amax(np.abs(orb_change))+1)
        plt.plot([0,offset],[0,0],color="black")
        plt.xlim(0,offset)
        plt.xticks(x_label, uniq_elems,fontsize=12)
        plt.legend(loc='upper left', ncols=3,fontsize=11)
        plt.title("Orbital radius changes from COGITO",fontsize=12)
        plt.savefig(dir+"orb_change"+tag+".png")
        print("Saved the orbital change figure.")
        if show_fig:
            plt.show()
        plt.close()

    # next check errors:
    #print(np.around(radial_change_periter*100,decimals=4))
    #print(np.around(orb_change_periter*100,decimals=4))
    iserror = False
    iswarn = False
    # check whether num_outer should be increased
    if (radial_change_periter[-1] > 0.05).any():
        iserror = True
        print("ERROR:   The radial part is not fully converged with some radial change over 5%: ",radial_change_periter[-1][radial_change_periter[-1] > 0.05])
        print("         To resolve issue, run COGITO generate_TBmodel with a num_outer +2 or +3 higher.")
    elif (radial_change_periter[-1] > 0.01).any():
        iswarn = True
        print("Warning: The radial part may not be fully converged with some radial change over 1%: ",radial_change_periter[-1][radial_change_periter[-1] > 0.01])
        print("         To improve convergence, run COGITO generate_TBmodel with a num_outer +1 or +2 higher.")

    if (np.abs(orb_change_periter[-1]) > 0.05).any():
        iserror = True
        print("ERROR:   The COGITO radii is not fully converged with some orbital radius change over 5%: ", orb_change_periter[-1][radial_change_periter[-1] > 0.05])
        print("         To resolve issue, run COGITO generate_TBmodel with a num_outer +2 or +3 higher.")
    elif (np.abs(orb_change_periter[-1]) > 0.01).any():
        iswarn = True
        print("Warning: The COGITO radii may not be fully converged with some orbital radius change over 1%: ", orb_change_periter[-1][radial_change_periter[-1] > 0.01])
        print("         To improve convergence, run COGITO generate_TBmodel with a num_outer +1 or +2 higher.")

    # check whether num_outer shouldn't be increased because convergence has stopped (this is highly unlikely + very bad (for some terms))
    avg_nrmse = np.average(bloch_nrmse_iter,axis=1)
    avg_radial_chang = np.average(np.abs(radial_change_periter),axis=1)
    avg_radius_chang = np.average(np.abs(orb_change_periter),axis=1)
    #print("avg nrmse per loop:",avg_nrmse)
    #print("avg radial change per loop:",avg_radial_chang)
    #print("avg radius change per loop:",avg_radius_chang)

    if ((avg_radial_chang[1:]-avg_radial_chang[:-1]) > 0).any():
        iserror = True
        print("ERROR:   The change in radial part is larger in later step. This means the convergence procedure isn't working properly.")
        print("         The average change in radial part per loop is: ", avg_radial_chang)
        print("         Solution:")
        print("         Check that this doesn't result from an especially large num_outer (>5) such that COGITO proceeds when already maximally converged. If so, decrease num_outer or ignore error message.")
        print("         No single fix. One could try adding more bands (NBANDS in vasp INCAR), decreasing min_proj (for generate_TBmodel), ")
        print("         or changing orbital initialization (include_excited in generate_TBmodel).")
        print("         If everything else works fine, proceed but interpret results cautiously.")

    if ((avg_radius_chang[1:] - avg_radius_chang[:-1]) > 0).any():
        iserror = True
        print("ERROR:   The change in orbital radius is larger in later step. This means the convergence procedure isn't working properly.")
        print("         The average change in radius part per loop is: ", avg_radius_chang)
        print("         Solution:")
        print("         Check that this doesn't result from an especially large num_outer (>5) such that COGITO proceeds when already maximally converged. If so, decrease num_outer or ignore error message.")
        print("         No single fix. One could try adding more bands (NBANDS in vasp INCAR), decreasing min_proj (for generate_TBmodel), ")
        print("         or changing orbital initialization (include_excited in generate_TBmodel).")
        print("         If everything else works fine, proceed but interpret results cautiously.")

    # Finally check that Bloch nrmse/nme and final gaussian fit errors are reasonably low.
    #print(bloch_nrmse_iter, bloch_nme_iter, gaus_fit_error)
    if (gaus_fit_error > 0.10).any():
        val = np.around(gaus_fit_error[gaus_fit_error > 0.10]*100,decimals = 2)
        iserror = True
        print("ERROR:   The final gaussian fit is more than 10% off of the gaus+exp fit. This could be fine, useless ", val, "is especially large.")
        print("         Check 'fit_cogito.png' for visual difference between gaus+exp fit (dark purple) and final gaus fit (pink). Generate file with 'save_orb_figs=True' in generate_TBmodel()")
        print("         The main solution for this involves tuning hyperparameters for the fitting, which is generally a big pain.")
        print("         Please submit issue to COGITO github repo or email olipemil@umich.edu.")

    avg_nrmse = np.average(bloch_nrmse_iter,axis=1)
    avg_nme = np.average(bloch_nme_iter,axis=1)
    if avg_nrmse[-1] > 0.15:
        iserror = True
        print("ERROR:   The average nmrse between the converged numerical Bloch orbital and the analytical fit COGITO Bloch orbital is above 15%: ", avg_nrmse[-1])
        print("         If spilling and orbital mixing errors sufficiently low, this is not necessary to fix.")
        print("         There are several possible fixes for this. Within the COGITO run, try decreasing min_proj or include_excited in generate_TBmodel(). Or try increasing NBANDS in the vasp INCAR")

    if (bloch_nrmse_iter[-1] > 0.2).any():
        iserror = True
        print("ERROR:   Some of the nmrse between the converged numerical Bloch orbital and the analytical fit COGITO Bloch orbital are above 20%: ",bloch_nrmse_iter[-1][bloch_nrmse_iter[-1] > 0.2])
        print("         Consider removing the following orbitals (likely by reducing include_excited): ", np.array(orb_info["orb_type"])[bloch_nrmse_iter[-1] > 0.2])
        print("         Or if everything else if fine, and these are not high energy state (or you want them). Just continue knowing that these orbitals don't span the DFT wfs as well.")

    if not iserror and not iswarn:
        print("The orbital convergence is within expected range!")

def analyze_spill_error(dir: str='', tag: str=''):
    """
    This function serves to analyze the spillge and orbital mixing of the COGITO run using the 'error_output.txt' created
    when running COGITO.generate_TBmodel(). The function checks that the values are within range or return an error or warning
    with recommended adjustments to the COGITO inputs or VASP NBANDS.

    Args:
        dir (str): Directory of the 'error_output' file
        tag (str): Tag appended to 'error_output' file. Final path is (dir "error_output" + tag + ".txt").
    """
    with open(dir+"error_output"+tag+".txt") as f:
        lines = f.readlines()
    collect_spill = False
    avg_spill = []
    for line in lines:
        line = line.strip()
        if "percent charge spilling" in line:
            charge_spill = float(line.split()[-1][:-1])
        if "maximum band charge spill" in line:
            maxband_charge_spill = float(line.split()[-1][:-1])
        if "maximum charge spill" in line:
            max_charge_spill = float(line.split()[-1][:-1])
        if "average orbital mixing" in line:
            avg_max_orbmix = float(line.split()[-1][:-1])
        if "max orbital mixing" in line:
            max_max_orbmix = float(line.split()[-1][:-1])
        if line.startswith("avg spill per band:"):
            collect_spill = True
            continue
        if collect_spill:
            # stop if line no longer looks like numbers
            if not len(line)==0:
                if not line[0].isdigit() and not line[0] == '.':
                    collect_spill = False
                else:
                    avg_spill.extend(map(float, line.split()))

    avg_spill = np.array(avg_spill)

    iserror = False
    iswarn = False
    # charge spilling error block
    if charge_spill > 3:
        if maxband_charge_spill > 10:
            iserror = True
            print("ERROR:   Charge spilling and largest occupied band spilling are too high at: ", charge_spill, "% and ", maxband_charge_spill,"%")
            print("         This indicates that some high energy state is missing. Probably add more orbitals by increasing include_excited.")
            print("         Also take note of the maximum kpt,band spill: ",max_charge_spill, "%")
            print("         If maximum spill directly above is especially larger than maximum band spill, than the high energy state mixes more at certain k-points.")
        else:
            iserror = True
            print("ERROR:   Charge spilling is too high at: ", charge_spill, "%")
            print("         This could indicate that some states are missing. Possibly increase include_excited. Or try increasing quality in other ways.")
            print("         Also take note of the largest occupied band spill: ", maxband_charge_spill,"% and maximum kpt,band spill: ",max_charge_spill, "%")
            print("         If maximum spill is especially larger than maximum band spill, than the high energy state mixes more at certain k-points.")
    elif charge_spill > 1:
        if maxband_charge_spill > 10:
            iserror = True
            print("ERROR:   Largest occupied band spilling is too high at: ", maxband_charge_spill, "%")
            print("         This means some high energy states are mixing heavily with just the top few bands. Probably add more orbitals by increasing include_excited.")
            print("         The total charge spilling was not too high: ", charge_spill, "%, so the fit could be good enough if you don't care too much about the top few bands.")
            print("         Also take note of the maximum kpt,band spill: ", max_charge_spill, "%")
            print("         If maximum spill directly above is especially larger than maximum band spill, than the high energy state mixes more at certain k-points.")
        else:
            iswarn = True
            print("Warning: Charge spilling is higher than normal at: ", charge_spill, "%")
            print("         While this may not be too bad, consider expanding the orbital basis by increasing include_excited, or changing other parameters.")
    if maxband_charge_spill < 10 and maxband_charge_spill > 3:
        print("Warning: The Largest occupied band spilling is higher than normal at: ", maxband_charge_spill, "%")
        print("         While this could be fine, it's likely that your top valence bands will not be as well-described by COGITO TB.")
        print("         To improve, consider increasing include_excited. Although, this comes with tradeoffs in efficiency, chemical intpretability, worse orbital mixing, etc.")

    # orbital mixing error block
    if avg_max_orbmix > 10:
        iserror = True
        print("ERROR:   The orbital mixing is too high at: ", avg_max_orbmix , "%")
        print("         This can be resolved by setting higher NBANDS in the vasp INCAR (up to 20*numatom). Or reducing high energy orbitals included by decreasing include_excited.")
    elif avg_max_orbmix > 5:
        iswarn = True
        print("Warning: The orbital mixing is higher than normal at: ", avg_max_orbmix , "%")
        print("         This could be improved by setting higher NBANDS in the vasp INCAR (up to 20*numatom). Or reducing high energy orbitals included by decreasing include_excited.")
    elif avg_max_orbmix > 2:
        iswarn = True
        print("Warning: The orbital mixing is a bit higher than normal at: ", avg_max_orbmix , "%")
        print("         This could be improved by setting higher NBANDS in the vasp INCAR (up to 20*numatom) or decreasing min_proj in generate_TBmodel().")

    if not iserror and not iswarn:
        print("The band spillage and orbital mixing is within expected range!")

def analyze_band_error(dir:str='',tag: str=''):
    """
    This function serves to analyze the band interpolation error of the COGITO tight binding run using the 'DFT_band_error.txt' created
    when running COGITO_TB_Model.compare_to_DFT(). The function checks that the values are within range or return an error or warning
    with recommended adjustments to the COGITO inputs or VASP NBANDS.

    Args:
        dir (str): Directory of the 'DFT_band_error' file
        tag (str): Tag appended to 'DFT_band_error' file. Final path is (dir "DFT_band_error" + tag + ".txt").
    """
    with open(dir+"DFT_band_error"+tag+".txt") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if 'band distance in Valence Bands' in line:
            aboveVBM = line.split("eV:")[0][-1]
            if aboveVBM != ' ' and aboveVBM !='0' and aboveVBM !='V': # has another thing
                print("Hey! The DFT band errors seem to have been calculated with a cutoff above the Fermi energy.")
                print("While this is perfectly fine, the cutoffs used for the following warning and error generation will apply differently.")
                print("Please see Table 1 and Figure 7 in the COGITO paper for examples on errors with above Fermi cutoff.")
            band_dist = float(line.split(" ")[-2])
        if 'max error in valence band' in line:
            max_error = float(line.split(" ")[-2])
        if 'average error in Bottom 3' in line:
            lowcond_error = float(line.split(" ")[-2])
    import math
    iserror = False
    iswarn = False
    if math.isnan(band_dist) or math.isnan(max_error) or math.isnan(lowcond_error): # generally if one is nan, all is nan
        print("ERROR:   Catastrophic failure! There is a nan in the error file. This generally happens when the eigenvalues do not converge.")
        print("         It could be that you need to loosen the 'maximum_dist' or 'minimum_value' in COGITO_TB_Model.restrict_params.")
        print("         Or, more generally, something went wrong in the COGITO run. Maybe many not enough NBANDS, maybe the symmetry was wrong, maybe the orbitals included are not ideal, etc.")
        print("         Reference warnings and errors from analyze_orb_converg_info() and analyze_spill_error() for clues on what went wrong.")

    if band_dist > 0.03:
        iserror = True
        print("ERROR:   The band distance of the valence bands is too high at: ",np.around(band_dist*1000,decimals=2),"meV")
        print("         However, if it is not much above 30 meV, the COGITO run seems otherwise good, and you don't mind less accurate band interpolation; ")
        print("         the chemical insight should still work fine.")
        print("Note: For magnetic calculations, an EIGENVAL made from a seperate band structure VASP run won't line up (unless you specify exact final MAGMOM in INCAR), use EIGENVAL from static run.")
    elif band_dist > 0.01:
        iswarn = True
        print("Warning: The band distance is higher than normal at ",np.around(band_dist*1000,decimals=2),"meV")
        print("         Not a big deal unless your priority is good interpolation.")

    if max_error > 0.1:
        iserror = True
        print("ERROR:   The maximum band error in the valence bands is too high at: ",np.around(max_error*1000,decimals=2),"meV")
        print("         However, if it is not much above 100 meV, the COGITO run seems otherwise good, and you don't mind less accurate band interpolation; ")
        print("         the chemical insight should still work fine. UNLESS, there is substainal reordering of bands near the Fermi energy (check figure).")
    elif max_error > 0.03:
        iswarn = True
        print("Warning: The maximum band error is higher than normal at ",np.around(max_error*1000,decimals=2),"meV")
        print("         Not a big deal unless your priority is good interpolation. Perhaps check that no bands switch order near the Fermi energy.")

    if lowcond_error > 0.1:
        iserror = True
        print("ERROR:   The average band error in the lowest 3eV of the conductions is high at: ",np.around(lowcond_error*1000,decimals=2),"meV")
        print("         Mostly relavent for excited state properties or other extensions. (Doesn't directly effect ground state T=0 calculated values.)")
        print("         For better conduction band fits, increase the variables 'low' and 'high' in optimize_band_set() in COGITO.py or increase include_excited.")
    elif lowcond_error > 0.03:
        iswarn = True
        print("Warning: The average band error in the lowest 3eV of the conductions is higher than normal at ",np.around(lowcond_error*1000,decimals=2),"meV")
        print("         Mostly relavent for excited state properties or other extensions. (Doesn't directly effect ground state T=0 calculated values.)")
        print("         For better conduction band fits, increase the variables 'low' and 'high' in optimize_band_set() in COGITO.py or increase include_excited.")

    if iserror:
        print("")
        print("         To reduce errors, follow recommendations from analyze_spill_error().")
        print("         These options include: increasing NBANDS in vasp INCAR, decreasing min_proj, or increasing include_excited (although adding more orbitals generally reduces chemical interpretability).")
    elif iswarn:
        print("")
        print("         To improve fit, follow recommendations from analyze_spill_error().")
        print("         These options include: increasing NBANDS in vasp INCAR, decreasing min_proj, or increasing include_excited (although adding more orbitals generally reduces chemical interpretability).")
    else:
        print("The band interpolation is within expected bounds!")

    #print(band_dist,max_error,lowcond_error)

class Tee: # just to print to terminal and save output to file
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
        return len(data)

    def flush(self):
        for s in self.streams:
            s.flush()

def analyze_all(dir: str = '',tag: str = '',make_change_plot: bool = True, show_fig: bool=False, group_same_elem: bool = True):
    """
    This function serves to analyze the three files to monitor quality of the COGITO run. Each function checks that values
    are within range or returns error or warning messages with recommended adjustments to the COGITO inputs or VASP NBANDS to potentially improve quality.
    The analyze_orb_converg_info() function also plots the orbital radius changes for different elements.

    Args:
        dir (str): Directory of the files
        tag (str): Tag appended to file. Final path is (dir "****" + tag + ".*").
        make_change_plot (bool): Whether to create, save to (dir+"orb_change"+tag+".png"), and show the orbital radius change plot.
        show_fig (bool): Whether to plt.show() the orbital change figure.
        group_same_elem (bool): Whether to group atoms that are of the same element in the orbital change plot.
                    Set to False to observe differences between the same element in varying coordination.
    """
    # make code to try running each one. If it fails just give error message that file wasn't found.

    # Open log file for terminal output
    import sys
    from contextlib import redirect_stdout, redirect_stderr
    log_file = open(dir + "analysis" + tag + ".txt", "w", buffering=1)
    tee_out = Tee(sys.stdout, log_file)
    tee_err = Tee(sys.stderr, log_file)

    with redirect_stdout(tee_out), redirect_stderr(tee_err):
        try:
            analyze_orb_converg_info(dir=dir,tag=tag,make_change_plot=make_change_plot,show_fig=show_fig, group_same_elem=group_same_elem)
        except:
            print("Orbital convergence analysis failed. Check that (dir+'orb_converg_info'+tag+'.json') exists.")
        try:
            analyze_spill_error(dir=dir,tag=tag)
        except:
            print("Spill and orbital mixing analysis failed. Check that (dir+'error_output'+tag+'.txt') exists.")

        try:
            analyze_band_error(dir=dir,tag=tag)
        except:
            print("DFT band error analysis failed. Check that (dir+'DFT_band_error'+tag+'.txt') exists.")

def main(argv=None):
    import argparse
    analyzer_args = argparse.ArgumentParser(description="Analyze the COGITO run with just one line in terminal.")
    analyzer_args.add_argument("--dir",type=str,help="The directory that contains the COGITO output files.",default="")
    analyzer_args.add_argument("--tag",type=str,help="The tag appended to COGITO output files.",default="")
    analyzer_args.add_argument("--files",type=str,help="Files to analyze. Can be 'all', 'orbital','spill', 'band', or a mix (e.g. 'orbital spill').",default="all")
    analyzer_args.add_argument("--no_make_change_plot",help="If called, does not create and save the orbital change plot.",action='store_true') #not #
    analyzer_args.add_argument("--show_fig",help="If called, does plt.show() for the orbital change figure.",action='store_true') #
    analyzer_args.add_argument("--no_group_same_elem",help="If called, does not group atoms of the same element in the orbital plot.",action='store_true') #not #

    args = analyzer_args.parse_args(argv)
    if 'all' in args.files:
        analyze_all(args.dir,args.tag,not args.no_make_change_plot,args.show_fig,not args.no_group_same_elem)
    else:
        import sys
        from contextlib import redirect_stdout, redirect_stderr
        # Open log file for terminal output
        log_file = open(args.dir + "analysis" + args.tag + ".txt", "w", buffering=1)
        tee_out = Tee(sys.stdout, log_file)
        tee_err = Tee(sys.stderr, log_file)
        with redirect_stdout(tee_out), redirect_stderr(tee_err):
            if 'orbital' in args.files:
                try:
                    analyze_orb_converg_info(args.dir,args.tag,not args.no_make_change_plot,args.show_fig,not args.no_group_same_elem)
                except:
                    print("Orbital convergence analysis failed. Check that (dir+'orb_converg_info'+tag+'.json') exists.")
            if 'spill' in args.files:
                try:
                    analyze_spill_error(args.dir,args.tag)
                except:
                    print("Spill and orbital mixing analysis failed. Check that (dir+'error_output'+tag+'.txt') exists.")

            if 'band' in args.files:
                try:
                    analyze_band_error(args.dir,args.tag)
                except:
                    print("DFT band error analysis failed. Check that (dir+'DFT_band_error'+tag+'.txt') exists.")

if __name__ == "__main__":
    raise SystemExit(main())
