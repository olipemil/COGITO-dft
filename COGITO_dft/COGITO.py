#!/usr/bin/env python3

from pymatgen.io.vasp.outputs import Wavecar, Vasprun
from pymatgen.electronic_structure.core import Spin
from pymatgen.io.vasp.outputs import Eigenval, Kpoints
#from pymatgen.core import Structure
#from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import numpy as np
np.set_printoptions(precision = 15,suppress=True)
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit,fsolve
from scipy import integrate, interpolate,ndimage,linalg
from scipy.special import spherical_jn,factorial2
import math
from os.path import exists
import copy
#import time
from functools import partial
import plotly.graph_objects as go

dist_version = "0.1.0"

class COGITO(object):
    def __init__(self,wavecar_dir,readmode = False,spin=0,spin_polar = False):
        """
        intialize the VASP calculation info.

        Args:    
            wavecar_dir (str): The directory with all the VASP output files.
            readmode (bool): If true, the code should have been run with readmode=False to generate output files;
                        will not read VASP output files or run orbital convergence and projection.
        """
        print("initializing COGITO")
        print("is from new equivariant branch!!")
        if wavecar_dir[-1] != "/" and wavecar_dir[-1] != "\\": # won't work on windows...
            wavecar_dir = wavecar_dir + "/"
        self.directory = wavecar_dir
        allfiles = ["POSCAR","WAVECAR","POTCAR","vasprun.xml"]
        POSexists = exists(self.directory+"POSCAR")
        WAVEexists = exists(self.directory+"WAVECAR")
        POTexists = exists(self.directory+"POTCAR")
        OUTexists = exists(self.directory+"OUTCAR")
        VASPRUNexists = exists(self.directory+"vasprun.xml")
        self.readmode = readmode
        self.spin_polar = spin_polar
        self.spin = spin

        if readmode == False:
            if not POSexists:
                print("WARNING: you have not provided a VASP POSCAR!")
                exit()
            if not WAVEexists:
                print("WARNING: you have not provided a VASP WAVECAR!")
                exit()
            if not POTexists:
                print("WARNING: you have not provided a VASP POTCAR!")
                exit()
            if not OUTexists:
                print("WARNING: you have not provided a VASP OUTCAR!")
                exit()

        ## -- get data from POSCAR -- ##
        filedata = open(self.directory+"POSCAR")
        filelines = filedata.readlines()
        for line in range(len(filelines)):
            if "1.0" in filelines[line]:
                conv_start = line + 1
            if "Direct" in filelines[line]:
                atom_coord_start = line + 1
            if "direct" in filelines[line]:
                atom_coord_start = line + 1
        conv_start = 2
        # set conventional cell vectors
        #print(filelines[conv_start],filelines[conv_start+1],filelines[conv_start+2])
        self._a = np.array([[float(i) for i in filelines[conv_start].strip().split()],[float(i) for i in filelines[conv_start + 1].strip().split()],[float(i) for i in filelines[conv_start + 2].strip().split()]])
        self.vol = np.dot(self._a[0, :], np.cross(self._a[1, :], self._a[2, :]))
        #print(self._a)

        # calculate reciprocal lattice
        b = np.array([
                np.cross(self._a[1, :], self._a[2, :]),
                np.cross(self._a[2, :], self._a[0, :]),
                np.cross(self._a[0, :], self._a[1, :]),])
        b = 2 * np.pi * b / self.vol
        #print("b_vec calc",b,self.vol)
        self.b_vecs = b
        recip_vol = np.dot(self.b_vecs[0, :], np.cross(self.b_vecs[1, :], self.b_vecs[2, :]))
        #print("recip vol:",recip_vol)

        # set position and name of atoms
        single_elem = filelines[atom_coord_start-3].strip().split()
        elem_num = filelines[atom_coord_start-2].strip().split()
        self.elements = np.array(np.repeat(single_elem,elem_num),dtype=np.str_)
        print("is from new equivariant branch!!")
        print("check elems:",single_elem,elem_num,self.elements,len(self.elements))
        self.numAtoms = len(self.elements)#int(np.sum([int(i) for i in filelines[atom_coord_start-2].strip().split()]))
        #print("num atoms",self.numAtoms)
        self.primAtoms = np.zeros([self.numAtoms,3],dtype=np.float64)
        #self.elements = []
        for atom in range(self.numAtoms):
            index = atom_coord_start + atom
            #element = filelines[index].strip().split()[3]
            position = np.array([float(i) for i in filelines[index].strip().split()[:3]])
            #self.elements.append(element)
            # make sure the positions are in the first primitive cell
            og_center = np.around(position,decimals=12)#_cart_to_red((self._a[0], self._a[1], self._a[2]), [new_center])[0],decimals=8)
            prim_center =  copy.deepcopy(og_center)
            #print("first:",prim_center,self.primAtoms[atm])
            while (prim_center>=1).any() or (prim_center<0).any():
                translate = np.zeros(3)
                translate[prim_center>=1] = -1
                translate[prim_center<0] = 1
                prim_center = prim_center + translate
                #print(prim_center)
            translate = prim_center-og_center
            final_center = prim_center
            
            self.primAtoms[atom] = final_center

        self.cartAtoms = _red_to_cart((self._a[0],self._a[1],self._a[2]),self.primAtoms)
        self.get_local_environment()
        #print("a1:",self._a)
        #print("atom info:", self.primAtoms, self.elements)
        #print("cart atoms:", self.cartAtoms)

        ## -- get info from POTCAR -- ##
        #determine whether to include d orbitals based on POTCAR file
        self.POTCARatoms = []
        include_dorbs = np.full((self.numAtoms),True,dtype=bool)
        #print("initial dorbs:",include_dorbs,self.numAtoms)
        potcar = open(self.directory + "POTCAR", "r")
        lines = potcar.readlines()
        checkdorb = []
        psuepotname = lines[0].strip().split()[0]
        #checkdorb.append(lines[0].strip().split()[1].split("_"))
        #self.POTCARatoms.append([checkdorb[0][0],0])
        #speedstart = int(len(lines)/self.numAtoms/2)
        start = 0
        for lind, line in enumerate(lines):
            if "End of Dataset" in line:
                start = lind + 1
            if "TITEL" in line:
                info = line.strip().split()[3].split("_")
                print("check:",info)
                checkdorb.append(info)
                self.POTCARatoms.append([info[0],start])#lind])
                #if len(include_dorbs) == self.numAtoms:
                #    break
        #print(checkdorb)
        self.POTCARatoms.append(["",len(lines)])
        self.POTCARatm_d = np.full((len(checkdorb)),True,dtype=bool)
        for ind,potatom in enumerate(checkdorb):
            if len(potatom) == 1:
                self.POTCARatm_d[ind] = False
            else:
                self.POTCARatm_d[ind] = True
            for atm in range(self.numAtoms):
                if potatom[0] == self.elements[atm]:
                    #print()
                    if len(potatom) == 1:
                        include_dorbs[atm] = False
                    elif potatom[1] == "d":
                        include_dorbs[atm] = True
                    elif potatom[1] == "pv":
                        works = True
                    else:
                        print("ERROR: there is an extra electron state in the POTCAR that cannot be modeled")
        #print(include_dorbs)
        #print(self.POTCARatoms)
        self.include_dorbs = include_dorbs

        self.haveQab = False
        
        self.mnkcoefficients = [] #for check later
        
        # get the shift to make average potential energy zero from OUTCAR
        filedata = open(self.directory+"OUTCAR")
        filelines = filedata.readlines()
        energy_shift = 0
        for line in filelines:
            if "XC(G=0)" in line:
                things = line.strip().split()
                #print(things)
                for i,thing in enumerate(things):
                    if "XC(G=0):" in thing:
                        ind = i + 1
                    elif "XC(G=0)" in thing: #written as XC(G=0) : ##### instead of XC(G=0): #####
                        ind = i+2
                energy_shift = float(things[ind])
        self.energy_shift = energy_shift
        #print("energy shift",energy_shift)

        ## -- get info from VASPRUN -- ##
        #get vasprun info for plotting BS and DOS
        if VASPRUNexists == True:
            vasprun = Vasprun(self.directory + "vasprun.xml")
            self.spin_polar = vasprun.is_spin
            kpt_weights = vasprun.actual_kpoints_weights
            self.kpoints = np.around(np.array(vasprun.actual_kpoints),decimals=13)
            self.kpt_weights = np.array(kpt_weights)
            self.num_kpts = len(kpt_weights)
            self.efermi = vasprun.efermi
            self.bandgap = vasprun.eigenvalue_band_properties[0]
            self.sym_prec = vasprun.parameters["SYMPREC"]*10
            print("SYMPREC!", self.sym_prec)
            eigval = vasprun.eigenvalues
            #print("eigen keys:",list(eigval.keys()))
            eigval = np.array(eigval[list(eigval.keys())[spin]])  #change with spin
            self.eigval = eigval
            self.eigval[:, :, 0] = self.eigval[:, :, 0]+energy_shift#-9.3263#-5#+ 0.91587 # - self.efermi
            self.complete_dos = vasprun.complete_dos
            self.total_dos = vasprun.tdos
            #self.total_dos.densities
            #self.el_spd_dos = {}
            #self.spd_dos = {}
            #elements = {}
            #for atm in range(self.numAtoms):
            #    self.el_spd_dos[self.elements[atm]] = self.complete_dos.get_element_spd_dos(el=self.elements[atm])
            #self.spd_dos = self.complete_dos.get_spd_dos()
            #self.pymatenergies = list(list(self.el_spd_dos.values())[0].values())[0].energies+energy_shift
            #print("dos info",list(self.el_spd_dos["Pb"].values())[0].densities[Spin.up])
            #print("dos info",list(self.el_spd_dos["S"].values())[0].densities[Spin.up])
            #print("dos info",list(self.spd_dos.values())[0].densities[Spin.up])
            #print("dos info",self.total_dos.densities[Spin.up])
            #print(len(self.pymatenergies))
        #print("from vaspun:",self.eigval[0])
        ''' # don't always do this cause the EIGENVAL may be form a band structure calculation
        if exists(self.directory+"EIGENVAL"): # read in from here for 6 decimals instead of 4 from vasprun.xml
            
            pyeig = Eigenval(self.directory + 'EIGENVAL')
            DFTeigvals = np.array(list(pyeig.eigenvalues.values()))[spin,:,:,:]
            self.eigval = DFTeigvals
            self.eigval[:, :, 0] = self.eigval[:, :, 0]+energy_shift
        '''
        #print("from EIGENVAL:",self.eigval[0])
        
        
        #print(self.kpoints)
        if exists(self.directory+"IBZKPT"): # read in from here for 14 decimals instead of 8 from vasprun.xml
            kptclass = Kpoints.from_file(self.directory + 'IBZKPT')
            #print(kptclass)
            #print(kptclass.kpts_weights)
            kpts = np.array(kptclass.kpts)
            self.kpoints = kpts
            
        #print(self.kpoints)
            
        ## -- get data from WAVECAR  -- ##
        if readmode == False:
            file = self.directory + 'WAVECAR'
            print("getting wavecar")
            self.wavecar = Wavecar(file)
            self.kpoints = np.around(np.array(self.wavecar.kpoints),decimals=14)
            self.max_band = self.wavecar.nb
            #print("original max band:",self.max_band)
            #print("kpoints:", self.kpoints)
            ftwavefunc = self.wavecar.fft_mesh(0, 0,spin=spin)
            realwavefunc = np.fft.ifftn(ftwavefunc)
            self.gridxyz = np.array(realwavefunc.shape)

            # also save eigenenergies
            DFTeigvals = np.array(self.wavecar.band_energy)
            if self.spin_polar:
                DFTeigvals = DFTeigvals[spin]
            #print(DFTeigvals.shape)
            #self.eigval = DFTeigvals
            self.eigval[:, :, 0] = DFTeigvals[:,:,0]+energy_shift
            #print("wavefunction grid:",self.gridxyz)
        #print("final:",self.kpoints)
        #self.kpoints = np.around(self.kpoints,decimals=4)

        #crystalWF = {}
        if readmode == False:
            self.periodic_DFTwf = np.zeros([1,self.max_band,self.gridxyz[0],self.gridxyz[1],self.gridxyz[2]],dtype=np.complex128)
            set_integral = False
            self.gpoints = self.wavecar.Gpoints
            #print("gpoints:",self.gpoints)
            #print("wave coeffs shape:",len(self.wavecar.coeffs))
            if len(self.wavecar.coeffs) == 2:
                self.recip_WFcoeffs = self.wavecar.coeffs[spin]  #[kpt,band,gpoint]  # change with spin
            else:
                self.recip_WFcoeffs = self.wavecar.coeffs
            #print("num gpoints for first kpoint!",len(self.recip_WFcoeffs[0][0]))

            # save real space wavefunction only for first kpoint
            zerokpt = 0
            for band in range(self.max_band):
                ftwavefunc = self.wavecar.fft_mesh(zerokpt, band, spin=spin) # change with spin
                realwavefunc = np.fft.ifftn(ftwavefunc)

                # normalize the wavefunctions only to band 0 of kpt such that each band can still have different norms
                if set_integral == False:
                    set_integral = True
                    self.periodic_DFTwf[zerokpt][band], integral = normalize_wf(realwavefunc, self._a, self.gridxyz,return_integral=True)
                else:
                    self.periodic_DFTwf[zerokpt][band] = realwavefunc / integral

            # save all the gcoeffs into numpy arrays for each kpt, no extra norming
            #self.num_kpts = 250
            #self.max_band = 120
            self.newrecip_WFcoeffs = {}
            for kpt in range(self.num_kpts):
                self.newrecip_WFcoeffs[kpt] = np.array(self.recip_WFcoeffs[kpt],dtype=np.complex64)#[:self.max_band]#/ integral)
            self.recip_WFcoeffs = self.newrecip_WFcoeffs
            print("num plane waves:",len(self.recip_WFcoeffs[0][0]))

        self.min_rad={}


    def generate_TBmodel(self, invariant: bool = True, irreducible_grid=True, verbose=0, tag="", include_excited=1, calc_nrms=False, save_orb_converg_info:bool=True,
                         orbfactor=1.0, num_steps=50, num_outer=3, plot_orbs = False,
                         min_proj=0.01, band_opt=True,orb_opt=True,orb_orth=False,start_from_orbnpy = False,plot_projBS = False,plot_projDOS=False,orbs = None,save_orb_data=False,save_orb_figs=False,minimum_orb_energy: float=-60,min_duplicate_energy:float=-60):
        """
        Runs all the functions neccessary to generate the TB interpolation.
        REQUIRES UNIFORM KPT GRID WITH NO SYMMETRY OR FULL SYMMETRY FOR TB MODEL.


        Args:
            invariant (bool): Whether to construct the orbitals as eigenvectors of the local environment tensor.
            irreducible_grid (bool): Whether or not the kpoint grid is irreducible. True for ISYM=1``|``2``|``3; False for ISYM=-1 (does not work with ISYM=0).
            verbose (str): How much to ouput, includes 0,1,2,3. Higher numbers result in more output.
            tag (bool): Tag to be appended to the runs output files.
            include_excited (int): How much to include excited orbital states, includes 0, 1, 2. To do this best avoid using POTCARs with semi-core states.
                        (0) Includes no excited orbital states (only ones which are partially occupied in isolated atom).
                        (1) Includes some excited states (if the POTCAR has the excited states + another higher one of the same l) (generally includes p orbital for d block).
                        (2) Includes maximally recommended states (condition of 1 + if the energy listed is the same as in isolated atom) (generally includes p orbitals for 1&2nd row).
            calc_nrms (bool): Extra error analysis that requires more runtime to generate. Correlates well with the square root projection spilling. Both spilling and nrmse are output to "error_output"+tag+".txt" during all runs.
            save_orb_converg_info (bool): Creates the file 'orb_converg_info.json'.
            orbfactor (float): Multipled by the radial part of the pseudo radial orbital to either shrink or grow them.
            num_steps (int): Maximum number of steps to perform the convergence. Generally convergence is achieved around 5-20 steps.
                        Setting equal to 0 will run the standard direct algorithm where ``|X_a> = sum_n(c_na |Y_n>)``.
                        Where ``|Y_n>`` are the set of bands (equal to number of orbitals) of the highest projection.
            num_outer (bool): The number of outer loops in COGITO. Aka, the guassian fit from fit_to_atomic_orb is feed back into the convergence and fitting. Converges very quickly with some possibly of divergence in some systems if num_outer is too large.
                        Values of 2-5 are recommended. Running with 0 creates model from just VASP psuedo orbitals.
            plot_orbs (bool): Plot the radial converged orbital being fit to Gaussian functions.
            min_proj (bool): Sets the minimum projectability of a wavefunction to be included Hamiltonian, overlap, etc matrix construction.
            band_opt (bool): Whether or not to apply Lowdin+Gram+extras band optimization scheme for improved TB interpolation (improves band error from ~40meV to ~2meV).
            orb_opt (bool): Whether or not enforce zero orbital mixing for improved TB interpolation (ends up having little effect).
            orb_orth (bool): Make orthogonal tight binding (indirectly creates orthogonal basis) from Lowdin orthogonalized orbital coefficients.
            start_from_orbnpy (bool): Start from the self.directory+"orbitals"+self.tag+".npy" of a previous run.
            plot_projBS (bool): Plot the orbital projected bandstructure (not BS if the kpt grid is uniform).
            plot_projDOS (bool): Plot the orbital projected density of states (not correct DOS unless kpt grid is uniform).
            orbs (dictionary): The orbitals to plot the projection of. Defaults to all orbitals.
                        FORMAT: {"element1":[orbital types]} e.g. {"Si":["s","p"],"C":["s","p"]}.
            save_orb_data (bool): Saves the numerical orbital data for converging basis on line between central and nearest atom. Not generally useful but used to make COGITO variance figure. Best when uncommenting make_fit_wannier() below and running on dense grid.
            save_orb_figs (bool): Saves the plot of numerical orbital data for the converging basis. Not easily interpreted.
            minimum_orb_energy (float): The lower limit to add semi-core states in the POTCAR into the COGITO basis. Uses the atomic orbital energy listed in POTCAR.
            min_duplicate_energy (float): The lower limit to add semi-core states when there is another valence state of the same l quantum number in the POTCAR into the COGITO basis. Uses the atomic orbital energy listed in POTCAR.

        Usage:
            new_model = COGITO("silicon/")
            new_model.generate_TBmodel(plot_orbs=True)
        """
        
        import time

        start_time = time.time()
        self.verbose = verbose
        self.plot_orbs = plot_orbs
        self.save_orb_data = save_orb_data
        self.save_orb_figs = save_orb_figs
        self.include_excited = include_excited
        self.tag = tag
        if self.spin_polar:
            self.tag = tag + str(self.spin)

        self.minimum_orb_energy = minimum_orb_energy
        self.min_duplicate_energy = min_duplicate_energy
        self.min_proj = min_proj
        self.low_min_proj = 0.5 # if bands -5 below fermi have less than this, don't include
        self.use_pseudo = False
        if num_steps == 0:
            self.use_pseudo = True
        
        # get converged ideal atomic orbitals
        if self.verbose > 0:
            print("cartesian atom positions:",self.cartAtoms)
        print("setting up intitial orbitals")
        self.get_orbs_fromPOT(orbfactor=orbfactor,invariant=invariant)
        self.get_Qab()
        #self.fit_to_atomic_orb(plot_orbs=plot_orbs)
        self.get_kdep_recipprojs(0)
        time1 = time.time()
        # Calculate elapsed time
        elapsed_time1 = time1 - start_time
        print("elapsed time for initialization:",elapsed_time1)
        print("")

        #self.write_input_file()
        if not start_from_orbnpy:
            if save_orb_converg_info:
                all_orb_info = {}
                all_orb_info["orb_type"] = []
                all_orb_info["orb_rad"] = []
                all_orb_info["orb_change"] = []
                all_orb_info["orb_change_periter"] = []
                all_orb_info["radial_change_periter"] = []
                all_orb_info["bloch_nrmse_iter"] = []
                all_orb_info["bloch_nme_iter"] = []
                all_orb_info["gaus_fit_error"] = [] # just the error between the gaus+exp fit and gaus fit
            elapsed_time2 = 0
            elapsed_time3 = 0
            for outer in range(num_outer):
                time1 = time.time()
                self.converge_orbs_recip(num_steps=num_steps)
                #self.converge_orbs()
                time2 = time.time()
                # Calculate elapsed time
                elapsed_time2 += time2 - time1
                print("elapsed time for converged orbs:",elapsed_time2)
                print("")
                self.fit_to_atomic_orb(plot_orbs=plot_orbs,orbfactor=orbfactor)
                time3 = time.time()
                # Calculate elapsed time
                elapsed_time3 += time3 - time2
                print("elapsed time for fitting orbs:",elapsed_time3)
                print("")

                if save_orb_converg_info:
                    # save orbital info
                    all_orb_info["orb_change_periter"].append(list(self.all_orbloop_info[2]))
                    all_orb_info["radial_change_periter"].append(list(self.all_orbloop_info[3]))
                    all_orb_info["bloch_nrmse_iter"].append(list(self.all_orbloop_info[4]))
                    all_orb_info["bloch_nme_iter"].append(list(self.all_orbloop_info[5]))
            if save_orb_converg_info:
                all_orb_info["orb_type"] = list(np.char.add(self.elements[self.orbatomnum],np.char.add("-",np.char.add(self.exactorbtype,np.char.add("-",np.array(self.orbatomnum,dtype=str))))))
                all_orb_info["orb_rad"]=list(self.all_orbloop_info[0])
                all_orb_info["orb_change"]=list(self.all_orbloop_info[1])
                all_orb_info["gaus_fit_error"]=list(self.all_orbloop_info[6])
                all_orb_info["init_orb_energy"] = list(self.init_orb_energy)
                import json

                with open(self.directory+"orb_converg_info"+self.tag+".json", "w") as f:
                    json.dump(all_orb_info, f, indent=2)
                #print(all_orb_info)
                #print("average bloch nsmre per loop:", np.average(all_orb_info["bloch_nrmse_iter"],axis=1))
                #print("average bloch nme per loop:", np.average(all_orb_info["bloch_nme_iter"],axis=1))
            # write out orbital and input files
            self.write_recip_rad_orbs()
            self.write_input_file()
        else: 
            self.read_recip_rad_orbs()
            elapsed_time2 = 0
            time3 = time.time()
            # Calculate elapsed time
            elapsed_time3 = time3 - time1
            print("elapsed time for reading in orbs:",elapsed_time3)
            print("")


        # get all-election orbital overlap
        #self.get_Qab()
        self.get_ae_overlap_info()

        # project orbitals on DFT wavefunctions
        self.proj_all_kpoints(calc_nrms=calc_nrms)

        time4 = time.time()
        # Calculate elapsed time
        elapsed_time4 = time4 - time3
        print("elapsed time for projecting orbs:",elapsed_time4)
        print("")
        #self.optimize_band_set()

        if band_opt or orb_opt or orb_orth:  # if not (no optimize and nonorth orbitals)
            if not orb_opt:
                print("WARNING: it is a very bad idea to set orb_opt to False when the optimization occurs before expand_irred_kgrid()!")
                print("         This is because in expand_irred_kgrid() the orbital overlaps are set by the coefficients.")
            self.optimize_band_set(band_opt=band_opt, orb_opt=orb_opt, orb_orth=orb_orth) # doing before expand grid definitely needs orb_opt=True!!!

        # expand the irreducible grid to a full reducible grid
        if irreducible_grid == True:
            self.expand_irred_kgrid()
        
        #if band_opt or orb_opt or orb_orth: # if not (no optimize and nonorth orbitals)
        #    self.optimize_band_set(band_opt=band_opt,orb_opt=orb_opt,orb_orth=orb_orth)
        
        time5 = time.time()
        # Calculate elapsed time
        elapsed_time5 = time5 - time4
        print("elapsed time for optimize and symmetry:",elapsed_time5)
        print("")

        if plot_projBS == True:
            if orbs == None:
                orbs = [self.elements, ["s", "p","d"]]
            self.plot_projectedBS(orbs)

        if plot_projDOS == True:
            if orbs == None:
                orbs = [self.elements, ["s", "p"]]
            self.plot_projectedDOS(orbs)

        #self.make_fit_wannier()
        
        for loop in range(0):
            self.make_fit_wannier()
            if irreducible_grid:
                print("reduced the kpoint grid back")
                self.num_kpts = self.irred_num_kpts
                self.kpoints = self.irred_kpoints
                #self.mnkcoefficients = self.irred_mnkcoefficients
                self.eigval = self.irred_eigval
                #self.mixed_eigval = all_reduc_newval
                self.gpoints = self.irred_gpoints
                #self.total_nrms = new_nrms
                #self.all_band_spillage = self.irred_all_band_spillage
                self.recip_WFcoeffs = self.irred_recip_WFcoeffs
                self.kpt_weights = self.irred_kpt_weights
                #self.Sij = self.irred_Sij
                #self.Aij = self.irred_Aij
            self.write_recip_rad_orbs()
            # project orbitals on DFT wavefunctions
            self.get_ae_overlap_info()
            self.proj_all_kpoints()
            #self.optimize_band_set()
            
            # expand the irreducible grid to a full reducible grid
            if irreducible_grid == True:
                self.expand_irred_kgrid()

            self.optimize_band_set()
            #self.make_fit_wannier()
        
        # finally generate the kdep hamiltonians and the real dep TB params
        self.get_hamiltonian()
        self.get_TBparameter()

        #if orb_orth:
        #    print("orthogonalizing basis")
        #    self.orthogonalize_basis()
        time6 = time.time()
        # Calculate elapsed time
        elapsed_time6 = time6 - time5
        print("elapsed time for calculating hams:",elapsed_time6)
        
        print("finished!")
        print("")
        print("elapsed time for initialization:",elapsed_time1)
        print("elapsed time for converged orbs:",elapsed_time2)
        print("elapsed time for fitting orbs:",elapsed_time3)
        print("elapsed time for projecting orbs:",elapsed_time4)
        print("elapsed time for optimize and symmetry:",elapsed_time5)
        print("elapsed time for calculating hams:",elapsed_time6)
        total_time = elapsed_time1+elapsed_time2+elapsed_time3+elapsed_time4+elapsed_time5+elapsed_time6
        print("total time:",total_time)

    def test_initialorbs(self,verbose=0, plot_orbs = False,include_excited=1, low_factor=0.8 ,high_factor=1.2, num_fac=9,num_outer=2,tag='',min_proj=0.02,num_steps=50,minimum_orb_energy: float=-60,min_duplicate_energy:float=-60):
        """
        Tests dependance of orbital radius and quality on the size of initial orbitals.
        Only runs the convergence of the orbitals without projecting them or generating the TB model.


        Args:    
            verbose (str): How much to ouput, includes 0,1,2,3. Higher numbers result in more output.
            plot_orbs (bool): Plot the radial converged orbital being fit to Gaussian functions.
            low_factor (float): Minimum to multiply the orbital size by; Default is 80%.
            high_factor (float): Maximum to multiply the orbital size by; Default is 120%.
            num_fac (int): The number of steps between low_factor and high_factor to try.

        Usage:
            new_model = COGITO("silicon/")
            new_model.generate_TBmodel(plot_orbs=True)
        """
        
        import time
        
        start_time = time.time()
        self.verbose = verbose
        self.plot_orbs = plot_orbs
        self.save_orb_data = False
        self.save_orb_figs = False
        self.include_excited = include_excited
        self.tag = tag
        if self.spin_polar:
            self.tag = tag + str(self.spin)

        self.minimum_orb_energy = minimum_orb_energy
        self.min_duplicate_energy = min_duplicate_energy
        self.min_proj = min_proj
        self.low_min_proj = 0.5 # if bands -5 below fermi have less than this, don't include
        
        #num_steps = 50
        
        self.use_pseudo = False
        if num_steps == 0:
            self.use_pseudo = True
        
        self.get_orbs_fromPOT() # just to initialize global variables
        self.get_Qab()
        #num_fac = 11
        orb_factors = np.linspace(low_factor,high_factor,num_fac)
        print("orb factors:",orb_factors)
        
        # one for my converging algothim
        converg_allnrmsorb = np.zeros((num_outer,num_fac,self.num_orbs))
        converg_allnrmsradial = np.zeros((num_outer,num_fac,self.num_orbs))
        converg_allnmeorb = np.zeros((num_outer,num_fac,self.num_orbs))
        converg_allnmeradial = np.zeros((num_outer,num_fac,self.num_orbs))
        #converg_allnrmsband = np.zeros((num_fac,self.max_band))
        converg_allradius = np.zeros((num_outer+1,num_fac,self.num_orbs))
        
        for ind,orbfactor in enumerate(orb_factors):
            # get converged ideal atomic orbitals
            print("orb factor:",ind,orbfactor)
            self.get_orbs_fromPOT(orbfactor=orbfactor)
            converg_allradius[0,ind] = self.orb_radius
            print("setting up intitial orbitals")
            time1 = time.time()
            # Calculate elapsed time
            elapsed_time1 = time1 - start_time
            print("elapsed time for initialization:",elapsed_time1)
            print("")
            try:
                elapsed_time2 = 0
                elapsed_time3 = 0
                for outer in range(num_outer):
                    time1 = time.time()
                    self.converge_orbs_recip(num_steps=50)
                    time2 = time.time()
                    # Calculate elapsed time
                    elapsed_time2 += time2 - time1
                    print("elapsed time for converged orbs:",elapsed_time2)
                    print("")
                    self.fit_to_atomic_orb(plot_orbs=plot_orbs,orbfactor=orbfactor)
                    time3 = time.time()
                    # Calculate elapsed time
                    elapsed_time3 += time3 - time2
                    print("elapsed time for fitting orbs:",elapsed_time3)
                    print("")
                    
                    # save the nrms and radius data
                    converg_allnrmsorb[outer,ind] = self.orb_nrms
                    converg_allnrmsradial[outer,ind] = self.radial_nrmse
                    converg_allnmeorb[outer,ind] = self.orb_nme
                    converg_allnmeradial[outer,ind] = self.radial_nme
                    converg_allradius[outer+1,ind] = self.orb_radius
                    print("average nrms:", np.average(self.orb_nrms))#,np.average(converg_allnrmsband[ind][:self.num_orbs]))
            except:
                print("failed (likely convergence). Save zeros for this orbfactor")
                converg_allnrmsorb[:,ind] = 0
                converg_allnrmsradial[:,ind] = 0
                converg_allnmeorb[:,ind] = 0
                converg_allnmeradial[:,ind] = 0
                converg_allradius[:,ind] = 0


            # get all-election orbital overlap
            #self.get_Qab()
            #self.get_ae_overlap_info()

            # project orbitals on DFT wavefunctions
            #self.proj_all_kpoints()
            #time4 = time.time()
            # Calculate elapsed time
            #elapsed_time4 = time4 - time3
            #print("elapsed time for projecting orbs:",elapsed_time4)
            #print("")
            #self.optimize_band_set()
            
            
            #band_spill = self.all_band_spillage
            #converg_allnrmsband[ind] = np.average(band_spill,axis=0,weights=self.kpt_weights[:self.num_kpts])
        
        print("got values!", converg_allnrmsorb)
        print("radial nrmse:", converg_allnrmsradial)
        print("orbital radius:", converg_allradius)
        #print(converg_allnrmsband)
        
        #dir = self.directory
        #for outer in range(num_outer):
        np.save(self.directory+"converg_orbnrmse"+tag,converg_allnrmsorb)
        np.save(self.directory+"converg_radialnrmse"+tag,converg_allnrmsradial)
        np.save(self.directory+"converg_orbnme"+tag,converg_allnmeorb)
        np.save(self.directory+"converg_radialnme"+tag,converg_allnmeradial)
        np.save(self.directory+"converg_rad"+tag,converg_allradius)
        #np.save(dir+"converg_bandnrms",converg_allnrmsband)
        '''
        # one for the direct algothim
        converg_allnrmsorb = np.zeros((num_fac,self.num_orbs))
        converg_allnrmsband = np.zeros((num_fac,self.max_band))
        converg_allradius = np.zeros((num_fac,self.num_orbs))
        num_steps = 0
        for ind,orbfactor in enumerate(orb_factors):
            # get converged ideal atomic orbitals
            print("orb factor:",ind,orbfactor)
            self.get_orbs_fromPOT(orbfactor=orbfactor)
            print("setting up intitial orbitals")
            time1 = time.time()
            # Calculate elapsed time
            elapsed_time1 = time1 - start_time
            print("elapsed time for initialization:",elapsed_time1)
            print("")

            self.converge_orbs_recip(num_steps=0)
            time2 = time.time()
            # Calculate elapsed time
            elapsed_time2 = time2 - time1
            print("elapsed time for converged orbs:",elapsed_time2)
            print("")
            self.fit_to_atomic_orb(plot_orbs=plot_orbs)
            time3 = time.time()
            # Calculate elapsed time
            elapsed_time3 = time3 - time2
            print("elapsed time for fitting orbs:",elapsed_time3)
            print("")

            # get all-election orbital overlap
            #self.get_Qab()
            #self.get_ae_overlap_info()

            # project orbitals on DFT wavefunctions
            #self.proj_all_kpoints()
            time4 = time.time()
            # Calculate elapsed time
            elapsed_time4 = time4 - time3
            print("elapsed time for projecting orbs:",elapsed_time4)
            print("")
            #self.optimize_band_set()
            
            
            # save the nrms and radius data
            converg_allnrmsorb[ind] = self.orb_nrms
            converg_allradius[ind] = self.orb_radius
            #band_spill = self.all_band_spillage
            #converg_allnrmsband[ind] = np.average(band_spill,axis=0,weights=self.kpt_weights[:self.num_kpts])
            print("average nrms:", np.average(self.orb_nrms))#,np.average(converg_allnrmsband[ind][:self.num_orbs]))
        
        print("got values!",converg_allnrmsorb,converg_allradius)
        #print(converg_allnrmsband)
        
        dir = self.directory
        np.save(dir+"direct_orbnrms",converg_allnrmsorb)
        np.save(dir+"direct_rad",converg_allradius)
        #np.save(dir+"direct_bandnrms",converg_allnrmsband)
        '''
        
        
        
    def get_WFdata_fromPOT(self):
        """
        Function which reads the pseudo and ae orbital information from the POTCAR.
        Creates the global variables self.atmProjectordata, self.atmRecipProjdata, and self.atmPsuedoAeData.
        FORMATs:
        self.atmProjectordata [=] {"element",[float(cutoff_rad),{"orbtypes":[proj_real_radial_data]}]} e.g. {"Si",[1.45,{"s":[],"s_ex":[],"p":[],"p_ex":[]}}
        self.atmRecipProjdata [=] {"element",[float(proj_gmax),{"orbtypes":[proj_recip_radial_data]}]}
        self.atmPsuedoAeData [=] {"element",[radial_grid, {"orbtypes":[pseudo_radial_data]}}, {"orbtypes":[ae_radial_data]}]} e.g. {"Si,[[0,0.01,...,1.45],{"s":[],"s_ex":[],"p":[],"p_ex":[]},{"s":[],"s_ex":[],"p":[],"p_ex":[]}]
        """
        potcar = open(self.directory + "POTCAR", "r")
        lines = potcar.readlines()
        atmpsuedodata = {}  # POT data that should be used in the interpolation
        atmgaus = {}
        atmprojector_data = {}
        atm_recipproj_data = {}
        orbsphkey = {}
        orb_energy = {}
        orbtype = {}
        orb_spectype = {}
        alex_orbtype = {}
        alex_orbsphkey = {}
        elec_count = {}

        # get info from POTCAR
        for atmind, potatom in enumerate(self.POTCARatoms[:-1]):
            psuedoind = []
            aeind = []
            projind = []
            recipprojind = []
            orb_info = []
            set_max = False
            if self.verbose > 1:
                print("POTCAR info for element",potatom[0])
            atm_lines = lines[potatom[1]:self.POTCARatoms[atmind + 1][1]]
            elem_info = atm_lines[0].strip().split()[1].split("_")
            elec_count[potatom[0]] = float(atm_lines[1])
            extra_orbs = ""
            if len(elem_info) > 1:
                extra_orbs = elem_info[1]
            for lind, line in enumerate(atm_lines):
                lind = lind + potatom[1]
                if "Description" in line:
                    descript_start = lind + 2
                if "Atomic configuration" in line:
                    config_start = lind + 3
                    num_config = int(lines[lind+1].strip().split()[0])
                    #print("number of atomic states:", num_config)
                if "grid" in line:
                    gridstart = lind + 1
                    # print(gridstart)
                if "RDEP" in line:
                    rad_cutoff = float(line.strip().split()[2])
                if "aepotential" in line:
                    gridlength = lind - gridstart
                    # print(gridlength)
                    # print("found aepotential")
                if "pseudo wavefunction" in line:
                    psuedoind.append(lind + 1)
                    # print(psuedoind)
                if "ae wavefunction" in line:
                    aeind.append(lind + 1)
                if "Real Space Part" in line:
                    projind.append(lind + 1)
                if "Reciprocal Space Part" in line:
                    recipprojind.append(lind + 1)
                if "Non local Part" in line:
                    if set_max == False:
                        proj_max_ind = lind + 1
                        set_max = True
                    orb_l = int(lines[lind + 1].strip().split()[0])
                    num_orbs = int(lines[lind + 1].strip().split()[1])
                    orb_info.append([orb_l,num_orbs])
            
            # get all the energies of the atomic states
            atomic_energies = []
            atomic_occup = []
            last_orb_energy = [0,0,0,0] # energies for [s,p,d,f]
            for st in range(num_config):
                energ = float(lines[config_start+st].strip().split()[3])
                occupation = float(lines[config_start+st].strip().split()[4])
                if self.include_excited > 1: # append everything!
                    atomic_energies.append(energ)
                    atomic_occup.append(occupation)
                else: # only append energy if occupied
                    if occupation != 0:
                        atomic_energies.append(energ)
                        atomic_occup.append(occupation)
                l = int(lines[config_start+st].strip().split()[1])
                last_orb_energy[l] = energ
            atomic_energies = np.around(np.array(atomic_energies),decimals=4)
            atomic_occup = np.array(atomic_occup)
            if self.verbose > 0:
                print("atomic energies:",atomic_energies)
            
            #get grid for psuedo and ae wavefunctions
            grid = []
            for line in lines[gridstart:gridstart + gridlength]:
                points = [float(i) for i in line.strip().split()]
                for point in points:
                    grid.append(point)
            grid = np.array(grid)#*rad_cutoff/grid[-1]

            #get reciprocal grid for projectors
            proj_gmax = float(lines[proj_max_ind-2].strip().split()[0])
            if self.verbose > 2:
                print("proj recip max:", proj_gmax)
            recip_grid = np.arange(100) * proj_gmax / 100  #there are 100 points for the recip projs
            
            # write pseudo orbs, ae orbs, real projector, and recip projector into arrays
            psuedodata = {}
            aedata = {}
            projdata = {}
            recip_projdata = {}
            for state in range(len(psuedoind)):
                # pseudo orbs
                psuedodata[state] = []
                for line in lines[psuedoind[state]:psuedoind[state] + gridlength]:
                    points = [float(i) for i in line.strip().split()]
                    for point in points:
                        psuedodata[state].append(point)
                psuedodata[state] = np.array(psuedodata[state])
                # ae orbs
                aedata[state] = []
                for line in lines[aeind[state]:aeind[state] + gridlength]:
                    points = [float(i) for i in line.strip().split()]
                    for point in points:
                        aedata[state].append(point)
                aedata[state] = np.array(aedata[state])
                # real projectors
                projdata[state] = []
                for line in lines[projind[state]:projind[state] + 20]: #seems like there are 20 lines for any POTCAR element
                    points = [float(i) for i in line.strip().split()]
                    for point in points:
                        projdata[state].append(point)
                projdata[state] = np.array(projdata[state])
                # recip projectors
                recip_projdata[state] = []
                for line in lines[recipprojind[state]:recipprojind[state] + 20]:  # seems like there are 20 lines for any POTCAR element
                    points = [float(i) for i in line.strip().split()]
                    for point in points:
                        recip_projdata[state].append(point)
                recip_projdata[state] = np.array(recip_projdata[state])

            # more generalized way to get the pseudopotential data
            # go through each line after "Description" and save
            # also if the describtion has an _sv include both s orbital states
            # but when you do that, you need to change the converging algo since they will have an overlap from Qij
            
            # for each state find the angular part and determine if it should be included in the valence or excited states
            # use the "Non local Part" to determine the stuffs
            # always have one p orbital, include a d orbital if there, always include an s orbital and include extra one if _sv
            minimum_orb_energy = self.minimum_orb_energy
            min_duplicate_energy = self.min_duplicate_energy
            cutoff_rad = lines[proj_max_ind].split()[-1]
            atmprojector_data[potatom[0]] = [float(cutoff_rad),{}]
            atm_recipproj_data[potatom[0]] = [float(proj_gmax),{}]
            atmpsuedodata[potatom[0]] = [grid, {}, {},rad_cutoff,{}] # radial grid, pseudo orbital, ae orbital, actual potcar cutoff, orbital energies
            atmgaus[potatom[0]] = {}
            
            if self.verbose > 0:
                print("orb info:",orb_info,extra_orbs)
            atm_orbsphharmkey = []
            atm_orbsenergy = []
            atm_orb_type = []
            atm_orb_spectype = []
            ex_orb_type = []
            ex_orbsphkey = []
            ltoorbtype = ['s', 'p', 'd', 'f']  # used to give correct s,p,d for 0,1,2 l quantum num
            sph_ham = [[0],[1,2,3],[4,5,6,7,8],[9,10,11,12,13,14,15]]
            specific_orb = [['s'],['px','py','pz'],['dz2','dxz','dyz','dx2y2','dxy'],['fz3', 'fxz2', 'fyz2', 'fxyz', 'fzx2y2', 'fxxy22', 'fyx2y2']]
            gen_orb = [np.array(['s']),np.array(['p','p','p']),np.array(['d','d','d','d','d']),np.array(['f','f','f','f','f','f','f'])]
            exgen_orb = [['s_ex'],['p_ex','p_ex','p_ex'],['d_ex','d_ex','d_ex','d_ex','d_ex'],['f_ex','f_ex','f_ex','f_ex','f_ex','f_ex','f_ex']]
            # have s, p, d, f and s2 (which is just a second s orbital) and s_ex, p_ex, d_ex, f_ex
            data_ind = 0
            for orbs_lset in orb_info:
                key = orbs_lset[0]
                if key == 3:
                    print("Congrats! f orbitals are now supported by COGITO (still testing)")
                    #raise Exception("f orbitals are not supported by COGITO at the moment, please try again later")
                orb_type = ltoorbtype[key]
                set_val = min(orbs_lset[1] - 1,1)
                #if orb_type == "s" and (extra_orbs == "sv"):
                #    set_val += 1
                #if (orb_type == "p") and (potatom[0] =="Co") and (extra_orbs!="pv"):
                #    set_val -= 1
                #if (orb_type == "s") and (potatom[0]=="C" or potatom[0]=="N"):
                #    set_val += 1
                excluded_l_state = False
                for orb in range(orbs_lset[1]):
                    state_energy = np.around(float(lines[descript_start+data_ind].strip().split()[1]),decimals=4)
                    includ_state = (state_energy > -70) & (state_energy==atomic_energies).any()
                    #if includ_state:
                    #    atomic_energies[state_energy==atomic_energies] = 0
                    includ_val = set_val >= 1
                    if self.verbose > 0:
                        print("psuedo state_energy:", state_energy)
                        print("state included:",includ_state,includ_val)
                    if self.include_excited > 0:
                        final_include = includ_state or includ_val
                    else:
                        final_include = includ_state
                    exclude_state = False

                    if final_include: # in the valance states
                        if self.include_excited > 1 and state_energy < -30 and orb_type == "p": # mostly for adding excited p orbitals in d block
                            set_val = set_val + 1
                        if self.include_excited > 1 and state_energy < -40 and orb_type == "s": # mostly for adding excited s orbitals in d block
                            set_val = set_val + 1
                        if self.include_excited > 2 and state_energy < -15 and orb_type != 'd': # mostly for adding extra s orbital for latter p block
                            set_val = set_val + 1

                        # exclude some low energy states (do this after final_include so that set_val is updated!)
                        if state_energy < minimum_orb_energy:
                            exclude_state = True
                            excluded_l_state = True
                            print("excluded state:",key,state_energy)
                            # remove the electrons from atom
                            elec_count[potatom[0]] = elec_count[potatom[0]] - atomic_occup[state_energy==atomic_energies]
                        elif (state_energy < min_duplicate_energy) and (state_energy != last_orb_energy[key]):
                            exclude_state = True
                            excluded_l_state = True
                            print("excluded state:",key,state_energy)
                            # remove the electrons from atom
                            elec_count[potatom[0]] = elec_count[potatom[0]] - atomic_occup[state_energy==atomic_energies]
                        if not exclude_state:
                            cur_state = orb_type
                            to_add = gen_orb[key]
                            if excluded_l_state: # a previous state was excluded which means this one needs a node but doesn't need to be orthed to lower state
                                cur_state = orb_type + "node"
                                to_add = np.char.add(gen_orb[key],"node")
                            if cur_state in atmprojector_data[potatom[0]][1].keys():
                                cur_state = cur_state + "2"
                                if self.verbose > 0:
                                    print("added second valence of same l")
                                atm_orb_type.append(np.char.add(gen_orb[key],"2"))
                            else:
                                atm_orb_type.append(to_add)

                            atm_orbsphharmkey.append(sph_ham[key])
                            atm_orb_spectype.append(specific_orb[key])
                            atm_orbsenergy.append(list(np.ones(len(sph_ham[key]))*state_energy))

                            atmprojector_data[potatom[0]][1][cur_state] = projdata[data_ind]
                            atm_recipproj_data[potatom[0]][1][cur_state] = recip_projdata[data_ind]
                            atmpsuedodata[potatom[0]][1][cur_state] = psuedodata[data_ind] / grid
                            atmpsuedodata[potatom[0]][2][cur_state] = aedata[data_ind] / grid
                            atmpsuedodata[potatom[0]][4][cur_state] = state_energy

                    if not final_include or exclude_state: # in the excited states
                        cur_state = orb_type
                        if cur_state+"_ex" in atmprojector_data[potatom[0]][1].keys():
                            cur_state = cur_state + "2"
                            if self.verbose > 0:
                                print("added second excited state of the same l")
                            ex_orb_type.append(np.char.add(gen_orb[key],"2_ex"))
                        else:
                            ex_orb_type.append(np.char.add(gen_orb[key],"_ex"))
                            
                        atmprojector_data[potatom[0]][1][cur_state+"_ex"] = projdata[data_ind]
                        atm_recipproj_data[potatom[0]][1][cur_state+"_ex"] = recip_projdata[data_ind]
                        atmpsuedodata[potatom[0]][1][cur_state+"_ex"] = psuedodata[data_ind] / grid
                        atmpsuedodata[potatom[0]][2][cur_state+"_ex"] = aedata[data_ind] / grid
                        atmpsuedodata[potatom[0]][4][cur_state+"_ex"] = state_energy
                        
                        ex_orbsphkey.append(sph_ham[key])
                        #ex_orb_type.append(cur_state+"_ex")#exgen_orb[key])
                    '''
                    # old method to create dictionaries
                    if set_val == 2: # this is the extra s orb
                        print("addding first of 2 s orbitals")
                        atmprojector_data[potatom[0]][1][orb_type] = projdata[data_ind]
                        atm_recipproj_data[potatom[0]][1][orb_type] = recip_projdata[data_ind]
                        atmpsuedodata[potatom[0]][1][orb_type] = psuedodata[data_ind] / grid
                        atmpsuedodata[potatom[0]][2][orb_type] = aedata[data_ind] / grid
                        
                        atm_orbsphharmkey.append(sph_ham[key])
                        atm_orb_type.append([orb_type])
                        atm_orb_spectype.append(specific_orb[key])
                    elif set_val == 1:
                        if orb_type == "d" and potatom[0] == 'nothing':#Dy':# "Ni": # don't include the p orbitals in the valence orbitals
                            atmprojector_data[potatom[0]][1][orb_type] = projdata[data_ind]
                            atm_recipproj_data[potatom[0]][1][orb_type] = recip_projdata[data_ind]
                            atmpsuedodata[potatom[0]][1][orb_type] = psuedodata[data_ind] / grid
                            atmpsuedodata[potatom[0]][2][orb_type] = aedata[data_ind] / grid

                            ex_orbsphkey.append(sph_ham[key])
                            ex_orb_type.append(gen_orb[key])
                        elif orb_type == "s" and "s" in list(atmpsuedodata[potatom[0]][1].keys()): # name this s2
                            print("adding second s orbital")
                            atmprojector_data[potatom[0]][1][orb_type+"2"] = projdata[data_ind]
                            atm_recipproj_data[potatom[0]][1][orb_type+"2"] = recip_projdata[data_ind]
                            atmpsuedodata[potatom[0]][1][orb_type+"2"] = psuedodata[data_ind] / grid
                            atmpsuedodata[potatom[0]][2][orb_type+"2"] = aedata[data_ind] / grid

                            atm_orbsphharmkey.append(sph_ham[key])
                            atm_orb_type.append([orb_type+"2"])
                            atm_orb_spectype.append(specific_orb[key])
                        else:
                            atmprojector_data[potatom[0]][1][orb_type] = projdata[data_ind]
                            atm_recipproj_data[potatom[0]][1][orb_type] = recip_projdata[data_ind]
                            atmpsuedodata[potatom[0]][1][orb_type] = psuedodata[data_ind] / grid
                            atmpsuedodata[potatom[0]][2][orb_type] = aedata[data_ind] / grid

                            atm_orbsphharmkey.append(sph_ham[key])
                            atm_orb_type.append(gen_orb[key])
                            atm_orb_spectype.append(specific_orb[key])
                    else:
                        atmprojector_data[potatom[0]][1][orb_type+"_ex"] = projdata[data_ind]
                        atm_recipproj_data[potatom[0]][1][orb_type+"_ex"] = recip_projdata[data_ind]
                        atmpsuedodata[potatom[0]][1][orb_type+"_ex"] = psuedodata[data_ind] / grid
                        atmpsuedodata[potatom[0]][2][orb_type+"_ex"] = aedata[data_ind] / grid
                        
                        ex_orbsphkey.append(sph_ham[key])
                        ex_orb_type.append(exgen_orb[key])
                    '''
                    set_val += -1
                    data_ind += 1
                    
            orbsphkey[potatom[0]] = [x for xs in atm_orbsphharmkey for x in xs]
            orbtype[potatom[0]] = [x for xs in atm_orb_type for x in xs]
            orb_spectype[potatom[0]] = [x for xs in atm_orb_spectype for x in xs]
            alex_orbsphkey[potatom[0]] = [x for xs in ex_orbsphkey for x in xs]
            alex_orbtype[potatom[0]] = [x for xs in ex_orb_type for x in xs]
            
            orb_energy[potatom[0]] = [x for xs in atm_orbsenergy for x in xs]
            
            sphkey_to_l = [0,1,1,1,2,2,2,2,2,3,3,3,3,3,3,3]
            
            # fit all the psuedo data to guassians
            keys = atmpsuedodata[potatom[0]][1].keys()
            type_to_l = {}
            type_to_l['s'] = 0
            type_to_l['p'] = 1
            type_to_l['d'] = 2
            type_to_l['f'] = 3
            gaus_params = {}#np.array((len(projdata),6))
            for key in keys:
                state_ener = atmpsuedodata[potatom[0]][4][key]
                cutoff = rad_cutoff*0.9+np.tanh((state_ener+8)/5)*0.5#self.cutoff_rad[atom]*0.8#**(1/2)#*1.4
                expcutoff = cutoff**(1/2)
                new_cutoff = cutoff**(1/2)

                #cutoff = rad_cutoff * 0.8
                #expcutoff = 1/cutoff
                #new_cutoff = 1/expcutoff**(1/2)
                min_cut = new_cutoff/15
                
                # first fit the pseudo orbital to get a good consistant guess for parameters
                pseudo_rad = grid[5:]#np.append(grid[5:],[grid[-1]*2]) # np.linspace(0,5,100)
                pseudo = atmpsuedodata[potatom[0]][1][key][5:]#np.append(atmpsuedodata[potatom[0]][1][key][5:],[0.000000001]) # self.orig_radial[orbgroup[0]]
                weights = 1/(pseudo_rad**(1/2)+0.2)#0.0001+pseudo_rad
                #weights[-1] = weights[-1]*2
                not_zero = pseudo != 0
                use_rad = pseudo_rad[not_zero]#np.append(pseudo_rad[not_zero],[pseudo_rad[not_zero][-1]*2])
                use_data = pseudo[not_zero]#np.append(pseudo[not_zero],[0])
                weights = 0.0001 + use_rad # 1/(use_rad**(1/2)+0.2)#0.0001 + use_rad
                cut_ind = np.argmin(np.abs(use_rad-cutoff)) # the closest point to cutoff value
                end_gaus = use_data[cut_ind]
                if self.verbose > 1:
                    print("cutoff:",cutoff,end_gaus)
                #weights[np.abs(use_rad-cutoff) < 1] = weights[np.abs(use_rad-cutoff) < 1] * 4 / (1+ np.abs(use_rad-cutoff)[np.abs(use_rad-cutoff) < 1])**2 #* (use_data[use_rad>cutoff]/end_gaus)**(1/4) * 4
                no_node = key[0] == key
                if self.verbose > 2:
                    print("pseudo data:",use_data,use_rad)
                    print("has a node?",not no_node)

                # initial parameters
                l = type_to_l[key[0]] #orb_to_l[orbperatom]
                if l==0:
                    weights =  0.1 + use_rad
                '''
                con1 = new_cutoff/2
                con2 = new_cutoff/3
                pa = 1
                pb = new_cutoff/1.5
                pc = 1/2
                pd = abs(pb-con1-min_cut/2) if con1+min_cut*1.5 < pb else (con1+min_cut*2.5)%pb
                pg = 0
                ph = abs(pb-con2-min_cut/2) if con2+min_cut*1.5 < pb else (con2+min_cut*2.5)%pb
                if no_node:
                    con3 = new_cutoff/2
                    con4 = 1/10
                    pf = abs(pb-con3-min_cut/2) if con3+min_cut*1.5 < pb else (con3+min_cut*2.5)%pb
                else:
                    con3 = new_cutoff
                    con4 = -2/3
                    pf = abs(pb-con3-min_cut/2)
                pe = (con4-pa/pb)*pf
                '''
                # try fitting with contraints built into function so can use curve_fit)
                #max_aceg = np.amax(np.abs([pa,pc,pe,pg]))
                pa = 0.6
                pb = new_cutoff/1.5
                pc = 1.2
                con1 = 1/2
                pd = pb * con1
                if no_node:
                    con3 = 1/3
                    pf = pb * con3
                    pe =  -0.1
                else:
                    con3 = 1/3
                    pf = pb * con3
                    pe = -0.5
                pg = 0
                ph = pb*1/2
                init_con1 = pd/pb #np.log( - 1)
                init_con4 = 1/2 #np.log(- 1)
                init_con3 = pf/pb #np.log(pb/pf - 1)
                init_con2 = pe/pf+pa/pb
                gaus_init = [pa,pb,init_con1,init_con2,init_con3,pc]#,pg,init_con4]
                gaus_fit = partial(func_for_rad_fit,g=0, con4=init_con4, l=l)
                #weights = np.append(weights,[0.02,0.1])
                sig = 1/(weights+0.00001) #np.append(weights,[0.02,0.05,.1,.2])
                #max_aceg = np.max(np.abs([pa,pc,pe,pg]))
                low_bounds = np.ones(6)*0.1#[0,0,0,0,0,0]#,0,0]
                low_bounds = [0.2,0.1,0.1,0.1,0.1,0.2]
                if l==0:
                    low_bounds = [0.2,0.1,0.1,0.1,0.1,0.01]
                high_bounds = [5.,new_cutoff*3,0.9,5.,0.8,5.]#,np.inf,0.9]
                if not no_node: # allow node in fit
                    low_bounds[3] = -5. # remove low bound for con2
                    high_bounds[4] = 5 # can be larger than b now, but generally is still not
                x = use_rad #np.append(use_rad,[0,0])
                y = use_data #np.append(use_data,[cutoff+2,cutoff+5])
                try:
                    popt,pcov = curve_fit(gaus_fit,x,y,p0=gaus_init,sigma=sig,bounds=(low_bounds,high_bounds),ftol=0.000001, xtol=0.000001,method="trf",max_nfev=2000)
                    [a,b,con1,con2,con3,c] = popt
                    d = con1 * b
                    f = con3 * b
                    e = (con2 - a/b) * f
                    g = 0
                    h = ph
                    #h = con4 * b
                    if self.verbose > 1:
                        print("fit succeeded:",l,[a,b,c,d,e,f])
                    
                except:
                    if self.verbose > 1:
                        print("failed to find a good fit of the pseudo orbital ",potatom[0], key)
                    [a, b, c, d, l, f, e, g, h] = [pa, pb, pc, pd, l, pf, pe, pg, ph]

                    #print(pseudo_result.fit_report())
                fitted_pseudo = func_for_rad(use_rad,a,b,c,d,e,f,g,h,l)
                average_error = np.sum(np.abs(fitted_pseudo[1:]-use_data[1:]) * (use_rad[1:]-use_rad[:-1]))
                if (average_error > 0.05) and (len(key) < 2): # has highish error and is included
                    print("WARNING: the pseudo orbital", potatom[0], key, "did not achieve a good fit with an error of",average_error)
                if self.verbose > 1:
                    print("error:",average_error)
                
                if self.verbose > 1:
                    print("Has a node?",not no_node)
                    print("gaus params fit to pseudo:",[a,b,c,d,e,f,g,h,l])
                fitted_gaus = func_for_rad(pseudo_rad,a,b,c,d,e,f,g,h,l)
                if self.verbose > 1 and self.plot_orbs:
                    plt.plot(pseudo_rad,pseudo)
                    plt.plot(use_rad,use_data)
                    plt.plot(pseudo_rad, fitted_gaus)
                    plt.show()
                    plt.close()
                new_rad = np.linspace(0,5,100)
                new_orbrad = func_for_rad(new_rad,a,b,c,d,e,f,g,h,l)
                max_val = np.amax(new_orbrad)
                max_rad = np.argmax(new_orbrad)
                tenth_radcut = new_rad[max_rad:][np.abs(new_orbrad[max_rad:])<max_val/10]
                if len(tenth_radcut) > 0:
                    tenth_radcut = tenth_radcut[0]
                else:
                    tenth_radcut = 5
                #print("tenth cutoff value:",tenth_radcut,new_rad[max_rad])
                gaus_params[key] = [[a,b,c,d,e,f,l],tenth_radcut,new_rad[max_rad]]
                
            
            atmgaus[potatom[0]] = gaus_params

        # save data to global variables
        self.atmProjectordata = atmprojector_data
        self.atmRecipProjdata = atm_recipproj_data
        self.atmPsuedoAeData = atmpsuedodata  # ["atm"][0] is grid, ["atm"][1] is psuedo, ["atm"][2] is ae, ['atm'] is the pseudo cutoff,['atm'][4] is the state energy
        self.atmGausPseudo = atmgaus
        
        self.atmOrbsphkey = orbsphkey
        self.atmOrbEnergy = orb_energy
        self.atmOrbtype = orbtype
        self.atmOrb_spectype = orb_spectype
        self.atmEx_orbtype = alex_orbtype
        self.atmEx_orbsphkey = alex_orbsphkey
        self.elem_elec_count = elec_count
        if self.verbose > 0:
            print("valence orb info:",orbsphkey,orbtype,orb_spectype)
            print("excited orb info:",alex_orbsphkey,alex_orbtype)

        print("got POT info")

    def get_local_environment(self):
        """
        This function gets the local environments of each atom in terms of local atom theta, phi, and radius.
        This is used to project onto the spherical harmonics to find the optimal rotation.
        Return:
        """
        # build larger list of atoms (in 27 neighboring translation cells)
        all_atoms = []
        for atm in self.cartAtoms:
            for trans1 in range(3):
                for trans2 in range(3):
                    for trans3 in range(3):
                        vec = (trans1 - 1) * self._a[0] + (trans2 - 1) * self._a[1] + (trans3 - 1) * self._a[2]
                        all_atoms.append(vec + atm)
        all_atoms = np.array(all_atoms)
        all_env = []
        # find closest atoms and then find all atoms that are within closest * 1.2 and add for atom
        for atm in self.cartAtoms:
            diff_atms = all_atoms - atm[None,:]
            dist_atms = np.linalg.norm(diff_atms, axis=1)
            closest_rad = np.sort(dist_atms)[1]
            all_close_atms = diff_atms[np.arange(len(diff_atms))[(dist_atms != 0) & (dist_atms < closest_rad*1.2)]]
            #print("closest atm radius:", closest_rad)
            rad = np.linalg.norm(all_close_atms, axis=1)
            theta = np.arccos(all_close_atms[:,2] / rad)
            theta[rad == 0.0000000001] = 0
            phi = np.arctan2(all_close_atms[:,1], all_close_atms[:,0])
            #print("all atoms:", all_close_atms, rad, phi, theta)
            all_env.append(np.array([phi,theta,rad]))
        self.atom_environ = all_env



    def get_orbs_fromPOT(self,orbfactor=1.0,invariant=True):
        """
        Creates the 3D initial orbitals in real space from the pseudo radial orbitals.
        Also sets the amount of orbitals by looping through atoms and their orbitals, thus intializes many orbital-dependent variables.

        Args:    
            orbfactor (float): Multipled by the radial part of the pseudo radial orbital to either shrink or grow them.
        """

        self.get_WFdata_fromPOT()
        #self.projectorWF = {}
        self.min_radphitheta = {}
        self.vector_toallatoms = {}
        self.psuedoWF = {}
        self.ratio_atomic_total = {}
        self.tot_orbs = {}
        self.orig_radial = {}
        self.orig_expparams = {}
        self.orig_gausparams = {}
        self.tenth_cutoff = {}
        self.orig_maxrad = {}
        # info for the reciprocal grid projectors
        self.recip_orbitalWF = {}
        self.recip_projs = {}
        self.one_orbitalWF = {}
        #self.ex_pseudos = {}
        # the key for the spherical harmonics
        self.sph_harm_key = []
        self.atm_elec = np.zeros(self.numAtoms)
        cart_gpoints = _red_to_cart((self.b_vecs[0], self.b_vecs[1], self.b_vecs[2]), np.array(self.gpoints[0],dtype=np.int_))
        vec = cart_gpoints.transpose()  # point - np.array([atm]).transpose()
        recip_rad = np.linalg.norm(vec, axis=0)
        recip_rad[recip_rad == 0] = 0.0000000001
        recip_theta = np.arccos(vec[2] / recip_rad)
        recip_theta[recip_rad == 0.0000000001] = 0
        recip_phi = np.arctan2(vec[1], vec[0])
        orb_to_l = [0,1,1,1,2,2,2,2,2,3,3,3,3,3,3,3]

        psuedo_norm = {}
        #build the POTCAR orbitals in the correct grid
        X, Y, Z = np.mgrid[0:(1 - 1 / self.gridxyz[0]):self.gridxyz[0] * 1j,
                  0:(1 - 1 / self.gridxyz[1]):self.gridxyz[1] * 1j, 0:(1 - 1 / self.gridxyz[2]):self.gridxyz[2] * 1j]
        self.X = X
        self.Y = Y
        self.Z = Z
        self.cartXYZ = _red_to_cart((self._a[0], self._a[1], self._a[2]),
                                    np.array([X.flatten(), Y.flatten(), Z.flatten()]).transpose())
        self.cartXYZ = self.cartXYZ.transpose()

        self.min_xyz = {}
        self.min_xyzreduced = {}


        orbcount = 0
        exorbcount = 0
        self.cutoff_rad = np.zeros(self.numAtoms)
        
        # make a version that gives the minradphitheta for atom at [0,0,0]
        zeroall_atoms = []
        for trans1 in range(3):
                for trans2 in range(3):
                    for trans3 in range(3):
                        vec = (trans1 - 1) * self._a[0] + (trans2 - 1) * self._a[1] + (trans3 - 1) * self._a[2]
                        zeroall_atoms.append(vec)
        # print(all_atoms)
        point = self.cartXYZ
        zeromin_rad = np.ones((len(self.cartXYZ[0]))) * 100
        zeromin_phi = np.ones((len(self.cartXYZ[0]))) * 100
        zeromin_theta = np.ones((len(self.cartXYZ[0]))) * 100
        zerovectors_to_atoms = np.ones((len(self.cartXYZ[0]), 3, 27)) * 100
        numatm = 0
        # get vectors to surrounding atoms in spherical coordinates
        for atm in zeroall_atoms:
            vec = point - np.array([atm]).transpose()
            rad = np.linalg.norm(vec, axis=0)
            rad[rad==0] = 0.0000000001
            theta = np.arccos(vec[2] / rad)
            theta[rad == 0.0000000001] = 0
            phi = np.arctan2(vec[1], vec[0])
            bool1 = rad < zeromin_rad
            zeromin_rad[bool1] = rad[bool1]
            zeromin_theta[bool1] = theta[bool1]
            zeromin_phi[bool1] = phi[bool1]
            zerovectors_to_atoms[:, :, numatm] = np.array([rad, phi, theta]).transpose()
            numatm += 1
        self.zerovector_atms = zerovectors_to_atoms
        self.zeromin_rad = zeromin_rad
        self.zeromin_theta = zeromin_theta
        self.zeromin_phi = zeromin_phi
        
        #make data for the orbitals of atoms
        self.orbpos = []
        self.lastatom = []
        self.orbatomnum = []
        self.orbatomname = []
        self.exorbpos = []
        self.exorbatomnum = []
        allatm_orbsphharm = []
        allatm_orb_energy = []
        allatm_orbtype = []
        allatm_orbspectype = []
        exallatm_orbsphharm = []
        exall_orbvecs = []
        all_orbvecs = []
        exallatm_orbtype = []
        environ_tensors = {} # dictionary of local environment projected on l states
        #find the radial coordinate at each point to the closest translated atom
        for atomnum in range(self.numAtoms):
            atom = self.cartAtoms[atomnum]
            self.atm_elec[atomnum] = self.elem_elec_count[self.elements[atomnum]]
            all_atoms = []
            cutoff_radius = self.atmPsuedoAeData[self.elements[atomnum]][0][-1]
            real_radius_cutoff = self.atmPsuedoAeData[self.elements[atomnum]][3] 
            self.cutoff_rad[atomnum] = real_radius_cutoff
            #print("cutoff_rad:",cutoff_radius,real_radius_cutoff)
            if real_radius_cutoff < cutoff_radius:
                cutoff_radius = real_radius_cutoff 
            psuedodat = self.atmPsuedoAeData[self.elements[atomnum]]
            gausdat = self.atmGausPseudo[self.elements[atomnum]]

            #recip projector stuff
            recip_proj_dat = self.atmRecipProjdata[self.elements[atomnum]]
            proj_gmax = recip_proj_dat[0]
            recip_proj_grid = np.arange(100) * proj_gmax / 100
            recip_rad_cut = recip_rad[recip_rad < proj_gmax]
            recip_rad_out = recip_rad[recip_rad >= proj_gmax]
            #print("is above gmax?",recip_rad_out)

            # get atom positions translated to surrounding unit cells
            for trans1 in range(3):
                for trans2 in range(3):
                    for trans3 in range(3):
                        vec = (trans1 - 1) * self._a[0] + (trans2 - 1) * self._a[1] + (trans3 - 1) * self._a[2]
                        all_atoms.append(vec + atom)
            # print(all_atoms)
            point = self.cartXYZ
            min_rad = np.ones((len(self.cartXYZ[0]))) * 100
            min_phi = np.ones((len(self.cartXYZ[0]))) * 100
            min_theta = np.ones((len(self.cartXYZ[0]))) * 100

            min_x = np.ones((len(self.cartXYZ[0]))) * 100
            min_y = np.ones((len(self.cartXYZ[0]))) * 100
            min_z = np.ones((len(self.cartXYZ[0]))) * 100
            atomWF = np.zeros((len(self.cartXYZ[0])), dtype=np.complex128)
            vectors_to_atoms = np.ones((len(self.cartXYZ[0]), 3, 27)) * 100
            numatm = 0
            # get vectors to surrounding atoms in spherical coordinates
            testatm = all_atoms[0]
            #print("point shape?",point.shape,np.array([testatm]).transpose().shape)
            for atm in all_atoms:
                vec = point - np.array([atm]).transpose()
                rad = np.linalg.norm(vec, axis=0)
                rad[rad==0] = 0.0000000001
                theta = np.arccos(vec[2] / rad)
                theta[rad == 0.0000000001] = 0
                phi = np.arctan2(vec[1], vec[0])

                bool1 = rad < min_rad
                min_rad[bool1] = rad[bool1]
                min_theta[bool1] = theta[bool1]
                min_phi[bool1] = phi[bool1]
                min_x[bool1] = vec[0][bool1]
                min_y[bool1] = vec[1][bool1]
                min_z[bool1] = vec[2][bool1]
                vectors_to_atoms[:, :, numatm] = np.array([rad, phi, theta]).transpose()
                numatm += 1
            self.min_rad[atomnum] = min_rad
            self.min_radphitheta[str(atomnum)] = [min_rad,min_phi,min_theta]
            self.vector_toallatoms[str(atomnum)] = vectors_to_atoms.transpose()
            self.min_xyz[atomnum] = np.array([min_x,min_y,min_z])
            self.min_xyzreduced[str(atomnum)] = _cart_to_red((self._a[0], self._a[1], self._a[2]), np.array([min_x,min_y,min_z]).transpose()).transpose()
            #print("min check:",self.min_xyzreduced[str(atomnum)][:25])
            #min_rad[min_rad==0] = 0.00001
            
            
            #construct the orbitals
            if self.elements[atomnum] == "H":
                num_orbs = 1
            elif self.include_dorbs[atomnum] == True:
                num_orbs = 9
            else:
                num_orbs = 4
            allsph_key = self.atmOrbsphkey[self.elements[atomnum]]
            allorb_ener = self.atmOrbEnergy[self.elements[atomnum]]
            allorb_type = self.atmOrbtype[self.elements[atomnum]]
            allspec_orbtype = self.atmOrb_spectype[self.elements[atomnum]]
            allex_orbtype = self.atmEx_orbtype[self.elements[atomnum]]
            allex_sphkey = self.atmEx_orbsphkey[self.elements[atomnum]]
            num_orbs = len(allsph_key)
            num_exorbs = len(allex_sphkey)
            allatm_orbsphharm.append(allsph_key)
            allatm_orb_energy.append(allorb_ener)
            allatm_orbtype.append(allorb_type)
            allatm_orbspectype.append(allspec_orbtype)
            exallatm_orbsphharm.append(allex_sphkey)
            exallatm_orbtype.append(allex_orbtype)

            # project spherical harmonics onto local environment
            atom_env = self.atom_environ[atomnum]
            new_orb_vector = [] # this will be a vector of vectors, the inside vectors can vary in size
            environ_tensors[atomnum] = {}
            for key in allsph_key: # key will be between 0 and 8 (for d) or even 15 (for f)
                if key == 0: # case of s orbitals. just copy over
                    Mab = np.sum(atom_env[2,:])
                    environ_tensors[atomnum][0] = Mab
                    new_orb_vector.append([1])
                if key == 1: # case of p orbitals (next two will be other ones)
                    init = np.array([[1,0,0],[0,1,0],[0,0,1]])
                    Mab = np.zeros((3,3))
                    for p1 in range(3):
                        for p2 in range(3):
                            Mab[p1,p2] = np.sum(atom_env[2,:]*complex128funs(atom_env[0,:], atom_env[1,:],init[p1]) *
                                                complex128funs(atom_env[0,:], atom_env[1,:],init[p2]))
                    #print(Mab)
                    if np.sum(np.abs(Mab - np.diag(np.diag(Mab)))) < 0.0000001:
                        #print("already best!")
                        vecs = init.T
                        vals, _ = np.linalg.eigh(Mab)
                    else:
                        vals, vecs = np.linalg.eigh(Mab)
                    #print("eigen info!",vals, vecs.T)
                    best_p = vecs.T
                    environ_tensors[atomnum][1] = Mab
                    new_orb_vector.append(best_p[0].tolist())

                if key == 2:
                    new_orb_vector.append(best_p[1].tolist()) # best_p will have already been created from key==1
                if key == 3:
                    new_orb_vector.append(best_p[2].tolist()) # best_p will have already been created from key==1

                if key == 4: # case of d orbitals (next four will be other ones)
                    init = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]])
                    Mab = np.zeros((5, 5))
                    for d1 in range(5):
                        for d2 in range(5):
                            Mab[d1, d2] = np.sum(atom_env[2,:]*complex128funs(atom_env[0, :], atom_env[1, :], init[d1]) *
                                                 complex128funs(atom_env[0, :], atom_env[1, :], init[d2]))
                    #print(Mab)
                    environ_tensors[atomnum][2] = Mab
                    zero_row_inds = np.arange(5)[(np.abs(Mab-0) <= 0.000001).all(axis=1)]
                    keep_rows = np.arange(5)[~(np.abs(Mab-0) <= 0.000001).all(axis=1)]
                    Mab = Mab[keep_rows,:][:,keep_rows]

                    if np.sum(np.abs(Mab - np.diag(np.diag(Mab)))) < 0.0000001:
                        #print("already best!")
                        vecs = init.T[keep_rows,:][:,keep_rows]
                        vals, _ = np.linalg.eigh(Mab)
                    else:
                        vals, vecs = np.linalg.eigh(Mab)
                    #print("eigen info!", vals, vecs.T)
                    best_d = vecs.T
                    for row in zero_row_inds:
                        leng = len(best_d)
                        best_d = np.insert(np.insert(best_d, row, np.zeros(leng), axis=0), row, np.zeros(leng+1), axis=1)
                        best_d[row,row] = 1
                    new_orb_vector.append(best_d[0].tolist())
                if key == 5:
                    new_orb_vector.append(best_d[1].tolist()) # best_d will have already been created from key==4
                if key == 6:
                    new_orb_vector.append(best_d[2].tolist()) # best_d will have already been created from key==4
                if key == 7:
                    new_orb_vector.append(best_d[3].tolist()) # best_d will have already been created from key==4
                if key == 8:
                    new_orb_vector.append(best_d[4].tolist()) # best_d will have already been created from key==4
                if key == 9: # case of f orbitals (next four will be other ones)
                    init = np.array([[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0],
                                     [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]])
                    Mab = np.zeros((7, 7))
                    for f1 in range(7):
                        for f2 in range(7):
                            Mab[f1, f2] = np.sum(atom_env[2,:]*complex128funs(atom_env[0, :], atom_env[1, :], init[f1]) *
                                                 complex128funs(atom_env[0, :], atom_env[1, :], init[f2]))
                    #print(Mab)
                    environ_tensors[atomnum][3] = Mab
                    # remove zero rows (much more common in f orbital cases in high symmetry lattices)
                    zero_row_inds = np.arange(7)[(np.abs(Mab-0) <= 0.000001).all(axis=1)]
                    keep_rows = np.arange(7)[~(np.abs(Mab-0) <= 0.000001).all(axis=1)]
                    #print(((np.abs(Mab-0) <= 0.000001)).all(axis=1))
                    #print(zero_row_inds)
                    #print(keep_rows)
                    Mab = Mab[keep_rows,:][:,keep_rows]

                    if np.sum(np.abs(Mab - np.diag(np.diag(Mab)))) < 0.0000001:
                        #print("already best!")
                        vecs = init.T[keep_rows,:][:,keep_rows]
                        vals, _ = np.linalg.eigh(Mab)
                    else:
                        vals, vecs = np.linalg.eigh(Mab)
                    #print("eigen info!", vals, vecs.T)
                    best_f = vecs.T
                    for row in zero_row_inds:
                        leng = len(best_f)
                        best_f = np.insert(np.insert(best_f, row, np.zeros(leng), axis=0), row, np.zeros(leng+1), axis=1)
                        best_f[row,row] = 1
                    #print(best_f)
                    new_orb_vector.append(best_f[0].tolist())
                if key == 10:
                    new_orb_vector.append(best_f[1].tolist()) # best_d will have already been created from key==9
                if key == 11:
                    new_orb_vector.append(best_f[2].tolist()) # best_d will have already been created from key==9
                if key == 12:
                    new_orb_vector.append(best_f[3].tolist()) # best_d will have already been created from key==9
                if key == 13:
                    new_orb_vector.append(best_f[4].tolist()) # best_d will have already been created from key==9
                if key == 14:
                    new_orb_vector.append(best_f[5].tolist()) # best_d will have already been created from key==9
                if key == 15:
                    new_orb_vector.append(best_f[6].tolist()) # best_d will have already been created from key==9
            #print("final new vector:",new_orb_vector)

            # then also for the excited states
            switch_to_vecs = [[1],[1,0,0],[0,1,0],[0,0,1],[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1],
                              [1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]]
            if not invariant: # switch back to just default orientation
                new_orb_vector = []
                for key in allsph_key:
                    new_orb_vector.append(switch_to_vecs[key])
            all_orbvecs.append(new_orb_vector)
            if self.verbose > 1:
                print("Equivariant combo of sph harm:", atomnum, self.elements[atomnum], new_orb_vector)
            ex_orbvec = []
            for key in allex_sphkey:
                ex_orbvec.append(switch_to_vecs[key])
            exall_orbvecs.append(ex_orbvec)

            tot_orbWF = np.zeros((num_orbs,len(self.cartXYZ[0])))
            #projector stuff
            proj_orbWF = np.zeros((num_orbs,len(self.cartXYZ[0])))
            exproj_orbWF = np.zeros((num_exorbs,len(self.cartXYZ[0])))
            valencePsuedos = np.zeros((num_orbs,len(self.cartXYZ[0])))
            excitedPsuedos = np.zeros((num_exorbs,len(self.cartXYZ[0])))
            #proj_data = self.atmProjectordata[self.elements[atomnum]]
            #proj_cutoff = proj_data[0]
            #print(proj_cutoff)
            #proj_grid = np.linspace(0,proj_cutoff,num=100, endpoint=False)
            
            # --  all of this is for constructing initial guess at ratio --
            #include only the center atom which is at [0,0,0]
            atomic_WF = np.zeros((num_orbs,len(self.cartXYZ[0])))
            simple_rad = np.linspace(0,5,num=100,endpoint=False)
            og_radial = np.zeros((num_orbs,100))
            orbgaus_params = np.zeros((num_orbs,7))
            tenth_cutoff = np.zeros((num_orbs))
            orig_maxrad = np.zeros((num_orbs))
            og_expparams = np.zeros((num_orbs,2))
            radial_part = np.zeros((len(self.cartXYZ[0])))
            rad_inside_cutoff = zeromin_rad<cutoff_radius
            rad_outide_cutoff = zeromin_rad>=cutoff_radius
            for orb in range(num_orbs):
                orb_sphkey = allsph_key[orb]
                orb_vec = new_orb_vector[orb]
                orbtype = allorb_type[orb]
                l = orb_to_l[orb_sphkey]
                # remove any extra zeros at the end of the pseudo data
                pseudo_data = np.array(psuedodat[1][orbtype])
                pseudo_rad = np.array(psuedodat[0])
                notzero = pseudo_data !=0
                pseudo_data = pseudo_data[notzero]
                pseudo_rad = pseudo_rad[notzero]
                if (len(pseudo_data) != len(notzero)) or ((pseudo_rad[-1] - cutoff_radius) < -0.0001): # change the cutoff radius
                    new_cutrad = pseudo_rad[-1]
                    rad_inside_cutoff = zeromin_rad<new_cutrad
                    rad_outide_cutoff = zeromin_rad>=new_cutrad
                else:
                    new_cutrad = cutoff_radius
                change_size = True
                if change_size == True:
                    pseudo_rad = pseudo_rad*orbfactor
                    new_cutrad = pseudo_rad[-1]
                    rad_inside_cutoff = zeromin_rad<new_cutrad
                    rad_outide_cutoff = zeromin_rad>=new_cutrad
                radial_part[rad_inside_cutoff] = interpolate.griddata(pseudo_rad,pseudo_data,zeromin_rad[rad_inside_cutoff],method='cubic', fill_value=pseudo_data[0])
                if self.verbose > 1:
                    print("actual cutrad:",new_cutrad)
                enddata = pseudo_data[pseudo_rad <= new_cutrad]
                endrad = pseudo_rad[pseudo_rad <= new_cutrad]
                phir = enddata[-1]
                delphir = (enddata[-1]-enddata[-2])/(endrad[-1]-endrad[-2])
                c = phir
                d = delphir*1
                r = new_cutrad
                b = -(d*r-l*c)/(2*c*r**2)#<--for gaus  #for exp: -(d*r-l*c)/(c*r)
                a = c/(np.exp(-b*r**2)*r**l)#<--for gaus  #for exp: c/(np.exp((d*r-l*c)/c)*r**l)
                radial_part[rad_outide_cutoff] = a*zeromin_rad[rad_outide_cutoff]**l*np.exp(-b*zeromin_rad[rad_outide_cutoff]**2) 
                angular_part = complex128funs(zeromin_phi,zeromin_theta,orb_vec)
                atomic_WF[orb] = radial_part*angular_part
                
                og_radial[orb][simple_rad<new_cutrad] = interpolate.griddata(pseudo_rad,pseudo_data,simple_rad[simple_rad<new_cutrad],method='cubic', fill_value=pseudo_data[0])
                if not self.use_pseudo or True:
                    og_radial[orb][simple_rad>=new_cutrad] = a*simple_rad[simple_rad>=new_cutrad]**l*np.exp(-b*simple_rad[simple_rad>=new_cutrad]**2) 
                    #og_radial[orb][-1] = 0 # still bound to zero
                    '''
                    if new_cutrad < 1.5:
                        new_cutrad = 1.5
                        r = new_cutrad
                        c2 = a*new_cutrad**l*np.exp(-b*new_cutrad**2)
                        d2 = -a*new_cutrad**(l-1)*(2*b*new_cutrad**2-l)*np.exp(-b*new_cutrad**2)
                        d2 = -abs(d2)*20
                        b2 = -(d2*r-l*c2)/(2*c2*r**2)#<--for gaus  #for exp: -(d*r-l*c)/(c*r)
                        a2 = c2/(np.exp(-b2*r**2)*r**l)#<--for gaus  #for exp: c/(np.exp((d*r-l*c)/c)*r**l)
                        rad_farout = min_rad>new_cutrad
                        radial_part[rad_farout] = a2*min_rad[rad_farout]**l*np.exp(-b2*min_rad[rad_farout]**2)
                    '''
                og_expparams[orb] = [a,b]
                orbgaus_params[orb] = gausdat[orbtype][0]
                tenth_cutoff[orb] = gausdat[orbtype][1] * orbfactor
                orig_maxrad[orb] = gausdat[orbtype][2] * orbfactor
                
            # include all atoms but still center at [0,0,0]
            zerotot_orbWF = np.zeros((num_orbs,len(self.cartXYZ[0])))
            if self.vol < 200:
                zero_atoms = zerovectors_to_atoms.transpose() # go over 27 translated atoms 
            else:
                zero_atoms = np.array([[zeromin_rad, zeromin_phi, zeromin_theta]]) # just use the one in the primitive cell
            for vec in zero_atoms:
                min_rad = vec[0]
                min_phi = vec[1]
                min_theta = vec[2]
                radial_part = np.zeros((len(self.cartXYZ[0])))
                rad_inside_cutoff = min_rad<cutoff_radius
                rad_outide_cutoff = min_rad>=cutoff_radius
                for orb in range(num_orbs):
                    orb_sphkey = allsph_key[orb]
                    orb_vec = new_orb_vector[orb]
                    orbtype = allorb_type[orb]
                    l = orb_to_l[orb_sphkey]
                    # remove any extra zeros at the end of the pseudo data
                    pseudo_data = np.array(psuedodat[1][orbtype])
                    pseudo_rad = np.array(psuedodat[0])
                    notzero = pseudo_data !=0
                    pseudo_data = pseudo_data[notzero]
                    pseudo_rad = pseudo_rad[notzero]
                    if (len(pseudo_data) != len(notzero)) or ((pseudo_rad[-1] - cutoff_radius) < -0.0001): # change the cutoff radius
                        new_cutrad = pseudo_rad[-1]
                        rad_inside_cutoff = min_rad<new_cutrad
                        rad_outide_cutoff = min_rad>=new_cutrad
                    else:
                        new_cutrad = cutoff_radius
                    if change_size == True:
                        pseudo_rad = pseudo_rad*orbfactor
                        new_cutrad = pseudo_rad[-1]
                        rad_inside_cutoff = min_rad<new_cutrad
                        rad_outide_cutoff = min_rad>=new_cutrad
                    radial_part[rad_inside_cutoff] = interpolate.griddata(pseudo_rad,pseudo_data,min_rad[rad_inside_cutoff],method='cubic', fill_value=pseudo_data[0])
                    enddata = pseudo_data[pseudo_rad <= new_cutrad]
                    endrad = pseudo_rad[pseudo_rad <= new_cutrad]
                    phir = enddata[-1]
                    delphir = (enddata[-1]-enddata[-2])/(endrad[-1]-endrad[-2])
                    c = phir
                    d = delphir*1
                    r = new_cutrad
                    b = -(d*r-l*c)/(2*c*r**2)#<--for gaus  #for exp: -(d*r-l*c)/(c*r)
                    a = c/(np.exp(-b*r**2)*r**l)#<--for gaus  #for exp: c/(np.exp((d*r-l*c)/c)*r**l)
                    radial_part[rad_outide_cutoff] = a*min_rad[rad_outide_cutoff]**l*np.exp(-b*min_rad[rad_outide_cutoff]**2)
                    '''
                    if new_cutrad < 1.5:
                        new_cutrad = 1.5
                        r = new_cutrad
                        c2 = a*new_cutrad**l*np.exp(-b*new_cutrad**2)
                        d2 = -a*new_cutrad**(l-1)*(2*b*new_cutrad**2-l)*np.exp(-b*new_cutrad**2)
                        d2 = -abs(d2)*20
                        b2 = -(d2*r-l*c2)/(2*c2*r**2)#<--for gaus  #for exp: -(d*r-l*c)/(c*r)
                        a2 = c2/(np.exp(-b2*r**2)*r**l)#<--for gaus  #for exp: c/(np.exp((d*r-l*c)/c)*r**l)
                        rad_farout = min_rad>new_cutrad
                        radial_part[rad_farout] = a2*min_rad[rad_farout]**l*np.exp(-b2*min_rad[rad_farout]**2)
                    '''
                    angular_part = complex128funs(min_phi,min_theta,orb_vec)
                    zerotot_orbWF[orb] += radial_part*angular_part

                    
            # -- now get data for initial orbitals, pseudo orbitals, and projectors -- 
            # include all the surrounding atoms
            if self.vol < 200 and False:
                loop_atoms = vectors_to_atoms.transpose() # go over 27 translated atoms 
            else:
                loop_atoms = np.array([self.min_radphitheta[str(atomnum)]]) # just use the one in the primitive cell
            for vec in loop_atoms:
                min_rad = vec[0]
                min_phi = vec[1]
                min_theta = vec[2]
                radial_part = np.zeros((len(self.cartXYZ[0])))
                rad_inside_cutoff = min_rad<cutoff_radius
                rad_outide_cutoff = min_rad>=cutoff_radius
                # get valence states to include in TB model
                for orb in range(num_orbs):
                    orb_sphkey = allsph_key[orb]
                    orb_vec = new_orb_vector[orb]
                    orbtype = allorb_type[orb]
                    l = orb_to_l[orb_sphkey]
                    # remove any extra zeros at the end of the pseudo data
                    pseudo_data = np.array(psuedodat[1][orbtype])
                    pseudo_rad = np.array(psuedodat[0])
                    notzero = pseudo_data !=0
                    pseudo_data = pseudo_data[notzero]
                    pseudo_rad = pseudo_rad[notzero]
                    if (len(pseudo_data) != len(notzero)) or ((pseudo_rad[-1] - cutoff_radius) < -0.0001): # change the cutoff radius
                        new_cutrad = pseudo_rad[-1]
                        rad_inside_cutoff = min_rad<new_cutrad
                        rad_outide_cutoff = min_rad>=new_cutrad
                    else:
                        new_cutrad = cutoff_radius
                    if change_size == True:
                        pseudo_rad = pseudo_rad*orbfactor
                        new_cutrad = pseudo_rad[-1]
                        rad_inside_cutoff = min_rad<new_cutrad
                        rad_outide_cutoff = min_rad>=new_cutrad
                    radial_part[rad_inside_cutoff] = interpolate.griddata(pseudo_rad,pseudo_data,min_rad[rad_inside_cutoff],method='cubic', fill_value=pseudo_data[0])
                    enddata = pseudo_data[pseudo_rad <= new_cutrad]
                    endrad = pseudo_rad[pseudo_rad <= new_cutrad]
                    phir = enddata[-1]
                    delphir = (enddata[-1]-enddata[-2])/(endrad[-1]-endrad[-2])
                    c = phir
                    d = delphir*1
                    r = new_cutrad
                    b = -(d*r-l*c)/(2*c*r**2)#<--for gaus  #for exp: -(d*r-l*c)/(c*r)
                    a = c/(np.exp(-b*r**2)*r**l)#<--for gaus  #for exp: c/(np.exp((d*r-l*c)/c)*r**l)
                    radial_part[rad_outide_cutoff] = a*min_rad[rad_outide_cutoff]**l*np.exp(-b*min_rad[rad_outide_cutoff]**2)
                    '''
                    if new_cutrad < 1.5:
                        new_cutrad = 1.5
                        r = new_cutrad
                        c2 = a*new_cutrad**l*np.exp(-b*new_cutrad**2)
                        d2 = -a*new_cutrad**(l-1)*(2*b*new_cutrad**2-l)*np.exp(-b*new_cutrad**2)
                        d2 = -abs(d2)*20
                        b2 = -(d2*r-l*c2)/(2*c2*r**2)#<--for gaus  #for exp: -(d*r-l*c)/(c*r)
                        a2 = c2/(np.exp(-b2*r**2)*r**l)#<--for gaus  #for exp: c/(np.exp((d*r-l*c)/c)*r**l)
                        rad_farout = min_rad>new_cutrad
                        radial_part[rad_farout] = a2*min_rad[rad_farout]**l*np.exp(-b2*min_rad[rad_farout]**2)
                    '''
                    # apply a strict cutoff radius  
                    #radial_part[min_rad>=real_radius_cutoff] = 0
                    
                    angular_part = complex128funs(min_phi,min_theta,orb_vec)
                    tot_orbWF[orb] += radial_part*angular_part

                    #psuedo orbitals
                    #radial_part[rad_inside_cutoff] = interpolate.griddata(psuedodat[0][notzero], psuedodat[1][orbtype], min_rad[rad_inside_cutoff], method='cubic',fill_value=psuedodat[1][orbtype][0])
                    radial_part[rad_outide_cutoff] = psuedodat[1][orbtype][-1]*np.exp(-4*(min_rad[rad_outide_cutoff]-new_cutrad))
                    valencePsuedos[orb] += radial_part * angular_part
                
                # get excited states which will not be included in TB model
                for orbex in range(num_exorbs):
                    orb_sphkey = allex_sphkey[orbex]
                    orb_vec = ex_orbvec[orbex]
                    orbtype = allex_orbtype[orbex]
                    l = orb_to_l[orb_sphkey]
                    # remove any extra zeros at the end of the pseudo data
                    pseudo_data = np.array(psuedodat[1][orbtype])
                    pseudo_rad = np.array(psuedodat[0])
                    notzero = pseudo_data != 0
                    pseudo_data = pseudo_data[notzero]
                    pseudo_rad = pseudo_rad[notzero]
                    if (len(pseudo_data) != len(notzero)) or (
                            (pseudo_rad[-1] - cutoff_radius) < -0.0001):  # change the cutoff radius
                        new_cutrad = pseudo_rad[-1]
                        rad_inside_cutoff = min_rad < new_cutrad
                        rad_outide_cutoff = min_rad >= new_cutrad
                    else:
                        new_cutrad = cutoff_radius
                    if change_size == True:
                        pseudo_rad = pseudo_rad * orbfactor
                        new_cutrad = pseudo_rad[-1]
                        rad_inside_cutoff = min_rad < new_cutrad
                        rad_outide_cutoff = min_rad >= new_cutrad

                    radial_part[rad_inside_cutoff] = interpolate.griddata(pseudo_rad,pseudo_data,min_rad[rad_inside_cutoff],method='cubic', fill_value=pseudo_data[0])
                    enddata = pseudo_data[pseudo_rad <= new_cutrad]
                    endrad = pseudo_rad[pseudo_rad <= new_cutrad]
                    phir = enddata[-1]
                    delphir = (enddata[-1]-enddata[-2])/(endrad[-1]-endrad[-2])
                    c = phir
                    d = delphir*1
                    r = new_cutrad
                    b = -(d*r-l*c)/(2*c*r**2)#<--for gaus  #for exp: -(d*r-l*c)/(c*r)
                    a = c/(np.exp(-b*r**2)*r**l)#<--for gaus  #for exp: c/(np.exp((d*r-l*c)/c)*r**l)
                    radial_part[rad_outide_cutoff] = a*min_rad[rad_outide_cutoff]**l*np.exp(-b*min_rad[rad_outide_cutoff]**2)
                    '''
                    if new_cutrad < 1.5:
                        #print("something is being made smaller!")
                        new_cutrad = 1.5
                        r = new_cutrad
                        c2 = a*new_cutrad**l*np.exp(-b*new_cutrad**2)
                        d2 = -a*new_cutrad**(l-1)*(2*b*new_cutrad**2-l)*np.exp(-b*new_cutrad**2)
                        d2 = -abs(d2)*20
                        b2 = -(d2*r-l*c2)/(2*c2*r**2)#<--for gaus  #for exp: -(d*r-l*c)/(c*r)
                        a2 = c2/(np.exp(-b2*r**2)*r**l)#<--for gaus  #for exp: c/(np.exp((d*r-l*c)/c)*r**l)
                        rad_farout = min_rad>new_cutrad
                        radial_part[rad_farout] = a2*min_rad[rad_farout]**l*np.exp(-b2*min_rad[rad_farout]**2)
                    '''
                    # apply a strict cutoff radius  
                    #radial_part[min_rad>=real_radius_cutoff] = 0
                    
                    angular_part = complex128funs(min_phi,min_theta,orb_vec)
                    excitedPsuedos[orbex] += radial_part*angular_part

                    
                    #radial_part[rad_inside_cutoff] = interpolate.griddata(psuedodat[0], psuedodat[1][orbtype + "_ex"], min_rad[rad_inside_cutoff], method='cubic',fill_value=psuedodat[1][orbtype + "_ex"][0])
                    #radial_part[rad_outide_cutoff] = psuedodat[1][orbtype + "_ex"][-1]*np.exp(-4*(min_rad[rad_outide_cutoff]-cutoff_radius))
                    #excitedPsuedos[orb] += radial_part * angular_part

                    '''
                    #get projectors
                    inside_proj_cutoff = min_rad<proj_cutoff
                    outside_proj_cutoff = min_rad>=proj_cutoff
                    radial_part[inside_proj_cutoff] = interpolate.griddata(proj_grid,proj_data[1][orbtype],min_rad[inside_proj_cutoff],method='cubic',fill_value=proj_data[1][orbtype][0])
                    radial_part[outside_proj_cutoff] = 0
                    proj_orbWF[orb] += radial_part*angular_part
                    radial_part[inside_proj_cutoff] = interpolate.griddata(proj_grid,proj_data[1][orbtype+"_ex"],min_rad[inside_proj_cutoff],method='cubic', fill_value=proj_data[1][orbtype+"_ex"][0])
                    exproj_orbWF[orb] += radial_part * angular_part
                    '''
            '''
            #get recip projector info
            recip_orbs = {}
            recip_projs = {}
            exrecip_orbs={}
            for orb in range(num_orbs):
                orb_sphkey = allsph_key[orb]
                orbtype = allorb_type[orb]
                #also real quick get save the parameter type for later use
                self.sph_harm_key.append(orb)
                # get projector info
                l = orb_to_l[orb_sphkey]
                radial_part = np.zeros((len(recip_rad)),dtype=np.complex128)
                radial_part[recip_rad*2<proj_gmax] = interpolate.griddata(recip_proj_grid,recip_proj_dat[1][orbtype], recip_rad[recip_rad*2 < proj_gmax]*2,method='cubic',fill_value=recip_proj_dat[1][orbtype][0])
                #plt.scatter(recip_rad_cut,radial_part[recip_rad<proj_gmax])
                #plt.show()
                angular_part = complex128funs(recip_phi, recip_theta, orb_sphkey)
                center = self.primAtoms[atomnum]
                exp_term = np.exp(-2j * np.pi * np.dot(self.gpoints[0], center))  # center the projector correctly
                recip_orbs[str(orb)] = (-1j) ** (l) * radial_part * angular_part * exp_term
                recip_projs[str(orb)] = (-1j) ** (l) * radial_part * angular_part #values centered at 0,0,0

                radial_part[recip_rad<proj_gmax] = interpolate.griddata(recip_proj_grid,recip_proj_dat[1][orbtype + "_ex"], recip_rad_cut, method='cubic',fill_value=recip_proj_dat[1][orbtype + "_ex"][0])
                exrecip_orbs[str(orb)] = (-1j) ** (l) * radial_part * angular_part * exp_term
            '''
            # save into global variables
            for orb in range(num_orbs):
                self.one_orbitalWF[str(orbcount)] = np.reshape(tot_orbWF[orb], self.gridxyz)

                #include excited states in the projectors
                #self.projectorWF[str(orbcount)] = np.reshape(proj_orbWF[orb], self.gridxyz)
                #self.projectorWF[str(orbcount) + "_ex"] = np.reshape(exproj_orbWF[orb], self.gridxyz)
                self.psuedoWF[str(orbcount)] = np.reshape(valencePsuedos[orb], self.gridxyz)
                #holdWF = np.reshape(excitedPsuedos[orb], self.gridxyz)
                #normholdWF, integral = normalize_wf(holdWF, self._a, self.gridxyz, return_integral=True)
                #self.psuedoWF[str(orbcount) + "_ex"] = holdWF #normholdWF / integral

                # get projector info
                #need to scale projectors by volume * grid / 2/pi
                nz = self.gridxyz[2]
                ny = self.gridxyz[1]
                nx = self.gridxyz[0]
                cross_ab = np.cross(self._a[0], self._a[1])
                supercell_volume = abs(np.dot(cross_ab, self._a[2]))
                #print("cell volume:",supercell_volume,nz*ny*nx)
                #self.recip_orbitalWF[str(orbcount)] = np.array(recip_orbs[str(orb)]) #*/(4*np.pi/nx/ny/nz)
                
                # get the ratio between the nearest atomic orb and all atomic orbs
                ratio = atomic_WF[orb]/(zerotot_orbWF[orb]+0.00000001)
                #print("how much is zero:",len(ratio[np.abs(zerotot_orbWF[orb])<0.00005]))
                ratio[np.abs(zerotot_orbWF[orb])<0.00005] = 0
                #ratio[zeromin_rad<2] = 1*(1-np.tanh(zeromin_rad[zeromin_rad<2]))+ratio[zeromin_rad<2]*np.tanh(zeromin_rad[zeromin_rad<2])
                #ratio[(zeromin_rad<2) & (ratio<0.2)] = 1
                #ratio[ratio<0.3] = 0
                ratio[np.abs(ratio)>15] = 0
                self.ratio_atomic_total[orbcount] = ratio
                self.orig_radial[orbcount] = og_radial[orb]
                self.orig_gausparams[orbcount] = orbgaus_params[orb]
                self.tenth_cutoff[orbcount] = tenth_cutoff[orb]
                self.orig_maxrad[orbcount] =orig_maxrad[orb]
                self.orig_expparams[orbcount] = og_expparams[orb]
                self.tot_orbs[orbcount] = zerotot_orbWF[orb]

                orbcount += 1
            
            for addorb in range(num_orbs):
                self.orbpos.append(self.primAtoms[atomnum])
                self.orbatomnum.append(atomnum)
                self.orbatomname.append(self.elements[atomnum])
            for addexorb in range(num_exorbs):
                self.exorbpos.append(self.primAtoms[atomnum])
                self.exorbatomnum.append(atomnum)
                #self.ex_pseudos[str(exorbcount)] = np.reshape(excitedPsuedos[addexorb], self.gridxyz)
                exorbcount += 1
            self.lastatom.append(orbcount)
        
        # just do excited states
        #if self.verbose > 1:
        #    print("check excited pseudo keys:",self.ex_pseudos.keys())
        #for key in self.ex_pseudos.keys():
        #    self.one_orbitalWF[key+"ex"] = self.ex_pseudos[key]
        #self.one_orbitalWF = self.ex_pseudos
        # resave all the orbital infomation
        self.num_exorbs = len(self.exorbpos)
        self.num_bands = len(self.orbpos)
        self.num_orbs = len(self.orbpos)
        self.orbpos = np.array(self.orbpos)
        if not hasattr(self, 'max_band'):
            self.max_band = self.num_bands
        self.sph_harm_key = [x for xs in allatm_orbsphharm for x in xs]
        self.init_orb_energy = [x for xs in allatm_orb_energy for x in xs]
        #print("initial orb energies:",self.init_orb_energy)
        self.orbtype = [x for xs in allatm_orbtype for x in xs]
        self.orbtype = np.array(self.orbtype)
        self.exactorbtype = [x for xs in allatm_orbspectype for x in xs]
        self.ex_orbtype = [x for xs in exallatm_orbtype for x in xs]
        self.ex_sph_harm_key = [x for xs in exallatm_orbsphharm for x in xs]
        self.ex_orb_vec = [x for xs in exall_orbvecs for x in xs]
        self.orb_vec = [x for xs in all_orbvecs for x in xs]
        self.environ_tensors = environ_tensors
        #print("orb vectors:",self.orb_vec)
        #print(self.ex_orb_vec)
        print("orb types!",self.orbtype)
        #print(self.one_orbitalWF.keys())
        #psuedoWF = copy.deepcopy(self.one_orbitalWF)
        self.psuedoint = np.zeros(self.num_orbs)
        #self.one_orbitalWF = self.lowdin_orth(self.one_orbitalWF)
        #normalize orbs
        for orb,key in enumerate(self.one_orbitalWF.keys()):
            self.one_orbitalWF[key],integral = normalize_wf(self.one_orbitalWF[key], self._a, self.gridxyz,return_integral=True)
            self.orig_radial[orb] = self.orig_radial[orb]/integral
            self.orig_expparams[orb][0] = self.orig_expparams[orb][0]/integral
            self.orig_gausparams[orb][[0,2,4]] = self.orig_gausparams[orb][[0,2,4]]/integral
            self.tot_orbs[orb] = self.tot_orbs[orb]/integral
            #print("check norm!",orb,integral,supercell_volume,nz*ny*nx)
        self.cur_radial = self.orig_radial
        # get original radius for atomic orbital
        orb_radius = np.zeros(self.num_orbs)
        for orb in range(self.num_orbs):
            orig_radial = self.orig_radial[orb]
            orb_rad = np.linspace(0,8,num=100,endpoint=False)
            simple_rad = np.linspace(0,5,num=100,endpoint=False)
            gaus_orb = interpolate.griddata(simple_rad,orig_radial,orb_rad, method='cubic', fill_value=0) # for calculating pseudo norm and rad
            
            gaus_norm = integrate.trapezoid((gaus_orb*gaus_orb) * orb_rad ** 2, x=orb_rad)
            gaus_rad = integrate.trapezoid((gaus_orb*gaus_orb)/gaus_norm * orb_rad ** 3, x=orb_rad)
            orb_radius[orb] = gaus_rad
        self.orb_radius = orb_radius
        self.pseudo_radius = orb_radius
        print("orbital radius",np.around(orb_radius,decimals=4))
        #have added excited states to the projectorWF?
        #overlap = self.get_overlap_matrix(self.psuedoWF)
        #print("projector keys:",self.projectorWF.keys())
        #print("psuedo keys:",self.atmPsuedoAeData["Pb"][1].keys())
        #if self.verbose > 2:
        #    print("overlap of psuedo orbs:",overlap[:10])

        #self.projpsuedo_overlap = self.get_overlap_matrix(self.projectorWF, self.psuedoWF)
        keys = self.one_orbitalWF.keys()
        one_orbitalvec = np.zeros((len(keys),self.gridxyz[0],self.gridxyz[1],self.gridxyz[2]),dtype=np.complex128)
        for oi,orb in enumerate(keys):
            one_orbitalvec[oi] = self.one_orbitalWF[orb]
        self.one_orbitalWF = one_orbitalvec
        #keys = self.ex_pseudos.keys()
        #one_orbitalvec = np.zeros((len(keys),self.gridxyz[0],self.gridxyz[1],self.gridxyz[2]),dtype=np.complex128)
        #for oi,orb in enumerate(keys):
        #    one_orbitalvec[oi] = self.ex_pseudos[orb]
        #self.ex_pseudos = one_orbitalvec
        self.orig_one_orbitalWF = copy.deepcopy(self.one_orbitalWF)
        self.one_orbWFrecip = self.real_to_recip(self.one_orbitalWF)
        #if self.verbose > 2:
        #    print("should be identity:",self.projpsuedo_overlap[:10])

        print("pseudo electrons per atom:",self.atm_elec)

    def get_coefficients(self,orbitalWF, crystalWF, overlap_matrix,recip=False, band=None):
        """
        Find the coefficients for the amount of each pseudo orbital (Φ) in the pseudo wavefunction (Ψ).
        If the overlap is identity, the coefficients would just be the integral of the orbital-wavefunction overlap: Φa*Ψn = O_an.
        When the overlap is not identity S_ab, obtaining the coefficients (C_bn) requires solving the linear problem S_ab*C_bn = O_an.


        Args:    
            orbitalWF (dict): of length M: All the orbital functions in a dictionary. The orbital may be defined in real 3D space on reciprocal G vectors.
            crystalWF (dict): of length N: All the DFT wavefunctions in a dictionary. The wavefunction is defined on the same space as orbitalWF.
            overlap_matrix (MxM): matrix of complex float: The overlap of the orbitals. NOTE: The orbitals and their overlaps have a k-dependence.
            recip (bool): Whether the wavefunctions are defined in real or reciprocal space.
            band (int): If the coefficents of only one band in the crystalWF dict is needed, pass that band as an integer here.

        Returns:    
            unknown: The coefficients C_an of orbital a in band n.
        """

        num_orbs = len(orbitalWF)
        coeff = [[0]]
        # bands to range
        if band != None:
            bands = [band] #[str(band)]
        else:
            bands = range(len(crystalWF)) #crystalWF.keys()
        all_wavefun_overlap = np.zeros((num_orbs,len(bands)), dtype=np.complex128)

        if recip == True:
            # convert WF from dictionaries to np arrays
            num_gpnts = len(orbitalWF[0])#list(orbitalWF.keys())[0]])
            firstWF = orbitalWF # np.zeros((num_orbs,num_gpnts),dtype=np.complex128)
            #for orb1 in range(num_orbs):
            #    firstWF[orb1] = orbitalWF[list(orbitalWF.keys())[orb1]]
            all_wavefun_overlap = np.matmul(np.conj(firstWF),crystalWF.T) # return orbs x bands #reciprocal_integral(np.conj(WF1) * WF2)
        else:
            for bi,band in enumerate(bands):
                realwavefunc = crystalWF[band]
                for orb in range(num_orbs):
                    WF = orbitalWF[0]#list(orbitalWF.keys())[orb]]
                    if recip == False:
                        all_wavefun_overlap[orb][bi] = periodic_integral_3d(np.conj(WF) * realwavefunc,self._a,self.gridxyz)
                    else:
                        all_wavefun_overlap[orb][bi] = reciprocal_integral(np.conj(WF) * realwavefunc)
        all_coeffs = np.linalg.solve(overlap_matrix,all_wavefun_overlap).T
        return all_coeffs#coeff[1:]

    def get_aecoefficients(self,orbitalWF, crystalWF, aeoverlap_matrix,kpt,recip=False, band=None,full_kpt=False,prints = False,gpnts=None,set_gpnts=False):
        """
        Find the coefficients for the amount of each ae orbital (Φ) in the ae wavefunction (Ψ).
        If the overlap is identity, the coefficients would just be the integral of the orbital-wavefunction overlap: Φa*Ψn = O_an.
        When the overlap is not identity S_ab, obtaining the coefficients (C_bn) requires solving the linear problem S_ab*C_bn = O_an.
        The orbital-wavefunction overlap is modified from the pseudo overlap using standard PAW methods.


        Args:    
            orbitalWF (dict): of length M: All the pseudo orbital functions in a dictionary. The orbital may be defined in real 3D space on reciprocal G vectors.
            crystalWF (dict): of length N: All the pseudo DFT wavefunctions in a dictionary. The wavefunction is defined on the same space as orbitalWF.
            aeoverlap_matrix (MxM): matrix of complex float: The overlap of the ae orbitals. NOTE: The orbitals and their overlaps have a k-dependence.
            recip (bool): Whether the wavefunctions are defined in real or reciprocal space.
            band (int): If the coefficents of only one band in the crystalWF dict is needed, pass that band as an integer here.

        Returns:    
            unknown: The coefficients C_na of orbital a in band n.
        """

        num_orbs = len(orbitalWF)
        coeff = [[0]]
        # bands to range
        if band != None:
            bands = [band] #[str(band)]
        else:
            bands = range(len(crystalWF)) #crystalWF.keys()
        all_wavefun_overlap = np.zeros((num_orbs,len(bands)), dtype=np.complex128)

        if recip == True:
            # constuct the adjusted orbitals
            krecip_orb = orbitalWF #self.get_kdep_reciporbs(kpt)
            krecip_proj = self.get_kdep_recipprojs(kpt,full_kpt=full_kpt,gpnts=gpnts,set_gpnts=set_gpnts)
            krecip_WF = crystalWF

            # construct k-dependant Sij and Aij, only different if psuedo basis orbital are not orthogonal
            # have to do the whole psuedo->ae PAW procedure :(
            # construct proj-orb overlap
            proj_orb_overlap = self.get_overlap_matrix(krecip_proj,krecip_orb,recip=True)
            #print("proj orb-lap:",proj_orb_overlap[:2])
            proj_WF_overlap = self.get_overlap_matrix(krecip_proj,krecip_WF,secondisarray=True,recip=True)

            # now get the ae overlap between orbitals
            psuedo_orbWFoverlap = self.get_overlap_matrix(krecip_orb,krecip_WF,secondisarray=True,recip=True)
            if prints == True:
                print("pseudo overlap:",psuedo_orbWFoverlap[-2:])
            Qab = self.Qab
            
            # starting the monitoring
            #tracemalloc.start()
            num_projs = len(krecip_proj)
            sum_overi = np.zeros((len(krecip_orb),len(krecip_WF)),dtype=np.complex128)
            count_val = np.arange((num_projs*num_projs))[Qab.flatten() != 0]
            #print(len(count_val),num_projs*num_projs)
            #i = np.array(np.floor(count_val/num_projs),dtype=np.int_)
            #j = np.array(count_val%num_projs,dtype=np.int_)
            all_orbatm = np.append(self.orbatomnum, self.exorbatomnum)
            orbatm =  np.array(self.orbatomnum)
            for ij in count_val:
                i = int(np.floor(ij/num_projs))
                j = ij%num_projs
                bool1 = orbatm == all_orbatm[i]
                #bool2 = orbatm == all_orbatm[j]
                sum_overi += np.conj(proj_orb_overlap[i, :,None]) * Qab[i,j]*proj_WF_overlap[j,None, :]
                #sum_overi += np.conj(proj_orb_overlap[i, :,None]) * Qab[i,j]*proj_WF_overlap[j,None, :]
            #print(sum_overi[0])
            #sum_overi = np.zeros((len(krecip_orb),len(krecip_WF)),dtype=np.complex128)
            #for i in range(num_projs):
            #    for j in range(num_projs):
            #        sum_overi += np.conj(proj_orb_overlap[i, :,None]) * Qab[i,j]*proj_WF_overlap[j,None, :]
            #sum_overi = np.sum(np.conj(proj_orb_overlap[:,None, :,None]) * Qab[:,:,None,None]*proj_WF_overlap[None,:,None, :],axis=(0,1))
            #sum_overi = all_paw
            
            # displaying the memory
            #print("memory:",tracemalloc.get_traced_memory())

            # stopping the library
            #tracemalloc.stop()
            
            if prints == True:
                print("ae overlap:",sum_overi[-2:])
            orb_WF_aeoverlap = psuedo_orbWFoverlap + sum_overi

        else:
            print("must do reciprocal integration!!")
        if prints == True:
            print("ae overlap:",orb_WF_aeoverlap[-2:])
        all_coeffs = (np.linalg.inv(aeoverlap_matrix) @ orb_WF_aeoverlap).T #np.linalg.solve().T
        return all_coeffs#coeff[1:]
    
    def get_ae_overlap_matrix(self,orbitalWF,secondWF = None,secondisarray=False,recip = True,kpt=0):
        """
        Finds the overlap matrix S_ab = Φa*φb. If secondWF is not defined, φ = Φ.


        Args:    
            orbitalWF (dict): of length M: All the orbital functions in a dictionary. The orbital may be defined in real 3D space on reciprocal G vectors.
            secondWF (dict): of length N: The second orbital functions in a dictionary. Defined on the same space as orbitalWF.
            recip (bool): Whether the orbitals are defined in real or reciprocal space.

        Returns:    
            unknown: The overlap matrix S_ab for orbitals a and b.
        """
        if secondisarray == False:
            try:
                if secondWF == None:
                    secondWF = orbitalWF
            except:
                do_nothing = True
        num_orbs = len(orbitalWF)
        num_orbs2 = len(secondWF)

        if recip == True:
            firstWF = orbitalWF
            otherWF = secondWF
            
            overlap_matrix = np.matmul(np.conj(firstWF),otherWF.T)
            
            krecip_orb = orbitalWF #self.get_kdep_reciporbs(kpt)
            krecip_proj = self.get_kdep_recipprojs(kpt)
            krecip_WF = secondWF

            # construct k-dependant Sij and Aij, only different if psuedo basis orbital are not orthogonal
            # have to do the whole psuedo->ae PAW procedure :(
            # construct proj-orb overlap
            proj_orb_overlap = self.get_overlap_matrix(krecip_proj,krecip_orb,recip=True)
            #print("proj orb-lap:",proj_orb_overlap[:2])
            proj_WF_overlap = self.get_overlap_matrix(krecip_proj,krecip_WF,secondisarray=True,recip=True)

            # now get the ae overlap between orbitals
            psuedo_orbWFoverlap = self.get_overlap_matrix(krecip_orb,krecip_WF,secondisarray=True,recip=True)
            if self.verbose > 2:
                print("pseudo overlap:",psuedo_orbWFoverlap[-2:])
            Qab = self.Qab
            
            # starting the monitoring
            #tracemalloc.start()
            num_projs = len(krecip_proj)
            sum_overi = np.zeros((len(krecip_orb),len(krecip_WF)),dtype=np.complex128)
            count_val = np.arange((num_projs*num_projs))[Qab.flatten() != 0]
            #print(len(count_val),num_projs*num_projs)
            #i = np.array(np.floor(count_val/num_projs),dtype=np.int_)
            #j = np.array(count_val%num_projs,dtype=np.int_)
            all_orbatm = np.append(self.orbatomnum, self.exorbatomnum)
            orbatm =  np.array(self.orbatomnum)
            for ij in count_val:
                i = int(np.floor(ij/num_projs))
                j = ij%num_projs
                bool1 = orbatm == all_orbatm[i]
                #bool2 = orbatm == all_orbatm[j]
                sum_overi += np.conj(proj_orb_overlap[i, :,None]) * Qab[i,j]*proj_WF_overlap[j,None, :]
                #sum_overi += np.conj(proj_orb_overlap[i, :,None]) * Qab[i,j]*proj_WF_overlap[j,None, :]
            #print(sum_overi[0])
            #sum_overi = np.zeros((len(krecip_orb),len(krecip_WF)),dtype=np.complex128)
            #for i in range(num_projs):
            #    for j in range(num_projs):
            #        sum_overi += np.conj(proj_orb_overlap[i, :,None]) * Qab[i,j]*proj_WF_overlap[j,None, :]
            #sum_overi = np.sum(np.conj(proj_orb_overlap[:,None, :,None]) * Qab[:,:,None,None]*proj_WF_overlap[None,:,None, :],axis=(0,1))
            #sum_overi = all_paw
            
            # displaying the memory
            #print("memory:",tracemalloc.get_traced_memory())

            # stopping the library
            #tracemalloc.stop()
            
            if self.verbose > 2:
                print("ae overlap:",sum_overi[-2:])
            orb_WF_aeoverlap = psuedo_orbWFoverlap + sum_overi


        return orb_WF_aeoverlap
        

    def get_overlap_matrix(self,orbitalWF,secondWF = None,secondisarray=False,recip = False):
        """
        Finds the overlap matrix S_ab = Φa*φb. If secondWF is not defined, φ = Φ.


        Args:    
            orbitalWF (dict): of length M: All the orbital functions in a dictionary. The orbital may be defined in real 3D space on reciprocal G vectors.
            secondWF (dict): of length N: The second orbital functions in a dictionary. Defined on the same space as orbitalWF.
            recip (bool): Whether the orbitals are defined in real or reciprocal space.

        Returns:    
            unknown: The overlap matrix S_ab for orbitals a and b.
        """
        if secondisarray == False:
            try:
                if secondWF == None:
                    secondWF = orbitalWF
            except:
                do_nothing = True
        num_orbs = len(orbitalWF)
        num_orbs2 = len(secondWF)

        overlap_matrix = np.zeros((num_orbs, num_orbs2), dtype=np.complex128)

        if recip == True:
            # convert WF from dictionaries to np arrays so calculation can be fully vectorized
            num_gpnts = len(orbitalWF[0])#list(orbitalWF.keys())[0]])
            firstWF = orbitalWF#np.zeros((num_orbs,num_gpnts),dtype=np.complex128)
            #otherWF = orbitalWF # np.zeros((num_orbs2,num_gpnts),dtype=np.complex128)
            #for orb1 in range(num_orbs):
            #    firstWF[orb1] = orbitalWF[list(orbitalWF.keys())[orb1]]
            #if secondisarray==False:
            #    for orb2 in range(num_orbs2):
            #        otherWF[orb2] = secondWF[list(secondWF.keys())[orb2]]
            #else:
            otherWF = secondWF
            overlap_matrix = np.matmul(np.conj(firstWF),otherWF.T)

        else: # not well vectorized but don't usually use
            for orb1 in range(num_orbs):
                WF1 = orbitalWF[orb1]#list(orbitalWF.keys())[orb1]]
                for orb2 in range(num_orbs2):
                    WF2 = secondWF[orb2]#list(secondWF.keys())[orb2]]
                    grid = orbitalWF[0].shape #list(orbitalWF.keys())[0]].shape
                    overlap_matrix[orb1][orb2] = periodic_integral_3d(np.conj(WF1) * WF2, self._a, grid)

        return overlap_matrix

    def lowdin_orth(self,low_orbitals,set_overlap=False, overlap=None,recip = False):
        """
        Orthogonalized the orbital based on the Lowdin scheme.
        The new orbitals Ψ are defined by the original orbitals Φ as Ψ_b = conj(S_ab)^(-1/2)*Φ_a.


        Args:    
            low_orbitals (dict): of length M: All the orbital functions in a dictionary. The orbital may be defined in real 3D space on reciprocal G vectors.
            set_overlap (bool): Whether the orbital overlap is being passed to the function (True) or should be calculated (False).
            overlap (MxM): matrix of complex float: The overlap of the orbitals. NOTE: The orbitals and their overlaps have a k-dependence.
            recip (bool): Whether the orbitals are defined in real or reciprocal space.

        Returns:    
            unknown: Lowdin orthogonalized orbitals.
        """

        # may not work if orbital are not real --> actually does now that I included the * of inv_sprt_overlap below
        if set_overlap == False:
            low_overlap = self.get_overlap_matrix(low_orbitals,recip=recip)
        else:
            low_overlap = overlap
        # make the inverse sqrt overlap matrix
        # Computing diagonalization
        #evalues, evectors = np.linalg.eigh(coeff_overlap)
        #sqrt_matrix = evectors * np.sqrt(evalues) @ np.linalg.inv(evectors)
        sqrt_overlap =  linalg.sqrtm(low_overlap) #sqrt_matrix
        sqrt_overlap = np.array(sqrt_overlap,dtype=np.complex128)
        inv_sqrt_overlap = np.linalg.inv(sqrt_overlap)
        # create the new orthogonal orbitals
        low_orth_orbs = np.zeros((low_orbitals.shape),dtype=np.complex128) # {}
        for orb1 in range(len(low_orbitals)):
            key1 = orb1 #list(low_orbitals.keys())[orb1]
            add_to_orb = np.zeros((low_orbitals[key1].shape), dtype=np.complex128)
            for orb2 in range(len(low_orbitals)):
                key2 = orb2 #list(low_orbitals.keys())[orb2]
                add_to_orb += np.conj(inv_sqrt_overlap[orb1, orb2]) * low_orbitals[key2]
            low_orth_orbs[key1] = add_to_orb
        return low_orth_orbs

    def converge_orbs_recip(self,num_steps=50):
        """
        Converge to the atomic-like Bloch orbitals which best fit the DFT wavefunction.
        Please reference COGITO paper for theoretical description.

        Creates global variable one_orbitalWF which is referenced in the Bloch to atomic orbital fitting.

        Args:    
            num_steps (int): Maximum number of steps to perform the convergence.
                        Setting equal to 0 with run the standard direct algorithm where ``|X_a> = sum_n(c_na |Y_n>)``.
                        Where ``|Y_n>`` are the set of band (equal to number of orbitals) of the highest projection.
        """
        
        # once orbitals are found, orbitals on the same atom are Lowdin orthogonalized using the ae overlap
        
        converg_orbs = self.one_orbWFrecip #self.real_to_recip(self.one_orbitalWF) # recip_orbs_noex
        
        # now refine the orbital
        exp_term = np.exp(-2j * np.pi * np.matmul(np.array(self.gpoints[0],dtype=np.int_), self.orbpos.T)).T # [orb,gpnt]
        centered_orbs = converg_orbs/exp_term
        # take only real or imaginary part
        sphkey_to_l = np.array([0,1,1,1,2,2,2,2,2,3,3,3,3,3,3,3])
        orb_l = sphkey_to_l[self.sph_harm_key]
        centered_orbs = (centered_orbs*(-1j)**orb_l[:,None]).real*(1j)**orb_l[:,None]
        converg_orbs = centered_orbs*exp_term #str()
        
        #excited_orbs = self.real_to_recip(self.ex_pseudos)
        #elf.ex_pseudos = excited_orbs
        all_orbs = converg_orbs # np.append(converg_orbs,excited_orbs,axis=0)
        #print("orbital shape:",all_orbs.shape)
        #orb_keys = list(converg_orbs.keys())
            
        # calculate the all-electron orbital overlap
        overlap = self.get_overlap_matrix(all_orbs,recip=True)
        #aeoverlap = self.get_ae_overlap_info(all_orbs,just_ae_overlap = True)
        aeoverlap = self.get_ae_overlap_matrix(all_orbs,secondWF = all_orbs,secondisarray=True)
        #print("ae overlap:",aeoverlap)
        norm = np.diag(aeoverlap)
        # now renormalize the orbitals to the ae overlap values
        for orb in range(len(all_orbs)):#ind,  enumerate(list(converg_orbs.keys())):
            normed_orb = all_orbs[orb] / ((norm[orb]) ** (1 / 2))
            all_orbs[orb] = normed_orb
        #aeoverlap = self.get_ae_overlap_info(all_orbs,just_ae_overlap = True)
        aeoverlap = self.get_ae_overlap_matrix(all_orbs,secondWF = all_orbs,secondisarray=True)
        #print("ae overlap:",aeoverlap)
        converg_orbs = all_orbs[:len(converg_orbs)]

        # orthogonalize orbitals on the same atom
        # most important if there are two orbital with the same quantum numbers l and m
        # fix this later
        '''
        previous_orb = 0
        for atms in self.lastatom:  #: [9, 18, 22, 26]
            orb_types = self.orbtype[previous_orb:atms]
            is_s2 = orb_types == "s2"
            if (is_s2).any():
                s2_num = np.arange(previous_orb,atms)[is_s2][0]
                s_num = np.arange(previous_orb,atms)[orb_types == "s"][0]
                old_s2 = converg_orbs[str(s2_num)]
                s_orb = converg_orbs[str(s_num)]
                # working on! 
                # now gram schidt orthogonalize the s2 orbital to the s orbital
                ss2_overlap = aeoverlap[s2_num,s_num] #np.sum(np.conj(old_s2)*s_orb)
                ss_overlap = aeoverlap[s_num,s_num] #np.sum(np.conj(s_orb)*s_orb)
                new_s2 = old_s2 - ss2_overlap/ss_overlap*s_orb
                converg_orbs[str(s2_num)] = new_s2
                new_overlap = np.sum(np.conj(new_s2)*s_orb)
                print("made new s2 orbital",ss2_overlap,new_overlap)
            
            previous_orb = atms
        aeoverlap = self.get_ae_overlap_info(converg_orbs,just_ae_overlap = True)
        print(aeoverlap[s2_num,s_num])
        '''
        '''
        previous_orb = 0
        for atms in self.lastatom:  #: [9, 18, 22, 26]
            for_gram = {}
            for orb in range(previous_orb, atms):
                for_gram[str(orb)] = converg_orbs[str(orb)]
            holder = self.lowdin_orth(for_gram, recip=True)  # ghj
            for orb in range(previous_orb, atms):
                converg_orbs[str(orb)] = holder[str(orb)]
            previous_orb = atms
        '''

        # normalize each band wavefunction. This defines the PAW treatment but is fine for converging to the best pseudo orbitals, which are also normalized
        # ...shit this won't work in case of very hybridized bands with orbital of very different PAW normalizations -- will change soon
        kpt = 0 # do at the first kpoint, which should be Gamma
        periodic_WFs = copy.deepcopy(self.recip_WFcoeffs[kpt])
        #if self.verbose > 1:
        #    print("previous norm for the band wavefunctions")
        #for wf in range(self.num_bands):
        #    periodic_WFs[wf],integral = normalize_wf(periodic_WFs[wf], self._a, self.gridxyz,recip=True,return_integral=True)
        #    if self.verbose > 1:
        #        print(wf,integral)
        periodic_WFs = np.array(periodic_WFs)
        num_gpnts = len(periodic_WFs[0])
        if self.verbose > 1:
            print("number of gpoints:", num_gpnts)
        
        #print("tenth orbital cutoff:",self.tenth_cutoff)

        # select the highest bands for converging algorithm
        orbitals = copy.deepcopy(converg_orbs)
        crystalWFs = periodic_WFs
        orth_orbs = converg_orbs #self.lowdin_orth(orbitals,recip=True)
        overlap = self.get_overlap_matrix(converg_orbs,recip=True)
        #aeoverlap = self.get_ae_overlap_info(orth_orbs,just_ae_overlap = True)
        aeoverlap = self.get_ae_overlap_matrix(orth_orbs,secondWF = orth_orbs,secondisarray=True)
        orth_coeff = self.get_aecoefficients(orth_orbs, crystalWFs, aeoverlap,kpt,recip=True)
        spillage = np.zeros(self.max_band)
        for band in range(self.max_band):
            spillage[band] = 1 - np.matmul(np.conj(orth_coeff[band]), np.matmul(aeoverlap, orth_coeff[band].transpose())).real
        #sqr_coeff = np.square(np.real(orth_coeff)) + np.square(np.imag(orth_coeff))
        #spillage = np.sum(sqr_coeff, axis=1)
        lowest_bands = np.argsort(spillage)[:self.num_orbs]
        #print(self.num_orbs)
        #print(lowest_bands)
        sortedLoBds = np.sort(lowest_bands).astype(np.int_)
        include_lowband = (((1-spillage)>self.low_min_proj) | (self.eigval[kpt][:,0]>=(self.efermi+self.energy_shift-5))) # only False for bad low energy bands
        include_bands = (1-spillage)>self.min_proj
        sortedLoBds = np.arange(self.max_band)[include_lowband & include_bands]
        
        inv_orth_coeff = np.matmul(np.conj(orth_coeff),aeoverlap).T  # np.linalg.inv(orth_coeff) #np.conj(orth_coeff.T)
        orb_mixing = np.matmul(orth_coeff.T,inv_orth_coeff.T)
        np.fill_diagonal(orb_mixing,0)
        if self.verbose > 0:
            print("initial orb mixing:",np.amax(np.abs(orb_mixing)))
        
        orth_coeff = orth_coeff[sortedLoBds]
        if self.verbose > 0:
            print("band spilling:",spillage)
        
        if self.verbose > 0:
            print("bands to converge:", sortedLoBds)
            print("num orbs:",self.num_orbs,len(sortedLoBds))
        if self.num_orbs > len(sortedLoBds): # failed to find enough bands
            raise Exception("ERROR: not enough bands with orbitals character! Inrease NBANDS in the static calculation.")
        prev_avgspill = 0.8
        prev_diff_spill = 0.5
        prev_coeff = orth_coeff[:self.num_orbs,:]
        prev_invcoeff = np.linalg.inv(orth_coeff[:self.num_orbs,:])
        
        new_orbs = converg_orbs
        og_orbs = converg_orbs
        og_orth_coeff = orth_coeff
        og_aeoverlap = aeoverlap
        
        orth_extra_wf = np.zeros([len(sortedLoBds), num_gpnts], dtype=np.complex128)
        # run the convergence of the orbitals!
        for justloop in range(num_steps):
            # if k!=0 would need to do more than this; Gamma_point = True
            #adjusted_orbs = converg_orbs
            # get overlap and coefficients of orbitals
            all_orbs = converg_orbs #np.append(converg_orbs,excited_orbs,axis=0)
            
            overlap = self.get_overlap_matrix(all_orbs,recip=True)
            # use PAW formalism instead
            #aeoverlap = self.get_ae_overlap_info(all_orbs,just_ae_overlap = True)
            aeoverlap = self.get_ae_overlap_matrix(all_orbs,secondWF = all_orbs,secondisarray=True)
            #print("check same:",aeoverlap,ae_overlap)
            #overlap = aeoverlap
            orth_coeff = self.get_aecoefficients(all_orbs, periodic_WFs[sortedLoBds], aeoverlap,kpt,recip=True)
                
            '''
            
            #orth_extra_wf = np.zeros([self.num_bands, num_gpnts], dtype=np.complex128)
            for bind, band in enumerate(sortedLoBds):
                trueWF = periodic_WFs[band]
                orb_coeff = np.zeros((trueWF.shape), dtype=np.complex128)
                for temporb in range(self.num_orbs):
                    orbWF = converg_orbs[temporb]#orb_keys[]
                    orb_coeff += orbWF * orth_coeff[bind][temporb]
                orth_extra_wf[bind] = trueWF - orb_coeff
            # include second loop to exclude the residual wavefunction in the periodic_WFs when calculating coeffs
            print("check how coefficients update:")
            for con_coeff in range(3):
                WFs_forcoeffs = periodic_WFs[sortedLoBds] - orth_extra_wf
                ps_orb_extra = self.get_overlap_matrix(converg_orbs,secondWF = orth_extra_wf,secondisarray=True,recip=True)
                proj_extra = self.get_aecoefficients(converg_orbs, orth_extra_wf, aeoverlap,kpt,recip=True,prints=True)
                orth_coeff = self.get_aecoefficients(converg_orbs, WFs_forcoeffs, aeoverlap,kpt,recip=True)
                
                #orth_extra_wf = np.zeros([self.num_bands, num_gpnts], dtype=np.complex128)  # different for recip
                #orb_coeff = np.sum(converg_orbs[None,:,:]*orth_coeff[:,:,None],axis=1) 
                for bind, band in enumerate(sortedLoBds):
                    trueWF = periodic_WFs[band]
                    orb_coeff = np.zeros((trueWF.shape), dtype=np.complex128)
                    for temporb in range(self.num_orbs):
                        orbWF = converg_orbs[temporb]#orb_keys[]
                        orb_coeff += orbWF * orth_coeff[bind][temporb]
                    orth_extra_wf[bind] = trueWF - orb_coeff
                print(ps_orb_extra[-2:])
            '''
            
            #check spillage and determine if converged to last loops value
            spillage = np.zeros(len(sortedLoBds))
            for band in range(len(sortedLoBds)):
                spillage[band] = 1-np.matmul(np.conj(orth_coeff[band]), np.matmul(aeoverlap, orth_coeff[band].transpose())).real
            if self.verbose > 0:
                print("spill", spillage)
                print("avg spill:", np.average(np.abs(spillage)))
                #coeff_diff = np.abs(orth_coeff-prev_coeff).flatten()
                #coeff_per = coeff_diff[orth_coeff.flatten()>0.1]/orth_coeff.flatten()[orth_coeff.flatten()>0.1]
                #print("avg coeff percent change:",np.average(np.abs(coeff_per)))
                #print("avg coeff change:",np.average(np.abs(coeff_diff)))
                #coeff_inv = np.linalg.inv(orth_coeff)
                #diff_inv = np.abs(coeff_inv-prev_invcoeff).flatten()
                #inv_per = diff_inv[coeff_inv.flatten()>0.1]/coeff_inv.flatten()[coeff_inv.flatten()>0.1]
                #print("avg inv percent change:",np.average(np.abs(inv_per)))
                #print("avg inv change:",np.average(np.abs(diff_inv)))
            cur_avgspill = np.average(np.abs(spillage))
            diff_spill = np.abs(cur_avgspill - prev_avgspill)
            if diff_spill < 0.0001 and justloop > 2: # converged 
                print("achieved convergence at loop", justloop+1,cur_avgspill)
                break
            if diff_spill > prev_diff_spill * 4 and justloop > 3:
                raise Exception("ERROR: convergence procedure is not working!")
            
            prev_avgspill = cur_avgspill
            prev_diff_spill = diff_spill


            
            # now orthogonalize the eigenvectors
            if justloop == 0:
                
                normalized = spillage #np.diagonal(np.matmul(np.conj(current_coeff), current_coeff.transpose()).transpose())#self.total_nrms[kpt,:self.num_orbs]#
                band_ind = np.argsort(normalized)
                norm_sorted = normalized[band_ind]
                energies = self.eigval[kpt][sortedLoBds][:, 0]
                diff = np.array([abs(normalized[i + 1] - normalized[i]) * np.tanh(abs(energies[i + 1] - energies[i])) for i in range(len(normalized) - 1)])
                is_valence = self.eigval[kpt][sortedLoBds[1:]][:,0]<=(self.efermi+self.energy_shift)
                avg = np.sum(diff[is_valence]) / len(diff[is_valence]) * 2
                #print(avg)
                m = [[0]]  # normalized[0]
                index = [[0]]
                for x in range(len(normalized) - 1):
                    val = diff[x]
                    ind = x + 1
                    groupsize = 1
                    if val < avg:  # or len(m[-1]) < groupsize:
                        m[-1].append(val)
                        index[-1].append(ind)
                    else:
                        m.append([val])
                        index.append([ind])
            if self.verbose > 1:
                print("did grouping work?", m, index)
            #index = [[0,1,2,3,4,5,6,7]]
            #index = [np.arange(self.num_orbs)]
            # do it with the nonorth coeff
            start = index[0][0]
            end = index[0][-1] + 1
            orb_overlap = aeoverlap
            
            for loop in range(1):
                nonorth_coeff = copy.deepcopy(orth_coeff)
                
                start = index[0][0]
                end = index[0][-1] + 1
                nonorth_coeff[start:end]= lowdin_orth_vectors_orblap(nonorth_coeff[start:end],orb_overlap) 
                # also calculate the new energy matrix, will no longer be diagonal
                for ind in index[1:]:
                    start = ind[0]
                    end = ind[-1] + 1
                    nonorth_coeff[start:end] = lowdin_orth_vectors_orblap(nonorth_coeff[start:end],orb_overlap)
                    nonorth_coeff[start:end] = GS_orth_twoLoworthSets_orblap(nonorth_coeff[0:start],nonorth_coeff[start:end],orb_overlap)

                #nonorth_coeff[:] = np.array(lowdin_orth_vectors_orblap(nonorth_coeff[:],orb_overlap))

                # actually just make all valence bands orthonormal and make a conduction bands orthogonal to VB (conduction bands do not need to be orthogonal to themselves)
                new_coeff = copy.deepcopy(orth_coeff)
                
                is_valence = self.eigval[kpt][sortedLoBds][:,0]<=(self.efermi+self.energy_shift)
                is_conduction = self.eigval[kpt][sortedLoBds][:,0]>(self.efermi+self.energy_shift)
                #print("valence bands:",np.arange(len(is_valence))[is_valence])
                new_coeff[is_valence] = nonorth_coeff[is_valence] # lowdin_orth_vectors_orblap(nonorth_coeff[is_valence],orb_overlap)
                new_coeff[is_conduction] = GS_orth_twoLoworthSets_orblap(new_coeff[is_valence],new_coeff[is_conduction],orb_overlap)

                # with the new scheme in calculating the Hamiltonian instead of orthogonalizing a square coefficient matrix we need to find some way to make the rectangular matrix 'exact'
                # my new idea is instead of Lowdin orthogonalizing the bands, Lowdin orthogonalize the orbitals, or for a nonorth basis, ensure that S_ab = c_an * c_nb^*
                
                sqrt_overlap =  linalg.sqrtm(aeoverlap) 
                sqrt_overlap = np.array(sqrt_overlap, dtype=np.complex128)
                #print(coeff_overlap[:2])
                inv_sqrt_overlap = np.linalg.inv(sqrt_overlap)
                #kdep_Aij = inv_sqrt_overlap
                test_orth_coeff = np.matmul(sqrt_overlap,new_coeff.transpose()) # [orb, band]
                overlap = np.matmul(test_orth_coeff,np.conj(test_orth_coeff.T))
                #print("Check mixing:",overlap)
                forc_orthcoeff = np.conj(lowdin_orth_vectors(np.conj(test_orth_coeff)))
                #overlap = np.matmul(forc_orthcoeff,np.conj(forc_orthcoeff.T))
                #print("Check ident:",overlap)
                forc_coeff = np.matmul(inv_sqrt_overlap,forc_orthcoeff).transpose()
                
                #print(loop,orth_coeff[is_valence],new_coeff[is_valence],forc_coeff[is_valence])
                orth_coeff = forc_coeff
                #orth_coeff[is_valence] = new_coeff[is_valence]
            
            #check spillage and determine if converged to last loops value
            fake_spillage = np.zeros(len(sortedLoBds))
            for band in range(len(sortedLoBds)):
                fake_spillage[band] = 1-np.matmul(np.conj(orth_coeff[band]), np.matmul(aeoverlap, orth_coeff[band].transpose())).real
            if self.verbose > 2:
                print("new spill:",fake_spillage)
            # redefine the extraWF
            if justloop == 0 or justloop >= 0:
                orth_extra_wf = np.zeros([len(sortedLoBds), num_gpnts], dtype=np.complex128)  # different for recip
                #orb_coeff = np.sum(converg_orbs[None,:,:]*orth_coeff[:,:,None],axis=1) 
                for bind, band in enumerate(sortedLoBds):
                    trueWF = periodic_WFs[band]
                    orb_coeff = np.zeros((trueWF.shape), dtype=np.complex128)
                    for temporb in range(len(all_orbs)):
                        orbWF = all_orbs[temporb]#orb_keys[]
                        orb_coeff += orbWF * orth_coeff[bind][temporb]
                    orth_extra_wf[bind] = trueWF - orb_coeff
                #    if self.verbose > 2:
                #        print("new extra:", band, np.average(abs(orth_extra_wf[bind].real)),
                #             np.max(abs(orth_extra_wf[bind].real)), np.max(abs(orth_extra_wf[bind].imag)))
                
            
            #find the best iteration matrix
            #eigenvalj,Dij = np.linalg.eigh(overlap)
            # check correctness of eigen
            # construct Aij
            #Aij = np.zeros((self.num_orbs, self.num_orbs), dtype=np.complex128)
            #for j in range(self.num_orbs):
            #    Aij[:, j] = Dij[:, j] / (eigenvalj[j]) ** (1 / 2)
            
            #sqrt_overlap =  linalg.sqrtm(aeoverlap) 
            #sqrt_overlap = np.array(sqrt_overlap, dtype=np.complex128)
            #print(coeff_overlap[:2])
            #inv_sqrt_overlap = np.linalg.inv(sqrt_overlap)
            #Aij = inv_sqrt_overlap
            
            #orth_orig = np.matmul(inv_sqrt_overlap,orth_coeff.transpose()).transpose()
            
            #parse the remaining wavefunction into the correct orbitals
            #extra_orth_orb_wf = np.zeros((converg_orbs.shape), dtype=np.complex128) #{}
            #inv_orth_coeff = np.matmul(np.conj(orth_orig),sqrt_overlap).transpose()
            inv_orth_coeff = np.matmul(np.conj(orth_coeff),aeoverlap[:,:]).T  # np.linalg.inv(orth_coeff) #np.conj(orth_coeff.T)
            orb_mixing = np.matmul(orth_coeff[:,:].T,inv_orth_coeff.T) - np.identity(self.num_orbs)
            np.fill_diagonal(orb_mixing,0)
            if self.verbose > 0:
                print("orb mixing:",np.amax(np.abs(orb_mixing)))
            #print("Check coeffs same:",inv_orth_coeff[:2],orth_coeff.T[:2])
            #inv_orth_coeff = np.linalg.inv(orth_coeff) # now should be [orb, band]
            #extra_orth_orb_wf = np.sum(inv_orth_coeff[:,:,None]*orth_extra_wf[None,:,:],axis=1)
            shape = np.append(self.num_orbs,periodic_WFs[0].shape)
            extra_orth_orb_wf = np.zeros(tuple(shape), dtype=np.complex128)
            wan_orbs = np.zeros(tuple(shape), dtype=np.complex128)
            for orb in range(self.num_orbs): #str()
                for bind, band in enumerate(sortedLoBds): #fine to not have actually band because only care about indices of coeff and extra wf
                    #extra_orth_orb_wf[str(orb)] += np.conj(orth_coeff[bind][orb])*orth_extra_wf[bind] # previous way whichs tends towards orthogonality
                    extra_orth_orb_wf[orb] += inv_orth_coeff[orb][bind] * orth_extra_wf[bind] # new way which adds the conj.T instead of inv to the orthogonal eigenvecs instead of nonorthogonal eigenvecs #str()
                    wan_orbs[orb] += inv_orth_coeff[orb][bind] * periodic_WFs[band]
                    
            # anti mixing part
            #mixed_orbs = np.zeros(tuple(shape), dtype=np.complex128)
            #print("orb mixing:",orb_mixing)
            #for orb1 in range(self.num_orbs): #str()
            #    for orb2 in range(self.num_orbs):
            #        mixed_orbs[orb1] += 1/2*orb_mixing[orb1,orb2]*converg_orbs[orb2]
            
            #test_mixed = np.zeros(tuple(shape), dtype=np.complex128)
            #wan_old_overlap = np.abs(self.get_ae_overlap_matrix(np.abs(wan_orbs-all_orbs),secondWF = all_orbs,secondisarray=True))
            #for wan in range(self.num_orbs):
            #    for old in range(self.num_orbs):
            #        if wan != old:
            #            test_mixed[wan] += wan_old_overlap[wan,old] * all_orbs[old] * np.sign(aeoverlap[wan,old]) * 2
            
            #new_orbs = np.zeros((converg_orbs.shape), dtype=np.complex128) # {}
            #for orb in range(self.num_orbs):
            #    # only for k==0; need Gamma_point=False for more general formalism
            #    orb_to_add = extra_orth_orb_wf[orb]#*1.5 #str()
            #    if justloop < 0:
            #        new_orbs[orb] = orb_to_add + orbitals[orb] #normalize_wf(, self._a, self.gridxyz,recip=True) #str()
            #    else:
            #new_orbs = wan_orbs #+ test_mixed# wan_orbs #+ mixed_orbs#extra_orth_orb_wf + converg_orbs #str()
            diff_wan = wan_orbs - converg_orbs
            symmetrize = False
            if symmetrize:
                new_orbs = self.symmetrize_orbs(wan_orbs)
            else:
                new_orbs = extra_orth_orb_wf + all_orbs #wan_orbs
            diff_mix = new_orbs - wan_orbs
            if self.verbose > 1:
                print("maximum diff between wan and mix orbs:",np.amax(diff_mix),np.amax(wan_orbs),np.average(np.abs(wan_orbs)))
            diff_sym = new_orbs - converg_orbs
            #print(converg_orbs)
            #print("bad diff:",diff_wan)
            #print("new diff:",diff_sym)
            #print("guess diff:",test_mixed)
            
            # now refine the orbital
            exp_term = np.exp(-2j * np.pi * np.matmul(np.array(self.gpoints[0],dtype=np.int_), self.orbpos.T)).T # [orb,gpnt]
            centered_orbs = new_orbs/exp_term
            # take only real or imaginary part
            sphkey_to_l = np.array([0,1,1,1,2,2,2,2,2,3,3,3,3,3,3,3])
            orb_l = sphkey_to_l[self.sph_harm_key]
            centered_orbs = (centered_orbs*(-1j)**orb_l[:,None]).real*(1j)**orb_l[:,None]
            cut_outer = False
            if cut_outer:
                # cutoff outer part
                cutoff = self.tenth_cutoff#[orb]) #2.0
                real_orbs = self.recip_to_real(centered_orbs)
                shape = real_orbs[0].shape
                flat_orbs = np.reshape(real_orbs,(self.num_orbs,shape[0]*shape[1]*shape[2])) #.flatten()
    
                min_rad = self.zeromin_rad
                #min_theta = self.zeromin_theta
                #min_phi = self.zeromin_phi
                #orbperatom = self.sph_harm_key[orb]
                for orb in range(self.num_orbs):
                    angular_part = complex128funs(self.zeromin_phi, self.zeromin_theta, self.orb_vec[orb])
                    #change points very close to zero rad for p and d orbs
                    if self.sph_harm_key[orb] != 0:
                        angular_part[min_rad < 0.001] = 0.0001 # just for the limit so point is never included
                    largeenough = np.abs(angular_part)>0.01
                    radunk = flat_orbs[orb][largeenough]/angular_part[largeenough]
                    og_maxrad = copy.deepcopy(self.orig_maxrad[orb])
                    #count_max = min_rad < og_maxrad*2
                    #print(min_rad[largeenough][np.argsort(np.abs(radunk))[::-1][:20]])
                    largest_rad = np.median(min_rad[largeenough][np.argsort(np.abs(radunk))[::-1][:20]])
                    rad_max = min_rad[np.argmax(flat_orbs[orb])]
                    if og_maxrad > 0.2 and rad_max > 0.2 and largest_rad > 0.2: # rescale cutoff based on new maximum
                        newcutoff = cutoff[orb]*largest_rad/og_maxrad
                        #print("radii:",rad_max,og_maxrad,largest_rad)
                        #print("rescale:",orb,cutoff,newcutoff)
                        cutoff[orb] = newcutoff
                
                    large_rad = min_rad>cutoff[orb]+justloop/5
                    flat_orbs[orb][large_rad] = flat_orbs[orb][large_rad]*np.exp(-1.5*np.abs(min_rad[large_rad]-(cutoff[orb]+justloop/5)))#/ratio[change_rad]
    
                real_orbs = np.reshape(flat_orbs,(self.num_orbs,shape[0],shape[1],shape[2]))
                centered_orbs = self.real_to_recip(real_orbs)
            
            new_orbs = centered_orbs*exp_term #str()
            '''
            # plot new_orbs and wan_orbs
            real_wan = self.recip_to_real(wan_orbs)
            real_new = self.recip_to_real(new_orbs)
            real_og = self.recip_to_real(og_orbs)
            wan_dists1, wan_vals1 = extract_line_data(real_wan[0].flatten(), self.min_xyz[1], np.array([0,0,0]), self.cartAtoms[0]-self.cartAtoms[1], tolerance=0.1)
            #wan_dists2, wan_vals2 = extract_line_data(real_wan[4].flatten(), self.cartXYZ, self.cartAtoms[1], self.cartAtoms[0], tolerance=0.1)
            
            cog_dists1, cog_vals1 = extract_line_data(real_new[0].flatten(), self.min_xyz[1], np.array([0,0,0]), self.cartAtoms[0]-self.cartAtoms[1], tolerance=0.1)
            #cog_dists2, cog_vals2 = extract_line_data(real_new[4].flatten(), self.cartXYZ, self.cartAtoms[1], self.cartAtoms[0], tolerance=0.1)

            og_dists1, og_vals1 = extract_line_data(real_og[0].flatten(), self.min_xyz[1], np.array([0,0,0]), self.cartAtoms[0]-self.cartAtoms[1], tolerance=0.1)

            #plt.figure(figsize=(8, 5))
            plt.plot(wan_dists1, wan_vals1, "blue")
            #plt.plot(wan_dists2, wan_vals2, "black")
            plt.plot(cog_dists1, cog_vals1, "pink")
            #plt.plot(cog_dists2, cog_vals2, "purple")
            plt.plot(og_dists1, og_vals1, "black")
            plt.xlabel("Distance along line (Å)")
            plt.ylabel("Orbital magnitude")
            plt.title("Orbital Magnitude vs. Distance")
            #plt.grid(True)
            plt.show()
            '''
            '''
            for orb in range(self.num_orbs):
                center = self.orbpos[orb]
                exp_term = np.exp(-2j * np.pi * np.dot(self.gpoints[0], center))
                #other_atm  = [0.5,0.5,0.5] # Pb atom
                #if orb > 8:
                #    other_atm  = [0,0,0] # S atom
                #other_term = np.exp(-2j * np.pi * np.dot(self.gpoints[0], other_atm))
                centered_orb = new_orbs[orb]/exp_term
                centered_orig = converg_orbs[orb]/exp_term
                #other_center = new_orbs[str(orb)]/other_term
                #print(new_orbs[str(orb)])
                #print(exp_term)
                #print(other_term)
                #print(centered_orb)
                #print(other_center)
                
                # get gpoint coords
                #gplusk_coords = self.gpoints[0]
                #cart_gplusk = _red_to_cart((self.b_vecs[0], self.b_vecs[1], self.b_vecs[2]), gplusk_coords)
                #vec = cart_gplusk.transpose()  # point - np.array([atm]).transpose()
                #recip_rad = np.linalg.norm(vec, axis=0)
                #recip_rad[recip_rad == 0] = 0.0000001
                #recip_theta = np.arccos(vec[2] / recip_rad)
                #recip_phi = np.arctan2(vec[1], vec[0])
                #orb_to_l = [0, 1, 1, 1, 2, 2, 2, 2, 2]
                #angular_part = complex128funs(recip_phi, recip_theta, self.sph_harm_key[orb])
                if self.sph_harm_key[orb] == 1 or self.sph_harm_key[orb] == 2 or self.sph_harm_key[orb] == 3:
                    centered_orb = centered_orb.imag*1j  #- other_center.imag*other_term*1j
                    no_good = np.sign(centered_orig.imag) != np.sign(centered_orb.imag)
                    #print(len(no_good),no_good)
                    integral =  np.sum(centered_orb.imag)
                    #print("integral for orb", orb, integral)
                    #centered_orb[no_good] = 0# np.abs(centered_orb[no_good].imag)*np.sign(centered_orig[no_good].imag)*1j*justloop/50
                    #centered_orb = np.abs(angular_part*centered_orb.imag)
                    #centered_orb[centered_orb < 0] = 0
                    #centered_orb = centered_orb*1j
                    #small_part = np.abs(angular_part)<0.1
                    #big_part = np.abs(angular_part)>=0.1
                    #angular_part[angular_part==0] = 0.0001
                    #centered_orb[big_part] = centered_orb[big_part]/np.abs(angular_part[big_part])
                    #centered_orb = centered_orb*np.sign(angular_part)
                    
                else:
                    centered_orb = centered_orb.real #- other_center.real*other_term
                    no_good = np.sign(centered_orig.real) != np.sign(centered_orb.real)
                    #print(len(no_good),no_good)
                    integral =  np.sum(centered_orb.real)
                    #print("integral for orb", orb, integral)
                    #centered_orb[no_good]= 0# np.abs(centered_orb[no_good].real)*np.sign(centered_orig[no_good].real)*justloop/50
                    #centered_orb = np.abs(angular_part*centered_orb.real)
                    #centered_orb[centered_orb < 0] = 0
                    #small_part = angular_part<0.1
                    #big_part = angular_part>=0.1
                    #angular_part[angular_part==0] = 0.0001
                    #centered_orb[big_part] = centered_orb[big_part]/np.abs(angular_part[big_part])
                    #centered_orb = centered_orb*np.sign(angular_part)
                #also make sure that the orbital has the right sign
                #print(centered_orig)
                
                if justloop < 50:
                    cutoff = copy.deepcopy(self.tenth_cutoff[orb]) #2.0
                    real_orb = self.recip_to_real(np.array([centered_orb]))[0]
                    shape = real_orb.shape
                    flat_orb = real_orb.flatten()

                    min_rad = self.zeromin_rad
                    #min_theta = self.zeromin_theta
                    #min_phi = self.zeromin_phi
                    #orbperatom = self.sph_harm_key[orb]
                    angular_part = complex128funs(self.zeromin_phi, self.zeromin_theta, self.sph_harm_key[orb])
                    #change points very close to zero rad for p and d orbs
                    if self.sph_harm_key[orb] != 0:
                        angular_part[min_rad < 0.001] = 0.0001 # just for the limit so point is never included
                    largeenough = np.abs(angular_part)>0.01
                    radunk = flat_orb[largeenough]/angular_part[largeenough]
                    #angular_part[angular_part == 0] = 0.001
                    #og_radunk = og_orb/angular_part
                    #og_radunk = og_radunk[largeenough]
                    #if sph_key > 3: # is a d orbital
                    #    rad_max = min_rad[np.argmax(flat_orb)]
                    #    cutoff = rad_max*2.5
                    #    if cutoff < 2:
                    #        if self.verbose > 1:
                    #            print(self.sph_harm_key[orb],cutoff)
                    #        #if cutoff < 1.2: # small orbital
                    #        #ratio = self.ratio_atomic_total[orb] # ratio of how much is from center atomic part
                    #        #change_rad = large_rad & (ratio < 1.25) & (ratio > 0.75)
                    #        #print(len(np.arange(len(large_rad))[large_rad]))
                    #        #print("with ratio:",orb,len(np.arange(len(large_rad))[change_rad]))
                    og_maxrad = copy.deepcopy(self.orig_maxrad[orb])
                    #count_max = min_rad < og_maxrad*2
                    #print(min_rad[largeenough][np.argsort(np.abs(radunk))[::-1][:20]])
                    largest_rad = np.median(min_rad[largeenough][np.argsort(np.abs(radunk))[::-1][:20]])
                    rad_max = min_rad[np.argmax(flat_orb)]
                    if og_maxrad > 0.2 and rad_max > 0.2 and largest_rad > 0.2: # rescale cutoff based on new maximum
                        newcutoff = cutoff*largest_rad/og_maxrad
                        #print("radii:",rad_max,og_maxrad,largest_rad)
                        #print("rescale:",orb,cutoff,newcutoff)
                        cutoff = newcutoff
                    large_rad = min_rad>cutoff+justloop/5
                    
                    #ratio = self.ratio_atomic_total[orb] # ratio of how much is from center atomic part
                    #change_rad = large_rad & (ratio < 1.25) & (ratio > 0.75)
                    flat_orb[large_rad] = flat_orb[large_rad]*np.exp(-1.5*np.abs(min_rad[large_rad]-(cutoff+justloop/5)))#/ratio[change_rad]

                    real_orb = np.reshape(flat_orb,shape)
                    centered_orb = self.real_to_recip(np.array([real_orb]))[0]
                
                new_orbs[orb] = centered_orb*exp_term #str()
                    
            '''
            # also normalize wrt AE overlap
            # calculate the all-electron orbital overlap
            #aeoverlap = self.get_ae_overlap_info(new_orbs,just_ae_overlap = True)
            #norm = np.diag(aeoverlap)
            ## now renormalize the orbitals to the ae overlap values
            #for ind, orb in enumerate(list(new_orbs.keys())):
            #    normed_orb = new_orbs[orb]/ ((norm[ind]) ** (1 / 2))
            #    new_orbs[orb] = normed_orb
                
            # test just solving the orbitals with the coefficents matrix
            #coeff_inv = np.linalg.inv(orth_coeff) # now should be [orb, band]
            #for orb in range(self.num_orbs):
            #    new_orbs[str(orb)] = np.zeros(trueWF.shape, dtype=np.complex128)
            #    for band in range(self.num_bands): 
            #        new_orbs[str(orb)] += coeff_inv[orb][band] * periodic_WFs[band]
            #for orb in range(self.num_orbs):
            #    new_orbs[str(orb)] = normalize_wf(new_orbs[str(orb)], self._a, self.gridxyz,recip=True)
            
            # only keep certain components
            #if justloop < 5:
            #from pymatgen.core import Structure
            #from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
            # get symmetry operations
            #structure = Structure.from_file(self.directory + 'POSCAR')
            # pass "structure" to define class into "space"
            #space = SpacegroupAnalyzer(structure,symprec=0.00001,angle_tolerance=-1.0)
            #point_groups = space.get_symmetry_operations()#get_point_group_operations()
            #cart_point_groups = space.get_symmetry_operations(cartesian=True)#get_point_group_operations(cartesian=True)
            #operations = np.zeros((len(point_groups),4,4),dtype=np.float64)
            #cart_operations = np.zeros((len(point_groups),4,4),dtype=np.float64)
            #for ind in range(len(point_groups)):
            #    operations[ind] = np.around(point_groups[ind].affine_matrix,decimals=6) #[:3,:3]
            #    cart_operations[ind] = np.around(cart_point_groups[ind].affine_matrix,decimals=6) #[:3,:3]
            
            '''
                # now symmetrize the p orbitals 
                sph_key = self.sph_harm_key[orb]
                
                if sph_key == 0 or sph_key == 1 or sph_key == 2 or sph_key == 3: # pz orbital
                    symz = np.array([[1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,1]])
                    symx = np.array([[-1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
                    symy = np.array([[1,0,0,0],[0,-1,0,0],[0,0,1,0],[0,0,0,1]])
                    symzxy = [symz,symx,symy]
                    neg = [1,2,3]
                    #if sph_key == 1: # pz orbital
                    #    sym = 
                    #elif sph_key == 2: # px orbital
                    #    sym = 
                    #elif sph_key == 3: # px orbital
                    #    sym = 
                    for sind,sym in enumerate(symzxy):
                        for opt_ind,opt in enumerate(cart_operations): 
                            if (opt == sym).all():
                                #print("found opt:", opt)
                                #print("now apply symmetry")
                                irred_opt = operations[opt_ind][:3,:3]
                                # get new gpoints
                                #k2 = reduc_kpts[kpt]
                                #gpnts = self.generate_gpnts(k2)
                                #all_gpoints.append(gpnts)#old_gpoints[reduc_toir[kpt]])#np.matmul(prim_opt.T,np.array(old_gpoints[reduc_toir[kpt]]).T).T)

                                # calculate the G coefficients of wavefunction at kpoint k2 from the IBZ k1 wavefunction
                                Gcoeff1 = centered_orb #np.array(self.recip_WFcoeffs[reduc_toir[kpt]])  # [band,igpnt]
                                og_gpnts = np.array(self.gpoints[0]) # [igpnt,b1/b2/b3]
                                # sort the original gpoints and coeffs
                                ind = np.lexsort((og_gpnts[:,0],og_gpnts[:,1],og_gpnts[:,2])) # index which converts the symmetrized gpnts to the normal gpnt ordering
                                back_to_og_ind = np.argsort(ind)
                                #print(ind.shape)
                                og_gpnts = og_gpnts[ind]
                                Gcoeff1 = Gcoeff1[ind]

                                prim_opt = np.linalg.inv(irred_opt).T # symrec
                                #prim_shift = operations[symopt_ind[kpt]][:3,3]
                                #phase = np.exp(2j*np.pi*np.dot(k2[None,:]+og_gpnts, prim_shift))
                                #Gcoeff1 = Gcoeff1 #*phase[None,:]


                                # generate from irred gpnts
                                shift_gpnts = og_gpnts #+ kpt_shifts[kpt][None,:]
                                sym_gpnts = np.matmul(prim_opt,shift_gpnts.T).T

                                ind = np.lexsort((sym_gpnts[:,0],sym_gpnts[:,1],sym_gpnts[:,2])) # index which converts the symmetrized gpnts to the normal gpnt ordering
                                #print(ind.shape)
                                sorted_symg = sym_gpnts[ind]

                                #if (kpt_shifts[kpt] != 0).any():
                                #print("gpnts same?")
                                #print(og_gpnts[back_to_og_ind])
                                #print(og_gpnts)
                                #print(kpt_shifts[kpt])
                                #print(sorted_symg)
                                #print(is_opposite[kpt])

                                #old_Cnkg = np.array(self.recip_WFcoeffs[reduc_toir[kpt]])  # [band,gpoint]
                                #new_Cnkg = old_Cnkg[:,ind] * np.exp(-2j*np.pi* np.dot(gpnts, shift))
                                new_Cnkg = Gcoeff1[ind]
                                if sph_key == neg[sind]:
                                    new_Cnkg = -new_Cnkg
                                centered_orb = (Gcoeff1 + new_Cnkg)/2

                                centered_orb = centered_orb[back_to_og_ind]

                
                
                # remove components far from atom
                if justloop < 50:
                    cutoff = 2.0
                    real_orb = self.recip_to_real({0:centered_orb})[0]
                    shape = real_orb.shape
                    flat_orb = real_orb.flatten()

                    min_rad = self.zeromin_rad
                    if sph_key > 3: # is a d orbital
                        rad_max = min_rad[np.argmax(flat_orb)]
                        cutoff = rad_max*2.5
                        if cutoff < 2:
                            if self.verbose > 1:
                                print(self.sph_harm_key[orb],cutoff)
                            #if cutoff < 1.2: # small orbital
                            large_rad = min_rad>cutoff+justloop/5
                            ratio = self.ratio_atomic_total[orb] # ratio of how much is from center atomic part
                            change_rad = large_rad & (ratio < 1.25) & (ratio > 0.75)
                            print(len(np.arange(len(large_rad))[large_rad]))
                            print("with ratio:",orb,len(np.arange(len(large_rad))[change_rad]))
                            flat_orb[change_rad] = flat_orb[change_rad]*np.exp(-2*np.abs(min_rad[change_rad]-(cutoff+justloop/5)))#/ratio[change_rad]

                            real_orb = np.reshape(flat_orb,shape)
                            centered_orb = self.real_to_recip({0:real_orb})[0]
                    
                    elif sph_key > 0: # is p orbital
                        rad_max = min_rad[np.argmax(flat_orb)]
                        cutoff = rad_max*3
                        if cutoff < 2:
                            if self.verbose > 1:
                                print(self.sph_harm_key[orb],cutoff)
                            #if cutoff < 1.2: # small orbital
                            large_rad = min_rad>cutoff+justloop/5
                            ratio = self.ratio_atomic_total[orb] # ratio of how much is from center atomic part
                            change_rad = large_rad & (ratio < 1.25) & (ratio > 0.75)
                            print(len(np.arange(len(large_rad))[large_rad]))
                            print("with ratio:",orb,len(np.arange(len(large_rad))[change_rad]))
                            flat_orb[change_rad] = flat_orb[change_rad]*np.exp(-2*np.abs(min_rad[change_rad]-(cutoff+justloop/5)))#/ratio[change_rad]

                            real_orb = np.reshape(flat_orb,shape)
                            centered_orb = self.real_to_recip({0:real_orb})[0]
                    else: # is s orbital
                        rad_max = min_rad[np.argmax(flat_orb)]
                        cutoff = 2.0 #rad_max*3
                        if cutoff < 3:
                            if self.verbose > 1:
                                print(self.sph_harm_key[orb],cutoff)
                            #if cutoff < 1.2: # small orbital
                            large_rad = min_rad>cutoff+justloop/5
                            ratio = self.ratio_atomic_total[orb] # ratio of how much is from center atomic part
                            change_rad = large_rad & (ratio < 1.25) & (ratio > 0.75)
                            print(len(np.arange(len(large_rad))[large_rad]))
                            print("with ratio:",orb,len(np.arange(len(large_rad))[change_rad]))
                            flat_orb[change_rad] = flat_orb[change_rad]*np.exp(-2*np.abs(min_rad[change_rad]-(cutoff+justloop/5)))#/ratio[change_rad]

                            real_orb = np.reshape(flat_orb,shape)
                            centered_orb = self.real_to_recip({0:real_orb})[0]

                    #else: # is s orbital
                    #    rad_max = min_rad[np.argmax(flat_orb)]
                    #    if rad_max > 0.5: # has hump you can use
                    #        cutoff = rad_max*3
                    #    else:
                    #        cutoff = self.cutoff_rad[self.orbatomnum[orb]]*1
            '''    
                
            #new_orbs[str(orb)] = centered_orb*exp_term
                
            '''
            # orthogonalize orbitals on the same atom #done not with ae
            previous_orb = 0
            for atms in self.lastatom:  #: [9, 18, 22, 26]
                for_gram = {}
                for orb in range(previous_orb, atms):
                    for_gram[str(orb)] = new_orbs[str(orb)]
                holder = self.lowdin_orth(for_gram,recip=True)
                for orb in range(previous_orb, atms):
                    new_orbs[str(orb)] = holder[str(orb)]
                previous_orb = atms
            
            if justloop > 2:
                previous_orb = 0
                for atms in self.lastatom:  #: [9, 18, 22, 26]
                    orb_types = self.orbtype[previous_orb:atms]
                    is_s2 = orb_types == "s2"
                    if (is_s2).any():
                        s2_num = np.arange(previous_orb,atms)[is_s2][0]
                        s_num = np.arange(previous_orb,atms)[orb_types == "s"][0]
                        old_s2 = new_orbs[str(s2_num)]
                        s_orb = new_orbs[str(s_num)]
                        # working on! 
                        # now gram schidt orthogonalize the s2 orbital to the s orbital
                        ss2_overlap = aeoverlap[s2_num,s_num] #np.sum(np.conj(old_s2)*s_orb)
                        ss_overlap = aeoverlap[s_num,s_num] #np.sum(np.conj(s_orb)*s_orb)
                        new_s2 = old_s2 - ss2_overlap/ss_overlap*s_orb
                        new_orbs[str(s2_num)] = new_s2
                        new_overlap = np.sum(np.conj(new_s2)*s_orb)
                        print("made new s2 orbital",ss2_overlap,new_overlap)

                    previous_orb = atms
                aeoverlap = self.get_ae_overlap_info(new_orbs,just_ae_overlap = True)
                print(aeoverlap[s2_num,s_num])
            '''
            # also normalize wrt AE overlap
            # calculate the all-electron orbital overlap
            #aeoverlap = self.get_ae_overlap_info(new_orbs,just_ae_overlap = True)
            aeoverlap = self.get_ae_overlap_matrix(new_orbs,secondWF = new_orbs,secondisarray=True)
            norm = np.diag(aeoverlap)
            # now renormalize the orbitals to the ae overlap values
            for orb in range(len(new_orbs)):#ind, enumerate(list(new_orbs.keys())):
                normed_orb = new_orbs[orb]/ ((norm[orb]) ** (1 / 2))
                new_orbs[orb] = normed_orb
            # redefine the orbital WF
            converg_orbs = copy.deepcopy(new_orbs)
            
        #direct algorithm    
        wan_orbs = np.zeros(new_orbs.shape, dtype=np.complex128)
        new_orbs[:] = 0
        if num_steps == 0 or True:
            #aeoverlap = self.get_ae_overlap_info(converg_orbs,just_ae_overlap = True)
            og_coeff_inv = np.matmul(np.conj(og_orth_coeff),og_aeoverlap[:,:self.num_orbs]).T #np.linalg.inv(orth_coeff) # now should be [orb, band]
            coeff_inv = np.matmul(np.conj(orth_coeff),aeoverlap[:,:self.num_orbs]).T
            orb_mixing = np.matmul(og_orth_coeff[:,:self.num_orbs].T,og_coeff_inv.T)
            np.fill_diagonal(orb_mixing,0)
            if self.verbose > 0:
                print("orb mixing:",np.amax(np.abs(orb_mixing)))
            for orb in range(self.num_orbs):
                for bind, band in enumerate(sortedLoBds):
                    wan_orbs[orb] += og_coeff_inv[orb][bind] * periodic_WFs[band]
                    new_orbs[orb] += coeff_inv[orb][bind] * periodic_WFs[band]
            #for orb in range(self.num_orbs):
            #    wan_orbs[orb] = normalize_wf(wan_orbs[orb], self._a, self.gridxyz,recip=True)

        if num_steps == 0:
            new_orbs = wan_orbs

        if (self.save_orb_data or self.save_orb_figs) and (len(self.cartAtoms) != 1) and self.verbose > 1:
            real_wan = self.recip_to_real(wan_orbs)
            real_new = self.recip_to_real(new_orbs)
            real_og = self.recip_to_real(og_orbs)
            real_conv = self.recip_to_real(converg_orbs)
    
            plots = []
            all_data = []
            if not hasattr(self, 'og_wan_data'):
                make_wan = True
                wan_data = []
                init_data = []
            else:
                make_wan = False
            for orb in range(self.num_orbs):
                atmnum = self.orbatomnum[orb]
                diff_atoms = np.array(self.cartAtoms)-np.array(self.cartAtoms[atmnum])[None,:]
                closed_atm = np.argsort(np.linalg.norm(diff_atoms,axis=1))[1]
                if self.verbose > 1:
                    print(self.cartAtoms[atmnum],closed_atm,diff_atoms[closed_atm])
    
                if make_wan:
                    wan_dists1, wan_vals1 = extract_line_data(real_wan[orb].real.flatten(), self.min_xyz[atmnum], np.array([0,0,0]), diff_atoms[closed_atm], tolerance=0.1)
                    og_dists1, og_vals1 = extract_line_data(real_og[orb].real.flatten(), self.min_xyz[atmnum], np.array([0,0,0]), diff_atoms[closed_atm], tolerance=0.1)
                    #wan_dists2, wan_vals2 = extract_line_data(real_wan[4].flatten(), self.cartXYZ, self.cartAtoms[1], self.cartAtoms[0], tolerance=0.1)
                    wan_data.append([wan_dists1, wan_vals1])
                    init_data.append([og_dists1, og_vals1])
                else:
                    wan_dists1, wan_vals1 = self.og_wan_data[orb]
                    og_dists1, og_vals1 = self.og_init_data[orb]
                cog_dists1, cog_vals1 = extract_line_data(real_new[orb].real.flatten(), self.min_xyz[atmnum], np.array([0,0,0]), diff_atoms[closed_atm], tolerance=0.1)
                #cog_dists2, cog_vals2 = extract_line_data(real_new[4].flatten(), self.cartXYZ, self.cartAtoms[1], self.cartAtoms[0], tolerance=0.1)
        
                con_dists1, con_vals1 = extract_line_data(real_conv[orb].real.flatten(), self.min_xyz[atmnum], np.array([0,0,0]), diff_atoms[closed_atm], tolerance=0.1)
    
                fig, ax = plt.subplots()
                #plt.figure(figsize=(8, 5))
                ax.plot(wan_dists1, wan_vals1, "blue",label="wannier")
                #plt.plot(wan_dists2, wan_vals2, "black")
                ax.plot(cog_dists1, cog_vals1, "red",label="cogito")
                ax.plot(con_dists1, con_vals1, "pink",label="cogito-sym")
                #plt.plot(cog_dists2, cog_vals2, "purple")
                ax.plot(og_dists1, og_vals1, "black",label="initial")
                #ax.set_xlim((np.amin(og_dists1),np.amax(og_dists1)))
                #ax.set_ylim((np.amin(og_vals1)-np.amax(og_vals1)*0.2,np.amax(og_vals1)*1.2))
                ax.set_xlabel("Distance along line (Å)")
                ax.set_ylabel("Orbital magnitude")
                ax.set_title("Orbital Magnitude vs. Distance")
                ax.legend()
                #plt.grid(True)
                plots.append(fig)
                all_data.append([[list(wan_dists1), list(wan_vals1)],[list(og_dists1), list(og_vals1)],[list(cog_dists1), list(cog_vals1)]])
                if self.plot_orbs and self.verbose > 1:
                    plt.show()
                else:
                    plt.close()
            if make_wan:
                self.og_wan_data = wan_data
                self.og_init_data = init_data
            if self.save_orb_figs:
                combine_and_save_plots(plots, filename=self.directory+"init_cogito"+self.tag+".png")
            if self.save_orb_data:
                import json
                with open(self.directory+"all_converge_data"+self.tag+".json", 'w') as f:
                    # indent=2 is not needed but makes the file human-readable 
                    # if the data is nested
                    json.dump(all_data, f, indent=2) # has real part
            
        converg_orbs = new_orbs

        # for loop ends
        # normalize back to pseudo for fitting
        for orb in range(self.num_orbs):
            # only for k==0; need Gamma_point=False for more general formalism
            converg_orbs[orb] = normalize_wf(new_orbs[orb], self._a, self.gridxyz,recip=True)
            wan_orbs[orb] = normalize_wf(wan_orbs[orb], self._a, self.gridxyz,recip=True)
            
        real_wan = self.recip_to_real(wan_orbs)
        real_new = self.recip_to_real(converg_orbs)
        # get radii for orbitals
        wan_rad = np.zeros(self.num_orbs)
        cog_rad = np.zeros(self.num_orbs)
        for orb in range(self.num_orbs):
            atm = self.orbatomnum[orb]
            min_rad = self.min_rad[atm]
            wan_rad[orb] = np.sum(min_rad * real_wan[orb].real.flatten()**2)*10000#self.vol*self.gridxyz[0]*self.gridxyz[1]*self.gridxyz[2]
            cog_rad[orb] = np.sum(min_rad * real_new[orb].real.flatten()**2)*10000#self.vol*self.gridxyz[0]*self.gridxyz[1]*self.gridxyz[2]
        if self.verbose > 1:
            print("avg normed rads!",wan_rad,cog_rad)
        # save real orbitals to orbitalWF
        self.one_orbWFrecip = converg_orbs
        #real_orbs = self.recip_to_real(converg_orbs)
        #self.one_orbitalWF = real_orbs
        #self.ex_pseudos = self.recip_to_real(excited_orbs)

    def fit_to_atomic_orb(self,plot_orbs = False,orbfactor=1.0):
        """
        This function is necessary when using reciprocal integrals
        because they need an individual orbital rather than the unk = sum_R(y(r-R)).
        The algorithm used here determines the percent of the center orbital
        which is exact if the orbital has a precise exponential decay with radius. See COGITO paper for more details.
        """
        #unkorbs = self.one_orbitalWF
        #og_orbs = self.one_orbitalWF
        #keys = list(unkorbs.keys())
        #print("spherical harmonics key:", self.sph_harm_key)
        #fig1 = plt.figure()
        #fig2 = plt.figure()
        orb_to_l = [0, 1, 1, 1, 2, 2, 2, 2, 2,3,3,3,3,3,3,3]
        orbital_coeffs = np.zeros((self.num_orbs,9))
        recip_orbcoeffs = np.zeros((self.num_orbs,9))
        avg_radius = np.zeros(self.num_orbs)
        radial_nrmse = np.zeros(self.num_orbs)
        radial_nme = np.zeros(self.num_orbs)
        final_gaus_fiterrors = np.zeros((self.num_orbs,4))
        # shift all the orbitals so that it's atom is at [0,0,0] on the grid

        recip_orbs = self.one_orbWFrecip
        gpoints = copy.deepcopy(np.array(self.gpoints[0], dtype=np.int_))
        center = self.orbpos # orb v 3
        exp_term = np.exp(2j * np.pi * np.dot(gpoints, center.T).T) # (num_gpnts v 3) . (3 v orb)
        centered_recip = recip_orbs * exp_term # orb v num_gpoints
        centered_orbs = self.recip_to_real(centered_recip) # self.center_real_orbs(unkorbs)

        # remake and minradphitheta
        all_atoms = []
        for trans1 in range(3):
                for trans2 in range(3):
                    for trans3 in range(3):
                        vec = (trans1 - 1) * self._a[0] + (trans2 - 1) * self._a[1] + (trans3 - 1) * self._a[2]
                        all_atoms.append(vec)
        # print(all_atoms)
        point = self.cartXYZ
        min_rad = np.ones((len(self.cartXYZ[0]))) * 100
        min_phi = np.ones((len(self.cartXYZ[0]))) * 100
        min_theta = np.ones((len(self.cartXYZ[0]))) * 100
        hold_atomvecs = np.ones((len(self.cartXYZ[0]), 3, 27)) * 100
        for atm in all_atoms:
            vec = point - np.array([atm]).transpose()
            rad = np.linalg.norm(vec, axis=0)
            rad[rad==0] = 0.0000000001
            theta = np.arccos(vec[2] / rad)
            theta[rad == 0.0000000001] = 0
            phi = np.arctan2(vec[1], vec[0])

            bool1 = rad < min_rad
            min_rad[bool1] = rad[bool1]
            min_theta[bool1] = theta[bool1]
            min_phi[bool1] = phi[bool1]

        # try much simplier method of getting groups by checking eigenvalues
        orb_groups = []
        uniq_type = []
        #print("environment tensor!",self.environ_tensors)

        # also include atom magnetic moment in determining symemtrically equivalent orbitals
        tot_magmoms = np.zeros(self.numAtoms)
        if self.spin_polar:
            if self.spin_polar:
                from pymatgen.io.vasp.outputs import Outcar
                outcar = Outcar(self.directory + "OUTCAR")
                mag = outcar.magnetization
                tot_magmoms = [m["tot"] for m in mag]

        for orb in range(self.num_orbs):
            sphkey = self.sph_harm_key[orb]
            l = [0,1,1,1,2,2,2,2,2,3,3,3,3,3,3,3][sphkey]
            atomnum = self.orbatomnum[orb]
            if l == 0:
                eigval = str(np.around(self.environ_tensors[atomnum][0],decimals=5))
                uniq_type.append(eigval + self.orbtype[orb] + self.elements[atomnum] + str(np.around(tot_magmoms[atomnum],decimals=3))) ## HERE
            if l == 1:
                eigval = max(np.abs(np.matmul(self.environ_tensors[atomnum][1],self.orb_vec[orb])))/max(np.abs(self.orb_vec[orb]))
                uniq_type.append( str(np.around(eigval,decimals=5)) + self.orbtype[orb] + self.elements[atomnum] + str(np.around(tot_magmoms[atomnum],decimals=3)))
            if l == 2:
                eigval = max(np.abs(np.matmul(self.environ_tensors[atomnum][2],self.orb_vec[orb])))/max(np.abs(self.orb_vec[orb]))
                uniq_type.append( str(np.around(eigval,decimals=5)) + self.orbtype[orb] + self.elements[atomnum] + str(np.around(tot_magmoms[atomnum],decimals=3)))
            if l == 3:
                eigval = max(np.abs(np.matmul(self.environ_tensors[atomnum][3],self.orb_vec[orb])))/max(np.abs(self.orb_vec[orb]))
                uniq_type.append( str(np.around(eigval,decimals=5)) + self.orbtype[orb] + self.elements[atomnum] + str(np.around(tot_magmoms[atomnum],decimals=3)))
        uniq_type = np.array(uniq_type)
        uniq_strs,uniq_ind = np.unique(uniq_type,return_index=True)
        uniq_strs = uniq_type[np.sort(uniq_ind)]
        for string in uniq_strs:
            group = np.arange(self.num_orbs)[uniq_type==string]
            orb_groups.append(group)

        if self.verbose > 0:
            print("found like orbital groups!",orb_groups)


        #if len(included_orbs) != self.num_orbs:
        #    print("error!!! not all orbitals are included in the fitting!")
        
        #orb_groups = np.array([np.arange(self.num_orbs)],dtype=np.int_).T

        # fit the orbitals to gaussian
        # first find the radial part of each orbital (including the extra part to try and get the atomic rather than Bloch)
        # than, for each orbital group, include all in all the orbitals and fit to function
        # update prediction and repeat
        
        all_radunk = {}
        all_good_min_rad = {}
        all_good_ind = {}
        all_weights = {}
        all_og_orbs = {}
        reconstruct_rad = {}
        all_ratio = self.ratio_atomic_total #[orb,nx*ny*nz]
        tot_orbs = self.tot_orbs #[orb,nx*ny*nz]
        orbgroup_gausparams = np.zeros((len(orb_groups),9))
        orbgroup_finalgausparams = np.zeros((len(orb_groups),9))
        orbgroup_pseudoparams = np.zeros((len(orb_groups),9))
        orbgroup_expparams = np.zeros((len(orb_groups),7))
        
        num_loop = 3
        abc = np.linalg.norm(self._a,axis=1)
        min_abc = np.amin(abc)
        #print("abc:",abc,min_abc)

        if min_abc > 10:
            num_loop = 1
        elif min_abc > 4:
            num_loop = 3
        else:
            num_loop = 4
        #print("num loops:",num_loop)
         
        import time
        
        start_time = time.time()
        #num_loop = 1
        print("fitting orbitals over", num_loop, "loops")

        for loop in range(num_loop):
            end_time = time.time()

            # Calculate elapsed time
            elapsed_time = end_time - start_time
            # Record start time
            start_time = time.time()
            if self.verbose > 0:
                print(loop,elapsed_time)
            # get the orbital radunk, good_min_rad, and weights
            
            start_time = time.time()
            for orb in range(self.num_orbs):#orbind, enumerate(keys):
                atom = self.orbatomnum[orb]
                ratio = all_ratio[orb].real
                unk = centered_orbs[orb].flatten().real
                og_orb = centered_orbs[orb].flatten().real

                # check how close the orbital is to the last one constructed
                diff_orb = np.abs((tot_orbs[orb] - unk)/unk)
                # test using the ratio to get only nearest atomic
                unk = unk * ratio
                # fit the exponential decay
                orbperatom = self.sph_harm_key[orb]
                angular_part = complex128funs(min_phi, min_theta, self.orb_vec[orb])
                #change points very close to zero rad for p and d orbs
                if orbperatom != 0:
                    angular_part[min_rad < 0.001] = 0.0001 # just for the limit so point is never included
                largeenough = np.abs(angular_part)>0.01
                radunk = unk[largeenough]/angular_part[largeenough]
                angular_part[angular_part == 0] = 0.001
                og_radunk = og_orb/angular_part
                og_radunk = og_radunk[largeenough]
                good_min_rad = min_rad[largeenough]
                
                # change p and d orbs for the center point is zero
                if orbperatom != 0:
                    radunk[good_min_rad < 0.001] = 0.0 # just for the limit later


                positive = (radunk != 0.0) & (good_min_rad != 0) #& (np.abs(ratio[largeenough])<20) & (np.abs(ratio[largeenough])>0.05)#& (np.abs(diff_orb[largeenough]) < 1.0)
                radunk = radunk[positive]
                good_min_rad = good_min_rad[positive]
              
                weight_ratio = np.abs(ratio[largeenough][positive])**(1/2)*np.exp(-1/2*np.abs(ratio[largeenough][positive]-1)**3)
                
                l=orb_to_l[orbperatom]
                weights = (1/(good_min_rad+0.001)**(1/2/(l+1)))*weight_ratio*np.abs(angular_part[largeenough][positive])**(1/2)#+radunk*5000
                orbkey = self.orbtype[orb]
                no_node = orbkey[0] == orbkey
                                
                # randomly select points to fit based on (1-(x/cutoff)^2) to speed up fit and have more even radial distribution of points
                rand = np.random.rand(len(radunk))
                func = 1/good_min_rad**(3/2) #(1-(good_min_rad/(self.cutoff_rad[atom]*2))**(2))
                keep_point = rand<func
                radunk = radunk[keep_point]
                good_min_rad = good_min_rad[keep_point]
                weights = weights[keep_point]
                #print("reduced points:",len(rand),len(radunk))
                
                # cutoff after certain large distance
                # this helps make the fitting scale linearly with the number of atoms rather than supercell size
                cutoff = self.cutoff_rad[atom]*0.9+np.tanh((self.init_orb_energy[orb]+8)/5)*0.5#self.cutoff_rad[atom]*0.8#**(1/2)#*1.4
                #print("new cut:",cutoff,self.init_orb_energy[orb])
                inside_cut = good_min_rad < 4*cutoff
                radunk = radunk[inside_cut]
                good_min_rad = good_min_rad[inside_cut]
                weights = weights[inside_cut]

                # add extra points to help with the fitting of orbitals with a node (usually second s orbital)
                if not no_node and orbkey[0] == 's': # only do for s orbitals
                    included_ratio = ratio[largeenough][positive][keep_point][inside_cut]
                    #good_ratio = (included_ratio > 0.6) & (included_ratio < 2)
                    #below_cut = good_min_rad[good_ratio] < cutoff*1.1
                    gd_rat_low = (included_ratio > 0.6) & (included_ratio < 2) & (good_min_rad < cutoff)#*1.1)
                    max_unk = np.average(np.sort(radunk[gd_rat_low])[-15:]) # average max point
                    rad_at_max = np.average(good_min_rad[gd_rat_low][np.abs(radunk[gd_rat_low]-max_unk) < max_unk*0.2])
                    final_max = np.average(np.sort(radunk[gd_rat_low][np.abs(good_min_rad[gd_rat_low]-rad_at_max) < 0.05])[-10:])
                    if self.verbose > 1:
                        print("original max, rad, final max:",max_unk,rad_at_max,final_max)
                    #rad_at_max = np.average(good_min_rad[gd_rat_low][(radunk[gd_rat_low] < final_max*1.2) & (radunk[gd_rat_low] > final_max)])
                    #final_max = np.average(np.sort(radunk[gd_rat_low][np.abs(good_min_rad[gd_rat_low]-rad_at_max) < 0.05])[-5:])
                    #print("original max, rad, final max:",max_unk,rad_at_max,final_max)
                    if math.isnan(rad_at_max) or math.isnan(final_max):
                        if self.verbose > 1:
                            print("caught nan")
                        radunk = np.append(radunk,[0,0])
                        good_min_rad = np.append(good_min_rad,[cutoff+2,cutoff+2])
                        weights = np.append(weights,[0.01,0.01])
                    else:
                        radunk = np.append(radunk,[final_max,0])
                        good_min_rad = np.append(good_min_rad,[rad_at_max,cutoff+2])
                        weights = np.append(weights,[2,0])
                        

                #append some high rad values to maintain a small tail
                radunk = np.append(radunk,np.zeros(20))#[0,0,0,0])
                good_min_rad = np.append(good_min_rad,np.linspace(cutoff+0,cutoff+4,20))#[cutoff+0.5,cutoff+0.9,cutoff+1.4,cutoff+2.5])
                    
                if no_node:
                    weights = np.append(weights,np.linspace(0,20,20))#[4,8,12,20])
                else:
                    weights = np.append(weights,np.linspace(0,20,20))#[4,8,12,20])
                all_radunk[orb] = radunk
                all_good_min_rad[orb] = good_min_rad
                all_weights[orb] = weights
                if not no_node and orbkey[0] == 's':
                    all_og_orbs[orb] = np.append(og_radunk[positive][keep_point][inside_cut],np.zeros(22))#[0,0,0,0,0,0])
                else:
                    all_og_orbs[orb] = np.append(og_radunk[positive][keep_point][inside_cut],np.zeros(20))#[0,0,0,0])
                all_good_ind[orb] = np.arange(len(og_orb))[largeenough][positive][keep_point][inside_cut]

            end_time = time.time()
            make_rad = end_time - start_time
            
            orb_nrms = np.zeros(self.num_orbs)
            orb_nme = np.zeros(self.num_orbs)
            start_time = time.time()
            for group_ind,orbgroup in enumerate(orb_groups):
                first_time = time.time()
                atom = self.orbatomnum[orbgroup[0]]
                orbperatom = self.sph_harm_key[orbgroup[0]]
                # append all values for each orb in orbgroup
                good_min_rad = all_good_min_rad[orbgroup[0]]
                radunk = all_radunk[orbgroup[0]]
                og_radunk = all_og_orbs[orbgroup[0]]
                weights = all_weights[orbgroup[0]]
                for orbind,orb in enumerate(orbgroup[1:]):
                    radunk = np.append(radunk,all_radunk[orb])
                    good_min_rad = np.append(good_min_rad,all_good_min_rad[orb])
                    weights = np.append(weights,all_weights[orb])
                    og_radunk = np.append(og_radunk,all_og_orbs[orb])

                # use the previously fitted gaus params to the pseudo orbital as initial condition
                [pa,pb,pc,pd,pe,pf,l] = self.orig_gausparams[orbgroup[0]]
                #[pg,ph] = [0,new_cutoff/4]
                # rescale the magnitude of the converged orbital
                simprad = np.linspace(0,5,100)
                simporb = func_for_rad(simprad,pa,pb,pc,pd,pe,pf,0,1/2,l=l)
                max_ind = np.argmax(np.abs(simporb))
                max_rad = simprad[max_ind]
                max_val = np.abs(simporb[max_ind])
                
                l=orb_to_l[orbperatom]
                max_pos_ind = np.argmax(simporb)
                max_pos = np.amax(simporb)
                test_cut_ind = np.argmin(np.abs(simporb[max_pos_ind:]-max_pos/3))
                test_cutoff = simprad[max_pos_ind:][test_cut_ind]
                cutoff = self.cutoff_rad[atom]*0.9+np.tanh((self.init_orb_energy[orbgroup[0]]+8)/5)*0.5#self.cutoff_rad[atom]*0.8#**(1/2)#*1.4
                #print("testing cutoffs:",test_cutoff,cutoff)
                if test_cutoff > cutoff:
                    test_cutoff = test_cutoff*0.25+cutoff*0.75
                #cutoff = test_cutoff
                expcutoff = cutoff**(1/2)
                new_cutoff = cutoff**(1/2)
                orbkey = self.orbtype[orbgroup[0]]
                no_node = orbkey[0] == orbkey # only is one character: 's', 'p', or 'd'
                #if not no_node: # make cutoff a bit larger
                #    cutoff = cutoff*1.1

                fourth_time = time.time()
                # go back to curve_fit
                #e = 0
                #f = expcutoff/2
                [pg,ph] = [0,new_cutoff/4]
                g = 0
                h = expcutoff/2
                min_cut = new_cutoff/15 # expcutoff/8
                if l == 2:
                    min_cut = expcutoff/4
                l=orb_to_l[orbperatom]
                x = good_min_rad[good_min_rad<=test_cutoff]
                y = radunk[good_min_rad<=test_cutoff]
                               
                truemaxval = np.abs(np.average(y[np.abs(x-max_rad) < 0.1/(l+1)]))#,weights = weights[good_min_rad<=cutoff][np.abs(x-max_rad) < 0.1/(l+1)]))
                if np.isnan(truemaxval):
                    truemaxval = max_val
                #print("normalizing values:",max_rad,max_val,truemaxval)
                #print("before norm:",[pa,pc,pe])
                [pa,pc,pe] = np.array([pa,pc,pe])/max_val*truemaxval

                # try fitting with contraints built into function so can use curve_fit
                init_con1 = pd/pb #np.log( - 1)
                init_con4 = 2/3 #np.log(- 1)
                init_con3 = pf/pb #np.log(pb/pf - 1)
                init_con2 = pe/pf+pa/pb
                max_aceg = np.amax(np.abs([pa,pc,pe,pg]))
                if no_node:
                    gaus_init = [pa,pb,init_con1,init_con2,init_con3]#,pg,init_con4] 
                    partial_gaus_fit = partial(func_for_rad_fit,c=pc,g=0,con4=init_con4, l=l)
                    low_bounds = np.ones(5)*0.000001 #[0,0,0,0,0]
                    high_bounds = [max_aceg*10,new_cutoff*3,0.95,max_aceg*10,0.9]#,max_aceg*20,0.9]
                else: # allow node in fit
                    gaus_init = [pa,pb,init_con1,init_con2,init_con3,pc]#,pg,init_con4]
                    partial_gaus_fit = partial(func_for_rad_fit,g=0,con4=init_con4, l=l)
                    low_bounds = np.ones(6)*0.000001
                    low_bounds[3] = -max_aceg*10
                    high_bounds = [max_aceg*10,new_cutoff*3, 0.95,max_aceg*10, 5, max_aceg*10]#,max_aceg*20,0.9]
                    #low_bounds[3] = -max_aceg*20 # remove low bound for con2
                    #high_bounds[4] = 5 # can be larger than b now, but generally is still not
                if self.verbose > 1:
                    print(gaus_init,low_bounds,high_bounds)
                gaus_init = np.array(gaus_init) + 0.000001
                x = np.append(x,[cutoff+1.25,cutoff+2.5])
                y = np.append(y,[0,0])
                init_weight = np.append(weights[good_min_rad<=test_cutoff],[5,10])
                sig = 1/(init_weight+0.00001)
                #print('check bounds:',gaus_init,low_bounds,high_bounds)
                popt,pcov = curve_fit(partial_gaus_fit,x,y,p0=gaus_init,sigma=sig,bounds=(low_bounds,high_bounds),ftol=0.000001, xtol=0.000001,method="trf",max_nfev=5000)
                if no_node:
                    [a,b,con1,con2,con3] = popt
                    c = pc
                else:
                    [a,b,con1,con2,con3,c] = popt
                d = con1 * b
                f = con3 * b
                e = (con2 - a/b) * f
                g = 0
                h = ph
                
                if self.verbose > 0:
                    print("init gaus terms:",a,b,c,d,e,f)

                # fit end part to exponential function
                end_gaus = func_for_rad(test_cutoff,a,b,c,d,e,f,g,h,l)
                c_e = 0
                d_e = expcutoff
                e_e = 0
                f_e = expcutoff/2
                l=orb_to_l[orbperatom]
                exp_fit = partial(func_for_rad_exp,e=e_e,f=f_e,l=l)
                x = np.append(good_min_rad[good_min_rad>test_cutoff],[test_cutoff])
                y = np.append(radunk[good_min_rad>test_cutoff],[end_gaus])
                weights[np.abs(weights) <= 0.0000001] = 0.0000001
                sig = np.append(1/weights[good_min_rad>test_cutoff],[0.1])
                orig_expparams = self.orig_expparams[orbgroup[0]]
                pbe = 1/orig_expparams[1]/2
                pae = orig_expparams[0]*pbe
                pinit = [orig_expparams[0],orig_expparams[1],orig_expparams[0]/10,orig_expparams[1]/2]
                [pbe,pde] = [1/pinit[1]/2,1/pinit[3]/2]
                [pae,pce] = [pinit[0]*pbe,pinit[2]*pde]
                pinit = [0.0005,expcutoff/2,0.001,expcutoff]
                low_bounds = [0.000001,expcutoff*0.05,0.000001,expcutoff*0.05]
                popt,pcov = curve_fit(exp_fit,x,y,p0=pinit,sigma=sig,bounds=(low_bounds,expcutoff*3),ftol=0.00000001, xtol=0.00000001,method="trf",max_nfev=5000)
                [a_e,b_e,c_e,d_e] = popt

                if self.verbose > 0:
                    print("exp fitting factors:",a_e,b_e,c_e,d_e)

                #plot
                if plot_orbs == True and self.verbose>2: # no need to print all these
                    plt.scatter(good_min_rad,og_radunk,c="black",s=1)
                    #plt.scatter(min_rad[big_ang],new_rad_part,c="red",s=1)
                    plt.scatter(good_min_rad,radunk,c="blue",s=weights*10)
                    new_rad = np.linspace(0,5,100)
                    #plot the original starting radial part
                    plt.plot(new_rad,self.orig_radial[orbgroup[0]],color="green",linewidth=1.0)
                    #plt.scatter(min_rad[big_ang],test_new[big_ang],facecolors="None",edgecolors="red")
                    just_one = func_for_rad(new_rad,a,b,c,d,e,f,g,h,l=l)
                    just_one[new_rad>test_cutoff] = func_for_rad_exp(new_rad[new_rad>test_cutoff],a_e,b_e,c_e,d_e,e_e,f_e,l=l)
                    plt.plot(new_rad,just_one,color="black",linewidth=1.0)
                    print("showing plot")
                    plt.show()
                    plt.close()

                #save fitted values
                orbgroup_gausparams[group_ind] = [a,b,c,d,e,f,g,h,l]
                orbgroup_pseudoparams[group_ind] = [pa,pb,pc,pd,pe,pf,pg,ph,l]
                orbgroup_expparams[group_ind] = [a_e,b_e,c_e,d_e,e_e,f_e,l]

                # fit to one gaussian first before making new ratio values
                [a,b,c,d,e,f,g,h,l] = orbgroup_gausparams[group_ind]
                [a_e,b_e,c_e,d_e,e_e,f_e,l] = orbgroup_expparams[group_ind]
                atom = self.orbatomnum[orbgroup[0]]
                #cutoff = self.cutoff_rad[atom]*0.8#**(1/2)#*1.4
                #expcutoff = 1/cutoff
                # orbital dependant section
                # now fit to just gaussian functions
                new_rad = np.linspace(0,(cutoff/2)**(1/2)*5,100)
                just_one = func_for_rad(new_rad,a,b,c,d,e,f,g,h,l=l)
                just_one[new_rad>test_cutoff] = func_for_rad_exp(new_rad[new_rad>test_cutoff],a_e,b_e,c_e,d_e,e_e,f_e,l=l)
    
                #use the original WF
                #just_one = self.orig_radial[orbind]
                weights = np.ones(len(new_rad))#1/(new_rad**(1/2)+0.2)
                #if l != 0: # de-weight near zero
                #    weights = 10*np.tanh(new_rad/2+0.1)/(new_rad**(3)+5)
                #weights[new_rad>cutoff] = weights[new_rad>cutoff] * (just_one[new_rad>cutoff]/end_gaus)**(1/4) * 1.5
                just_one = np.append(just_one,np.zeros(20))
                new_rad = np.append(new_rad,np.linspace(cutoff+1,cutoff+3,20))#[cutoff+1.5,cutoff+1.8,cutoff+2.2,cutoff+3])
                weights = np.append(weights,np.linspace(0,4,20))#[2,4,8,15])
    
                #set initial conditions
                l=l
                x = new_rad
                y = just_one
                min_cut = expcutoff/8
                if l == 2:
                    min_cut = expcutoff/4
    
                
                # use the pseudo parameters
                [pa,pb,pc,pd,pe,pf,pg,ph,l] = orbgroup_pseudoparams[group_ind]
                #pe = -abs(pa)/10
                #pf = pb/10
                #if abs(pe/pf) > abs(pa/pb):
                #    pe = -pa/pb*pf/10
                # constrain the gaussian to not have nodes UNLESS it is an excited state or a second s orbital (which is labeled in orbtype as s_ex or s2)
                orbkey = self.orbtype[orbgroup[0]]
                no_node = orbkey[0] == orbkey # only is one character: 's', 'p', or 'd'
                # if the s orbital has shifted such that is has node, allow node
                #if not no_node:
                #    # use the previously found start
                #    [pa,pb,pc,pd,pe,pf,pg,ph] = [a,b,c,d,e,f,g,h]
                # try fitting with contraints built into function so can use curve_fit)
                max_aceg = np.amax(np.abs([pa,pc,pe,pg]))
                pg = max_aceg/5
                init_con1 = pd/pb #np.log( - 1)
                init_con4 = 2/3 #np.log(- 1)
                init_con3 = pf/pb #np.log(pb/pf - 1)
                init_con2 = pe/pf+pa/pb
                gaus_init = [pa,pb,init_con1,init_con2,init_con3,pc,pg,init_con4]
                gaus_fit = partial(func_for_rad_fit, l=l)
                sig = 1/(weights+0.00001)
                #max_aceg = np.max(np.abs([pa,pc,pe,pg]))
                low_bounds = np.ones(8)*0.000001 #[0,0,0,0,0,0,0,0]
                high_bounds = [max_aceg*10,new_cutoff*3,0.95,max_aceg*10,0.9,max_aceg*10,max_aceg*10,0.9]
                if not no_node: # allow node in fit
                    low_bounds[3] = -max_aceg*10 # remove low bound for con2
                    high_bounds[4] = 5 # can be larger than b now, but generally is still not
                gaus_init = np.array(gaus_init) + 0.000001
                popt,pcov = curve_fit(gaus_fit,x,y,p0=gaus_init,sigma=sig,bounds=(low_bounds,high_bounds),ftol=0.000001, xtol=0.000001,method="trf",max_nfev=2000)
                # check how close output is to bounds
                close_low = np.abs((popt-low_bounds)/popt)
                close_high = np.abs((popt-high_bounds)/popt)
                if ((close_low<0.05).any() or (close_high<0.05).any()) and self.verbose > 0:
                    print("warning very close to bounds!",l,np.arange(len(close_low))[close_low<0.05],close_low[close_low<0.05],np.arange(len(close_high))[close_high<0.05],close_high[close_high<0.05])
                [t_a,t_b,con1,con2,con3,t_c,t_g,con4] = popt
                t_d = con1 * t_b
                t_f = con3 * t_b
                t_e = (con2 - t_a/t_b) * t_f
                t_h = con4 * t_b
                #print("check lmfit params:",r_a,r_b,r_c,r_d,r_e,r_f,r_g,r_h)
                #print("check scipy params:",t_a,t_b,t_c,t_d,t_e,t_f,t_g,t_h)
                r_a,r_b,r_c,r_d,r_e,r_f,r_g,r_h = [t_a,t_b,t_c,t_d,t_e,t_f,t_g,t_h]
                orbgroup_finalgausparams[group_ind] = [r_a,r_b,r_c,r_d,r_e,r_f,r_g,r_h,l]

                # check error of final guassian fit
                newgaus_one = func_for_rad(new_rad, r_a, r_b, r_c, r_d, r_e, r_f, r_g, r_h, l=l)
                newgaus_diff = just_one-newgaus_one
                origin_error = np.average(np.abs(newgaus_diff[new_rad<test_cutoff/3]))/np.average(np.abs(just_one[new_rad<test_cutoff/3]))
                # do allow pos + neg error cancelation in tail because often the gaus overshots while the exp undershots the tail area
                tail_error = np.abs(np.average(newgaus_diff[new_rad>test_cutoff*0.9]))/np.average(just_one[new_rad>test_cutoff*0.9])
                abstail_error = np.average(np.abs(newgaus_diff[new_rad>test_cutoff*0.9]))/np.average(np.abs(just_one[new_rad>test_cutoff*0.9]))
                total_error = np.average(np.abs(newgaus_diff))/np.average(np.abs(just_one))
                #print("all final fit error: (origin,tail,all)",origin_error,abstail_error,tail_error,total_error)
                final_gaus_fiterrors[orb_groups[group_ind],:] = np.array([origin_error,abstail_error,tail_error,total_error])[None,:]
                
                if self.verbose > 0:
                    print("final gaus terms:",l,r_a,r_b,r_c,r_d,r_e,r_f,r_g,r_h)

                
                # make values new values for the ratio of center orbital to others for next loop
                second_time = time.time()
                if loop != num_loop:
                    for orb in orbgroup:
                        orbperatom = self.sph_harm_key[orb]

                        # instead of creating Bloch orbital from many sums, should just make in recip space and fourier transform back
                        # switch to a*e^-bx^2 format form a/b*e^-1/2(x/b)^2 format
                        [a, c, e, g] = [r_a / r_b, r_c / r_d, r_e / r_f, r_g / r_h]
                        [b, d, f, h] = 1 / 2 / np.array([r_b, r_d, r_f, r_h]) ** 2

                        # convert to reciprocal space
                        # g(k) = ∫j_l(kr)*g(r)*r^2*dr was done in mathematica to get the symbol conversion for analytical calc
                        ak = np.pi ** (1 / 2) * 2 ** (-2 - l) * a * b ** (-3 / 2 - l) * (2 / np.pi) ** (1 / 2)
                        bk = 1 / 4 / b
                        ck = np.pi ** (1 / 2) * 2 ** (-2 - l) * c * d ** (-3 / 2 - l) * (2 / np.pi) ** (1 / 2)
                        dk = 1 / 4 / d
                        ek = np.pi ** (1 / 2) * 2 ** (-2 - l) * e * f ** (-3 / 2 - l) * (2 / np.pi) ** (1 / 2)
                        fk = 1 / 4 / f
                        gk = np.pi ** (1 / 2) * 2 ** (-2 - l) * g * h ** (-3 / 2 - l) * (2 / np.pi) ** (1 / 2)
                        hk = 1 / 4 / h

                        [bk, dk, fk, hk] = 1 / (2 * np.array([bk, dk, fk, hk])) ** (1 / 2)
                        [ak, ck, ek, gk] = [ak * bk, ck * dk, ek * fk, gk * hk]
                        # make recip orb
                        orb_params = np.array([[ak,bk,ck,dk,ek,fk,gk,hk,l,self.sph_harm_key[orb]]])
                        recip_orb = self.get_kdep_reciporbs(0, orb_params=orb_params,orb_vecs=[self.orb_vec[orb]])
                        tot_orbWF = self.recip_to_real(recip_orb)[0]
                        shape = tot_orbWF.shape
                        const = shape[0]*shape[1]*shape[2]/self.vol**(1/2)
                        tot_orbWF = tot_orbWF.flatten() * const

                        angular_part = complex128funs(min_phi, min_theta, self.orb_vec[orb])
                        #big_ang = np.abs(angular_part)>0.1
                        #new_rad_part = tot_orbWF[big_ang]/angular_part[big_ang]
                        just_one = np.zeros(len(min_rad))
                        just_one = func_for_rad(min_rad, r_a,r_b,r_c,r_d,r_e,r_f,r_g,r_h,l=l)
                        #just_one[min_rad<=cutoff] = func_for_rad(min_rad[min_rad<=cutoff],a,b,c,d,e,f,g,h,l=l)
                        #just_one[(min_rad>cutoff) & (min_rad<5*cutoff)] = func_for_rad_exp(min_rad[(min_rad>cutoff) & (min_rad<5*cutoff)],a_e,b_e,c_e,d_e,e_e,f_e,l=l)
                        tot_orbWF[tot_orbWF==0] = 0.000000000001
                        if loop != num_loop-1:
                            new_ratio = just_one*angular_part/tot_orbWF
                            #new_ratio[np.abs(tot_orbWF)<0.00005] = 0
                            #new_ratio[new_ratio<0.] = 0
                            new_ratio[np.abs(new_ratio)>15] = 0
                            #new_ratio[new_ratio < 0.1] = 0
                            all_ratio[orb] = new_ratio
                        tot_orbs[orb] = tot_orbWF
                        hold = tot_orbWF[all_good_ind[orb]]/angular_part[all_good_ind[orb]]
                        if not no_node and orbkey[0] == 's':
                            reconstruct_rad[orb] = np.append(hold,np.zeros(22))#[0,0,0,0,0,0])
                        else:
                            reconstruct_rad[orb] = np.append(hold,np.zeros(20))#[0,0,0,0])
                        #get the error
                        unk = centered_orbs[orb].flatten().real
                        diff_orb = np.abs((tot_orbWF - unk))
                        orb_nrms[orb] = np.sqrt(np.average(np.square(diff_orb)))/np.sqrt(np.average(np.square(np.abs(unk))))
                        orb_nme[orb] = np.average(np.abs(diff_orb))/np.average(np.abs(unk))
                        #print("NRMS:",orb,orb_nrms[orb])

                third_time = time.time()
               
            end_time = time.time()
            fit_rad = end_time - start_time
            if self.verbose > 2:
                print("timing:",loop,make_rad,fit_rad,third_time-second_time,second_time-first_time)
                

        # finally fit the total gaussian + exponential fit to a gaussian basis
        
        plt_subplots = []
        # and properly normalize the orbitals within the PAW basis
        for group_ind, orbgroup in enumerate(orb_groups):
            [a,b,c,d,e,f,g,h,l] = orbgroup_gausparams[group_ind]
            [a_e,b_e,c_e,d_e,e_e,f_e,l] = orbgroup_expparams[group_ind]
            [r_a,r_b,r_c,r_d,r_e,r_f,r_g,r_h,l] = orbgroup_finalgausparams[group_ind]
            atom = self.orbatomnum[orbgroup[0]]
            cutoff = self.cutoff_rad[atom]*0.9+np.tanh((self.init_orb_energy[orbgroup[0]]+8)/5)*0.5#self.cutoff_rad[atom]*0.8#**(1/2)#*1.4
            expcutoff = 1/cutoff
            # orbital dependant section
            # now fit to just gaussian functions
            new_rad = np.linspace(0,(cutoff/2)**(1/2)*5,100)
            
            # already fit the gaussian parameters so only do this for s2 states for orthing
            # labels are all for s orbitals but works for other types too
            orbkey = self.orbtype[orbgroup[0]]
            no_node = orbkey[0] == orbkey 
            if orbkey[0]+"2" == orbkey: #not no_node: # only do if this is a s2, p2, d2 state
                # first get the overlap between the s semicore and s2 orbital
                orbatm = self.orbatomnum[orb]
                sphkey = self.sph_harm_key[orb]
                sorb = np.arange(self.num_orbs)[(np.array(self.orbatomnum) == orbatm) & (np.array(self.orbtype) == orbkey[0]) & (np.array(self.sph_harm_key) == sphkey)][0]
                for gind in range(len(orb_groups)):
                    if (np.array(orb_groups[gind])==sorb).any():
                        sgroup_ind = gind
                [s_a,s_b,s_c,s_d,s_e,s_f,s_g,s_h,l] = orbgroup_finalgausparams[sgroup_ind]
                #converg_orbs[orb] = converg_orbs[orb] - aeoverlap[orb,sorb]/aeoverlap[sorb,sorb]*converg_orbs[sorb]
   
                s2_orbital = func_for_rad(new_rad,r_a,r_b,r_c,r_d,r_e,r_f,r_g,r_h,l=l)
                s_orbital = func_for_rad(new_rad,s_a,s_b,s_c,s_d,s_e,s_f,s_g,s_h,l=l)
                #just_one[new_rad>cutoff] = func_for_rad_exp(new_rad[new_rad>cutoff],a_e,b_e,c_e,d_e,e_e,f_e,l=l)
                # calculate the ae_overlap between the orbitals
                # only do projectors that have the same center angular momentum as the orbital
                num_projs = self.num_orbs + self.num_exorbs
                all_sphharm = np.append(self.sph_harm_key,self.ex_sph_harm_key)
                all_orbtype = np.append(self.orbtype, self.ex_orbtype)
                all_orbatm = np.append(self.orbatomnum, self.exorbatomnum)
                orb1 = orbgroup[0] # each orbital in the group should have the same norm
                sphharm1 = all_sphharm[orb1]
                atm = all_orbatm[orb1]
                nonzero = (all_sphharm == sphharm1) & (all_orbatm == atm) # only nonzero if the orbs are on the same atom and have the same l and ml quantum nums
                orbs = np.arange(num_projs,dtype=np.int_)[nonzero]

                real_proj_data = self.atmProjectordata[self.elements[atom]]
                proj_cutoff = real_proj_data[0]
                real_proj_grid = np.linspace(0, proj_cutoff, num=100,endpoint=False)
                #gaus_orb = func_for_rad(real_proj_grid,*old_params)
                s2o = func_for_rad(real_proj_grid,r_a,r_b,r_c,r_d,r_e,r_f,r_g,r_h,l=l)
                so = func_for_rad(real_proj_grid,s_a,s_b,s_c,s_d,s_e,s_f,s_g,s_h,l=l)
                #gaus_gaus_orbs = gaus_orb * gaus_orb
                s2s_aeoverlap = integrate.trapezoid((so * s2o) * real_proj_grid ** 2, x=real_proj_grid)
                ss_aeoverlap = integrate.trapezoid((so * so) * real_proj_grid ** 2, x=real_proj_grid)
                for orb1 in orbs:
                    type1 = all_orbtype[orb1]
                    gaus_proj_orbs1 = so * real_proj_data[1][type1]
                    gaus_proj_norm1 = integrate.trapezoid((gaus_proj_orbs1) * real_proj_grid ** 2, x=real_proj_grid)
                    for orb2 in orbs:
                        type2 = all_orbtype[orb2]
                        # for s-s2 overlap
                        gaus_proj_orbs2 = s2o * real_proj_data[1][type2]
                        gaus_proj_norm2 = integrate.trapezoid((gaus_proj_orbs2) * real_proj_grid ** 2, x=real_proj_grid)
                        s2s_aeoverlap += gaus_proj_norm1 * Qab[orb1,orb2] * gaus_proj_norm2
                        # for s-s overlap for norming
                        gaus_proj_orbs2 = so * real_proj_data[1][type2]
                        gaus_proj_norm2 = integrate.trapezoid((gaus_proj_orbs2) * real_proj_grid ** 2, x=real_proj_grid)
                        ss_aeoverlap += gaus_proj_norm1 * Qab[orb1,orb2] * gaus_proj_norm2

                psuedo_overlap = gaus_norm
                individ_norm =  ae_overlap
                just_one = s2_orbital - s2s_aeoverlap/ss_aeoverlap * s_orbital

                #use the original WF
                #just_one = self.orig_radial[orbind]
                weights = np.ones(len(new_rad))# 1/(new_rad**(1/2)+0.5)
                end_gaus = func_for_rad(cutoff,r_a,r_b,r_c,r_d,r_e,r_f,r_g,r_h,l)
                #weights[new_rad>cutoff] = weights[new_rad>cutoff] * (just_one[new_rad>cutoff]/end_gaus)**(1/4) * 2
                #just_one = np.append(just_one,[0,0,0,0])
                #new_rad = np.append(new_rad,[cutoff+3,cutoff+4,cutoff+6,cutoff+8])
                #weights = np.append(weights,[0.1,0.2,0.4,1.0])

                #set initial conditions
                l=l
                x = new_rad
                y = just_one
                min_cut = expcutoff/8
                if l == 2:
                    min_cut = expcutoff/4

                
                # use the pseudo parameters
                [pa,pb,pc,pd,pe,pf,pg,ph,l] = orbgroup_pseudoparams[group_ind]
                # constrain the gaussian to not have nodes UNLESS it is an excited state or a second s orbital (which is labeled in orbtype as s_ex or s2)
                orbkey = self.orbtype[orbgroup[0]]
                no_node = orbkey[0] == orbkey # only is one character: 's', 'p', or 'd'
                # if the s orbital has shifted such that is has node, allow node
                #if not no_node:
                # use the previously found start
                [pa,pb,pc,pd,pe,pf,pg,ph] = [r_a,r_b,r_c,r_d,r_e,r_f,r_g,r_h]#[a,b,c,d,e,f,g,h] 
                
                # try fitting with contraints built into function so can use curve_fit)
                max_aceg = np.amax(np.abs([pa,pc,pe,pg]))
                pg = max_aceg/5
                init_con1 = pd/pb #np.log( - 1)
                init_con4 = 2/3 #np.log(- 1)
                init_con3 = pf/pb #np.log(pb/pf - 1)
                init_con2 = pe/pf+pa/pb
                gaus_init = [pa,pb,init_con1,init_con2,init_con3,pc,pg,init_con4]
                gaus_fit = partial(func_for_rad_fit, l=l)
                sig = 1/(weights+0.00001)
                #max_aceg = np.max(np.abs([pa,pc,pe,pg]))

                # redefine new_cutoff
                cutoff = self.cutoff_rad[atom] * 0.9 + np.tanh((self.init_orb_energy[orbgroup[0]] + 8) / 5) * 0.5
                new_cutoff = cutoff ** (1 / 2)

                low_bounds = np.ones(8)*0.000001 #[0,0,0,0,0,0,0,0]
                high_bounds = [max_aceg*10,new_cutoff*3,0.95,max_aceg*10,1,max_aceg*10,max_aceg*10,1]
                if not no_node: # allow node in fit
                    low_bounds[3] = -max_aceg*10 # remove low bound for con2
                    high_bounds[4] = 5 # can be larger than b now, but generally is still not
                popt,pcov = curve_fit(gaus_fit,x,y,p0=gaus_init,sigma=sig,bounds=(low_bounds,high_bounds),ftol=0.000001, xtol=0.000001,method="trf",max_nfev=2000)
                # check how close output is to bounds
                close_low = np.abs((popt-low_bounds)/popt)
                close_high = np.abs((popt-high_bounds)/popt)
                if ((close_low<0.05).any() or (close_high<0.05).any()) and self.verbose > 0:
                    print("warning very close to bounds!",l,np.arange(len(close_low))[close_low<0.05],close_low[close_low<0.05],np.arange(len(close_high))[close_high<0.05],close_high[close_high<0.05])
                [t_a,t_b,con1,con2,con3,t_c,t_g,con4] = popt
                t_d = con1 * t_b
                t_f = con3 * t_b
                t_e = (con2 - t_a/t_b) * t_f
                t_h = con4 * t_b
                #print("check lmfit params:",r_a,r_b,r_c,r_d,r_e,r_f,r_g,r_h)
                #print("check scipy params:",t_a,t_b,t_c,t_d,t_e,t_f,t_g,t_h)
                r_a,r_b,r_c,r_d,r_e,r_f,r_g,r_h = [t_a,t_b,t_c,t_d,t_e,t_f,t_g,t_h]
                if self.verbose > 0:
                    print("fitted params:",r_a,r_b,r_c,r_d,r_e,r_f,r_g,r_h,l)
                
            # get values for plotting
            good_min_rad = all_good_min_rad[orbgroup[0]]
            radunk = all_radunk[orbgroup[0]]
            weights = all_weights[orbgroup[0]]
            og_radunk = all_og_orbs[orbgroup[0]]
            fullrecon_rad = reconstruct_rad[orbgroup[0]]
            for orbind,orb in enumerate(orbgroup[1:]):
                radunk = np.append(radunk,all_radunk[orb])
                good_min_rad = np.append(good_min_rad,all_good_min_rad[orb])
                weights = np.append(weights,all_weights[orb])
                og_radunk = np.append(og_radunk,all_og_orbs[orb])
                fullrecon_rad = np.append(fullrecon_rad,reconstruct_rad[orb])

            if self.save_orb_figs or plot_orbs:
                #plt.scatter(min_rad[big_ang],og_radunk[big_ang],c="black",s=2)
                #plt.scatter(min_rad[big_ang],new_rad_part,c="red",s=1)
                fig, ax = plt.subplots()
                ax.plot(good_min_rad,radunk,'o',color="blue",markersize=1/4,label="From approximate atomic")
                ax.plot(good_min_rad,og_radunk,'o',color="green",markersize=1/2,label="From Bloch orbital")
                ax.plot(good_min_rad,fullrecon_rad,'o',color="red",markersize=1/2,label="reconstructed")
                new_rad = np.linspace(0,5,100)
                #plot the original starting radial part
                [nx,ny,nz] = self.gridxyz
                ax.plot(new_rad,self.orig_radial[orbgroup[0]],color="black",linewidth=2.5,label="Pseudo orbital")
                #plt.scatter(min_rad[big_ang],test_new[big_ang],facecolors="None",edgecolors="red")
                just_one = func_for_rad(new_rad,a,b,c,d,e,f,g,h,l=l)
                just_one[new_rad>cutoff] = func_for_rad_exp(new_rad[new_rad>cutoff],a_e,b_e,c_e,d_e,e_e,f_e,l=l)
                #plt.ylim((0,0.0005))
                ax.set_xlim((0,5))
                ax.plot(new_rad,just_one,color="purple",linewidth=2.5,label="Gauss+exp fit")#c=c,d=d,*popt

                [pa,pb,pc,pd,pe,pf,pg,ph,l] = orbgroup_pseudoparams[group_ind]
                new_rad = np.linspace(0,8,100)
                ax.plot(new_rad,func_for_rad(new_rad,pa,pb,pc,pd,pe,pf,pg,ph,l=l),color="red",linewidth=1,label="Pseudo orbital")


                # plot the just guassian fit
                ax.plot(new_rad,func_for_rad(new_rad,r_a,r_b,r_c,r_d,r_e,r_f,r_g,r_h,l=l),color='#db61ed',linewidth = 2.5,label="Final gaus fit")
                plt.ylim((-0.002,0.002))
                ax.set_xlim((0,4))
                #print("showing plot")

                ax.set_xlabel("Radius (Å)")
                ax.set_ylabel("Radial part of orbital")
                ax.legend()
                plt_subplots.append(fig)
                if plot_orbs:
                    plt.show()
                plt.close()

            [a,b,c,d,e,f,g,h] = [r_a,r_b,r_c,r_d,r_e,r_f,r_g,r_h]
            if self.use_pseudo:
                [pa,pb,pc,pd,pe,pf,pg,ph,l] = orbgroup_pseudoparams[group_ind]
                [a,b,c,d,e,f,g,h] = [pa,pb,pc,pd,pe,pf,pg,ph]
                [b,d,f,h] = np.array([b,d,f,h])*orbfactor

            # now normalize for the PAW treatment
            # find orbital projector overlap (overlap_OrbProj) and ae-psuedo diff (Qab)
            if self.haveQab == False:
                self.get_Qab()
                self.haveQab = True
            Qab = self.Qab

            # change the coeffs so they normalize correctly
            const = self.gridxyz[0]*self.gridxyz[1]*self.gridxyz[2]/self.vol**(1/2)
            if self.verbose > 1:
                print("volume:", self.vol)
                print("const",const)
            [a,c,e,g] = np.array([a,c,e,g])*const #analytical calc
            
            if self.use_pseudo or True:
                orig_radial = self.cur_radial[orbgroup[0]]
                orig_radial = orig_radial*const

            # switch to a*e^-bx^2 format form a/b*e^-1/2(x/b)^2 format
            old_params = copy.deepcopy([a,b,c,d,e,f,g,h,l])
            #print("check same:",[a,b,c,d,e,f,g,h,l])
            [a,c,e,g] = [a/b,c/d,e/f,g/h]
            [b,d,f,h] = 1/2/np.array([b,d,f,h])**2
            #print("as:", old_params,[a,b,c,d])

            # normalize the orbital just with the onsite overlap
            l= int(l)
            orbind = orbgroup[0]
            real_proj_data = self.atmProjectordata[self.elements[atom]]
            proj_cutoff = real_proj_data[0]
            real_proj_grid = np.linspace(0, proj_cutoff, num=100,endpoint=False)
            gaus_orb = func_for_rad(real_proj_grid,*old_params)
            
            # analytical integral of Gaussian funcs
            #true_gauss_norm = [np.pi**(1/2)/4,1/2,3*np.pi**(1/2)/8][l] * (a*d**((3+l)/2)+c*b**((3+l)/2))/d**((3+l)/2)/b**((3+l)/2)
            #gaus_gaus_int = [1,3,15][l]*np.pi**(1/2)*(a*c/(2**(1+l)*(d+b)**(3/2+l))+(a**2*d**(3/2+l)+c**2*b**(3/2+l))/(2**(7/2+2*l)*b**(3/2+l)*d**(3/2+l))) #analytical calc
            fnt_coef = np.array([a,c,e,g])
            exp_coef =np.array([b,d,f,h])
            #gaus_norm = factorial2(1+2*l)*np.pi**(1/2)*(2**(-7/2-2*l)*np.sum(fnt_coef**2*exp_coef**(-3/2-l))+2**(-1-l)*np.prod(fnt_coef)*np.sum(exp_coef)**(-3/2-l))
            gaus_norm = factorial2(1+2*l)*np.pi**(1/2)*2**(-2-l)*np.sum(fnt_coef[:,None]*fnt_coef[None,:]*(exp_coef[:,None]+exp_coef[None,:])**(-3/2-l))
            #gaus_gaus_int = gaus_gaus_int *(self.gridxyz[0]*self.gridxyz[1]*self.gridxyz[2])**(2)/self.vol
            #true_gauss_norm = true_gauss_norm * self.gridxyz[0]* self.gridxyz[1]* self.gridxyz[2]

            # only do projectors that have the same center angular momentum as the orbital
            num_projs = self.num_orbs + self.num_exorbs
            all_sphharm = np.append(self.sph_harm_key,self.ex_sph_harm_key)
            all_orbtype = np.append(self.orbtype, self.ex_orbtype)
            all_orbatm = np.append(self.orbatomnum, self.exorbatomnum)
            orb1 = orbgroup[0] # each orbital in the group should have the same norm
            sphharm1 = all_sphharm[orb1]
            atm = all_orbatm[orb1]
            nonzero = (all_sphharm == sphharm1) & (all_orbatm == atm) # only nonzero if the orbs are on the same atom and have the same l and ml quantum nums
            orbs = np.arange(num_projs,dtype=np.int_)[nonzero]
            
            real_proj_data = self.atmProjectordata[self.elements[atom]]
            proj_cutoff = real_proj_data[0]
            real_proj_grid = np.linspace(0, proj_cutoff, num=100,endpoint=False)
            
            if self.use_pseudo or True: # this is normalizing the initial pseudo orbitals correctly
                # first get norm and radius
                orb_rad = np.linspace(0,8,num=100,endpoint=False)
                simple_rad = np.linspace(0,5,num=100,endpoint=False)
                pseudo_orb = interpolate.griddata(simple_rad,orig_radial,orb_rad, method='cubic', fill_value=0) # for calculating pseudo norm and rad
                
                pseudo_norm = integrate.trapezoid((pseudo_orb*pseudo_orb) * orb_rad ** 2, x=orb_rad)
                pseudo_rad = integrate.trapezoid((pseudo_orb*pseudo_orb)/pseudo_norm * orb_rad ** 3, x=orb_rad)
                
                pseudo_orb = interpolate.griddata(simple_rad,orig_radial,real_proj_grid, method='cubic', fill_value=0) # for projecting on projectors
                ae_overlap = pseudo_norm
                for orb1 in orbs:
                    type1 = all_orbtype[orb1]
                    gaus_proj_orbs1 = pseudo_orb * real_proj_data[1][type1]
                    gaus_proj_norm1 = integrate.trapezoid((gaus_proj_orbs1) * real_proj_grid ** 2, x=real_proj_grid)
                    for orb2 in orbs:
                        type2 = all_orbtype[orb2]
                        gaus_proj_orbs2 = pseudo_orb * real_proj_data[1][type2]
                        gaus_proj_norm2 = integrate.trapezoid((gaus_proj_orbs2) * real_proj_grid ** 2, x=real_proj_grid)
                        ae_overlap += gaus_proj_norm1 * Qab[orb1,orb2] * gaus_proj_norm2
    
                orig_radial = orig_radial / ae_overlap ** (1 / 2)

            # back to the fitted gaussian orbital
            gaus_orb = func_for_rad(real_proj_grid,*old_params)
            
            #gaus_gaus_orbs = gaus_orb * gaus_orb
            ae_overlap = gaus_norm
            for orb1 in orbs:
                type1 = all_orbtype[orb1]
                gaus_proj_orbs1 = gaus_orb * real_proj_data[1][type1]
                gaus_proj_norm1 = integrate.trapezoid((gaus_proj_orbs1) * real_proj_grid ** 2, x=real_proj_grid)
                for orb2 in orbs:
                    type2 = all_orbtype[orb2]
                    gaus_proj_orbs2 = gaus_orb * real_proj_data[1][type2]
                    gaus_proj_norm2 = integrate.trapezoid((gaus_proj_orbs2) * real_proj_grid ** 2, x=real_proj_grid)
                    ae_overlap += gaus_proj_norm1 * Qab[orb1,orb2] * gaus_proj_norm2

            psuedo_overlap = gaus_norm
            individ_norm =  ae_overlap

            if self.verbose > 0:
                print("pseudo norm and ae adjustments:",orbind, psuedo_overlap,individ_norm-psuedo_overlap)# sum_overi)
            [a,c,e,g] = [a,c,e,g] / individ_norm ** (1 / 2)
            
            #if self.use_pseudo or True:
            #    orig_radial = orig_radial / individ_norm ** (1 / 2)

            #calculate the average (Bohr) radius # the orbital should be normed to pseudo since you want the pseudo-orbital radius
            fnt_coef = np.array([a, c, e, g]) * individ_norm ** (1 / 2) / gaus_norm ** (1 / 2)
            exp_coef = np.array([b, d, f, h])
            avg_rad = 1/2*math.factorial(1+l)*np.sum(fnt_coef[:,None]*fnt_coef[None,:]*(exp_coef[:,None]+exp_coef[None,:])**(-2-l)) #analytical calc
            old_rad = 1/8*math.factorial(1+l)*(2**(-l)*a**2*b**(-2-l)+2**(-l)*c**2*d**(-2-l)+8*a*c*(b+d)**(-2-l))
            if self.verbose > 0:
                print("average radius!",avg_rad,old_rad)
            
            fnt_coef = np.array([a, c, e, g])
            newgaus_gaus_int = factorial2(1+2*l)*np.pi**(1/2)*2**(-2-l)*np.sum(fnt_coef[:,None]*fnt_coef[None,:]*(exp_coef[:,None]+exp_coef[None,:])**(-3/2-l))

            # convert to reciprocal space
            # g(k) = ∫j_l(kr)*g(r)*r^2*dr was done in mathematica to get the symbol conversion for analytical calc
            ak = np.pi**(1/2)*2**(-2-l)*a*b**(-3/2-l) #*(self.num_kpts/self.vol)**(1/4)#* self.vol**(1/4)/np.pi**(3/4)
            bk = 1/4/b
            ck = np.pi**(1/2)*2**(-2-l)*c*d**(-3/2-l) # *(self.num_kpts/self.vol)**(1/4)#* self.vol**(1/4)/np.pi**(3/4)
            dk = 1/4/d
            ek = np.pi**(1/2)*2**(-2-l)*e*f**(-3/2-l)
            fk = 1/4/f
            gk = np.pi**(1/2)*2**(-2-l)*g*h**(-3/2-l)
            hk = 1/4/h
            
            #newgaus_gaus_int = [1,3,15][l]*np.pi**(1/2)*(a*c/(2**(1+l)*(d+b)**(3/2+l))+(a**2*d**(3/2+l)+c**2*b**(3/2+l))/(2**(7/2+2*l)*b**(3/2+l)*d**(3/2+l)))
            fnt_coefk = np.array([ak, ck, ek, gk])
            exp_coefk = np.array([bk, dk, fk, hk])
            recipgaus_gaus_int = factorial2(1+2*l)*np.pi**(1/2)*2**(-2-l)*np.sum(fnt_coefk[:,None]*fnt_coefk[None,:]*(exp_coefk[:,None]+exp_coefk[None,:])**(-3/2-l))

            # convert back to a/b*e^-1/2(x/b)^2 format from a*e^-bx^2 format
            [b,d,f,h] =  1/(2*np.array([b,d,f,h]))**(1/2) # 1/2/np.array([b,d,f,h])**2
            [a,c,e,g] = [a*b,c*d,e*f,g*h]
            
            [bk,dk,fk,hk] =  1/(2*np.array([bk,dk,fk,hk]))**(1/2) # 1/2/np.array([b,d,f,h])**2
            [ak,ck,ek,gk] = [ak*bk,ck*dk,ek*fk,gk*hk]

            
            # finally get the NME and NRMSE between the new atomic radial part and the old one
            simple_rad = np.linspace(0,5,num=100,endpoint=False)
            new_radial = func_for_rad(simple_rad,a,b,c,d,e,f,g,h,l=l)
            # align the maximums
            radial_diff = np.abs(orig_radial*np.amax(new_radial)/np.amax(orig_radial) - new_radial) # scale orig_radial to align the new maximum (tends give smallest errors)
            orbrad_nrmse = np.sqrt(np.average(np.square(radial_diff)))/np.sqrt(np.average(np.square(np.abs(new_radial))))
            orbrad_nme = np.average(np.abs(radial_diff))/np.average(np.abs(new_radial))
            
            
            #save to orbital coeffs
            for orb in orbgroup:
                avg_radius[orb] = avg_rad
                orbital_coeffs[orb] = np.array([a,b,c,d,e,f,g,h,l])
                recip_orbcoeffs[orb] = np.array([ak,bk,ck,dk,ek,fk,gk,hk,l])
                radial_nrmse[orb] = orbrad_nrmse
                radial_nme[orb] = orbrad_nme
                
                if self.use_pseudo: # if using pseudo for other projection save back original pseudo
                    self.cur_radial[orb] = orig_radial
                else: # save the new radial part
                    self.cur_radial[orb] = new_radial
                

            
            #recipgaus_gaus_int = [1,3,15][l]*np.pi**(1/2)*(ak*ck/(2**(1+l)*(dk+bk)**(3/2+l))+(ak**2*dk**(3/2+l)+ck**2*bk**(3/2+l))/(2**(7/2+2*l)*bk**(3/2+l)*dk**(3/2+l)))
            if self.verbose > 0:
                print("new norms:",newgaus_gaus_int,recipgaus_gaus_int)
            #end orb dependant section
        
        # save figures
        if self.save_orb_figs:
            combine_and_save_plots(plt_subplots, filename=self.directory+"fit_cogito"+self.tag+".png")
        # get percentage changes in radius
        chang_from_pseudo =  (avg_radius - self.pseudo_radius)/self.pseudo_radius
        chang_from_last =  (avg_radius - self.orb_radius)/self.orb_radius

        print("orbital radius", np.around(avg_radius,decimals=4))
        print("orbital percent change pseudo:",np.around(chang_from_pseudo*100,decimals=2))
        if self.verbose > 0:
            print("orbital change last:",chang_from_last)
            print("bloch nme:", orb_nme)
        elif self.verbose > 1:
            print("bloch nrmse:", orb_nrms)
            print("radial nrmse:",radial_nrmse)
            print("radial nme:",radial_nme)
            print("final gaus errors:",final_gaus_fiterrors)
        self.orb_nrms = orb_nrms
        self.orb_nme = orb_nme
        self.radial_nrmse = radial_nrmse
        self.radial_nme = radial_nme
        self.orb_radius = avg_radius
        self.recip_orbcoeffs = recip_orbcoeffs
        self.real_orbcoeffs = orbital_coeffs
        all_loop_info = [avg_radius,chang_from_pseudo,chang_from_last,radial_nme,orb_nrms,orb_nme,final_gaus_fiterrors[:,3]] # orb_rad, orb_change_pseudo, orb_change_last, radial_change (nme), bloch_nrmse, bloch_nme,  final_gaus_fit_error
        self.all_orbloop_info = all_loop_info
        #print(self.sph_harm_key)
        #print(self.recip_orbcoeffs)
        #print(self.orb_radius)
        # self.spher_bessel_trans(orbital_coeffs) # this is only useful to printing the recip radial orbital to text file

        if self.use_pseudo:
            self.recip_pseudo = self.spher_bessel_trans(orbital_coeffs)
            self.recip_pseudo_grid = self.recip_rad_grid
        
        # remake the original self.one_orbitalWF
        recip_orbitalWF = self.get_kdep_reciporbs(0) # for kpt=[0,0,0]
        #self.one_orbitalWF = self.recip_to_real(recip_orbitalWF)
        self.one_orbWFrecip = recip_orbitalWF

    def spher_bessel_trans(self, orbital_coeffs, rad_grid_size = 500):
        # need to pass the orbitals as coeffs for func_for_rad
        # all of this needs to be done either angstorms or inverse angstorms (can't use prim)
        #keys = list(orbitals.keys())
        self.rad_grid_size = rad_grid_size
        self.rad_grid_max = 12
        [nx,ny,nz] = self.gridxyz
        rad_grid = np.linspace(0,10,num=rad_grid_size, endpoint=False)
        kpt_radgrid = np.linspace(0,self.rad_grid_max,num=rad_grid_size,endpoint=False)
        fourier_rad_orbs = np.zeros((self.num_orbs,rad_grid_size),dtype=np.complex128)
        orb_to_l = [0,1,1,1,2,2,2,2,2,3,3,3,3,3,3,3]
        for orb in range(self.num_orbs):
            orb_recip = np.zeros(rad_grid_size,dtype=np.complex128)
            orb_coeff = orbital_coeffs[orb]
            atomnum = self.orbatomnum[orb]
            orb_peratom = self.sph_harm_key[orb]
            l = orb_to_l[orb_peratom]
            if self.use_pseudo:
                pseudo_vals = self.cur_radial[orb]
                simple_rad = np.linspace(0,5,num=100,endpoint=False)
                orb_real = interpolate.griddata(simple_rad,pseudo_vals,rad_grid, method='cubic', fill_value=0)
            else:
                orb_real = func_for_rad(rad_grid,*orb_coeff)
            r2 = rad_grid**2
            for kind,kpt in enumerate(kpt_radgrid):
                kr = kpt*rad_grid
                bessel = spherical_jn(l,kr)
                integral = np.trapezoid(orb_real*bessel*r2,rad_grid)#*np.pi**(1/4)# *nx*ny*nz
                orb_recip[kind] = integral
            fourier_rad_orbs[orb] = orb_recip
            #plt.plot(kpt_radgrid,orb_recip)
            #plt.show()
        self.recip_rad_orbs = fourier_rad_orbs
        self.recip_rad_grid = kpt_radgrid
        return fourier_rad_orbs
    
    def center_real_orbs(self,real_orbs,kpt=0,make_real=False):
        #keys = list(real_orbs.keys())
        gpoints = copy.deepcopy(np.array(self.gpoints[kpt],dtype=np.int_))
        num_gpoints = len(self.recip_WFcoeffs[kpt][0])
        recip_orbs = {}
        centered_recip = {}
        for orb in range(len(real_orbs)):#orb_ind, enumerate(keys):
            recip_orbs[orb] = np.zeros((num_gpoints), dtype=np.complex128)
            mesh = np.fft.fftn(real_orbs[orb])
            mesh = np.fft.fftshift(mesh)
            for gind, gp in enumerate(gpoints):
                t = tuple(gp.astype(np.int_) + (self.gridxyz / 2).astype(np.int_))
                recip_orbs[orb][gind] = mesh[t]
            center = self.orbpos[orb]
            exp_term = np.exp(2j * np.pi * np.dot(gpoints, center))
            centered_recip[orb] = recip_orbs[orb] * exp_term
        
        centered_real = np.zeros((len(real_orbs),self.gridxyz[0],self.gridxyz[1],self.gridxyz[2]), dtype=np.complex128)#{}
        for orb in range(len(real_orbs)): #keys:
            coeffs = centered_recip[orb]
            mesh = np.zeros(tuple(self.gridxyz), dtype=np.complex128)
            for gp, coeff in zip(np.array(self.gpoints[kpt],dtype=np.int_), coeffs):
                t = tuple(gp.astype(np.int_) + (self.gridxyz / 2).astype(np.int_))
                mesh[t] = coeff
            mesh = np.fft.ifftshift(mesh)
            centered_real[orb] = np.fft.ifftn(mesh)
            if make_real == True:
                centered_real[orb] = centered_real[orb].real
        return centered_real

    def recip_to_real(self,recip_orbs,kpt=0,make_real = False):
        #keys = list(recip_orbs.keys())
        recip_coeff = recip_orbs# np.zeros((len(keys),len(recip_orbs[keys[0]])), dtype=np.complex128)
        all_mesh = np.zeros((len(recip_coeff),self.gridxyz[0],self.gridxyz[1],self.gridxyz[2]), dtype=np.complex128)
        #for oi,orb in enumerate(keys):
        #    recip_coeff[oi] = recip_orbs[orb]

        t0 = self.gpoints[kpt][:,0].astype(np.int_) + (self.gridxyz[0] / 2).astype(np.int_)
        t1 = self.gpoints[kpt][:,1].astype(np.int_) + (self.gridxyz[1] / 2).astype(np.int_)
        t2 = self.gpoints[kpt][:,2].astype(np.int_) + (self.gridxyz[2] / 2).astype(np.int_)
        
        all_mesh[:,t0,t1,t2] = recip_coeff[:,:]
        all_mesh = np.fft.ifftshift(all_mesh,axes=(1,2,3))
        allreal_orbs = np.fft.ifftn(all_mesh,axes=(1,2,3))
        
        real_orbs = allreal_orbs #{}
        #for oi,orb in enumerate(keys):
        #    real_orbs[orb] = allreal_orbs[oi]
        
        #for orb in keys:
        #    coeffs = recip_orbs[orb]
        #    mesh = np.zeros(tuple(self.gridxyz), dtype=np.complex128)
        #    mesh[t0,t1,t2] = coeffs
        #    #for gp, coeff in zip(self.gpoints[kpt], coeffs):
        #    #    print(gp.astype(np.int_) + (self.gridxyz / 2).astype(np.int_))
        #    #    t = tuple(gp.astype(np.int_) + (self.gridxyz / 2).astype(np.int_))
        #    #    print("t",t)
        #    #    mesh[t] = coeff
        #    mesh = np.fft.ifftshift(mesh)
        #    real_orbs[orb] = np.fft.ifftn(mesh)
        #    if make_real == True:
        #        real_orbs[orb] = real_orbs[orb].real
        return real_orbs

    def real_to_recip(self,real_orbs,kpt=0):
        #keys = list(real_orbs.keys())
        #gpoints = copy.deepcopy(self.gpoints[kpt])
        num_gpoints = len(self.recip_WFcoeffs[kpt][0])
        #shape = real_orbs[keys[0]].shape
        real_orb_vec = real_orbs# np.zeros((len(keys),shape[0],shape[1],shape[2]), dtype=np.complex128)
        #for oi,orb in enumerate(keys):
        #    real_orb_vec[oi] = real_orbs[orb]
        
        t0 = self.gpoints[kpt][:,0].astype(np.int_) + (self.gridxyz[0] / 2).astype(np.int_)
        t1 = self.gpoints[kpt][:,1].astype(np.int_) + (self.gridxyz[1] / 2).astype(np.int_)
        t2 = self.gpoints[kpt][:,2].astype(np.int_) + (self.gridxyz[2] / 2).astype(np.int_)

        mesh = np.fft.fftn(real_orb_vec,axes=(1,2,3))
        mesh = np.fft.fftshift(mesh,axes=(1,2,3))
        recip_orb_vec = mesh[:,t0,t1,t2] 
        
        recip_orbs = recip_orb_vec # {}
        #for oi,orb in enumerate(keys):
        #    recip_orbs[orb] = recip_orb_vec[oi]
            
        #recip_orbs = {}
        #for orb in keys:
        #    recip_orbs[orb] = np.zeros((num_gpoints), dtype=np.complex128)
        #    mesh = np.fft.fftn(real_orbs[orb])
        #    mesh = np.fft.fftshift(mesh)
        #    for gind, gp in enumerate(self.gpoints[kpt]):
        #        t = tuple(gp.astype(np.int_) + (self.gridxyz / 2).astype(np.int_))
        #        recip_orbs[orb][gind] = mesh[t]
        return recip_orbs

    def get_kdep_recipprojs(self,kpt,full_kpt = False,gpnts=None,set_gpnts=False):

        if set_gpnts==False:
            if full_kpt == False:
                all_gpnts = np.array(self.gpoints[kpt],dtype=np.int_)
                kpoint = self.kpoints[kpt]
            else:
                kpoint = kpt
                all_gpnts = np.array(self.generate_gpnts(kpoint))

        else:
            if full_kpt == True:
                print("used provided gpoints")
                all_gpnts = np.array(gpnts)
                kpoint = kpt
                #gplusk_coords = all_gpnts.transpose() + np.array([kpt]).transpose()

        gplusk_coords = all_gpnts.transpose() + np.array([kpoint]).transpose()

        cart_gplusk = _red_to_cart((self.b_vecs[0], self.b_vecs[1], self.b_vecs[2]), gplusk_coords.transpose())
        vec = cart_gplusk.transpose()  # point - np.array([atm]).transpose()
        recip_rad = np.linalg.norm(vec, axis=0)
        recip_rad[recip_rad == 0] = 0.0000000001
        recip_theta = np.arccos(vec[2] / recip_rad)
        recip_theta[recip_rad == 0.0000000001] = 0
        recip_phi = np.arctan2(vec[1], vec[0])
        orb_to_l = [0, 1, 1, 1, 2, 2, 2, 2, 2,3,3,3,3,3,3,3]


        recip_orbs = np.zeros((self.num_orbs+self.num_exorbs,len(recip_rad)),dtype=np.complex128)#{}
        # get the projectors which have the same quantum numbers as the orbitals basis
        for orb in range(self.num_orbs):
            atomnum = self.orbatomnum[orb]
            #real stuff
            real_proj_data = self.atmProjectordata[self.elements[atomnum]]
            proj_cutoff = real_proj_data[0]
            real_proj_grid = np.linspace(0, proj_cutoff, num=100,endpoint=False)

            #recip stuff
            recip_proj_dat = self.atmRecipProjdata[self.elements[atomnum]]
            proj_gmax = recip_proj_dat[0]
            recip_proj_grid = np.arange(100) * proj_gmax / 100
            recip_rad_cut = recip_rad[recip_rad < proj_gmax]

            # get projector info
            orb_peratom = self.sph_harm_key[orb]
            l = orb_to_l[orb_peratom]
            radial_part = np.zeros((len(recip_rad)), dtype=np.complex128)
            #if l == 2: # cubic spline not
            radial_part[recip_rad < proj_gmax] = interpolate.griddata(recip_proj_grid, recip_proj_dat[1][self.orbtype[orb]], recip_rad_cut, method='cubic', fill_value=recip_proj_dat[1][self.orbtype[orb]][0])
            # define them just in normal xyz directions
            sphkey = self.sph_harm_key[orb]
            switch_to_vecs = [[1], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0],
                              [0, 0, 0, 1, 0], [0, 0, 0, 0, 1],
                              [1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]]
            orb_vec = switch_to_vecs[sphkey]
            angular_part = complex128funs(recip_phi, recip_theta, orb_vec)
            center = self.primAtoms[atomnum]
            exp_term = np.exp(-2j * np.pi * np.dot(all_gpnts, center))  # center the projector correctly
            flat_recip_orb = (-1j) ** (l) * radial_part * angular_part * exp_term

            recip_orbs[orb] = flat_recip_orb / self.vol**(1/2) # guess that the vasp recip projectors already include the 4pi

        # get the projectors which have different same quantum numbers as the orbitals basis, generally excited states
        for orb in range(self.num_exorbs):
            # now get the excited projs
            atomnum = self.exorbatomnum[orb]
            #real stuff
            real_proj_data = self.atmProjectordata[self.elements[atomnum]]
            proj_cutoff = real_proj_data[0]
            real_proj_grid = np.linspace(0, proj_cutoff, num=100,endpoint=False)

            #recip stuff
            recip_proj_dat = self.atmRecipProjdata[self.elements[atomnum]]
            proj_gmax = recip_proj_dat[0]
            recip_proj_grid = np.arange(100) * proj_gmax / 100
            recip_rad_cut = recip_rad[recip_rad < proj_gmax]

            # get projector info
            orb_peratom = self.ex_sph_harm_key[orb]
            l = orb_to_l[orb_peratom]
            radial_part = np.zeros((len(recip_rad)), dtype=np.complex128)
            radial_part[recip_rad < proj_gmax] = interpolate.griddata(recip_proj_grid, recip_proj_dat[1][self.ex_orbtype[orb]], recip_rad_cut, method='cubic', fill_value=recip_proj_dat[1][self.ex_orbtype[orb]][0])
            angular_part = complex128funs(recip_phi, recip_theta, self.ex_orb_vec[orb])
            center = self.primAtoms[atomnum]
            exp_term = np.exp(-2j * np.pi * np.dot(all_gpnts, center))  # center the projector correctly
            ex_flat_orb = (-1j) ** (l) * radial_part * angular_part * exp_term

            recip_orbs[orb+self.num_orbs] = ex_flat_orb / self.vol**(1/2) # guess that the vasp recip projectors already include the 4pi
        return recip_orbs

    def get_kdep_reciporbs(self,kpt,full_kpt =False,gpnts=None,set_gpnts=False,orb_params=[],orb_vecs=[]):
        if set_gpnts==False:
            if full_kpt == False:
                all_gpnts = np.array(self.gpoints[kpt],dtype=np.int_)
                kpoint = self.kpoints[kpt]
            else:
                kpoint = kpt
                all_gpnts = np.array(self.generate_gpnts(kpoint))

        else:
            if full_kpt == True:
                print("used provided gpoints")
                all_gpnts = np.array(gpnts)
                kpoint = kpt
                #gplusk_coords = all_gpnts.transpose() + np.array([kpt]).transpose()

        gplusk_coords = all_gpnts.transpose() + np.array([kpoint]).transpose()
        
        cart_gplusk = _red_to_cart((self.b_vecs[0], self.b_vecs[1], self.b_vecs[2]), gplusk_coords.transpose())
        vec = cart_gplusk.transpose()  # point - np.array([atm]).transpose()
        recip_rad = np.linalg.norm(vec, axis=0)
        recip_rad[recip_rad == 0] = 0.0000000001
        recip_theta = np.arccos(vec[2] / recip_rad)
        recip_theta[recip_rad == 0.0000000001] = 0
        recip_phi = np.arctan2(vec[1], vec[0])

        # try out the recip orbital coeffs
        if len(orb_params) == 0:
            numo = self.num_orbs
            all_coeffs = self.recip_orbcoeffs
            sph_vec = self.orb_vec
            centers = self.primAtoms[self.orbatomnum]
            use_pseudo = self.use_pseudo
        else:
            numo = len(orb_params)
            all_coeffs = np.array(orb_params)[:,:9]
            sph_vec = orb_vecs #np.array(orb_params)[:,9]
            centers = np.zeros((numo,3)) # always center them on zero (that is what fit_to_atomic does)
            use_pseudo = False

        recip_orbs = np.zeros((numo,len(recip_rad)),dtype=np.complex128) # {}
        for orb in range(numo):
            o_vec = sph_vec[orb] #self.sph_harm_key[orb]
            angular_part = complex128funs(recip_phi, recip_theta, o_vec)
            # atomnum = self.orbatomnum[orb]
            center = centers[orb] #self.primAtoms[atomnum]  # center the projector correctly
            exp_term = np.exp(-2j * np.pi * np.dot(all_gpnts, center))
            
            recip_coeff = all_coeffs[orb] #self.recip_orbcoeffs[orb]
            [ak,bk,ck,dk,ek,fk,gk,hk,l] = recip_coeff
            if use_pseudo:
                # do interpolation of pseudo orbitals
                pseudo_vals = self.recip_pseudo[orb] # self.orig_radial[orb]
                simple_rad = self.recip_pseudo_grid # np.linspace(0,5,num=100,endpoint=False)
                radial_part = interpolate.griddata(simple_rad,pseudo_vals,recip_rad, method='cubic', fill_value=0)
                
            else: # use fitted guassian functions
                radial_part = func_for_rad(recip_rad,ak,bk,ck,dk,ek,fk,gk,hk,l)
            
            flat_recip_orb = (-1j) ** (l) * radial_part * angular_part * exp_term

            # structure constant
            const = 4*np.pi/(self.vol)**(1/2)
            recip_orbs[orb] = flat_recip_orb*const

        return recip_orbs

    def proj_all_kpoints(self,max_bandavg = None,calc_nrms=False):
        num_kpts = self.num_kpts
        allspillage = np.zeros((num_kpts,self.max_band))
        charge_spill = np.zeros((num_kpts))
        max_charge_spill = np.zeros((num_kpts))
        max_orbmixing = np.zeros((num_kpts))
        max_orbmixing_allbands = np.zeros((num_kpts))
        all_nrms = []
        if max_bandavg == None:
            max_bandavg = self.max_band
        self.max_band = max_bandavg
        #print("maxband",max_bandavg)
        self.mnkcoefficients = np.zeros((num_kpts,max_bandavg,self.num_orbs), dtype=np.complex128) #kpt, band,orb
        #print("coeff shape:",self.mnkcoefficients.shape)

        #original test
        #orb_overlap = self.get_overlap_matrix(self.recip_orbs,recip=True)
        #zero_coeff = self.get_aecoefficients(self.recip_orbs,np.array(self.recip_WFcoeffs[0]),orb_overlap,recip=True)
        #print("original zero kpt:",zero_coeff[0])

        wan_orbs = {}
        all_gplusk_x = np.array([])
        all_gplusk_y = np.array([])
        all_gplusk_z = np.array([])
        all_gvecs_x = np.array([])
        all_gvecs_y = np.array([])
        all_gvecs_z = np.array([])
        for orb in range(self.num_orbs):
            wan_orbs[orb] = np.array([],dtype=np.complex128)

        total_nrms = []
        total_nme = []
        print("projected orbital on wavefunctions")
        for kpt in range(num_kpts):
                print("-",end="")
        print("")

        for kpt in range(num_kpts):
            if self.verbose > 1:
                print("kpt:",kpt,self.kpoints[kpt])
            DFT_wavefunc = np.array(self.recip_WFcoeffs[kpt])  # [band,gpoint]
            adjusted_orbs = self.get_kdep_reciporbs(kpt)
            overlap = self.get_overlap_matrix(adjusted_orbs, recip=True)
            ae_overlap = self.Sij[kpt]
            coeff = self.get_aecoefficients(adjusted_orbs, DFT_wavefunc, ae_overlap,kpt,recip=True)

            inv_orth_coeff = np.matmul(np.conj(coeff), ae_overlap).T  # np.linalg.inv(orth_coeff) #np.conj(orth_coeff.T)
            orb_mixing = np.matmul(coeff.T, inv_orth_coeff.T)
            np.fill_diagonal(orb_mixing, 0)
            max_orbmixing_allbands[kpt] = np.amax(np.abs(orb_mixing))
            if self.verbose > 0:
                print("orb mixing of all bands:", len(coeff), np.amax(np.abs(orb_mixing)))


            '''
            orth_extra_wf = np.zeros([len(DFT_wavefunc), len(DFT_wavefunc[0])], dtype=np.complex128)
            # include second loop to exclude the residual wavefunction in the periodic_WFs when calculating coeffs
            print("check how coefficients update:")
            for con_coeff in range(3):
                WFs_forcoeffs = DFT_wavefunc - orth_extra_wf
                proj_extra = self.get_aecoefficients(adjusted_orbs, orth_extra_wf, ae_overlap,kpt,recip=True)
                coeff = self.get_aecoefficients(adjusted_orbs, WFs_forcoeffs, ae_overlap,kpt,recip=True)
                
                #orth_extra_wf = np.zeros([self.num_bands, num_gpnts], dtype=np.complex128)  # different for recip
                #orb_coeff = np.sum(adjusted_orbs[None,:,:]*coeff[:,:,None],axis=1) 
                
                spillage = np.zeros(self.max_band)
                orth_orb_wfs = np.zeros(DFT_wavefunc.shape,dtype=np.complex128)
                for band in range(self.max_band):
                    trueWF = DFT_wavefunc[band]
                    orb_coeff = np.zeros(trueWF.shape, dtype=np.complex128)
                    for temporb in range(self.num_orbs):
                        orbWF = adjusted_orbs[temporb]#orth_orbs
                        orb_coeff += orbWF * coeff[band][temporb]
                    orth_orb_wfs[band] = orb_coeff
                    orth_extra_wf[band] = trueWF - orb_coeff
                    
                    spillage[band] = np.matmul(np.conj(coeff[band]), np.matmul(ae_overlap, coeff[band].transpose())).real
                    if (spillage[band] > 0.3) & (spillage[band] < 0.8):
                        print(band,spillage[band])
                        print(proj_extra)
            '''
            #coeff = np.array(self.get_aecoefficients(adjusted_orbs, DFT_wavefunc, ae_overlap, kpt,recip=True))
            
            

            self.mnkcoefficients[kpt] = coeff

            #reconstruct the LCAO wavefunction and find error
            if calc_nrms == True:
                orth_extra_wf = np.zeros(DFT_wavefunc.shape,dtype=np.complex128)
                trueWFs = np.zeros(DFT_wavefunc.shape,dtype=np.complex128)
                orth_orb_wfs = np.zeros(DFT_wavefunc.shape,dtype=np.complex128)
                rms = []
                nrms = []
                nme = []
                for band in range(self.max_band):
                    trueWF = DFT_wavefunc[band]
                    orb_coeff = np.zeros(trueWF.shape, dtype=np.complex128)
                    for temporb in range(self.num_orbs):
                        orbWF = adjusted_orbs[temporb]#orth_orbs
                        orb_coeff += orbWF * coeff[band][temporb]
                    orth_orb_wfs[band] = orb_coeff
                    orth_extra_wf[band] = trueWF - orb_coeff
                    trueWFs[band] = trueWF
                    normalize_rms = np.sqrt(np.average(np.square(np.abs(DFT_wavefunc[band]))))
                    normalize_me = np.average(np.abs(DFT_wavefunc[band]))
                    rms.append(np.sqrt(np.average(np.square(orth_extra_wf[band].real)+np.square(orth_extra_wf[band].imag))))
                    nrms.append(np.sqrt(np.average(np.square(orth_extra_wf[band].real)+np.square(orth_extra_wf[band].imag)))/normalize_rms)
                    nme.append(np.average(np.abs(orth_extra_wf[band]))/normalize_me)

                if self.verbose > 0:
                    #print("rms:",rms)
                    print("nrms:",np.around(nrms,decimals=3))
                total_nrms.append(nrms)
                total_nme.append(nme)
                all_nrms.append(np.average(nrms[:self.num_orbs]))

            spillage = np.zeros(self.max_band)
            for band in range(self.max_band):
                spillage[band] = np.matmul(np.conj(coeff[band]), np.matmul(ae_overlap, coeff[band].transpose())).real

            allspillage[kpt] = 1 - spillage

            is_valence = self.eigval[kpt][:,0]<=(self.efermi+self.energy_shift)
            val_spill = allspillage[kpt][is_valence]
            charge_spill[kpt] = np.sum(np.abs(val_spill)) / len(val_spill)
            max_charge_spill[kpt] = np.amax(val_spill)
            if ((1-spillage)<0).any():
                if self.verbose > 1:
                    print("warning!!! your charge spillage is negative, which is unphysical")
                else:
                    print("!", end="")
            else:
                if self.verbose < 2:
                    print("-", end="")

            if self.verbose > 0:
                print("spill:",np.around(1-spillage,decimals=3))

            bands_spill = 1 - spillage
            include_lowband = (((1 - bands_spill) > self.low_min_proj) | (self.eigval[kpt][:, 0] >= (
                        self.efermi + self.energy_shift - 5)))  # only False for bad low energy bands
            include_bands = (include_lowband) & (bands_spill < (1.0 - self.min_proj))
            #coeff_mult = np.matmul(red_coeff[:, include_bands], np.conj(red_coeff.T)[include_bands])
            inv_orth_coeff = np.matmul(np.conj(coeff[include_bands]),ae_overlap).T  # np.linalg.inv(orth_coeff) #np.conj(orth_coeff.T)
            orb_mixing = np.matmul(coeff[include_bands].T,inv_orth_coeff.T)
            np.fill_diagonal(orb_mixing,0)
            max_orbmixing[kpt] = np.amax(np.abs(orb_mixing))
            if self.verbose > 0:
                print("orb mixing:",len(coeff),np.amax(np.abs(orb_mixing)))

        #avg spillage per band
        #print(np.sum(self.kpt_weights),self.kpt_weights)
        avgbandspillage = np.sum(allspillage*self.kpt_weights[:,None],axis=0)
        tot_charge_spill = np.sum(charge_spill*self.kpt_weights)
        self.all_band_spillage = allspillage
        #self.mnkcoefficients = np.trunc(self.mnkcoefficients.real*10**6)/10**6 + np.trunc(self.mnkcoefficients.imag*10**6)/10**6*1j
        #print("og coeff:",self.mnkcoefficients)
        print("")
        #print("kpts:",self.kpoints)
        #print("spill", spillage)
        #print("avg avg band spilling",np.average(avgbandspillage[:self.num_orbs]))
        print("percent charge spilling:",np.around(tot_charge_spill*100,decimals=2), "%")
        print("maximum band charge spill:", np.around(np.sum(max_charge_spill*self.kpt_weights)*100,decimals=2), "%")
        print("maximum charge spill:", np.around(np.amax(max_charge_spill)*100,decimals=2), "%")
        print("avg spill per band:",np.around(avgbandspillage,decimals=4))
        
        print("average orbital mixing:", np.around(np.sum(max_orbmixing*self.kpt_weights)*100,decimals=2),"%")
        print("max orbital mixing:", np.around(np.amax(max_orbmixing)*100,decimals=2),"%")
        print("")

        # save error info to txt file
        error_file = open(self.directory+"error_output"+self.tag+".txt", "w")
        error_file.write("File generated by COGITO\n")
        
        error_file.write("percent charge spilling: "+str(np.around(tot_charge_spill*100,decimals=2))+ "%\n")
        error_file.write("maximum band charge spill: "+ str(np.around(np.sum(max_charge_spill*self.kpt_weights)*100,decimals=2))+ "%\n")
        error_file.write("maximum charge spill: "+ str(np.around(np.amax(max_charge_spill)*100,decimals=2))+ "%\n")
        #wannier_hr.write("avg spill per band:",np.around(avgbandspillage,decimals=4))
        
        error_file.write("average orbital mixing: "+ str(np.around(np.sum(max_orbmixing*self.kpt_weights)*100,decimals=2))+"%\n")
        error_file.write("max orbital mixing: "+ str(np.around(np.amax(max_orbmixing)*100,decimals=2))+"%\n")

        error_file.write("If all bands are used:"+"%\n")
        error_file.write("all bands avg orbmix: "+ str(np.around(np.sum(max_orbmixing_allbands*self.kpt_weights)*100,decimals=2))+"%\n")
        error_file.write("all abnds max orbmix: "+ str(np.around(np.amax(max_orbmixing_allbands)*100,decimals=2))+"%\n")
        #wannier_hr.write(str(int(max1/2))+" "+str(int(max2/2))+" "+str(int(max3/2))+" "+str(self.num_orbs)+"\n")
        self.max_orb_mixing = max_orbmixing
        self.max_band_spilling = max_charge_spill

        error_file.write("avg spill per band:\n")
        rounded_array = np.around(avgbandspillage,decimals=4)
        for i in range(0, len(rounded_array), 6):
            #line = ' '.join(map(str, rounded_array[i:i+6]))
            line = ' '.join(f"{x:8.4f}" for x in rounded_array[i:i+6])
            error_file.write(line + '\n')
        
        if calc_nrms == True:
            error_file.write('\n')
            total_nrms = np.array(total_nrms)
            self.total_nrms = total_nrms
            self.total_nme = total_nme
            occ_nrms = np.zeros(num_kpts)
            max_occnrms = np.zeros(num_kpts)
            for kpt in range(num_kpts):
                is_valence = self.eigval[kpt][:,0]<=(self.efermi+self.energy_shift)
                val_nrms = total_nrms[kpt][is_valence]
                occ_nrms[kpt] = np.sum(np.abs(val_nrms)) / len(val_nrms)
                max_occnrms[kpt] = np.amax(val_nrms)
            avgbandnrms = np.sum(total_nrms*self.kpt_weights[:,None],axis=0)
            avg_occnrms = np.sum(occ_nrms*self.kpt_weights)
            
            print("average normalized rms:",np.around(avg_occnrms,decimals=4))
            print("maximum band nrms:", np.around(np.sum(max_occnrms*self.kpt_weights),decimals=4))
            print("maximum nrms:", np.around(np.amax(max_occnrms),decimals=4))
            print("avg nrms per band:",np.around(avgbandnrms,decimals=4))
        
            #print("avg nrms per band",avgbandnrms)
            #print("avg avg nrms:", np.average(avgbandnrms[:self.num_orbs]))#,all_nrms)
            
            error_file.write("average normalized rms: "+str(np.around(avg_occnrms,decimals=4))+"\n")
            error_file.write("maximum band nrms: "+ str(np.around(np.sum(max_occnrms*self.kpt_weights),decimals=4))+"\n")
            error_file.write("maximum nrms: "+ str(np.around(np.amax(max_occnrms),decimals=4))+"\n")
            #error_file.write("avg nrms per band: "+np.around(avgbandnrms,decimals=4)+"\n")
            
            #error_file.write("avg avg wavefunction nrms: "+ np.average(avgbandnrms[:self.num_orbs])+"%\n")
            #wannier_hr.write(str(int(max1/2))+" "+str(int(max2/2))+" "+str(int(max3/2))+" "+str(self.num_orbs)+"\n")
    
            error_file.write("avg nrms per band:\n")
            rounded_array = np.around(avgbandnrms,decimals=4)
            for i in range(0, len(rounded_array), 6):
                #line = ' '.join(map(str, rounded_array[i:i+6]))
                line = ' '.join(f"{x:8.4f}" for x in rounded_array[i:i+6])
                error_file.write(line + '\n')
            

    def optimize_band_set(self,band_opt=True,orb_opt=True,orb_orth=False):
        """
        This funciton refines the orbital coefficients for perfect 'spilling' in the valence bands and improving spilling and correct orthogonalize to VB in the conduction bands.
        This process significantly decreases the band error in the TB interpolation while making little to no change to the chemically interpretible metrics (i.e. COHP, COOP, etc).

        Returns:    
            unknown: Set of band which minimizes the difference between the orbital states left after and the band set orbital states;   maybe later: also minimize overlap between Lowdin orthogonalized band sets.
        """
        
        aeorth_coeffs = self.get_orth_coeffs(self.mnkcoefficients)
        self.mixed_eigval = np.zeros((self.num_kpts,self.num_orbs,self.num_orbs),dtype=np.complex128)
        for kpt in range(self.num_kpts):
            ae_coeff = aeorth_coeffs[kpt] #[band,orb]
            band_energies = self.eigval[kpt][:,0]
            band_occupancy = self.eigval[kpt][:,1]
            lumo = np.arange(len(band_occupancy))[band_occupancy==0][0]
            #print("LUMO band:",lumo)
            #print(band_energies)
            
            #calculate the norm and compare to the NRMS
            norm = 1-self.all_band_spillage[kpt]#np.sum(np.conj(ae_coeff)*ae_coeff,axis=1)
            og_norm = copy.deepcopy(norm)
            #norm_nrms = np.abs(1 - self.total_nrms[kpt])**(1/2)
            #error = np.array((norm-norm_nrms)/(norm_nrms+0.00001))
            
            if self.verbose > 1:
                print(np.abs(norm))
                #print(norm_nrms)

            # if error is too large, norm is inaccurate because of PAW d orbs and should be set to the NRMS
            #norm[error>0.1] = np.array(norm_nrms)[error>0.1]
            norm = np.abs(norm)
            if self.verbose > 1:
                print("final band norm",norm)
            #rescale all the coeffs to the nrms as well
            #ae_coeff[error>0.1] = ae_coeff[error>0.1]/np.array([og_norm[error>0.1]]).T**(1/2)*np.array([norm_nrms[error>0.1]]).T**(1/2)
            
            # select bands that are included immediately and ones to detangle
            bad_proj = np.arange(self.max_band)[norm < 0.9]
            lowest_bad_proj = lumo#bad_proj[0]
            set_bands = np.arange(lowest_bad_proj)[norm[:lowest_bad_proj] >= 0.6] #np.arange(self.max_band)[norm_nrms >= 0.8]
            low_remove_bands = np.arange(lowest_bad_proj)[norm[:lowest_bad_proj] < 0.6]
            num_lowremoved = len(low_remove_bands)
            num_set = len(set_bands)
            bands_avail = self.num_orbs - num_set
            high_bands = np.arange(lowest_bad_proj,self.max_band)
            #only disentangle bands that have >0.2 projection
            # only include bands with > 15% projection
            disent_bands = high_bands[norm[high_bands] > 0.15]
            # only include the number of bands available + 4
            #disent_bands = np.sort(disent_bands[np.argsort(norm[disent_bands])[::-1][:bands_avail+4]])
            '''
            if self.verbose > 0:
                print("band groups:",set_bands,disent_bands)
            
            if len(disent_bands) == bands_avail:
                new_bands = disent_bands
                # check for the band that have been added and removed
                added_bands = []
                removed_bands = list(range(lowest_bad_proj,self.num_orbs+num_lowremoved))
                for band in new_bands:
                    if band < self.num_orbs+num_lowremoved:
                        removed_bands.remove(band)
                    else:
                        added_bands.append(band)
                        if self.verbose > 1:
                            print("higher energy band added!")
                
                if self.verbose > 0:
                    print("new bands!!", added_bands, removed_bands)

                added_bands = np.sort(added_bands)
                old_index = np.array(np.append(added_bands, removed_bands), dtype=np.int_)  # include reassigning the removed bands to higher bands #[29,5] [13,29,5,7]
                new_index = np.array(np.append(removed_bands, added_bands), dtype=np.int_) #just the opposite of old_index # [5,29] #[5,7,13,29]
                temp_eig = self.eigval[kpt][old_index]
                # print("top eig:", temp_eig[-2:])
                # sorting = np.argsort(temp_eig[:,0])
                # print(sorting)
                sorted_temp_eig = temp_eig  # [sorting]
                self.eigval[kpt][new_index] = sorted_temp_eig
                temp_coeff = self.mnkcoefficients[kpt][old_index]
                sorted_temp_coeff = temp_coeff  # [sorting]
                self.mnkcoefficients[kpt][new_index] = sorted_temp_coeff
                sorted_spill = self.all_band_spillage[kpt][old_index]
                self.all_band_spillage[kpt][new_index] = sorted_spill
                #if self.readmode == False:
                temp_wf = self.recip_WFcoeffs[kpt][old_index]
                sorted_temp_wf = temp_wf  # [sorting]
                # print("new wf shape:",temp_wf.shape)
                self.recip_WFcoeffs[kpt][new_index] = sorted_temp_wf
                
                
            elif bands_avail > 0:
                #calculate the orbital states that should be in the remaining disentangled bands
                set_coeffs = ae_coeff[set_bands]#/np.array([norm[set_bands]]).T**(1/8)
                disent_coeffs = ae_coeff[disent_bands]#/np.array([norm[disent_bands]]).T**(1/4)
                num_disent = len(disent_bands)
                allideal_orb_state = np.identity(self.num_orbs) - np.matmul(set_coeffs.T,np.conj(set_coeffs))
                #print("check ideal",allideal_orb_state)
                bool_sparse = np.abs(allideal_orb_state.flatten()) > 0.1
                ideal_orb_state = allideal_orb_state.flatten()[bool_sparse]
                if len(ideal_orb_state) == 0:
                    set_max = np.amax(np.abs(allideal_orb_state.flatten()))
                    bool_sparse = np.abs(allideal_orb_state.flatten()) > set_max-0.01
                #if len(ideal_orb_state) == 0:
                #    bool_sparse = np.abs(allideal_orb_state.flatten()) > 0.01

                disband_orbstate = []
                for band in range(num_disent):
                    orbstate = np.matmul(np.array([disent_coeffs[band]]).T,[np.conj(disent_coeffs[band])])
                    #only use the diagonal now
                    #orbstate = np.conj(disent_coeffs[band])*disent_coeffs[band]
                    #print("orbstate",orbstate)
                    disband_orbstate.append(orbstate.flatten()[bool_sparse])
                #print("all orbstates:",disband_orbstate)
                
                #calculate band overlap
                set_disent_overlap = np.sum(np.abs(np.matmul(np.conj(set_coeffs),disent_coeffs.T)),axis=0)
                if self.verbose > 1:
                    print("overlap of disent with set:",set_disent_overlap)
                disent_overlap = np.abs(np.matmul(np.conj(disent_coeffs),disent_coeffs.T)*(np.array([norm[disent_bands]])/np.array([norm[disent_bands]]).T))
                weight_lower = np.tri(num_disent)*2.5+(1-np.tri(num_disent))/2.5
                #print(weight_lower)
                np.fill_diagonal(disent_overlap,0)
                summed_overlaps = np.sum(disent_overlap*weight_lower,axis=1)
                if self.verbose > 1:
                    print(summed_overlaps)

                # make everything absolute
                disband_orbstate = np.abs(disband_orbstate) #[band,orbstate]
                ideal = np.abs(ideal_orb_state)
                if self.verbose > 1:
                    print(np.sum(np.abs(disband_orbstate),axis=1))
                total_coeff = -np.sum(np.abs(disband_orbstate),axis=1)+set_disent_overlap/norm[disent_bands]**(1/4)+summed_overlaps/2 # + np.arange(num_disent)*0.03#/(disent_bands[-1]-disent_bands[0]-bands_avail+1) #+np.abs(norm[disent_bands])/4
                if self.verbose > 0:
                    print("summed coeff",total_coeff)
                # make the ones with a summed coeff of < 0.1 extra unlikely
                total_coeff[np.sum(np.abs(disband_orbstate),axis=1)< 0.1] += 1

                min_bands = np.argsort(total_coeff)
                all_bands = min_bands[:bands_avail]
                guess = np.append(np.ones(bands_avail),np.zeros(num_disent-bands_avail))
                
                #print(problem.success,problem.message)
                
                selected_bands = np.sort(all_bands)#np.argsort(all_bands)[::-1][:bands_avail]
                if self.verbose > 1:
                    print("selected bands!",kpt,selected_bands,disent_bands[selected_bands])
                # try just selected bands with highest projection
                min_bands = np.argsort(norm[disent_bands])[::-1]
                projall_bands = np.sort(min_bands[:bands_avail])
                if (selected_bands != projall_bands).any() and self.verbose > 0:
                    print("projection found different set:",disent_bands[selected_bands],disent_bands[projall_bands])
                    print("overlap of disent with set:",set_disent_overlap)
                    print("summed coeff",total_coeff)
                    print(norm[disent_bands])
                #selected_bands = projall_bands # just using the bands with highest projection!!!
                new_bands = disent_bands[selected_bands]

                # check for the band that have been added and removed
                added_bands = []
                removed_bands = list(range(lowest_bad_proj,self.num_orbs+num_lowremoved))
                for band in new_bands:
                    if band < self.num_orbs+num_lowremoved:
                        removed_bands.remove(band)
                    else:
                        added_bands.append(band)
                        if self.verbose > 1:
                            print("higher energy band added!")
                if self.verbose > 0:
                    print("new bands!!", added_bands, removed_bands)
                
                if len(added_bands) > 0:
                    added_bands = np.sort(added_bands)
                    old_index = np.array(np.append(added_bands, removed_bands), dtype=np.int_)  # include reassigning the removed bands to higher bands #[29,5] [13,29,5,7]
                    new_index = np.array(np.append(removed_bands, added_bands), dtype=np.int_) #just the opposite of old_index # [5,29] #[5,7,13,29]
                    temp_eig = self.eigval[kpt][old_index]
                    # print("top eig:", temp_eig[-2:])
                    # sorting = np.argsort(temp_eig[:,0])
                    # print(sorting)
                    sorted_temp_eig = temp_eig  # [sorting]
                    self.eigval[kpt][new_index] = sorted_temp_eig
                    temp_coeff = self.mnkcoefficients[kpt][old_index]
                    sorted_temp_coeff = temp_coeff  # [sorting]
                    self.mnkcoefficients[kpt][new_index] = sorted_temp_coeff
                    sorted_spill = self.all_band_spillage[kpt][old_index]
                    self.all_band_spillage[kpt][new_index] = sorted_spill
                    #if self.readmode == False:
                    temp_wf = self.recip_WFcoeffs[kpt][old_index]
                    sorted_temp_wf = temp_wf  # [sorting]
                    # print("new wf shape:",temp_wf.shape)
                    self.recip_WFcoeffs[kpt][new_index] = sorted_temp_wf
                #print("eig:",self.eigval[kpt][:self.num_orbs])
                
            # now also remove the lower bands
            if len(low_remove_bands) > 0:
                all_newbands = np.append(set_bands,np.arange(lowest_bad_proj,self.num_orbs+num_lowremoved))
                old_index = np.append(all_newbands,low_remove_bands)
                new_index = np.arange(0,self.num_orbs+num_lowremoved)

                temp_eig = self.eigval[kpt][old_index]
                self.eigval[kpt][new_index] = temp_eig
                temp_coeff = self.mnkcoefficients[kpt][old_index]
                self.mnkcoefficients[kpt][new_index] = temp_coeff
                sorted_spill = self.all_band_spillage[kpt][old_index]
                self.all_band_spillage[kpt][new_index] = sorted_spill
                #if self.readmode == False:
                temp_wf = self.recip_WFcoeffs[kpt][old_index]
                # print("new wf shape:",temp_wf.shape)
                self.recip_WFcoeffs[kpt][new_index] = temp_wf

            #print("new eig:",self.eigval[kpt][:self.num_orbs])
            '''
            
            # now orthogonalize the eigenvectors
            include_lowband = (((1 - self.all_band_spillage[kpt]) > self.low_min_proj) | (self.eigval[kpt][:, 0] >= (self.efermi + self.energy_shift - 5)))  # only False for bad low energy bands
            bands_spill = self.all_band_spillage[kpt][include_lowband]
            include_lowband = (include_lowband & (self.all_band_spillage[kpt] < (1.0 - self.min_proj)))
            normalized = self.all_band_spillage[kpt][include_lowband][:self.num_orbs]#np.diagonal(np.matmul(np.conj(current_coeff), current_coeff.transpose()).transpose())#self.total_nrms[kpt,:self.num_orbs]#
            band_ind = np.argsort(normalized)
            norm_sorted = normalized[band_ind]
            energies = self.eigval[kpt][include_lowband][:self.num_orbs, 0]
            diff = np.array([abs(normalized[i + 1] - normalized[i]) * np.tanh(abs(energies[i + 1] - energies[i])) for i in range(len(normalized) - 1)])
            is_valence = self.eigval[kpt][include_lowband][1:self.num_orbs][:,0]<=(self.efermi+self.energy_shift)
            avg = np.sum(diff[is_valence]) / len(diff[is_valence]) #* 2
            #avg = sum(diff) / len(diff)
            #print(avg)
            m = [[0]]  # normalized[0]
            index = [[0]]
            for x in range(len(normalized) - 1):
                val = diff[x]
                ind = x + 1
                groupsize = 1
                if val < avg:  # or len(m[-1]) < groupsize:
                    m[-1].append(val)
                    index[-1].append(ind)
                else:
                    m.append([val])
                    index.append([ind])
            if self.verbose > 1:
                print("did grouping work?", m, index)
            #index = [[0,1,2,3,4,5,6,7]]
            #index = [np.arange(self.num_orbs)]
            # do it with the nonorth coeff
            start = index[0][0]
            end = index[0][-1] + 1
            orb_overlap = self.Sij[kpt]
            
            ''' very important but test without'''
            #print(1-self.all_band_spillage[kpt])
            #print(self.eigval[kpt][:,0])
            new_energies = np.diag(self.eigval[kpt][include_lowband][:,0])
            if self.verbose > 1:
                print("og energies:",np.diag(new_energies))
            '''
            # adjust band energies by combining the bands that aren't in the set
            norm_out = 1-self.all_band_spillage[kpt][include_lowband][self.num_orbs+num_lowremoved:]
            norm_in = 1-self.all_band_spillage[kpt][include_lowband][:self.num_orbs]
            lowest = np.amin(norm_in)/2
            if self.verbose > 1:
                print("lowest:",np.amin(norm_in))
            out_bands = np.arange(self.num_orbs+num_lowremoved,len(self.all_band_spillage[kpt][include_lowband]))[norm_out > 0.1]
            lowest_en = abs(np.amin(self.eigval[kpt][include_lowband][self.num_orbs+num_lowremoved:,0])-(self.efermi+self.energy_shift))/1.5
            in_bands = np.arange(self.num_orbs)[(norm_in < 0.9) & (self.eigval[kpt][include_lowband][:self.num_orbs,0] > (self.efermi+self.energy_shift+lowest_en))]
            
            if self.verbose > 1:
                print(in_bands,out_bands)
                print("update energies:",np.diag(new_energies)[in_bands],np.diag(new_energies)[out_bands])
            if len(in_bands) > 0 and len(out_bands) > 0:
                _,enval = GS_combine_states_orblap(nonorth_coeff[out_bands],nonorth_coeff[in_bands],orb_overlap,
                                                                                                 new_energies[out_bands][:,out_bands],new_energies[in_bands][:,in_bands],return_energy=True)
                new_energies[in_bands,in_bands] = np.diag(enval)
                if self.verbose > 1:
                    print("updated energies:",np.diag(new_energies)[in_bands])
            '''

            for loop in range(1):
                # band optimization
                nonorth_coeff = np.array(copy.deepcopy(self.mnkcoefficients[kpt]))[include_lowband]#[:self.num_orbs,:]))
                '''
                band_mix = np.abs(np.matmul(np.conj(nonorth_coeff), np.matmul(orb_overlap, nonorth_coeff.transpose())))
                if (kpt==155 or kpt ==131 or kpt == 0): #self.max_band_spilling[kpt] > np.amax(self.max_band_spilling)*0.98 or
                    print("plotting at kpt:",kpt,self.max_band_spilling[kpt])
                    print(self.kpoints[kpt])
                    print("grouping:",index)
                    
                    print("min diag:", np.min(np.diag(band_mix[:self.num_orbs-4][:,:self.num_orbs-4])))
                    hold_mix = copy.deepcopy(band_mix)
                    np.fill_diagonal(hold_mix,0)
                    print("max off-diag:", np.amax(hold_mix))
                    bands_spill = self.all_band_spillage[kpt]
                    include_bands = bands_spill < (1.0-self.min_proj)
                    print("bands included",np.arange(len(bands_spill))[include_bands])
                    print("band energies:",self.eigval[kpt][:,0])
                    print("correct sort?", np.argsort(self.eigval[kpt][:,0]))
                    cut = 34 #self.num_orbs+10
                    plot_matrix(band_mix[:cut][:,:cut],low_center=0.3,high_center=0.9,filename=self.directory + 'matrix_preopt'+str(kpt)+'.png')
                orb_mixing = np.abs(np.matmul(nonorth_coeff.T,np.matmul(np.conj(nonorth_coeff),orb_overlap)))
                #if (kpt==155 or kpt ==131): #self.max_band_spilling[kpt] > np.amax(self.max_band_spilling)*0.98 or
                #    plot_matrix(orb_mixing)
                '''
                #orb_overlap = self.Sij[kpt]
                #inv_orth_coeff = np.matmul(np.conj(nonorth_coeff[:self.num_orbs]),orb_overlap).T  # np.linalg.inv(orth_coeff) #np.conj(orth_coeff.T)
                #orb_mixing = np.matmul(nonorth_coeff[:self.num_orbs].T,inv_orth_coeff.T)
                #np.fill_diagonal(orb_mixing,0)
                #print("initial orb mixing:",np.amax(np.abs(orb_mixing)))

                if np.isnan(nonorth_coeff.flatten()).any():
                    #print(kpt,sym_opt,prim_opt,reduc_kpts[kpt])
                    print("there are nans after select!!",nonorth_coeff)
                start = index[0][0]
                end = index[0][-1] + 1
                nonorth_coeff[start:end]= lowdin_orth_vectors_orblap(nonorth_coeff[start:end],orb_overlap) #,new_energies[start:end,start:end],return_energy=True),new_energies[start:end,start:end] 
                # also calculate the new energy matrix, will no longer be diagonal
                for ind in index[1:]:
                    start = ind[0]
                    end = ind[-1] + 1
                    nonorth_coeff[start:end] = lowdin_orth_vectors_orblap(nonorth_coeff[start:end],orb_overlap) #,new_energies[start:end,start:end],return_energy=True),new_energies[start:end,start:end]
                    nonorth_coeff[start:end] = GS_orth_twoLoworthSets_orblap(nonorth_coeff[0:start],nonorth_coeff[start:end],orb_overlap) #,  #,new_energies[start:end,start:end]
                    #                                                                                 new_energies[0:start,0:start],new_energies[start:end,start:end],return_energy=True)

                nonorth_coeff[:self.num_orbs] = np.array(lowdin_orth_vectors_orblap(nonorth_coeff[:self.num_orbs],orb_overlap))

                #inv_orth_coeff = np.matmul(np.conj(nonorth_coeff[:self.num_orbs]),orb_overlap).T  # np.linalg.inv(orth_coeff) #np.conj(orth_coeff.T)
                #orb_mixing = np.matmul(nonorth_coeff[:self.num_orbs].T,inv_orth_coeff.T)
                #np.fill_diagonal(orb_mixing,0)
                #print("orb mixing:",np.amax(np.abs(orb_mixing)))

                #print("new energies:",np.diag(new_energies))
                #self.eigval[kpt][:self.num_orbs,0] = new_energies
                self.mixed_eigval[kpt] = np.diag(np.diag(new_energies)[:self.num_orbs])
                if np.isnan(nonorth_coeff.flatten()).any():
                    #print(kpt,sym_opt,prim_opt,reduc_kpts[kpt])
                    print("there are nans after orth!!",nonorth_coeff)

                #self.mnkcoefficients[kpt] = nonorth_coeff
                

                #actually just make all valence bands orthonormal and make a conduction bands orthogonal to VB
                new_coeff = np.array(copy.deepcopy(self.mnkcoefficients[kpt]))[include_lowband]
                low_coeff = np.array(copy.deepcopy(self.mnkcoefficients[kpt]))[include_lowband]
                if band_opt:
                    #nrange = 0
                    #for num in range(nrange+1):
                    low = 2 # eV above fermi
                    high = 5 # eV above fermi
                    low_valence = self.eigval[kpt][include_lowband][:,0]<=(self.efermi+self.energy_shift + low)
                    low_conduction = self.eigval[kpt][include_lowband][:,0]>(self.efermi+self.energy_shift + low)
                    cond_between = (self.eigval[kpt][include_lowband][:,0][low_conduction]<=(self.efermi+self.energy_shift + high))
                    #print("valence bands:",np.arange(len(is_valence))[is_valence])
                    low_coeff[low_valence] = nonorth_coeff[low_valence] # lowdin_orth_vectors_orblap(nonorth_coeff[is_valence],orb_overlap)
                    low_coeff[low_conduction] = GS_orth_twoLoworthSets_orblap(new_coeff[low_valence],new_coeff[low_conduction],orb_overlap)

                    #is_between = (self.eigval[kpt][include_lowband][:,0]<=(self.efermi+self.energy_shift + high)) & (self.eigval[kpt][include_lowband][:,0]>(self.efermi+self.energy_shift + low))
                    is_valence = self.eigval[kpt][include_lowband][:,0]<=(self.efermi+self.energy_shift + high)
                    val_between = (self.eigval[kpt][include_lowband][:,0][is_valence]>(self.efermi+self.energy_shift + low))
                    is_conduction = self.eigval[kpt][include_lowband][:,0]>(self.efermi+self.energy_shift + high)
                    #print("valence bands:",np.arange(len(is_valence))[is_valence])
                    new_coeff[is_valence] = nonorth_coeff[is_valence] # lowdin_orth_vectors_orblap(nonorth_coeff[is_valence],orb_overlap)
                    ratio = np.ones(len(new_coeff[is_valence]))
                    ref_energies = self.eigval[kpt][include_lowband][:,0][is_valence][val_between]-(self.efermi+self.energy_shift + low) # between 0 and 2
                    ratio[val_between] = np.tanh((-ref_energies.real+1))/2+1/2 # ratio ~1 for ref_energies at 0 and ~0 for ref at 2.
                    # only keep part of the nonorth_coeff for in between region
                    new_coeff[is_valence][val_between] = new_coeff[is_valence][val_between]*ratio[val_between][:,None] + low_coeff[low_conduction][cond_between]*(1-ratio[val_between][:,None])
                    new_coeff[is_conduction] = GS_orth_twoLoworthSets_orblap(new_coeff[is_valence],new_coeff[is_conduction],orb_overlap,ratio=ratio)

                    band_spill = self.all_band_spillage[kpt][include_lowband]
                    #new_coeff = self.mnkcoefficients[kpt]
                    mcut = self.min_proj*4
                    #print("diff")
                    #new_coeff[band_spill<=mcut] = nonorth_coeff[band_spill<=mcut] # lowdin_orth_vectors_orblap(new_coeff[band_spill<=mcut],orb_overlap)
                    #new_coeff[band_spill>mcut] = GS_orth_twoLoworthSets_orblap(new_coeff[band_spill<=mcut],new_coeff[band_spill>mcut],orb_overlap)
                    #self.mnkcoefficients[kpt][include_lowband] = new_coeff

                    # also adapt the eigenenergies of conduction states
                    '''
                    new_energies = np.diag(self.eigval[kpt][include_lowband][:, 0])
                    _, enval = GS_orth_twoLoworthSets_orblap(new_coeff[is_valence], self.mnkcoefficients[kpt][include_lowband][is_conduction], orb_overlap,
                                                        new_energies[is_valence][:, is_valence],
                                                        new_energies[is_conduction][:, is_conduction], return_energy=True,energycut=self.efermi+self.energy_shift+10)
                    new_energies[is_conduction][:, is_conduction] = np.diag(enval)
                    #print("energies change:",self.eigval[kpt][include_lowband][:, 0][is_conduction],np.diag(enval))
                    self.eigval[kpt][include_lowband][:, 0][is_conduction] = np.diag(enval)
                    '''
                '''
                band_mix = np.abs(np.matmul(np.conj(new_coeff), np.matmul(orb_overlap, new_coeff.transpose())))
                if (kpt==155 or kpt ==131 or kpt == 0): #self.max_band_spilling[kpt] > np.amax(self.max_band_spilling)*0.98
                    cut = 34 #self.num_orbs+10
                    plot_matrix(band_mix[:cut][:,:cut],low_center=0.3,high_center=0.9,filename=self.directory+'matrix_postopt'+str(kpt)+'.png')
                orb_mixing = np.abs(np.matmul(new_coeff.T,np.matmul(np.conj(new_coeff),orb_overlap)))
                #if (kpt==155 or kpt ==131): #self.max_band_spilling[kpt] > np.amax(self.max_band_spilling)*0.98 and 
                #    plot_matrix(orb_mixing)
                '''

                # orbital optimization
                # with the new scheme in calculating the Hamiltonian instead of orthogonalizing a square coefficient matrix we need to find some way to make the rectangular matrix 'exact'
                # my new idea is instead of Lowdin orthogonalizing the bands, Lowdin orthogonalize the orbitals, or for a nonorth basis, ensure that S_ab = c_an * c_nb^*
                if orb_opt or orb_orth:
                    bands_spill = self.all_band_spillage[kpt][include_lowband]
                    include_bands = bands_spill < (1.0 - self.min_proj)

                    test_coeff = new_coeff[include_bands]
                    forc_coeff = new_coeff

                    kdep_Aij = self.Aij[kpt]
                    orth_coeff = np.matmul(np.linalg.inv(kdep_Aij), test_coeff.transpose())  # [orb, band]
                    overlap = np.matmul(orth_coeff, np.conj(orth_coeff.T))
                    # print("Check mixing:",kpt,overlap[:2])
                    forc_orthcoeff = np.conj(lowdin_orth_vectors(np.conj(orth_coeff)))
                    overlap = np.matmul(forc_orthcoeff, np.conj(forc_orthcoeff.T))
                    # print("Check ident:",overlap)
                    forc_coeff[include_bands] = np.matmul(kdep_Aij, forc_orthcoeff).transpose()
                    # print("compare low coeff:",test_coeff[1],forc_coeff[1])
                    # print("compare coeff:",test_coeff[5],forc_coeff[5])
                    # forc_coeff[is_conduction] = GS_orth_twoLoworthSets_orblap(forc_coeff[is_valence],forc_coeff[is_conduction],orb_overlap)
                    # print(forc_coeff[5])
                    if orb_orth:
                        # if orb_opt: # techniqually could do this but most anything that enforces orthogonality also enforces orthonormality for the orbitals (orb_opt), so just to both
                        #    self.mnkcoefficients[kpt][include_bands] = orth_coeff.T
                        # else:
                        self.mnkcoefficients[kpt][include_lowband][include_bands] = forc_orthcoeff.T  # orth_coeff.T
                        self.Aij[kpt] = np.identity(self.num_orbs)
                        self.Sij[kpt] = np.identity(self.num_orbs)
                    else:  # orbital are still nonorth but are orb optimized
                        self.mnkcoefficients[kpt][include_lowband] = forc_coeff
                else:  # orbs are nonorth and not orb optimized
                    self.mnkcoefficients[kpt][include_lowband] = new_coeff

    def expand_irred_kgrid(self):
        """
        This function does a couple things:
        1. Finds the kpoints of the reducible grid and the coorespond symmetry operations to get them from the irreducible points.
        2. Creates the eigenvalues, eigenvectors, and overlaps matrices for the new reducible kpoint grid.
        """
        
        ## -- 1 -- ##
        from pymatgen.core import Structure
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
        np.set_printoptions(precision = 15,suppress=True)
        # creating the k point grid - no weights b/c it is not the reduced form of the kpoint grid
        
        # get irreducible kpts and their weights
        irrec_kpts_count = self.kpt_weights
        irrec_kpts = self.kpoints#,decimals=14)
        #print(irrec_kpts)
        
        # get symmetry operations
        structure = Structure.from_file(self.directory + 'POSCAR') # TODO: make sure that the symmetry ops are correct for AFM
        # add the magnetic properties
        print("spin polar?", self.spin_polar)
        if self.spin_polar:
            from pymatgen.io.vasp.outputs import Outcar
            outcar = Outcar(self.directory + "OUTCAR")
            mag = outcar.magnetization
            tot_magmoms = [m["tot"] for m in mag]
            print(tot_magmoms)
            site_prop = {}
            site_prop['magmom'] = tot_magmoms
            structure = Structure(structure.lattice,structure.species,structure.frac_coords,site_properties=site_prop)

        # pass "structure" to define class into "space"
        space = SpacegroupAnalyzer(structure,symprec=self.sym_prec,angle_tolerance=-1.0)
        point_groups = space.get_symmetry_operations()#get_point_group_operations()
        cart_point_groups = space.get_symmetry_operations(cartesian=True)#get_point_group_operations(cartesian=True)
        operations = np.zeros((len(point_groups),4,4),dtype=np.float64)
        cart_operations = np.zeros((len(point_groups),4,4),dtype=np.float64)
        for ind in range(len(point_groups)):
            operations[ind] = point_groups[ind].affine_matrix #[:3,:3]
            cart_operations[ind] = cart_point_groups[ind].affine_matrix #[:3,:3]
        #print("check sym:",operations)
        #print(cart_operations)
        if self.verbose > 0:
            print("cart symmetry opts:",cart_operations)
            print("prim symmetry opts:",operations)
            print(len(cart_operations))
        
        #find redicable kpts
        # kpoint on edges of keeping: [0.5,0.5,0.5] [0.5,-0.5,0.5] [0.5,-0.5,-0.5] [
        reduc_kpts = [[0.,0.,0.]] #requires the first point is Gamma point
        kpt_shifts = [[0.,0.,0.]]
        symopt_ind = [0]
        reduc_toir = [0]
        
        kprim1 = np.sort(np.around(np.abs(irrec_kpts[:,0]),decimals=6))
        kprim2 = np.sort(np.around(np.abs(irrec_kpts[:,1]),decimals=6))
        kprim3 = np.sort(np.around(np.abs(irrec_kpts[:,2]),decimals=6))
        
        min_div1 = kprim1[kprim1 != 0][0] if len(kprim1[kprim1 != 0]) > 0 else 1
        min_div2 = kprim2[kprim2 != 0][0] if len(kprim2[kprim2 != 0]) > 0 else 1 
        min_div3 = kprim3[kprim3 != 0][0] if len(kprim3[kprim3 != 0]) > 0 else 1
        
        max1 = np.around(1/kprim1[kprim1 != 0][0],decimals=0) if len(kprim1[kprim1 != 0]) > 0 else 1 
        max2 = np.around(1/kprim2[kprim2 != 0][0],decimals=0) if len(kprim2[kprim2 != 0]) > 0 else 1 
        max3 = np.around(1/kprim3[kprim3 != 0][0],decimals=0) if len(kprim3[kprim3 != 0]) > 0 else 1 
        
        #print(min_div1,min_div2,min_div3)
        #min_div1 = np.amin(np.abs(irrec_kpts[:,0])[np.abs(irrec_kpts[:,0]) > 0.000000001])
        #min_div2 = np.amin(np.abs(irrec_kpts[:,1])[np.abs(irrec_kpts[:,1]) > 0.000000001])
        #min_div3 = np.amin(np.abs(irrec_kpts[:,2])[np.abs(irrec_kpts[:,2]) > 0.000000001])
        #print("min divides",min_div1,min_div2,min_div3)
        for ir_ind,ir_kpt in enumerate(irrec_kpts[1:]):
            # generate all possible kpoints with the symmetry operations
            all_kpts = np.zeros((len(point_groups),3))
            all_shifts = np.zeros((len(point_groups),3))
            for ind in range(len(point_groups)):
                sym_opt = np.linalg.inv(operations[ind][:3,:3])
                try_kpt = np.dot(sym_opt.T,ir_kpt)
                shift = np.zeros(3)
                #shift[np.abs(try_kpt) <= 0.5] = 0 # shift back into primitive reciprocal cell if out
                shift[np.around(try_kpt,decimals=8)< -0.5] = 1
                shift[np.around(try_kpt,decimals=8) > 0.5] = -1
                temp_kpt = try_kpt + shift
                
                #shift anything on the negative faces to the positive faces
                #print(np.around(temp_kpt,decimals=4) == -0.5)
                if (np.around(temp_kpt,decimals=8) == -0.5).any(): # determine if the point is on the positive or negative side of the cube
                    num_trues = 0
                    while num_trues < 2:
                        #print("edge point",temp_kpt)
                        temp_kpt = np.around(temp_kpt,decimals=12)
                        angle_xy = np.arctan2(temp_kpt[0],temp_kpt[1])/np.pi*180
                        angle_yz = np.arctan2(temp_kpt[1],temp_kpt[2])/np.pi*180
                        angle_zx = np.arctan2(temp_kpt[2],temp_kpt[0])/np.pi*180
                        #print(angle_xy,angle_yz,angle_zx)
                        in_xy = angle_xy < 135 and angle_xy > -45
                        in_yz = angle_yz < 135 and angle_yz > -45
                        in_zx = angle_zx < 135 and angle_zx > -45
                        num_trues = 0
                        if in_xy == True:
                            num_trues += 1
                        if in_yz == True:
                            num_trues += 1
                        if in_zx == True:
                            num_trues += 1
                        #print(in_xy,in_yz,in_zx,num_trues)
                        if num_trues < 2: # shift the point to a positive face
                            shift[np.argmin(temp_kpt)] += 1
                        temp_kpt = try_kpt + shift
                
                all_kpts[ind] = np.around(try_kpt + shift,decimals=14)
                all_shifts[ind] = shift
            # select only the unique points and save which symmetry operation they used
            new_kpts,new_kpt_ind = np.unique(np.around(all_kpts,decimals=10),axis=0,return_index=True)#
            # also only select point which are on the right grid
            fits_grid = (((np.abs(new_kpts[:,0])+0.00001)%min_div1) < 0.0001) & (((np.abs(new_kpts[:,1])+0.00001)%min_div2) < 0.0001) & (((np.abs(new_kpts[:,2])+0.00001)%min_div3) < 0.0001)
            #print(new_kpts[:,0]%min_div1,new_kpts[:,1]%min_div2,new_kpts[:,2]%min_div3)
            new_kpts = new_kpts[fits_grid]
            new_kpt_ind = new_kpt_ind[fits_grid]
            symopt_ind = np.append(symopt_ind,new_kpt_ind)
            reduc_toir = np.append(reduc_toir,np.ones(len(new_kpt_ind))*(ir_ind+1))
            reduc_kpts = np.append(reduc_kpts,all_kpts[new_kpt_ind],axis=0)
            kpt_shifts = np.append(kpt_shifts,all_shifts[new_kpt_ind],axis=0)
            #print(new_kpts,len(new_kpts))
            
            # check that the kpoints have the same multiplicity as expected
            total_kpts = 1331
            if self.verbose > 0:
                if len(new_kpt_ind) != int(np.around(irrec_kpts_count[ir_ind+1]*total_kpts,decimals=0)):
                    print("did not find the correct number of reducible kpoints!",len(new_kpt_ind),irrec_kpts_count[ir_ind+1]*total_kpts)
                    print(ir_kpt)
                    print(new_kpts)
                    #print(new_kpts-all_shifts[new_kpt_ind])
                else:
                    print("successfully found the correct number of reducible kpoints!",len(new_kpt_ind),irrec_kpts_count[ir_ind+1]*total_kpts)
                    print(ir_kpt)
                    print(new_kpts)

        cart_red_kpt = _red_to_cart((self.b_vecs[0], self.b_vecs[1], self.b_vecs[2]), reduc_kpts)
        #print(reduc_kpts)
        reduc_toir = np.array(reduc_toir,dtype=np.int_)
        num_reduc = len(reduc_toir)
        if self.verbose > -1:
            print("num kpts reconstructed from irred:",num_reduc)
            
        # also include time symmetry
        # for each of the original point check if there is an opposite, if not, add one a label as same symmetry operation but list things as being time sym or not
        is_opposite = np.full((num_reduc), False)
        old_kpts =np.around(copy.deepcopy(reduc_kpts),decimals=14)
        rndold_kpts =np.around(copy.deepcopy(reduc_kpts),decimals=10)
        new_red_kpts = copy.deepcopy(reduc_kpts)
        true_numkpt = max1*max2*max3
        
        if num_reduc != true_numkpt: # try time symmetry if the number of kpoint is not already correct; if self.spin_polar == False:
            for kind, red_kpt in enumerate(old_kpts):
                opp_kpt = -red_kpt
                # check that the opposite kpoint in not on the negative surface of the cube

                #shift anything on the negative faces to the positive faces
                #print(temp_kpt)
                shift = np.zeros(3)
                try_kpt = opp_kpt
                temp_kpt = copy.deepcopy(opp_kpt)
                #print(np.around(temp_kpt,decimals=4) == -0.5)
                if (np.around(temp_kpt,decimals=8) == -0.5).any(): # determine if the point is on the positive or negative side of the cube
                    num_trues = 0
                    while num_trues < 2:
                        #print("edge point",temp_kpt)
                        temp_kpt = np.around(temp_kpt,decimals=12)
                        angle_xy = np.arctan2(temp_kpt[0],temp_kpt[1])/np.pi*180
                        angle_yz = np.arctan2(temp_kpt[1],temp_kpt[2])/np.pi*180
                        angle_zx = np.arctan2(temp_kpt[2],temp_kpt[0])/np.pi*180
                        #print(angle_xy,angle_yz,angle_zx)
                        in_xy = angle_xy < 135 and angle_xy > -45
                        in_yz = angle_yz < 135 and angle_yz > -45
                        in_zx = angle_zx < 135 and angle_zx > -45
                        num_trues = 0
                        if in_xy == True:
                            num_trues += 1
                        if in_yz == True:
                            num_trues += 1
                        if in_zx == True:
                            num_trues += 1
                        #print(in_xy,in_yz,in_zx,num_trues)
                        if num_trues < 2: # shift the point to a positive face
                            #print(np.argmin(temp_kpt))
                            shift[np.argmin(temp_kpt)] += 1
                        temp_kpt = try_kpt + shift
                opp_kpt = try_kpt + shift
                '''
                negface = False
                if (np.around(opp_kpt,decimals=4) == -0.5).any(): # determine if the point is on the positive or negative side of the cube
                    temp_kpt = np.around(opp_kpt,decimals=8)
                    angle_xy = np.arctan2(temp_kpt[0],temp_kpt[1])/np.pi*180
                    angle_yz = np.arctan2(temp_kpt[1],temp_kpt[2])/np.pi*180
                    angle_zx = np.arctan2(temp_kpt[2],temp_kpt[0])/np.pi*180
                    #print(angle_xy,angle_yz,angle_zx)
                    in_xy = angle_xy < 135 and angle_xy > -45
                    in_yz = angle_yz < 135 and angle_yz > -45
                    in_zx = angle_zx < 135 and angle_zx > -45
                    num_trues = 0
                    if in_xy == True:
                        num_trues += 1
                    if in_yz == True:
                        num_trues += 1
                    if in_zx == True:
                        num_trues += 1
                    #print(in_xy,in_yz,in_zx,num_trues)
                    if num_trues < 2: # is on a negative face
                        negface = True
                '''
                #if negface == False:
                rnd_kpt = np.around(opp_kpt,decimals=10)
                in_opposite = (rndold_kpts[:,0] == rnd_kpt[0]) & (rndold_kpts[:,1] == rnd_kpt[1]) & (rndold_kpts[:,2] == rnd_kpt[2])
                opp_kpt_inold = np.arange(num_reduc)[in_opposite]
                #print(rnd_kpt)
                #print(opp_kpt_inold)
                #print(old_kpts[opp_kpt_inold])
                #print(len(opp_kpt_inold))
                if len(opp_kpt_inold) == 0: # no point which is opposite
                    new_red_kpts=np.append(new_red_kpts,[opp_kpt],axis=0)
                    is_opposite=np.append(is_opposite,True)
                    symopt_ind=np.append(symopt_ind,symopt_ind[kind])
                    reduc_toir=np.append(reduc_toir,reduc_toir[kind])
                    kpt_shifts=np.append(kpt_shifts,[kpt_shifts[kind]-shift],axis=0)
                #else:
                #    print("didn't pass negface:",opp_kpt)

        reduc_kpts = new_red_kpts
        num_reduc = len(reduc_toir)
        cart_red_kpt = _red_to_cart((self.b_vecs[0], self.b_vecs[1], self.b_vecs[2]), reduc_kpts)
        if self.verbose > -1:
            print("num kpts reconstructed after time symmetry:",num_reduc)
            #print(reduc_kpts[1])
            #print(reduc_kpts)
        
        true_numkpt = max1*max2*max3
        if num_reduc != true_numkpt:
            print("ERROR: number of kpoints found from symmetry",num_reduc,' is not correct ',true_numkpt)
            raise Exception('ERROR: number of kpoints found from symmetry',num_reduc,' is not correct ',true_numkpt)

        #test kpoints being generated
        #[n1,n2,n3] = [5,5,5]
        #grid = np.mgrid[-0.5+1/n1/2:0.5-1/n1/2:n1*1j,-0.5+1/n2/2:0.5-1/n2/2:n2*1j,-0.5+1/n3/2:0.5-1/n3/2:n3*1j]
        #test_red_kpts = np.array([grid[0].flatten(),grid[1].flatten(),grid[2].flatten()]).T
        #print("should be kpoints:",test_red_kpts)
        #num_uniq = len(np.unique(np.around(np.append(new_red_kpts,test_red_kpts,axis=0),decimals=4),axis=0))
        #print("num unique with test kpts:",num_uniq)
        
        if self.verbose > 0:
            print(symopt_ind)
            print(reduc_toir)
            print(reduc_kpts)
        
        
        ## -- 2 -- ##
        # get arrays for the p and d orbitals
        #print("orbtype!",self.exactorbtype)
        # get unique orbital types
        uniq_orbs,inverse = np.unique(self.orbtype,return_inverse=True)
        #print("unique orbs:",uniq_orbs,inverse)
        orb_index = {}
        atm_has_orb = {}
        orbatomnum = np.array(self.orbatomnum)
        for ind, orb in enumerate(uniq_orbs):
            orb_index[orb] = np.sort(np.arange(self.num_orbs)[inverse==ind])
            #print(ind,orb,orb_index[orb])
            atm_has_orb[orb] = np.unique(orbatomnum[orb_index[orb]])
        num_irkpt = len(irrec_kpts)
        #print("orb values!",uniq_orbs,atm_has_orb,orb_index)
        ireigvals =  self.eigval # last index has energy and occupation
        #irnewvals = self.eigval[:,:,0]# self.mixed_eigval
        
        if np.isnan(self.mnkcoefficients.flatten()).any():
            #print(kpt,sym_opt,prim_opt,reduc_kpts[kpt])
            print("there are nans in og coeffs!!",self.mnkcoefficients)
            
        ireigvecs = self.mnkcoefficients #[kpt,band,orb]
        #print("old coeffs:",ireigvecs)
        #print(ireigvecs[10])
        #print("coeff shape:",ireigvecs.shape)
        shape = ireigvecs.shape
        old_gpoints = self.gpoints
        #old_nrms = self.total_nrms
        old_spill = self.all_band_spillage
        old_mixing = self.max_orb_mixing
        old_max_spill = self.max_band_spilling
        old_recip_WFs = self.recip_WFcoeffs
        
        all_reduc_eigvec = np.zeros((num_reduc,shape[1],shape[2]),dtype=np.complex128)
        all_reduc_eigval = np.zeros((num_reduc,shape[1],2),dtype=np.complex128)
        #all_reduc_newval = np.zeros((num_reduc,shape[2],shape[2]),dtype=np.complex128)
        all_recip_wfs = []
        
        all_gpoints = []
        #new_nrms = np.zeros((num_reduc,len(old_nrms[0])))
        new_spill = np.zeros((num_reduc,len(old_spill[0])))
        new_max_spill = np.zeros(num_reduc)
        new_mixing = np.zeros(num_reduc)
        all_Aij = np.zeros((num_reduc, self.num_orbs, self.num_orbs), dtype=np.complex128)
        all_Sij = np.zeros((num_reduc, self.num_orbs, self.num_orbs), dtype=np.complex128)
        old_Aij = self.Aij
        old_Sij = self.Sij
        
        #transform to the reducible kpts
        for kpt in range(num_reduc):
            # get the eigenvecs the normal way
            cart_opt = cart_operations[symopt_ind[kpt]]
            full_opt = cart_opt[:3,:3]#np.linalg.inv()
            prim_opt = np.linalg.inv(operations[symopt_ind[kpt]][:3,:3])#.T
            prim_shift = operations[symopt_ind[kpt]][:3,3]
            sym_opt = full_opt
            shift = cart_opt[:3,3]
            '''
            U, s, Vh = np.linalg.svd(sym_opt)
            sym_opt_ortho = U @ Vh
            # Ensure proper determinant (±1)
            if np.linalg.det(sym_opt_ortho) * np.linalg.det(sym_opt) < 0:
                sym_opt_ortho = -sym_opt_ortho
            sym_opt = sym_opt_ortho
            print(sym_opt)
            print(sym_opt_ortho)
            '''
            #print(sym_opt)
            #print(prim_shift)
            #print(reduc_kpts[kpt])
            #print("sym:",symopt_ind[kpt])
            #print(full_opt)
            #if (kpt_shifts[kpt] !=0).any():
                #print("shift here!",kpt_shifts[kpt])
            ir_coeffs = np.around(ireigvecs[reduc_toir[kpt]].T,decimals=14) #[orb,band]
            if np.isnan(ir_coeffs.flatten()).any():
                print(kpt,sym_opt,prim_opt,reduc_kpts[kpt])
                print(reduc_toir[kpt],irrec_kpts[reduc_toir[kpt]])
                print("there are nans!!",ir_coeffs)
            #print(ir_coeffs)
            red_coeff = np.zeros(ir_coeffs.shape,dtype=np.complex128)
            temp_Sij = np.zeros((self.num_orbs, self.num_orbs), dtype=np.complex128)
            temp_Aij = np.zeros((self.num_orbs, self.num_orbs), dtype=np.complex128)
            # game plan:
            # figure out sign of p orbital now cause they don't depend on atom
            # then go through each atom, apply inv sym opt to it, and translate it back into the prim cell
            # finally, assign the atom the coefficients of the atom it got translated to with the right sign

            #print("d_index",d_index)
            #print(has_d)
            # loop over atoms with p orbitals
            old_orbs = []
            new_orbs = []
            old_pos = copy.deepcopy(self.orbpos)
            orb_trans = np.zeros(self.orbpos.shape)
            redo_from_wfcoeff = False # this is only necessary is orbital symmetry transformations aren't working (currently not for f orbitals)
            for atm in range(self.numAtoms):
                #print(atm)
                center = self.primAtoms[atm] - prim_shift
                new_center = np.matmul(prim_opt,center)# - prim_shift
                og_center = np.around(new_center,decimals=10)#_cart_to_red((self._a[0], self._a[1], self._a[2]), [new_center])[0],decimals=8)
                #print("first:",prim_center)
                prim_center =  copy.deepcopy(og_center)
                while (prim_center>=1).any() or (prim_center<0).any():
                    translate = np.zeros(3)
                    translate[prim_center>=1] = -1
                    translate[prim_center<0] = 1
                    prim_center = prim_center + translate
                translate = prim_center-og_center
                final_center = prim_center
                diff_atoms = self.primAtoms - final_center
                #dist_atoms = np.linalg.norm(diff_atoms,axis=1)
                dist_atoms = np.amin([np.linalg.norm(diff_atoms,axis=1), np.abs(np.linalg.norm(diff_atoms,axis=1)-1),np.abs(np.linalg.norm(diff_atoms,axis=1)-2**(1/2)),np.abs(np.linalg.norm(diff_atoms,axis=1)-3**(1/2))],axis=0)
                old_atom = np.argmin(np.abs(dist_atoms))
                #print(og_center,prim_center,translate)
                if np.sum(np.abs(dist_atoms)[old_atom]) > self.sym_prec:
                    print("WARNING: did not find an actual atom!!!",dist_atoms,old_atom)
                #print(dist_atoms,old_atom)
                
                if self.verbose > 1:
                    if atm != old_atom:
                        print(self.primAtoms[atm],final_center)
                        print("new and old atm indices:",atm,old_atom)
                
                for orb_type in uniq_orbs:
                    has_orb = atm_has_orb[orb_type]
                    orbind = orb_index[orb_type]
                    if (has_orb == atm).any():
                        if "s" in orb_type:
                            #set s orbitals
                            relative_atm = np.arange(len(has_orb))[has_orb==atm][0]
                            relative_oldatm = np.arange(len(has_orb))[has_orb==old_atom][0]
                            old_s = orbind[relative_oldatm]
                            new_s = orbind[relative_atm]
                            red_coeff[new_s] = ir_coeffs[old_s]
                            old_orbs.append(old_s)
                            new_orbs.append(new_s)
                            old_pos[new_s] = self.orbpos[old_s]
                            orb_trans[new_s] = translate


                            #print("Is same atom?", atm, old_atom)
                            #print("Is coefficient the same?", ir_coeffs[old_s],ir_coeffs[new_s])

                        elif "p" in orb_type:
                            old_patm = np.arange(len(has_orb))[has_orb==old_atom][0]
                            new_patm = np.arange(len(has_orb))[has_orb==atm][0]

                            old_pxpypz = orbind[old_patm*3:(old_patm+1)*3]
                            new_pxpypz = orbind[new_patm*3:(new_patm+1)*3]
                            # y z x
                            oldatm_matrix = np.array([self.orb_vec[old_pxpypz[0]],self.orb_vec[old_pxpypz[1]],self.orb_vec[old_pxpypz[2]]])
                            newatm_matrix = np.array([self.orb_vec[new_pxpypz[0]],self.orb_vec[new_pxpypz[1]],self.orb_vec[new_pxpypz[2]]])

                            old_pcoeff = ir_coeffs[old_pxpypz]
                            vasp_map = np.array([1,2,0])
                            full_mtrx = np.matmul(newatm_matrix,np.matmul(sym_opt[vasp_map][:,vasp_map],oldatm_matrix.T))
                            new_pcoeff = np.matmul(full_mtrx,old_pcoeff)

                            #print("check sym:", sym_opt)
                            #print(old_pcoeff[:,0],new_pcoeff[:,0])
                            red_coeff[new_pxpypz] = new_pcoeff
                            
                            old_pos[new_pxpypz] = self.orbpos[old_pxpypz]
                            orb_trans[new_pxpypz] = translate

                        elif "d" in orb_type:
                            
                            old_datm = np.arange(len(has_orb))[has_orb==old_atom][0]
                            new_datm = np.arange(len(has_orb))[has_orb==atm][0]
                            old_z2_xz_yz_x2y2_xy = orbind[old_datm*5:(old_datm+1)*5]
                            new_z2_xz_yz_x2y2_xy = orbind[new_datm*5:(new_datm+1)*5]
                            
                            sym = sym_opt.T # just need transpose cause I wrote all the [i,j] as [j,i] and don't feel like rewriting it
                    
                            # for x^2 - y^2
                            #x2y2_xi2 = sym[0,0]*sym[0,0]
                            #print(sym[0,0]*sym[0,0],sym[1,0]*sym[1,0],sym[0,1]*sym[0,1],sym[1,1]*sym[1,1])
                            #weirdval = ((sym[0,0]*sym[0,0] - sym[1,0]*sym[1,0]) - (sym[0,1]*sym[0,1]-sym[1,1]*sym[1,1]))/2
                            x2y2_xi2yi2 = ((sym[0,0]*sym[0,0] - sym[1,0]*sym[1,0]) - (sym[0,1]*sym[0,1] - sym[1,1]*sym[1,1]))/2 
                            x2y2_zi2 = ((2*sym[2,0]*sym[2,0]-sym[1,0]*sym[1,0]-sym[0,0]*sym[0,0]) - (2*sym[2,1]*sym[2,1]-sym[1,1]*sym[1,1]-sym[0,1]*sym[0,1]))/2/3**(1/2)#**(1/2)
                            x2y2_xiyi = (2*sym[0,0]*sym[1,0] - 2*sym[0,1]*sym[1,1])/2#**(1/2)
                            x2y2_yizi = (2*sym[1,0]*sym[2,0] - 2*sym[1,1]*sym[2,1])/2#**(1/2)
                            x2y2_zixi = (2*sym[2,0]*sym[0,0] - 2*sym[2,1]*sym[0,1])/2#**(1/2)

                            # for z^2
                            z2_xi2yi2 = (2*(sym[0,2]*sym[0,2] - sym[1,2]*sym[1,2]) - (sym[0,1]*sym[0,1] - sym[1,1]*sym[1,1]) - (sym[0,0]*sym[0,0] - sym[1,0]*sym[1,0]))/2/3**(1/2)#2/2**(1/2)
                            #z2_yi2 = sym[1,2]*sym[1,2]
                            z2_zi2 = (2*(sym[2,2]*sym[2,2]-sym[0,2]*sym[0,2]-sym[1,2]*sym[1,2]) - (sym[2,0]*sym[2,0]-sym[0,0]*sym[0,0]-sym[1,0]*sym[1,0]) - (sym[2,1]*sym[2,1]-sym[0,1]*sym[0,1]-sym[1,1]*sym[1,1]))/4
                            z2_xiyi = (2*(2*sym[0,2]*sym[1,2])-(2*sym[0,0]*sym[1,0])-(2*sym[0,1]*sym[1,1]))/2/(3)**(1/2)
                            z2_yizi = (2*(2*sym[1,2]*sym[2,2])-(2*sym[1,0]*sym[2,0])-(2*sym[1,1]*sym[2,1]))/2/(3)**(1/2)
                            z2_zixi = (2*(2*sym[2,2]*sym[0,2])-(2*sym[2,0]*sym[0,0])-(2*sym[2,1]*sym[0,1]))/2/(3)**(1/2)

                            # for xy
                            xy_xi2yi2 = (sym[0,0]*sym[0,1] - sym[1,0]*sym[1,1])#/2**(1/2)
                            #xy_yi2 = sym[1,0]*sym[1,1]
                            xy_zi2 = (2*sym[2,0]*sym[2,1]-sym[1,0]*sym[1,1]-sym[0,0]*sym[0,1])/(3)**(1/2)
                            xy_xiyi = sym[0,0]*sym[1,1]+sym[1,0]*sym[0,1]
                            xy_yizi = sym[1,0]*sym[2,1]+sym[2,0]*sym[1,1]
                            xy_zixi = sym[2,0]*sym[0,1]+sym[0,0]*sym[2,1]
                            # for yz
                            yz_xi2yi2 = (sym[0,1]*sym[0,2] - sym[1,1]*sym[1,2])#/2**(1/2)
                            #yz_yi2 = sym[1,1]*sym[1,2]
                            yz_zi2 = (2*sym[2,1]*sym[2,2]-sym[1,1]*sym[1,2]-sym[0,1]*sym[0,2])/(3)**(1/2)
                            yz_xiyi = sym[0,1]*sym[1,2]+sym[1,1]*sym[0,2]
                            yz_yizi = sym[1,1]*sym[2,2]+sym[2,1]*sym[1,2]
                            yz_zixi = sym[2,1]*sym[0,2]+sym[0,1]*sym[2,2]
                            # for zx
                            zx_xi2yi2 = (sym[0,2]*sym[0,0] - sym[1,2]*sym[1,0])#/2**(1/2)
                            #xz_yi2 = sym[1,0]*sym[1,2]
                            zx_zi2 = (2*sym[2,2]*sym[2,0]-sym[1,2]*sym[1,0]-sym[0,2]*sym[0,0])/(3)**(1/2)
                            zx_xiyi = sym[0,2]*sym[1,0]+sym[1,2]*sym[0,0]
                            zx_yizi = sym[1,2]*sym[2,0]+sym[2,2]*sym[1,0]
                            zx_zixi = sym[2,2]*sym[0,0]+sym[0,2]*sym[2,0]
                            # order is z2, zx, yz, x2-y2, xy
                            d_switching = np.array([[z2_zi2, z2_zixi, z2_yizi, z2_xi2yi2, z2_xiyi],
                                           [zx_zi2, zx_zixi, zx_yizi, zx_xi2yi2, zx_xiyi],
                                           [yz_zi2, yz_zixi, yz_yizi, yz_xi2yi2, yz_xiyi],
                                           [x2y2_zi2, x2y2_zixi, x2y2_yizi, x2y2_xi2yi2, x2y2_xiyi],
                                           [xy_zi2, xy_zixi, xy_yizi, xy_xi2yi2, xy_xiyi]])#np.around(,decimals=15)
                            old_dcoeff = ir_coeffs[old_z2_xz_yz_x2y2_xy]
                            #print("d_switching:",d_switching)
                            #print(sym)
                            # new order is dxy dyz dz2 dxz dx2y2
                            vasp_map = np.array([4,2,0,1,3])
                            oldatm_matrix = np.array([self.orb_vec[old_z2_xz_yz_x2y2_xy[0]],self.orb_vec[old_z2_xz_yz_x2y2_xy[1]],
                                                      self.orb_vec[old_z2_xz_yz_x2y2_xy[2]],self.orb_vec[old_z2_xz_yz_x2y2_xy[3]],self.orb_vec[old_z2_xz_yz_x2y2_xy[4]]])
                            newatm_matrix = np.array([self.orb_vec[new_z2_xz_yz_x2y2_xy[0]],self.orb_vec[new_z2_xz_yz_x2y2_xy[1]],
                                                      self.orb_vec[new_z2_xz_yz_x2y2_xy[2]],self.orb_vec[new_z2_xz_yz_x2y2_xy[3]],self.orb_vec[new_z2_xz_yz_x2y2_xy[4]]])

                            full_mtrx = np.matmul(newatm_matrix,np.matmul(d_switching[vasp_map][:,vasp_map],oldatm_matrix.T))

                            old_pos[new_z2_xz_yz_x2y2_xy] = self.orbpos[old_z2_xz_yz_x2y2_xy]
                            orb_trans[new_z2_xz_yz_x2y2_xy] = translate

                            new_dcoeff = np.matmul(full_mtrx,old_dcoeff)
                            red_coeff[new_z2_xz_yz_x2y2_xy] = new_dcoeff
                            #if (old_z2_xz_yz_x2y2_xy != new_z2_xz_yz_x2y2_xy).any():
                            #    print("d coeff:",old_dcoeff[:,0],new_dcoeff[:,0])

                        elif 'f' in orb_type:
                            #redo_from_wfcoeff = True
                            old_datm = np.arange(len(has_orb))[has_orb==old_atom][0]
                            new_datm = np.arange(len(has_orb))[has_orb==atm][0]
                            old_forbs = orbind[old_datm*7:(old_datm+1)*7]
                            new_forbs = orbind[new_datm*7:(new_datm+1)*7]
                            old_fcoeff = ir_coeffs[old_forbs]

                            oldatm_matrix = np.array([self.orb_vec[old_forbs[0]],self.orb_vec[old_forbs[1]],self.orb_vec[old_forbs[2]],self.orb_vec[old_forbs[3]],
                                                      self.orb_vec[old_forbs[4]],self.orb_vec[old_forbs[5]],self.orb_vec[old_forbs[6]]])
                            newatm_matrix = np.array([self.orb_vec[new_forbs[0]],self.orb_vec[new_forbs[1]],self.orb_vec[new_forbs[2]],self.orb_vec[new_forbs[3]],
                                                      self.orb_vec[new_forbs[4]],self.orb_vec[new_forbs[5]],self.orb_vec[new_forbs[6]]])

                            f_switching = D_real_l3_from_R(sym_opt)

                            full_mtrx = np.matmul(newatm_matrix, np.matmul(f_switching, oldatm_matrix.T))

                            old_pos[new_forbs] = self.orbpos[old_forbs]
                            orb_trans[new_forbs] = translate

                            new_fcoeff = np.matmul(full_mtrx, old_fcoeff)
                            red_coeff[new_forbs] = new_fcoeff

            # also include the kpt shift
            #print("check old and new orb pos:",old_pos,self.orbpos)
            phase_shift = np.exp(-2j*np.pi*np.dot(kpt_shifts[kpt],(self.orbpos).T))
            #print("phase shift:",phase_shift)
            red_coeff = (red_coeff.T*phase_shift).T
            # now try to include what abinit does for paw coeff
            kpt1= irrec_kpts[reduc_toir[kpt]]
            k2 = reduc_kpts[kpt]
            #print(orb_trans)
            #other_phase = np.exp(2j*np.pi*np.dot(k2,(orb_trans[:,:]-prim_shift[None,:]).T))
            #print("other phase:",other_phase)
            #red_coeff = (red_coeff.T*other_phase).T

            # also include time symmetry
            if is_opposite[kpt]:
                if self.verbose > 0:
                    print("at time sym point!",kpt)
                red_coeff = np.conj(red_coeff)
                
            if np.isnan(red_coeff.flatten()).any():
                print(kpt,sym_opt,prim_opt,reduc_kpts[kpt])
                print("there are nans!!",red_coeff)
            all_reduc_eigvec[kpt] = red_coeff.T #[kpt,band,orb]
            all_reduc_eigval[kpt] = ireigvals[reduc_toir[kpt]]
            #all_reduc_newval[kpt] = irnewvals[reduc_toir[kpt]]
            
            
            # get new gpoints
            k2 = reduc_kpts[kpt]
            gpnts = self.generate_gpnts(k2)
            #all_gpoints.append(gpnts)#old_gpoints[reduc_toir[kpt]])#np.matmul(prim_opt.T,np.array(old_gpoints[reduc_toir[kpt]]).T).T)
            
            # calculate the G coefficients of wavefunction at kpoint k2 from the IBZ k1 wavefunction
            Gcoeff1 = np.array(old_recip_WFs[reduc_toir[kpt]])  # [band,igpnt]
            og_gpnts = np.array(old_gpoints[reduc_toir[kpt]]) # [igpnt,b1/b2/b3]
            prim_opt = prim_opt.T #np.linalg.inv(operations[symopt_ind[kpt]][:3,:3]).T # symrec
            prim_shift = operations[symopt_ind[kpt]][:3,3]
            phase = np.exp(2j*np.pi*np.dot(og_gpnts, prim_shift))#k2[None,:]+og_gpnts
            Gcoeff1 = phase[None,:]*Gcoeff1
            
            
            # generate from irred gpnts
            shift_gpnts = og_gpnts 
            sym_gpnts = np.matmul(prim_opt,shift_gpnts.T).T - kpt_shifts[kpt]
            if is_opposite[kpt]:
                if self.verbose > 1:
                    print("at time sym point!",kpt)
                sym_gpnts = -sym_gpnts
                
            max_vals = np.amax(np.abs(sym_gpnts),axis=0)
            sort_gpnts = copy.deepcopy(sym_gpnts)
            sort_gpnts[sym_gpnts[:,0]<0] = sort_gpnts[sym_gpnts[:,0]<0] + np.array([[max_vals[0]*2 + 1,0,0]])
            sort_gpnts[sym_gpnts[:,1]<0] = sort_gpnts[sym_gpnts[:,1]<0] + np.array([[0,max_vals[1]*2 + 1,0]])
            sort_gpnts[sym_gpnts[:,2]<0] = sort_gpnts[sym_gpnts[:,2]<0] + np.array([[0,0,max_vals[2]*2 + 1]])
            
            ind = np.lexsort((sort_gpnts[:,0],sort_gpnts[:,1],sort_gpnts[:,2])) # index which converts the symmetrized gpnts to the normal gpnt ordering
            #ind = np.lexsort((sym_gpnts[:,0],sym_gpnts[:,1],sym_gpnts[:,2])) # index which converts the symmetrized gpnts to the normal gpnt ordering
            #print(ind.shape)
            sorted_symg = sym_gpnts[ind]
            all_gpoints.append(sorted_symg)
            if (kpt_shifts[kpt] != 0).any() and False:
                print("gpnts same?")
                print(len(gpnts),len(og_gpnts),len(sorted_symg))
                print(gpnts)
                print(kpt_shifts[kpt])
                print(og_gpnts)
                print(sorted_symg)
                print(is_opposite[kpt])
            
            #old_Cnkg = np.array(old_recip_WFs[reduc_toir[kpt]])  # [band,gpoint]
            #new_Cnkg = old_Cnkg[:,ind] * np.exp(-2j*np.pi* np.dot(gpnts, shift))
            new_Cnkg = Gcoeff1[:,ind]
            # also include time symmetry
            if is_opposite[kpt]:
                if self.verbose > -1:
                    print("at time sym point!",kpt)
                new_Cnkg = np.conj(new_Cnkg)
            
            # calculate the coefficients by projecting orbital on the symmetry transformed KS wavefunctions (very expensive)
            if redo_from_wfcoeff:
                adjusted_orbs = self.get_kdep_reciporbs(reduc_kpts[kpt],full_kpt =True,gpnts=sorted_symg,set_gpnts=True)
                ae_overlap = self.get_ae_overlap_info(adjusted_orbs,kpt=reduc_kpts[kpt], just_ae_overlap = True,full_kpt=True,gpnts=sorted_symg,set_gpnts=True)
                coeff = self.get_aecoefficients(adjusted_orbs, new_Cnkg, ae_overlap, kpt=reduc_kpts[kpt],recip=True,full_kpt=True,gpnts=sorted_symg,set_gpnts=True)

                diff_coeff = coeff-red_coeff.T
                if np.amax(np.abs(diff_coeff)) > 1e-4:# and self.verbose > 0:
                    print("gpnts same?")
                    print(gpnts)
                    print(og_gpnts)
                    print(kpt_shifts[kpt])
                    print(sorted_symg)

                    print(shift)
                    print("check if coefficients are same!!",coeff[5],red_coeff.T[5])
                    print("phase:",(coeff/red_coeff.T)[5])
                    print(is_opposite[kpt])
                    print(sym_opt)
                    print("diff coeff:",diff_coeff)
                    print("sym_opt: \n", sym_opt)
                    print("max abs diff:",np.amax(np.abs(diff_coeff)))
                    print("max diff position:", np.unravel_index(np.argmax(np.abs(diff_coeff)), diff_coeff.shape))
                    print("orb associated with max diff:",self.orbtype[np.unravel_index(np.argmax(np.abs(diff_coeff)), diff_coeff.shape)[1]])
                    print(self.orbtype)
                    print(np.shape(self.orbtype))
                    print(np.shape(diff_coeff))
                    for i in range(self.orbtype.shape[0]):
                        #if self.orbtype[i] == "f":
                        print(self.orbtype[i]," orb:",i,np.amax(np.abs(diff_coeff[:40,i])),np.argmax(np.abs(diff_coeff[:40,i])))

                all_reduc_eigvec[kpt] = coeff
                red_coeff = coeff.T

            all_recip_wfs.append(new_Cnkg)
            
            
            if self.verbose > 1:
                print(kpt,reduc_toir[kpt])
                print(ireigvals[reduc_toir[kpt]])
                print(sym_opt)
                print("old coeff:",ir_coeffs[:,5])
                print("new coeff:",red_coeff[:,5])
            
            #new_nrms[kpt] = old_nrms[reduc_toir[kpt]]
            new_spill[kpt] = old_spill[reduc_toir[kpt]]
            new_mixing[kpt] = old_mixing[reduc_toir[kpt]]
            new_max_spill[kpt] = old_max_spill[reduc_toir[kpt]]
            
            #coeff_inv = np.linalg.inv(red_coeff[:,:])#
            #use_coeff = red_coeff[:,:]
            #ogactual_orblap = np.matmul(np.conj(coeff_inv.transpose()), coeff_inv)
            
            bands_spill = old_spill[reduc_toir[kpt]]
            include_lowband = (((1 - bands_spill) > self.low_min_proj) | (self.eigval[reduc_toir[kpt]][:, 0] >= (self.efermi + self.energy_shift - 5)))  # only False for bad low energy bands
            include_bands = (include_lowband) & (bands_spill < (1.0-self.min_proj))
            coeff_mult = np.matmul(red_coeff[:,include_bands], np.conj(red_coeff.T)[include_bands])
            temp_Sij = np.conj(np.linalg.inv(coeff_mult)).T # ogactual_orblap#np.matmul(red_coeff,np.conj(red_coeff.T))
            #print("compare Sij:",temp_Sij,old_Sij[reduc_toir[kpt]])
            all_Sij[kpt] = temp_Sij
            #print(kpt)
            #print("old coeff:",ir_coeffs[:,5])
            #print("new coeff:",red_coeff[:,5])
            #print(temp_Sij)
            #eigenvalj, kdep_Dij = np.linalg.eigh(temp_Sij)# + np.conj(temp_Sij).T)/2)
            # check correctness of eigen
            # construct Aij
            #kdep_Aij = np.zeros((self.num_orbs, self.num_orbs), dtype=np.complex128)
            #for j in range(self.num_orbs):
            #    kdep_Aij[:, j] = kdep_Dij[:, j] / (eigenvalj[j]) ** (1 / 2)
            
            sqrt_overlap =  linalg.sqrtm(temp_Sij) 
            sqrt_overlap = np.array(sqrt_overlap, dtype=np.complex128)
            #print(coeff_overlap[:2])
            inv_sqrt_overlap = np.linalg.inv(sqrt_overlap)
            kdep_Aij = inv_sqrt_overlap
            
            all_Aij[kpt] = kdep_Aij
            
            
        # save old values
        self.irred_num_kpts = self.num_kpts
        self.irred_kpoints = self.kpoints
        self.irred_mnkcoefficients = self.mnkcoefficients
        self.irred_eigval = self.eigval
        #self.mixed_eigval = all_reduc_newval
        self.irred_gpoints = self.gpoints
        #self.total_nrms = new_nrms
        self.irred_all_band_spillage = self.all_band_spillage
        self.irred_max_orb_mixing = self.max_orb_mixing
        self.irred_recip_WFcoeffs = self.recip_WFcoeffs
        self.irred_kpt_weights = self.kpt_weights
        self.irred_Sij = self.Sij
        self.irred_Aij = self.Aij
        
        # save all the values
        self.num_kpts = num_reduc
        self.kpoints = reduc_kpts
        self.mnkcoefficients = all_reduc_eigvec
        self.eigval = all_reduc_eigval
        #self.mixed_eigval = all_reduc_newval
        self.gpoints = all_gpoints
        #self.total_nrms = new_nrms
        self.all_band_spillage = new_spill
        self.max_orb_mixing = new_mixing
        self.max_band_spilling = new_max_spill
        self.recip_WFcoeffs = all_recip_wfs
        self.kpt_weights = np.ones(self.num_kpts)/self.num_kpts
        
        # reget all the overlap info
        #print(self.num_kpts)
        #print(self.kpoints)
        #print(len(self.mnkcoefficients))
        #self.get_ae_overlap_info()
        
        #self.optimize_band_set()
        self.Sij = all_Sij
        self.Aij = all_Aij
        #print("check overlaps:")
        #print(self.Sij[0],all_Sij[0])
        #print(self.Sij[10],all_Sij[10])
        #print(self.Sij[-1],all_Sij[-1])
        
        #np.set_printoptions(suppress=True,precision=8)
        
    def symmetrize_orbs(self,recip_orbs):
        """
        This function is to symmetrize the iterated orbitals to decrease orbital mixing by ensuring orbital has s orbital symmetry.
        """
        
        ## -- 2 -- ##
        # get arrays for the p and d orbitals
        #print("orbtype!",self.exactorbtype)
        # get unique orbital types
        uniq_orbs,inverse = np.unique(self.orbtype,return_inverse=True)
        #print("unique orbs:",uniq_orbs,inverse)
        orb_index = {}
        atm_has_orb = {}
        orbatomnum = np.array(self.orbatomnum)
        for ind, orb in enumerate(uniq_orbs):
            orb_index[orb] = np.sort(np.arange(self.num_orbs)[inverse==ind])
            #print(ind,orb,orb_index[orb])
            atm_has_orb[orb] = np.unique(orbatomnum[orb_index[orb]])
        
        old_gpoints = self.gpoints[0]
        
        
        from pymatgen.core import Structure
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
        # get symmetry operations
        structure = Structure.from_file(self.directory + 'POSCAR')
        # pass "structure" to define class into "space"
        space = SpacegroupAnalyzer(structure,symprec=self.sym_prec,angle_tolerance=-1.0)
        point_groups = space.get_symmetry_operations()#get_point_group_operations()
        cart_point_groups = space.get_symmetry_operations(cartesian=True)#get_point_group_operations(cartesian=True)
        operations = np.zeros((len(point_groups),4,4),dtype=np.float64)
        cart_operations = np.zeros((len(point_groups),4,4),dtype=np.float64)
        for ind in range(len(point_groups)):
            operations[ind] = point_groups[ind].affine_matrix #[:3,:3]
            cart_operations[ind] = cart_point_groups[ind].affine_matrix #[:3,:3]
        #print("check sym:",operations)
        #print(cart_operations)
        
        
        #transform to the reducible kpts
        for opti in range(len(operations)):
            # get the eigenvecs the normal way
            cart_opt = cart_operations[opti]
            full_opt = cart_opt[:3,:3].T #np.linalg.inv()
            prim_opt = np.linalg.inv(operations[opti][:3,:3])#.T
            prim_shift = operations[opti][:3,3]
            sym_opt = full_opt
            shift = cart_opt[:3,3]
            
            # game plan:
            # figure out sign of p orbital now cause they don't depend on atom
            # then go through each atom, apply inv sym opt to it, and translate it back into the prim cell
            # finally, assign the atom the coefficients of the atom it got translated to with the right sign

            #print("d_index",d_index)
            #print(has_d)
            # loop over atoms with p orbitals
            old_orbs = []
            new_orbs = []
            old_pos = copy.deepcopy(self.orbpos)
            orb_trans = np.zeros(self.orbpos.shape)
            orb_transfer_matrix = np.zeros((self.num_orbs,self.num_orbs)) # will be real #[new orb, old orb]
                
            for atm in range(self.numAtoms):
                #print(atm)
                center = self.primAtoms[atm] - prim_shift
                new_center = np.matmul(prim_opt,center)# - prim_shift
                og_center = np.around(new_center,decimals=10)#_cart_to_red((self._a[0], self._a[1], self._a[2]), [new_center])[0],decimals=8)
                #print("first:",prim_center)
                prim_center =  copy.deepcopy(og_center)
                while (prim_center>=1).any() or (prim_center<0).any():
                    translate = np.zeros(3)
                    translate[prim_center>=1] = -1
                    translate[prim_center<0] = 1
                    prim_center = prim_center + translate
                translate = prim_center-og_center
                final_center = prim_center
                diff_atoms = self.primAtoms - final_center
                dist_atoms = np.amin([np.linalg.norm(diff_atoms,axis=1), np.abs(np.linalg.norm(diff_atoms,axis=1)-1),np.abs(np.linalg.norm(diff_atoms,axis=1)-2**(1/2)),np.abs(np.linalg.norm(diff_atoms,axis=1)-3**(1/2))],axis=0)
                old_atom = np.argmin(np.abs(dist_atoms))
                #print(og_center,prim_center,translate)
                if np.sum(np.abs(dist_atoms)[old_atom]) > self.sym_prec:
                    print("WARNING: did not find an actual atom!!!",dist_atoms,old_atom)
                #print(dist_atoms,old_atom)
                
                if self.verbose > 1:
                    if atm != old_atom:
                        print(self.primAtoms[atm],final_center)
                        print("new and old atm indices:",atm,old_atom)
                
                for orb_type in uniq_orbs:
                    has_orb = atm_has_orb[orb_type]
                    orbind = orb_index[orb_type]
                    if (has_orb == atm).any():
                        if "s" in orb_type:
                            #set s orbitals
                            relative_atm = np.arange(len(has_orb))[has_orb==atm][0]
                            relative_oldatm = np.arange(len(has_orb))[has_orb==old_atom][0]
                            old_s = orbind[relative_oldatm]
                            new_s = orbind[relative_atm]
                            #red_coeff[new_s] = ir_coeffs[old_s]
                            orb_transfer_matrix[new_s,old_s] = 1
                            old_orbs.append(old_s)
                            new_orbs.append(new_s)
                            old_pos[new_s] = self.orbpos[old_s]
                            orb_trans[new_s] = translate
                        elif "p" in orb_type:
                            old_patm = np.arange(len(has_orb))[has_orb==old_atom][0]
                            new_patm = np.arange(len(has_orb))[has_orb==atm][0]

                            old_pzpxpy = orbind[old_patm*3:(old_patm+1)*3]
                            new_pzpxpy = orbind[new_patm*3:(new_patm+1)*3]

                            #old_pcoeff = np.array([ir_coeffs[old_pzpxpy[1]],ir_coeffs[old_pzpxpy[2]],ir_coeffs[old_pzpxpy[0]]],dtype=np.complex128)
                            #new_pcoeff = np.matmul(sym_opt,old_pcoeff)
                            #print("check sym:", sym_opt)
                            #print(old_pcoeff[:,0],new_pcoeff[:,0])
                            #print("setting p:",new_pzpxpy,old_pzpxpy,orb_transfer_matrix[new_pzpxpy][:,old_pzpxpy],sym_opt[[2,0,1]][:,[2,0,1]])
                            #print(orb_transfer_matrix[new_pzpxpy,old_pzpxpy])
                            hold_matrix = orb_transfer_matrix[new_pzpxpy]
                            hold_matrix[:,old_pzpxpy] = sym_opt[[2,0,1]][:,[2,0,1]]
                            orb_transfer_matrix[new_pzpxpy] = hold_matrix
                            #print(orb_transfer_matrix[new_pzpxpy][:,old_pzpxpy])
                            #red_coeff[new_pzpxpy[0]] = new_pcoeff[2] #pz
                            #red_coeff[new_pzpxpy[1]] = new_pcoeff[0] #px
                            #red_coeff[new_pzpxpy[2]] = new_pcoeff[1] #py
                            
                            old_pos[new_pzpxpy] = self.orbpos[old_pzpxpy]
                            orb_trans[new_pzpxpy] = translate
                        elif "d" in orb_type:
                            
                            old_datm = np.arange(len(has_orb))[has_orb==old_atom][0]
                            new_datm = np.arange(len(has_orb))[has_orb==atm][0]
                            old_z2_xz_yz_x2y2_xy = orbind[old_datm*5:(old_datm+1)*5]
                            new_z2_xz_yz_x2y2_xy = orbind[new_datm*5:(new_datm+1)*5]
                            
                            sym = sym_opt.T # just need transpose cause I wrote all the [i,j] as [j,i] and don't feel like rewriting it
                    
                            # for x^2 - y^2
                            #x2y2_xi2 = sym[0,0]*sym[0,0]
                            #print(sym[0,0]*sym[0,0],sym[1,0]*sym[1,0],sym[0,1]*sym[0,1],sym[1,1]*sym[1,1])
                            #weirdval = ((sym[0,0]*sym[0,0] - sym[1,0]*sym[1,0]) - (sym[0,1]*sym[0,1]-sym[1,1]*sym[1,1]))/2
                            x2y2_xi2yi2 = ((sym[0,0]*sym[0,0] - sym[1,0]*sym[1,0]) - (sym[0,1]*sym[0,1] - sym[1,1]*sym[1,1]))/2 
                            x2y2_zi2 = ((2*sym[2,0]*sym[2,0]-sym[1,0]*sym[1,0]-sym[0,0]*sym[0,0]) - (2*sym[2,1]*sym[2,1]-sym[1,1]*sym[1,1]-sym[0,1]*sym[0,1]))/2/3**(1/2)#**(1/2)
                            x2y2_xiyi = (2*sym[0,0]*sym[1,0] - 2*sym[0,1]*sym[1,1])/2#**(1/2)
                            x2y2_yizi = (2*sym[1,0]*sym[2,0] - 2*sym[1,1]*sym[2,1])/2#**(1/2)
                            x2y2_zixi = (2*sym[2,0]*sym[0,0] - 2*sym[2,1]*sym[0,1])/2#**(1/2)

                            # for z^2
                            z2_xi2yi2 = (2*(sym[0,2]*sym[0,2] - sym[1,2]*sym[1,2]) - (sym[0,1]*sym[0,1] - sym[1,1]*sym[1,1]) - (sym[0,0]*sym[0,0] - sym[1,0]*sym[1,0]))/2/3**(1/2)#2/2**(1/2)
                            #z2_yi2 = sym[1,2]*sym[1,2]
                            z2_zi2 = (2*(sym[2,2]*sym[2,2]-sym[0,2]*sym[0,2]-sym[1,2]*sym[1,2]) - (sym[2,0]*sym[2,0]-sym[0,0]*sym[0,0]-sym[1,0]*sym[1,0]) - (sym[2,1]*sym[2,1]-sym[0,1]*sym[0,1]-sym[1,1]*sym[1,1]))/4
                            z2_xiyi = (2*(2*sym[0,2]*sym[1,2])-(2*sym[0,0]*sym[1,0])-(2*sym[0,1]*sym[1,1]))/2/(3)**(1/2)
                            z2_yizi = (2*(2*sym[1,2]*sym[2,2])-(2*sym[1,0]*sym[2,0])-(2*sym[1,1]*sym[2,1]))/2/(3)**(1/2)
                            z2_zixi = (2*(2*sym[2,2]*sym[0,2])-(2*sym[2,0]*sym[0,0])-(2*sym[2,1]*sym[0,1]))/2/(3)**(1/2)

                            # for xy
                            xy_xi2yi2 = (sym[0,0]*sym[0,1] - sym[1,0]*sym[1,1])#/2**(1/2)
                            #xy_yi2 = sym[1,0]*sym[1,1]
                            xy_zi2 = (2*sym[2,0]*sym[2,1]-sym[1,0]*sym[1,1]-sym[0,0]*sym[0,1])/(3)**(1/2)
                            xy_xiyi = sym[0,0]*sym[1,1]+sym[1,0]*sym[0,1]
                            xy_yizi = sym[1,0]*sym[2,1]+sym[2,0]*sym[1,1]
                            xy_zixi = sym[2,0]*sym[0,1]+sym[0,0]*sym[2,1]
                            # for yz
                            yz_xi2yi2 = (sym[0,1]*sym[0,2] - sym[1,1]*sym[1,2])#/2**(1/2)
                            #yz_yi2 = sym[1,1]*sym[1,2]
                            yz_zi2 = (2*sym[2,1]*sym[2,2]-sym[1,1]*sym[1,2]-sym[0,1]*sym[0,2])/(3)**(1/2)
                            yz_xiyi = sym[0,1]*sym[1,2]+sym[1,1]*sym[0,2]
                            yz_yizi = sym[1,1]*sym[2,2]+sym[2,1]*sym[1,2]
                            yz_zixi = sym[2,1]*sym[0,2]+sym[0,1]*sym[2,2]
                            # for zx
                            zx_xi2yi2 = (sym[0,2]*sym[0,0] - sym[1,2]*sym[1,0])#/2**(1/2)
                            #xz_yi2 = sym[1,0]*sym[1,2]
                            zx_zi2 = (2*sym[2,2]*sym[2,0]-sym[1,2]*sym[1,0]-sym[0,2]*sym[0,0])/(3)**(1/2)
                            zx_xiyi = sym[0,2]*sym[1,0]+sym[1,2]*sym[0,0]
                            zx_yizi = sym[1,2]*sym[2,0]+sym[2,2]*sym[1,0]
                            zx_zixi = sym[2,2]*sym[0,0]+sym[0,2]*sym[2,0]
                            # order is z2, zx, yz, x2-y2, xy
                            d_switching = [[z2_zi2, z2_zixi, z2_yizi, z2_xi2yi2, z2_xiyi],
                                           [zx_zi2, zx_zixi, zx_yizi, zx_xi2yi2, zx_xiyi],
                                           [yz_zi2, yz_zixi, yz_yizi, yz_xi2yi2, yz_xiyi],
                                           [x2y2_zi2, x2y2_zixi, x2y2_yizi, x2y2_xi2yi2, x2y2_xiyi],
                                           [xy_zi2, xy_zixi, xy_yizi, xy_xi2yi2, xy_xiyi]]#np.around(,decimals=15)
                            
                            #old_dcoeff = ir_coeffs[old_z2_xz_yz_x2y2_xy]
                            #print("d_switching:",d_switching)
                            #print(sym)
                            old_pos[new_z2_xz_yz_x2y2_xy] = self.orbpos[old_z2_xz_yz_x2y2_xy]
                            orb_trans[new_z2_xz_yz_x2y2_xy] = translate
                            
                            hold_matrix = orb_transfer_matrix[new_z2_xz_yz_x2y2_xy]
                            hold_matrix[:,old_z2_xz_yz_x2y2_xy] = d_switching
                            orb_transfer_matrix[new_z2_xz_yz_x2y2_xy] = hold_matrix
                            #new_dcoeff = np.matmul(d_switching,old_dcoeff)
                            #red_coeff[new_z2_xz_yz_x2y2_xy] = new_dcoeff
                            #if (old_z2_xz_yz_x2y2_xy != new_z2_xz_yz_x2y2_xy).any():
                            #    print("d coeff:",old_dcoeff[:,0],new_dcoeff[:,0])


            # also include the kpt shift
            #print("check old and new orb pos:",old_pos,self.orbpos)
            #phase_shift = np.exp(-2j*np.pi*np.dot(kpt_shifts[kpt],(self.orbpos).T))
            #print("phase shift:",phase_shift)
            #red_coeff = (red_coeff.T*phase_shift).T
            
            #gpnts = self.generate_gpnts([0,0,0])
            #print("gen gpnts:",gpnts)
            
            # calculate the G coefficients of wavefunction at kpoint k2 from the IBZ k1 wavefunction
            Gcoeff1 = np.array(recip_orbs)  # [old orb,igpnt]
            og_gpnts = np.array(old_gpoints) # [igpnt,b1/b2/b3]
            prim_opt = np.linalg.inv(operations[opti][:3,:3]).T # symrec
            prim_shift = operations[opti][:3,3]
            phase = np.exp(2j*np.pi*np.dot(og_gpnts, prim_shift))#k2[None,:]+og_gpnts
            Gcoeff1 = phase[None,:]*Gcoeff1
            
            # generate from irred gpnts
            shift_gpnts = og_gpnts 
            sym_gpnts = np.matmul(prim_opt,shift_gpnts.T).T
            max_vals = np.amax(np.abs(sym_gpnts),axis=0)
            sort_gpnts = copy.deepcopy(sym_gpnts)
            sort_gpnts[sym_gpnts[:,0]<0] = sort_gpnts[sym_gpnts[:,0]<0] + np.array([[max_vals[0]*2 + 1,0,0]])
            sort_gpnts[sym_gpnts[:,1]<0] = sort_gpnts[sym_gpnts[:,1]<0] + np.array([[0,max_vals[1]*2 + 1,0]])
            sort_gpnts[sym_gpnts[:,2]<0] = sort_gpnts[sym_gpnts[:,2]<0] + np.array([[0,0,max_vals[2]*2 + 1]])
            
            ind = np.lexsort((sort_gpnts[:,0],sort_gpnts[:,1],sort_gpnts[:,2])) # index which converts the symmetrized gpnts to the normal gpnt ordering
            #print(ind.shape)
            sorted_symg = sym_gpnts[ind]
            #old_Cnkg = np.array(old_recip_WFs[reduc_toir[kpt]])  # [band,gpoint]
            #new_Cnkg = old_Cnkg[:,ind] * np.exp(-2j*np.pi* np.dot(gpnts, shift))
            new_Cnkg = Gcoeff1[:,ind] # [new orb, igpnt]
            
            # calculate 
            '''
            adjusted_orbs = self.get_kdep_reciporbs(reduc_kpts[kpt],full_kpt =True,gpnts=gpnts,set_gpnts=True)
            ae_overlap = self.get_ae_overlap_info(adjusted_orbs,kpt=reduc_kpts[kpt], just_ae_overlap = True,full_kpt=True)
            coeff = self.get_aecoefficients(adjusted_orbs, new_Cnkg, ae_overlap, kpt=reduc_kpts[kpt],recip=True,full_kpt=True)
            
            # also include time symmetry
            if is_opposite[kpt]:
                if self.verbose > -1:
                    print("at time sym point!",kpt)
                coeff = np.conj(coeff)
            
            diff_coeff = coeff-red_coeff.T
            if np.sum(np.abs(diff_coeff)) > 0.01 and self.verbose > 1:
                print("gpnts same?")
                print(gpnts)
                print(kpt_shifts[kpt])
                print(sorted_symg)
                
                print(shift)
                print("check if coefficients are same!!",coeff[5],red_coeff.T[5])
                print("phase:",(coeff/red_coeff.T)[5])
                print(is_opposite[kpt])
                print(sym_opt)
            
            #all_reduc_eigvec[kpt] = coeff
            #red_coeff = coeff.T
            '''
            #print("transfer matrix:",orb_transfer_matrix.T)
            same_Gcoeff = np.matmul(orb_transfer_matrix.T, new_Cnkg)
            #print(recip_orbs,same_Gcoeff,new_Cnkg)
            recip_orbs = (recip_orbs + same_Gcoeff)/2
        
        return recip_orbs
            
        

    def get_Qab(self):
        #need to change this to just read it from the POTCAR!!!
        num_projs = self.num_orbs + self.num_exorbs
        all_sphharm = np.append(self.sph_harm_key,self.ex_sph_harm_key)
        all_orbtype = np.append(self.orbtype, self.ex_orbtype)
        all_orbatm = np.append(self.orbatomnum, self.exorbatomnum)
        Qab = np.zeros((num_projs,num_projs))
        
        for orb1 in range(num_projs):
            sphharm1 = all_sphharm[orb1]
            atm = all_orbatm[orb1]
            atmkey = self.elements[atm]
            nonzero = (all_sphharm == sphharm1) & (all_orbatm == atm) # only nonzero if the orbs are on the same atom and have the same l and ml quantum nums
            orbs2 = np.arange(num_projs,dtype=np.int_)[nonzero]
            #print(orbs2)
            for orb2 in orbs2:
                type1 = all_orbtype[orb1]
                type2 = all_orbtype[orb2]
                grid = self.atmPsuedoAeData[atmkey][0]
                # for valence state
                psuedoWF = self.atmPsuedoAeData[atmkey][1][type1] * self.atmPsuedoAeData[atmkey][1][type2]
                aeWF = self.atmPsuedoAeData[atmkey][2][type1] * self.atmPsuedoAeData[atmkey][2][type2]
                psuedoint = integrate.trapezoid((psuedoWF) * grid ** 2,x=grid)
                aeint = integrate.trapezoid((aeWF) * grid ** 2, x=grid)
                Qab[orb1, orb2] = (aeint - psuedoint) #* (1+np.random.rand(1)/1000)
        self.Qab = Qab
        if self.verbose > 1:
            #for i in range(len(Qab)):
            print("Qab:",Qab)
        
        #print("Qab:",Qab)

    def get_ae_overlap_info(self,orbs=None,kpt=0,just_ae_overlap = False,recip=False,test=False,full_kpt=False, gpnts=None,set_gpnts=False):
        try:
            if orbs == None:
                return_value = False
                recip_orbs = self.get_kdep_reciporbs(kpt,full_kpt=full_kpt,gpnts=gpnts,set_gpnts=set_gpnts)
            else:
                return_value = True
                recip_orbs = orbs #better be passing rreciprocal space orbitals!
        except:
            return_value = True
            recip_orbs = orbs
        
        kpt = kpt
        recip_proj = self.get_kdep_recipprojs(kpt,full_kpt=full_kpt,gpnts=gpnts,set_gpnts=set_gpnts)

        # find orbital projector overlap (overlap_OrbProj) and ae-psuedo diff (Qab)
        if self.haveQab == False:
            self.get_Qab()
            self.haveQab = True
        Qab = self.Qab
        
        #print(recip_proj.keys())
        projector_orbital_overlap = self.get_overlap_matrix(recip_proj,recip_orbs,recip=True)
        #print("check if orbs orth:", orb_psuedo_overlap[:5], orb_psuedo_overlap[22:])
        #print(Qab)
        #if self.verbose > 1:
        #    #print("proj-orb overlap:", projector_orbital_overlap[:11,:6])
        #    print("proj-orb overlap:",projector_orbital_overlap)

        # construct the overlap matrix for ae orbitals
        psuedo_overlap_mt = self.get_overlap_matrix(recip_orbs,recip=True)
        #psuedo_overlap_mt_real = self.get_overlap_matrix(orth_orbs)
        if self.verbose >1:
            print("orbital overlaps!",psuedo_overlap_mt[:5,:5])#,psuedo_overlap_mt_real[:5,:5]
        
        num_projs = len(recip_proj)
        sum_overi = np.zeros((len(recip_orbs),len(recip_orbs)),dtype=np.complex128)
        count_val = np.arange((num_projs*num_projs))[Qab.flatten() != 0]
        all_orbatm = np.append(self.orbatomnum, self.exorbatomnum)
        orbatm =  np.array(self.orbatomnum)
        for ij in count_val:
            i = int(np.floor(ij/num_projs))
            j = ij%num_projs
            bool1 = orbatm == all_orbatm[i]
            bool2 = orbatm == all_orbatm[j]
            sum_overi += np.conj(projector_orbital_overlap[i, :,None]) * Qab[i,j]*projector_orbital_overlap[j,None, :]
            #sum_overi += np.conj(projector_orbital_overlap[i, :,None]) * Qab[i,j]*projector_orbital_overlap[j,None, :]
        #all_paw = np.conj(projector_orbital_overlap[:,None, :,None]) * Qab[:,:,None,None]*projector_orbital_overlap[None,:,None, :]
        #sum_overi = np.sum(all_paw,axis=(0,1))
        
        #print("before:",sum_overi)
        #orbatm =  np.array(self.orbatomnum)#np.append(self.orbatomnum, self.exorbatomnum)
        #orbatm1 = orbatm[:,None]*np.ones((self.num_orbs,self.num_orbs))
        #orbatm2 = orbatm[None,:]*np.ones((self.num_orbs,self.num_orbs))
        #print(orbatm1)
        #print(orbatm2)
        #print(orbatm1.flatten() != orbatm2.flatten())
        #hold = sum_overi.flatten()
        #hold[orbatm1.flatten() != orbatm2.flatten()] = 0
        #sum_overi = np.reshape(hold,(self.num_orbs,self.num_orbs))
        #print("after:",sum_overi)
        
        aeorb_overlap = psuedo_overlap_mt + sum_overi
        if self.verbose >1:
            print("check second", aeorb_overlap)

        if return_value == True:
            return aeorb_overlap
        else:
            self.aeorb_overlap = aeorb_overlap


        if test==True:
            # FIRST: try to just correctly orthonormalize the wavefunctions
            # get non normalized WFs
            kpt = 0
            print("kpoint:", self.kpoints[kpt])
            # kpoint = self.kpoints[kpt]
            test_normWF = np.zeros((self.max_band, self.gridxyz[0], self.gridxyz[1], self.gridxyz[2]),
                                   dtype=np.complex128)
            #if recip == False:
            #    test_normWF = self.periodic_DFTwf[kpt, :self.max_band]
            #    realDFTWF = self.periodic_DFTwf[kpt, :self.max_band]
            #else:
            test_normWF = self.recip_WFcoeffs[kpt][:self.max_band]
            #realDFTWF = self.periodic_DFTwf[kpt, :self.max_band]

            # construct the interproduct
            try_coeffs = self.mnkcoefficients[kpt]
            #if recip == False:
            #    orb_overlap = self.get_overlap_matrix(self.one_orbitalWF)
            #else:
            orb_overlap = self.get_overlap_matrix(recip_orbs, recip=True)
            for band1 in range(self.max_band):
                for band2 in range(self.max_band):
                    wf1 = test_normWF[band1]
                    wf2 = test_normWF[band2]

                    psuedo_overlap = reciprocal_integral(np.conj(wf1) * wf2)

                    psuedo_overlap_coeff = np.matmul(np.conj(try_coeffs[band1]), np.matmul(orb_overlap, try_coeffs[band2].transpose()))

                    sum = 0
                    sum_coeff = 0
                    for i in range(self.num_orbs * 2):
                        keys = list(recip_proj.keys())
                        keyi = keys[i]
                        orb1 = int(np.floor(i / 2))
                        projector = recip_proj[keyi] # real space distorted projector
                        proj_overlap1 = reciprocal_integral(np.conj(wf1) * projector)#, self._a, self.gridxyz)
                        proj_overlap1_coeff = np.sum(np.conj(try_coeffs[band1]) * projector_orbital_overlap[i, :])
                        # print("check proj_wf overlap:", proj_overlap1,proj_overlap1_coeff)
                        for j in range(self.num_orbs * 2):
                            qij = Qab[i, j]
                            if abs(qij) > 0.000001:
                                keyj = keys[j]
                                orb2 = int(np.floor(j / 2))
                                projector = recip_proj[keyj] #* expon_term[orb2]
                                proj_overlap2 = reciprocal_integral(np.conj(projector) * wf2)#, self._a, self.gridxyz)
                                proj_overlap2_coeff = np.sum(try_coeffs[band2] * projector_orbital_overlap[j, :])
                                # print("check proj_wf overlap:", proj_overlap2, proj_overlap2_coeff)
                                toadd = proj_overlap1 * qij * proj_overlap2
                                toadd_coeff = proj_overlap1_coeff * qij * proj_overlap2_coeff
                                # if abs(toadd) > 0.000000000001:
                                #   print(i,j,toadd)
                                sum += toadd
                                sum_coeff += toadd_coeff
                    ae_overlap = psuedo_overlap + sum
                    ae_overlap_coeff = psuedo_overlap_coeff + sum_coeff
                    # if abs(psuedo_overlap) > 0.00005:
                    if band1 == band2:
                        print("bands:", band1, band2, ae_overlap, psuedo_overlap, sum)
                        print("bands:", band1, band2, ae_overlap_coeff, psuedo_overlap_coeff, sum_coeff)

        if just_ae_overlap == False:
            # construct matrix Aij to convert eigenvector from nonorthogonal to orthogonal basis set

            self.Aij = np.zeros((self.num_kpts, self.num_orbs, self.num_orbs), dtype=np.complex128)
            self.Sij = np.zeros((self.num_kpts, self.num_orbs, self.num_orbs), dtype=np.complex128)
            self.Sij_pseudo = np.zeros((self.num_kpts, self.num_orbs, self.num_orbs), dtype=np.complex128)

            #print load compare bar
            print("getting ae overlaps")
            for kpt in range(self.num_kpts):
                print("-",end="")
            print("")
            for kpt in range(self.num_kpts):
                #print(kpt)
                print("-",end="")
                # constuct the adjusted orbitals
                krecip_orb = self.get_kdep_reciporbs(kpt)
                krecip_proj = self.get_kdep_recipprojs(kpt)

                # construct k-dependant Sij and Aij, only different if psuedo basis orbital are not orthogonal
                # have to do the whole psuedo->ae PAW procedure :(
                # construct proj-orb overlap
                proj_orb_overlap = self.get_overlap_matrix(krecip_proj,krecip_orb,recip=True)
                '''
                # only do projectors that have the same center angular momentum as the orbital
                num_projs = self.num_orbs + self.num_exorbs
                all_sphharm = np.append(self.sph_harm_key,self.ex_sph_harm_key)
                all_orbtype = np.append(self.orbtype, self.ex_orbtype)
                all_orbatm = np.append(self.orbatomnum, self.exorbatomnum)
                mtrx_orbatm1 = all_orbatm[:,None]*np.ones((num_projs,self.num_orbs))
                mtrx_orbatm2 = np.array(self.orbatomnum)[None,:]*np.ones((num_projs,self.num_orbs))
                print(mtrx_orbatm1,mtrx_orbatm2)
                is_zero = mtrx_orbatm1.flatten() != mtrx_orbatm2.flatten()
                holdoverlap = proj_orb_overlap.flatten()
                holdoverlap[is_zero] = 0
                proj_orb_overlap = np.reshape(holdoverlap,(num_projs,self.num_orbs))
                '''
                #orb1 = orbgroup[0] # each orbital in the group should have the same norm
                #sphharm1 = all_sphharm[orb1]
                #atm = all_orbatm[orb1]
                #nonzero = (all_sphharm == sphharm1) & (all_orbatm == atm) # only nonzero if the orbs are on the same atom and have the same l and ml quantum nums
                #orbs = np.arange(num_projs,dtype=np.int_)[nonzero]

                # now get the ae overlap between orbitals
                psuedo_overlap_matrix = self.get_overlap_matrix(krecip_orb, recip=True)
                self.Sij_pseudo[kpt] = psuedo_overlap_matrix
                
                
                num_projs = len(krecip_proj)
                sum_overi = np.zeros((len(krecip_orb),len(krecip_orb)),dtype=np.complex128)
                count_val = np.arange((num_projs*num_projs))[(Qab.flatten() != 0)]
                
                all_orbatm = np.append(self.orbatomnum, self.exorbatomnum)
                orbatm =  np.array(self.orbatomnum)
                for ij in count_val:
                    i = int(np.floor(ij/num_projs))
                    j = ij%num_projs
                    bool1 = orbatm == all_orbatm[i]
                    bool2 = orbatm == all_orbatm[j]
                    sum_overi += np.conj(proj_orb_overlap[i, :,None]) * Qab[i,j]*proj_orb_overlap[j,None, :]
                #all_paw = np.conj(proj_orb_overlap[:,None, :,None]) * Qab[:,:,None,None]*proj_orb_overlap[None,:,None, :]
                #sum_overi = np.sum(all_paw,axis=(0,1))
                kdep_Sij = psuedo_overlap_matrix + sum_overi
                kdep_Sij = (kdep_Sij + np.conj(kdep_Sij).T)/2 # ensure that kdep_Sij is Hermitian
                self.Sij[kpt] = kdep_Sij# np.around(kdep_Sij,decimals=6)
                #eigenvalj, kdep_Dij = np.linalg.eigh(kdep_Sij)
                # check correctness of eigen
                # construct Aij
                #kdep_Aij = np.zeros((self.num_orbs, self.num_orbs), dtype=np.complex128)
                #for j in range(self.num_orbs):
                #    kdep_Aij[:, j] = kdep_Dij[:, j] / (eigenvalj[j]) ** (1 / 2)
                
                sqrt_overlap =  linalg.sqrtm(kdep_Sij) 
                sqrt_overlap = np.array(sqrt_overlap, dtype=np.complex128)
                #print(coeff_overlap[:2])
                inv_sqrt_overlap = np.linalg.inv(sqrt_overlap)
                kdep_Aij = inv_sqrt_overlap
                
                self.Aij[kpt] = kdep_Aij #np.around(kdep_Aij,decimals=6)
            
            print("\n")
            #print("Sij:",self.Sij)
            #print("Aij:",self.Aij)

    def get_orth_coeffs(self,coeff,kpoint=None):
        #coeff is ordered with [band][orb]
        #kpoint: reduced coordinates of kpoint for coeff
        #print("check psuedo int:",self.psuedoint)
        #print(self.psuedo_overlap)

        #actually get the stuff
        if kpoint == None: #passing the entire coeff array
            num_kpt = self.num_kpts
        else:
            num_kpt = 1
            coeff = [coeff]

        new_coeff = np.zeros(coeff.shape, dtype=np.complex128)
        #finally normalize/orthogonalize the coefficient!
        for kpt in range(num_kpt):
            kdep_Aij = self.Aij[kpt]
            new_coeff[kpt] = np.matmul(np.linalg.inv(kdep_Aij),coeff[kpt].transpose()).transpose()

        if kpoint != None:
            new_coeff = new_coeff[0]
        else:
            self.aecoefficients = new_coeff
        #print("check coeffs for gamma:",coeff[0][:3],new_coeff[0][:3],coeff[0][18:21],new_coeff[0][18:21])

        return new_coeff

    def get_hamiltonian(self, kpoints=None):
        if kpoints == None:
            kpoints = range(self.num_kpts)
        #self.get_WFdata_fromPOT() #get data to run convert_psuedo_toae

        self.Sij_coeffs = np.zeros((self.num_kpts, self.num_orbs, self.num_orbs), dtype=np.complex128)
        self.hamilton = np.zeros((self.num_orbs, self.num_orbs, len(kpoints)), dtype=np.complex128)


        wan_orbs = {}
        for orb in range(self.num_orbs):
            wan_orbs[orb] = np.array([],dtype=np.complex128)

        all_gplusk_x = np.array([])
        all_gplusk_y = np.array([])
        all_gplusk_z = np.array([])
        all_gvecs_x = np.array([])
        all_gvecs_y = np.array([])
        all_gvecs_z = np.array([])
        
        for kpt in kpoints:
            bands_spill = self.all_band_spillage[kpt]
            include_lowband = (((1 - bands_spill) > self.low_min_proj) | (self.eigval[kpt][:, 0] >= (self.efermi + self.energy_shift - 5)))  # only False for bad low energy bands
            include_bands = (include_lowband) & (bands_spill < (1.0-self.min_proj)) #0.9
            incld_spill = bands_spill[include_bands]
            nonorth_coeff = copy.deepcopy(self.mnkcoefficients[kpt][include_bands])
            energ_weight = np.ones(len(np.arange(len(include_bands))[include_bands]))
            energ_weight[incld_spill>0.3] = (1-incld_spill)[incld_spill>0.3]
            energies = self.eigval[kpt][include_bands][:,0]#*energ_weight

            kdep_Aij = self.Aij[kpt]
            orth_coeff = np.matmul(np.linalg.inv(kdep_Aij),nonorth_coeff.transpose()).transpose()
            loworth_coeff = np.array(orth_coeff).transpose()   # want columns to be eigenvectors
            AtHA = np.matmul(loworth_coeff, np.matmul(np.diag(energies), #self.mixed_eigval[kpt], 
                                                     np.conj(loworth_coeff).T))#np.diag(self.eigval[kpt, :self.num_orbs,0])
            #AtHA = (AtHA + np.conj(AtHA).T)/2 # ensure that AtHA is Hermitian

            #need to make it the Hamiltonian for the generalized eigenvalue problem!!
            conj_Aij = np.conj(self.Aij[kpt]).transpose()
            self.hamilton[:, :, kpt] = np.matmul(np.linalg.inv(conj_Aij),np.matmul(AtHA,np.linalg.inv(self.Aij[kpt])))

            '''
            # get the Cogitwo basis
            DFT_wavefunc = self.recip_WFcoeffs[kpt]
            
            #orth_orig = np.matmul(np.linalg.inv(kdep_Aij),og_nonorthcoeff.transpose()).transpose()
            #orig_coeff = np.matmul(kdep_Aij,orth_orig.transpose()).transpose()
            #inv_coeff = np.matmul(np.conj(orth_orig),np.linalg.inv(kdep_Aij)).transpose() #np.linalg.inv(og_nonorthcoeff)#nonorth_coeff)#og_nonorthcoeff)  # now should be [orb, band]
            #inv_coeff = np.linalg.inv(og_nonorthcoeff)
            band_coeff = np.matmul(np.conj(nonorth_coeff),self.Sij[kpt])
            for orb in range(self.num_orbs):
                cur_sum = DFT_wavefunc[0] * 0
                for band in range(len(band_coeff)):  # fine to not have actually band because only care about indices of coeff and extra wf
                    # way to converge immediately
                    cur_sum += band_coeff[band][orb] * DFT_wavefunc[band]
                wan_orbs[orb] = np.append(wan_orbs[orb], cur_sum)
            gvecs = np.array(self.gpoints[kpt])
            gplusk_coords = (gvecs.transpose() + np.array([self.kpoints[kpt]]).transpose())
            all_gplusk_x = np.append(all_gplusk_x, gplusk_coords[0])
            all_gplusk_y = np.append(all_gplusk_y, gplusk_coords[1])
            all_gplusk_z = np.append(all_gplusk_z, gplusk_coords[2])

            all_gvecs_x = np.append(all_gvecs_x, gvecs[:, 0])
            all_gvecs_y = np.append(all_gvecs_y, gvecs[:, 1])
            all_gvecs_z = np.append(all_gvecs_z, gvecs[:, 2])
            '''
        '''
        # check the Cogitwo basis
        print("close to cogitwo basis!!")
        print(len(wan_orbs[0]), len(all_gvecs_x), len(all_gvecs_y), len(all_gvecs_z))
        # try sorting
        arg_sort = np.lexsort((all_gplusk_x, all_gplusk_y, all_gplusk_z))

        print(arg_sort)
        print(all_gplusk_x[arg_sort])
        sort_gvecs = np.array([all_gvecs_x[arg_sort], all_gvecs_y[arg_sort], all_gvecs_z[arg_sort]]).T
        sort_gplusk = np.array([all_gplusk_x[arg_sort], all_gplusk_y[arg_sort], all_gplusk_z[arg_sort]]).T
        cart_gplusk = _red_to_cart((self.b_vecs[0], self.b_vecs[1], self.b_vecs[2]),
                                   sort_gplusk)  # gplusk_coords.transpose())
        vec = cart_gplusk.transpose()  # point - np.array([atm]).transpose()
        recip_rad = np.linalg.norm(vec, axis=0)
        # print("check recip rad",kpoint,recip_rad)
        recip_rad[recip_rad == 0] = 0.000000001
        # print(rad.shape)
        recip_theta = np.arccos(vec[2] / recip_rad)
        recip_phi = np.arctan2(vec[1], vec[0])

        orb_to_l = [0, 1, 1, 1, 2, 2, 2, 2, 2]

        for orbind, orb in enumerate(range(self.num_orbs)):
            atom = self.orbatomnum[orbind]
            min_rad = recip_rad  # self.min_radphitheta[str(atom)][0]
            min_phi = recip_phi  # self.min_radphitheta[str(atom)][1]
            min_theta = recip_theta  # self.min_radphitheta[str(atom)][2]
            unk = wan_orbs[orb][arg_sort]  # unkorbs[orb].flatten()
            # fit the exponential decay
            orbperatom = self.sph_harm_key[orbind]
            angular_part = complex128funs(min_phi, min_theta, orbperatom)
            # replace the first value which is G=0,0,0 with (1/4/np.pi)**(1/2)
            # angular_part[0] = (1/4/np.pi)**(1/2)
            largeenough = np.abs(angular_part) > 0.01

            l = orb_to_l[orbperatom]
            print(l)
            center = self.primAtoms[atom]
            exp_term = np.exp(-2j * np.pi * np.dot(sort_gvecs, center))
            radunk = unk[largeenough] / angular_part[largeenough] / exp_term[largeenough] / (-1j) ** (l)
            
            # plot

            plt.scatter(min_rad[largeenough], radunk, c="black")
            new_rad = np.linspace(0,8,100)
            #plt.scatter(min_rad[big_ang],test_new[big_ang],facecolors="None",edgecolors="red")
            [ak, bk, ck, dk, ek, fk,gk,hk, l] = self.recip_orbcoeffs[orb]
            const = ((2 * np.pi) ** 3 / self.vol) ** (1 / 2)
            plt.plot(new_rad,func_for_rad(new_rad,ak,bk,ck,dk,ek,fk,gk,hk,l)*const,color="blue")
            # plt.plot(min_rad[sort_rad],rad_fit[sort_rad])
            # fig2.scatter(min_rad,unk[largeenough])
            print("showing plot")
            plt.show()
            plt.close()
            
            # convert to real orbital
            coeffs = unk/exp_term #recip_orbs[orb]
            recip_coeff = coeffs # np.zeros((len(keys),len(recip_orbs[keys[0]])), dtype=np.complex128)
            # get dimension of box
            kprim1 = np.sort(np.around(np.abs(self.kpoints[:,0]),decimals=6))
            kprim2 = np.sort(np.around(np.abs(self.kpoints[:,1]),decimals=6))
            kprim3 = np.sort(np.around(np.abs(self.kpoints[:,2]),decimals=6))
            trans1 = int(np.around(1/kprim1[kprim1 != 0][0],decimals=0)) if len(kprim1[kprim1 != 0]) > 0 else 1 
            trans2 = int(np.around(1/kprim2[kprim2 != 0][0],decimals=0)) if len(kprim2[kprim2 != 0]) > 0 else 1 
            trans3 = int(np.around(1/kprim3[kprim3 != 0][0],decimals=0)) if len(kprim3[kprim3 != 0]) > 0 else 1 
            
            all_mesh = np.zeros((self.gridxyz[0]*trans1,self.gridxyz[1]*trans2,self.gridxyz[2]*trans3), dtype=np.complex128)
            #for oi,orb in enumerate(keys):
            #    recip_coeff[oi] = recip_orbs[orb]
            int_gplusk = sort_gplusk * np.array([trans1,trans2,trans3])
            t0 = int_gplusk[:,0].astype(np.int_) + (self.gridxyz[0]*trans1 / 2).astype(np.int_)
            t1 = int_gplusk[:,1].astype(np.int_) + (self.gridxyz[1]*trans2 / 2).astype(np.int_)
            t2 = int_gplusk[:,2].astype(np.int_) + (self.gridxyz[2]*trans3 / 2).astype(np.int_)

            all_mesh[t0,t1,t2] = recip_coeff
            all_mesh = np.fft.ifftshift(all_mesh,axes=(0,1,2))
            allreal_orbs = np.fft.ifftn(all_mesh,axes=(0,1,2))
            print("wan shape:",allreal_orbs.shape)
            
            X,Y = np.mgrid[0:trans1:self.gridxyz[0]*trans1*1j,0:trans2:self.gridxyz[1]*trans2*1j]
            X[X>(trans1)/2] = X[X>(trans1)/2] - trans1
            Y[Y>(trans1)/2] = Y[Y>(trans1)/2] - trans1
            plotZ = allreal_orbs[:,:,0].real
            print(X.shape,Y.shape,plotZ.shape)
            
            
            select = np.arange(-40,0)
            select = np.append(select,np.arange(0,40))
            fig = go.Figure(data=[go.Surface(x=X[select][:,select], y=Y[select][:,select], z=plotZ[select][:,select])])

            fig.update_layout(title='Wannier orbital', autosize=False,
                              width=500, height=500,
                              margin=dict(l=65, r=50, b=65, t=90))

            fig.show()
        '''

        
        
        if self.verbose > 1:
            print("hamiltonian!:", self.hamilton[:,:,0],self.hamilton[:,:,4])

        return self.hamilton
    
    def make_fit_wannier(self):
        
        wan_orbs = {}
        for orb in range(self.num_orbs):
            wan_orbs[orb] = np.array([],dtype=np.complex128)

        all_gplusk_x = np.array([])
        all_gplusk_y = np.array([])
        all_gplusk_z = np.array([])
        all_gvecs_x = np.array([])
        all_gvecs_y = np.array([])
        all_gvecs_z = np.array([])
        
        for kpt in range(self.num_kpts):
            bands_spill = self.all_band_spillage[kpt]
            include_lowband = (((1 - bands_spill) > self.low_min_proj) | (self.eigval[kpt][:, 0] >= (self.efermi + self.energy_shift - 5)))  # only False for bad low energy bands
            include_bands = (include_lowband) & (bands_spill < (1.0-self.min_proj))
            nonorth_coeff = copy.deepcopy(self.mnkcoefficients[kpt][include_bands])
            # get the Cogitwo basis
            DFT_wavefunc = self.recip_WFcoeffs[kpt][include_bands]
            
            #orth_orig = np.matmul(np.linalg.inv(kdep_Aij),og_nonorthcoeff.transpose()).transpose()
            #orig_coeff = np.matmul(kdep_Aij,orth_orig.transpose()).transpose()
            #inv_coeff = np.matmul(np.conj(orth_orig),np.linalg.inv(kdep_Aij)).transpose() #np.linalg.inv(og_nonorthcoeff)#nonorth_coeff)#og_nonorthcoeff)  # now should be [orb, band]
            #inv_coeff = np.linalg.inv(og_nonorthcoeff)
            band_coeff = np.matmul(np.conj(nonorth_coeff),self.Sij[kpt])
            
            for orb in range(self.num_orbs):
                #cur_sum = DFT_wavefunc[0] * 0
                cur_sum = np.sum(band_coeff[:,orb,None] * DFT_wavefunc,axis=0)
                #for band in range(len(band_coeff)):  # fine to not have actually band because only care about indices of coeff and extra wf
                #    # way to converge immediately
                #    cur_sum += band_coeff[band][orb] * DFT_wavefunc[band]
                wan_orbs[orb] = np.append(wan_orbs[orb], cur_sum)
            gvecs = np.array(self.gpoints[kpt])
            gplusk_coords = (gvecs.transpose() + np.array([self.kpoints[kpt]]).transpose())
            all_gplusk_x = np.append(all_gplusk_x, gplusk_coords[0])
            all_gplusk_y = np.append(all_gplusk_y, gplusk_coords[1])
            all_gplusk_z = np.append(all_gplusk_z, gplusk_coords[2])

            all_gvecs_x = np.append(all_gvecs_x, gvecs[:, 0])
            all_gvecs_y = np.append(all_gvecs_y, gvecs[:, 1])
            all_gvecs_z = np.append(all_gvecs_z, gvecs[:, 2])

        # check the Cogitwo basis
        
        orbital_coeffs = np.zeros((self.num_orbs,9))
        recip_orbcoeffs = np.zeros((self.num_orbs,9))
        avg_radius = np.zeros(self.num_orbs)
        
        print("close to cogitwo basis!!")
        print(len(wan_orbs[0]), len(all_gvecs_x), len(all_gvecs_y), len(all_gvecs_z))
        # try sorting
        arg_sort = np.lexsort((all_gplusk_x, all_gplusk_y, all_gplusk_z))

        print(arg_sort)
        print(all_gplusk_x[arg_sort])
        sort_gvecs = np.array([all_gvecs_x[arg_sort], all_gvecs_y[arg_sort], all_gvecs_z[arg_sort]]).T
        sort_gplusk = np.array([all_gplusk_x[arg_sort], all_gplusk_y[arg_sort], all_gplusk_z[arg_sort]]).T
        cart_gplusk = _red_to_cart((self.b_vecs[0], self.b_vecs[1], self.b_vecs[2]),
                                   sort_gplusk)  # gplusk_coords.transpose())
        vec = cart_gplusk.transpose()  # point - np.array([atm]).transpose()
        recip_rad = np.linalg.norm(vec, axis=0)
        # print("check recip rad",kpoint,recip_rad)
        recip_rad[recip_rad == 0] = 0.0000000001
        recip_theta = np.arccos(vec[2] / recip_rad)
        recip_theta[recip_rad == 0.0000000001] = 0
        recip_phi = np.arctan2(vec[1], vec[0])

        orb_to_l = [0, 1, 1, 1, 2, 2, 2, 2, 2,3,3,3,3,3,3,3]
        centered_realwan = {}
        for orbind, orb in enumerate(range(self.num_orbs)):
            atom = self.orbatomnum[orbind]
            min_rad = recip_rad  # self.min_radphitheta[str(atom)][0]
            min_phi = recip_phi  # self.min_radphitheta[str(atom)][1]
            min_theta = recip_theta  # self.min_radphitheta[str(atom)][2]
            unk = wan_orbs[orb][arg_sort]  # unkorbs[orb].flatten()
            # fit the exponential decay
            orbperatom = self.sph_harm_key[orbind]
            angular_part = complex128funs(min_phi, min_theta, self.orb_vec[orbind])
            # replace the first value which is G=0,0,0 with (1/4/np.pi)**(1/2)
            # angular_part[0] = (1/4/np.pi)**(1/2)
            largeenough = np.abs(angular_part) > 0.01

            l = orb_to_l[orbperatom]
            print(l)
            center = self.primAtoms[atom]
            exp_term = np.exp(-2j * np.pi * np.dot(sort_gvecs, center))
            radunk = unk[largeenough] / angular_part[largeenough] / exp_term[largeenough] / (-1j) ** (l)
            
            '''
            grid = np.array([self.gridxyz])*7
            mesh = np.zeros(grid, dtype=np.complex128)
            for gp, coeff in zip(sort_gvecs, coeffs):
                t = tuple(gp.astype(np.int_) + (grid / 2).astype(np.int_))
                mesh[t] = coeff
            mesh = np.fft.ifftshift(mesh)
            real_orbs = np.fft.ifftn(mesh)
            '''

            #plot_orbs = False
            if self.plot_orbs == True and False:
                # plot
                plt.scatter(min_rad[largeenough], radunk, c="black")
                new_rad = np.linspace(0,8,100)
                #plt.scatter(min_rad[big_ang],test_new[big_ang],facecolors="None",edgecolors="red")
                [ak, bk, ck, dk, ek, fk,gk,hk, l] = self.recip_orbcoeffs[orb]
                const = ((2 * np.pi) ** 3 / self.vol) ** (1 / 2)
                plt.plot(new_rad,func_for_rad(new_rad,ak,bk,ck,dk,ek,fk,gk,hk,l)*const,color="blue")
                # plt.plot(min_rad[sort_rad],rad_fit[sort_rad])
                # fig2.scatter(min_rad,unk[largeenough])
                print("showing plot")
                plt.show()
                plt.close()
            
            # convert to real orbital
            coeffs = unk/exp_term #recip_orbs[orb]
            recip_coeff = coeffs # np.zeros((len(keys),len(recip_orbs[keys[0]])), dtype=np.complex128)
            # get dimension of box
            kprim1 = np.sort(np.around(np.abs(self.kpoints[:,0]),decimals=6))
            kprim2 = np.sort(np.around(np.abs(self.kpoints[:,1]),decimals=6))
            kprim3 = np.sort(np.around(np.abs(self.kpoints[:,2]),decimals=6))
            trans1 = int(np.around(1/kprim1[kprim1 != 0][0],decimals=0)) if len(kprim1[kprim1 != 0]) > 0 else 1 
            trans2 = int(np.around(1/kprim2[kprim2 != 0][0],decimals=0)) if len(kprim2[kprim2 != 0]) > 0 else 1 
            trans3 = int(np.around(1/kprim3[kprim3 != 0][0],decimals=0)) if len(kprim3[kprim3 != 0]) > 0 else 1 
            
            all_mesh = np.zeros((self.gridxyz[0]*trans1,self.gridxyz[1]*trans2,self.gridxyz[2]*trans3), dtype=np.complex128)
            #for oi,orb in enumerate(keys):
            #    recip_coeff[oi] = recip_orbs[orb]
            int_gplusk = sort_gplusk * np.array([trans1,trans2,trans3])
            print(int_gplusk)
            t0 = int_gplusk[:,0].astype(np.int_) # + (self.gridxyz[0] / 2).astype(np.int_)*trans1
            t1 = int_gplusk[:,1].astype(np.int_) # + (self.gridxyz[1] / 2).astype(np.int_)*trans2
            t2 = int_gplusk[:,2].astype(np.int_) # + (self.gridxyz[2] / 2).astype(np.int_)*trans3

            all_mesh[t0,t1,t2] = recip_coeff
            #all_mesh = np.fft.ifftshift(all_mesh,axes=(0,1,2))
            allreal_orbs = np.fft.ifftn(all_mesh,axes=(0,1,2))
            const = self.gridxyz[0] * self.gridxyz[1] * self.gridxyz[2] / self.vol ** (1 / 2)
            centered_realwan[orbind] = allreal_orbs*const
            print("wan shape:",allreal_orbs.shape)
            lim1,lim2,lim3 = [trans1*(1-1/(self.gridxyz[0]*trans1)),trans2*(1-1/(self.gridxyz[1]*trans2)),trans3*(1-1/(self.gridxyz[2]*trans3))]
            X,Y,Z = np.mgrid[0:lim1:self.gridxyz[0]*trans1*1j,0:lim2:self.gridxyz[1]*trans2*1j,0:lim3:self.gridxyz[2]*trans3*1j]
            print("check periodic:",X[-2:,0,0])
            X[X>int(trans1/2)] = X[X>int(trans1/2)] - trans1
            Y[Y>int(trans2/2)] = Y[Y>int(trans2/2)] - trans2
            Z[Z>int(trans3/2)] = Z[Z>int(trans3/2)] - trans3
            plotZ = allreal_orbs[:,:,0].real
            cart_dim = X.shape
            print(X.shape,Y.shape,plotZ.shape)
            
            prim_XYZ = np.array([X.flatten(),Y.flatten(),Z.flatten()]).T
            cart_XYZ = _red_to_cart((self._a[0], self._a[1], self._a[2]), prim_XYZ)
            
            plot_grid = False
            if plot_grid == True:
                select = np.arange(-80,0)
                select = np.append(select,np.arange(0,80))
                fig = go.Figure(data=[go.Surface(x=X[select][:,select,0], y=Y[select][:,select,0], z=allreal_orbs[select][:,select,0].real)])

                fig.update_layout(title='Wannier orbital', autosize=False,
                                  width=500, height=500,
                                  margin=dict(l=65, r=50, b=65, t=90))

                fig.show()
            
            vec = cart_XYZ.transpose()  # point - np.array([atm]).transpose()
            rad = np.linalg.norm(vec, axis=0)
            # print("check recip rad",kpoint,recip_rad)
            rad[rad==0] = 0.0000000001
            theta = np.arccos(vec[2] / rad)
            theta[rad == 0.0000000001] = 0
            phi = np.arctan2(vec[1], vec[0])

            orb_to_l = [0, 1, 1, 1, 2, 2, 2, 2, 2,3,3,3,3,3,3,3]
            
            # only fit orbital less than 6 ang
            to_fitorb = allreal_orbs.flatten().real
            to_fitorb = to_fitorb[rad < 6]
            theta = theta[rad < 6]
            phi = phi[rad < 6]
            rad = rad[rad < 6]
            
            # now get fit in real space
            
            atom = self.orbatomnum[orbind]
            unk = to_fitorb
            og_orb = to_fitorb
            # check how close the orbital is to the last one constructed
            
            # fit the exponential decay
            orbperatom = self.sph_harm_key[orbind]
            angular_part = complex128funs(phi, theta, self.orb_vec[orbind])
            #change points very close to zero rad for p and d orbs
            if orbperatom != 0:
                angular_part[rad < 0.001] = 0.0001 # just for the limit so point is never included
            largeenough = np.abs(angular_part)>0.01
            radunk = unk[largeenough]/angular_part[largeenough]
            angular_part[angular_part == 0] = 0.001
            og_radunk = og_orb/angular_part
            og_radunk = og_radunk[largeenough]
            good_min_rad = rad[largeenough]

            # change p and d orbs for the center point is zero
            if orbperatom != 0:
                radunk[good_min_rad < 0.001] = 0.0 # just for the limit later


            positive = (radunk != 0.0) 
            radunk = radunk[positive]
            good_min_rad = good_min_rad[positive]

            l=orb_to_l[orbperatom]
            weights = (1/(good_min_rad+0.001)**(1/2/(l+1)))*np.abs(angular_part[largeenough][positive])**(1/2)#+radunk*5000


            # randomly select points to fit based on (1-(x/cutoff)^2) to speed up fit and have more even radial distribution of points
            rand = np.random.rand(len(radunk))
            func = 1/good_min_rad**(3/2) #(1-(good_min_rad/(self.cutoff_rad[atom]*2))**(2))
            keep_point = rand<func
            radunk = radunk[keep_point]
            good_min_rad = good_min_rad[keep_point]
            weights = weights[keep_point]
            #print("reduced points:",len(rand),len(radunk))

            # cutoff after certain large distance
            # this helps make the fitting scale linearly with the number of atoms rather than supercell size
            cutoff = self.cutoff_rad[atom]*0.9+np.tanh((self.init_orb_energy[orbind]+8)/5)*0.5 #self.cutoff_rad[atom]*0.8#**(1/2)#*1.4
            inside_cut = good_min_rad < 4*cutoff
            radunk = radunk[inside_cut]
            good_min_rad = good_min_rad[inside_cut]
            weights = weights[inside_cut]

            #append some high rad values to maintain a small tail
            #radunk = np.append(radunk,[0,0,0,0])
            #good_min_rad = np.append(good_min_rad,[cutoff+0.8,cutoff+1.2,cutoff+1.8,cutoff+3])
            #weights = np.append(weights,[4,8,12,20])
            og_radunk = og_radunk[positive][keep_point][inside_cut]
            #og_radunk = np.append(og_radunk,[0,0,0,0])

            l=orb_to_l[orbperatom]
            #cutoff = self.cutoff_rad[atom]*0.8#**(1/2)#*1.4
            orbkey = self.orbtype[orbind]
            no_node = orbkey[0] == orbkey # only is one character: 's', 'p', or 'd'
            #if not no_node:
            #    cutoff = cutoff * 1.1
            expcutoff = 1/cutoff
            new_cutoff = 1/expcutoff**(1/2)

            #fourth_time = time.time()
            # go back to curve_fit
            #e = 0
            #f = expcutoff/2
            g = 0
            h = expcutoff/2
            min_cut = new_cutoff/15 # expcutoff/8
            if l == 2:
                min_cut = expcutoff/4
            l=orb_to_l[orbperatom]
            x = good_min_rad[good_min_rad<=cutoff]
            y = radunk[good_min_rad<=cutoff]
            
            # use the previously fitted gaus params to the pseudo orbital as initial condition
            [pa,pb,pc,pd,pe,pf,l] = self.orig_gausparams[orbind]
            [pg,ph] = [0,new_cutoff/4]
            # rescale the magnitude of the converged orbital
            simprad = np.linspace(0,cutoff,100)
            simporb = func_for_rad(simprad,pa,pb,pc,pd,pe,pf,pg,ph,l=l)
            max_ind = np.argmax(simporb)
            max_rad = simprad[max_ind]
            max_val = simporb[max_ind]
            truemaxval = np.average(y[np.abs(x-max_rad) < 0.1/(l+1)],weights = weights[good_min_rad<=cutoff][np.abs(x-max_rad) < 0.1/(l+1)])
            #print("normalizing values:",max_rad,max_val,truemaxval)
            [pa,pc,pe,pg] = np.array([pa,pc,pe,pg])/max_val*truemaxval
            
            # try fitting with contraints built into function so can use curve_fit
            init_con1 = pd/pb #np.log( - 1)
            init_con4 = 1/2 #np.log(- 1)
            init_con3 = pf/pb #np.log(pb/pf - 1)
            init_con2 = pe/pf+pa/pb
            max_aceg = np.amax(np.abs([pa,pc,pe,pg]))
            if no_node:
                gaus_init = [pa,pb,init_con1,init_con2,init_con3]#,pg,init_con4]
                partial_gaus_fit = partial(func_for_rad_fit,c=pc,g=0,con4=init_con4, l=l)
                low_bounds = np.ones(5)*0.000001 #[0,0,0,0,0]
                high_bounds = [max_aceg*10,new_cutoff*3,0.95,max_aceg*10,0.9]#,max_aceg*5,0.9]
            else: # allow node in fit
                gaus_init = [pa,pb,init_con1,init_con2,init_con3,pc]#,pg,init_con4]
                partial_gaus_fit = partial(func_for_rad_fit,g=0,con4=init_con4, l=l)
                low_bounds = np.ones(6)*0.000001 #[0,0,0,-max_aceg*20,0,0]
                low_bounds[3] = -max_aceg*10
                high_bounds = [max_aceg*10,new_cutoff*3, 0.95,max_aceg*10, 5, max_aceg*10]#,max_aceg*5,0.9]
                #low_bounds[3] = -max_aceg*20 # remove low bound for con2
                #high_bounds[4] = 5 # can be larger than b now, but generally is still not
            print(gaus_init,low_bounds,high_bounds)
            gaus_init = np.array(gaus_init) + 0.000001
            x = np.append(x,[cutoff+1.25,cutoff+2.5])
            y = np.append(y,[0,0])
            init_weight = np.append(weights[good_min_rad<=cutoff],[5,10])
            sig = 1/(init_weight+0.00001)
            popt,pcov = curve_fit(partial_gaus_fit,x,y,p0=gaus_init,sigma=sig,bounds=(low_bounds,high_bounds),ftol=0.00000001, xtol=0.00000001,method="trf",max_nfev=2000)
            if no_node:
                [a,b,con1,con2,con3] = popt
                c = pc
            else:
                [a,b,con1,con2,con3,c] = popt
            d = con1 * b
            f = con3 * b
            e = (con2 - a/b) * f
            g = 0
            h = ph
            
            if self.verbose > 0:
                print("terms:",a,b,c,d,e,f)

            # fit end part to exponential function
            end_gaus = func_for_rad(cutoff,a,b,c,d,e,f,g,h,l)
            c_e = 0
            d_e = expcutoff
            e_e = 0
            f_e = expcutoff/2
            l=orb_to_l[orbperatom]
            exp_fit = partial(func_for_rad_exp,e=e_e,f=f_e,l=l)
            x = np.append(good_min_rad[good_min_rad>cutoff],[cutoff])
            y = np.append(radunk[good_min_rad>cutoff],[end_gaus])
            weights[weights == 0] = 0.0001
            sig = np.append(1/weights[good_min_rad>cutoff],[0.1])
            orig_expparams = self.orig_expparams[orbind]
            pbe = 1/orig_expparams[1]/2
            pae = orig_expparams[0]*pbe
            pinit = [orig_expparams[0],orig_expparams[1],orig_expparams[0]/10,orig_expparams[1]/2]
            [pbe,pde] = [1/pinit[1]/2,1/pinit[3]/2]
            [pae,pce] = [pinit[0]*pbe,pinit[2]*pde]
            pinit = [0.0005,expcutoff/2,0.001,expcutoff]
            popt,pcov = curve_fit(exp_fit,x,y,p0=pinit,sigma=sig,bounds=(0,expcutoff*10),ftol=0.000001, xtol=0.000001,method="trf",max_nfev=2000)
            [a_e,b_e,c_e,d_e] = popt

            if self.verbose > 2:
                print("exp fitting factors:",a_e,b_e,c_e,d_e)

            #plot
            #plot_orbs = False
            if self.plot_orbs == True and False:# self.verbose>2: # no need to print all these
                plt.scatter(good_min_rad,og_radunk,c="black",s=1)
                #plt.scatter(min_rad[big_ang],new_rad_part,c="red",s=1)
                plt.scatter(good_min_rad,radunk,c="blue",s=weights*10)
                new_rad = np.linspace(0,5,100)
                #plot the original starting radial part
                plt.plot(new_rad,self.orig_radial[orbind],color="green",linewidth=1.0)
                #plt.scatter(min_rad[big_ang],test_new[big_ang],facecolors="None",edgecolors="red")
                just_one = func_for_rad(new_rad,a,b,c,d,e,f,g,h,l=l)
                just_one[new_rad>cutoff] = func_for_rad_exp(new_rad[new_rad>cutoff],a_e,b_e,c_e,d_e,e_e,f_e,l=l)
                plt.plot(new_rad,just_one,color="black",linewidth=1.0)
                [ak, bk, ck, dk, ek, fk,gk,hk, l] = self.real_orbcoeffs[orb]
                const = ((2 * np.pi) ** 3 / self.vol) ** (1 / 2)*self.num_kpts
                const = self.gridxyz[0]*self.gridxyz[1]*self.gridxyz[2]/self.vol**(1/2)
                plt.plot(new_rad,func_for_rad(new_rad,ak,bk,ck,dk,ek,fk,gk,hk,l)/const/2,color="blue")
                print("showing plot")
                plt.show()
                plt.close()

            #save fitted values
            #orbgroup_gausparams[group_ind] = [a,b,c,d,e,f,g,h,l]
            #orbgroup_pseudoparams[group_ind] = [pa,pb,pc,pd,pe,pf,pg,ph,l]
            #orbgroup_expparams[group_ind] = [a_e,b_e,c_e,d_e,e_e,f_e,l]

        

            # finally fit the total gaussian + exponential fit to a gaussian basis
            # and properly normalize the orbitals within the PAW basis
            #[a,b,c,d,e,f,g,h,l] = orbgroup_gausparams[group_ind]
            #[a_e,b_e,c_e,d_e,e_e,f_e,l] = orbgroup_expparams[group_ind]
            #atom = self.orbatomnum[orbgroup[0]]
            #cutoff = self.cutoff_rad[atom]*0.8#**(1/2)#*1.4
            #expcutoff = 1/cutoff
            # orbital dependant section
            # now fit to just gaussian functions
            new_rad = np.linspace(0,(cutoff/2)**(1/2)*5,100)
            just_one = func_for_rad(new_rad,a,b,c,d,e,f,g,h,l=l)
            just_one[new_rad>cutoff] = func_for_rad_exp(new_rad[new_rad>cutoff],a_e,b_e,c_e,d_e,e_e,f_e,l=l)

            #use the original WF
            #just_one = self.orig_radial[orbind]
            weights = np.ones(len(new_rad))#1/(new_rad**(1/2)+0.5)
            end_gaus = func_for_rad(cutoff,a,b,c,d,e,f,g,h,l)
            weights[new_rad>cutoff] = weights[new_rad>cutoff] * (just_one[new_rad>cutoff]/end_gaus)**(1/4) * 3
            #just_one = np.append(just_one,[0,0,0,0])
            #new_rad = np.append(new_rad,[cutoff+1.5,cutoff+2,cutoff+2.5,cutoff+3])
            #weights = np.append(weights,[0.1,0.2,0.5,1.0])

            #set initial conditions
            l=l
            x = new_rad
            y = just_one
            min_cut = expcutoff/8
            if l == 2:
                min_cut = expcutoff/4

            
            # try fitting with contraints built into function so can use curve_fit)
            max_aceg = np.amax(np.abs([pa,pc,pe,pg]))
            pg = max_aceg/5
            init_con1 = pd/pb #np.log( - 1)
            init_con4 = 2/3 #np.log(- 1)
            init_con3 = pf/pb #np.log(pb/pf - 1)
            init_con2 = pe/pf+pa/pb
            gaus_init = [pa,pb,init_con1,init_con2,init_con3,pc,pg,init_con4]
            gaus_fit = partial(func_for_rad_fit, l=l)
            sig = 1/(weights+0.00001)
            #max_aceg = np.max(np.abs([pa,pc,pe,pg]))
            low_bounds = np.ones(8)*0.000001 #[0,0,0,0,0,0,0,0]
            high_bounds = [max_aceg*10,new_cutoff*3,0.95,max_aceg*10,0.9,max_aceg*10,max_aceg*10,0.9]
            if not no_node: # allow node in fit
                low_bounds[3] = -max_aceg*10 # remove low bound for con2
                high_bounds[4] = 5 # can be larger than b now, but generally is still not
            gaus_init = np.array(gaus_init) + 0.000001
            popt,pcov = curve_fit(gaus_fit,x,y,p0=gaus_init,sigma=sig,bounds=(low_bounds,high_bounds),ftol=0.00000001, xtol=0.00000001,method="trf")
            # check how close output is to bounds
            close_low = np.abs(popt-low_bounds)/popt
            close_high = np.abs(popt-high_bounds)/popt
            if ((close_low<0.05).any() or (close_high<0.05).any()) and self.verbose > 0:
                print("warning very close to bounds!",l,close_low[close_low<0.05],close_high[close_high<0.05])
            [t_a,t_b,con1,con2,con3,t_c,t_g,con4] = popt
            t_d = con1 * t_b
            t_f = con3 * t_b
            t_e = (con2 - t_a/t_b) * t_f
            t_h = con4 * t_b
            #print("check lmfit params:",r_a,r_b,r_c,r_d,r_e,r_f,r_g,r_h)
            #print("check scipy params:",t_a,t_b,t_c,t_d,t_e,t_f,t_g,t_h)
            r_a,r_b,r_c,r_d,r_e,r_f,r_g,r_h = [t_a,t_b,t_c,t_d,t_e,t_f,t_g,t_h]
            
            if self.verbose > 1:
                print("fitted params:",r_a,r_b,r_c,r_d,r_e,r_f,r_g,r_h,l)
            
            # get values for plotting
            #good_min_rad = all_good_min_rad[orbgroup[0]]
            #radunk = all_radunk[orbgroup[0]]
            #weights = all_weights[orbgroup[0]]
            #og_radunk = all_og_orbs[orbgroup[0]]
            #plot_orbs = True
            if self.plot_orbs == True:
                #plt.scatter(min_rad[big_ang],og_radunk[big_ang],c="black",s=2)
                #plt.scatter(min_rad[big_ang],new_rad_part,c="red",s=1)
                plt.scatter(good_min_rad,radunk,c="blue",s=1/4,label="From approximate atomic")
                plt.scatter(good_min_rad,og_radunk,c="green",s=1/2,label="From Bloch orbital")
                new_rad = np.linspace(0,5,100)
                #plot the original starting radial part
                [nx,ny,nz] = self.gridxyz
                other_const = ((2 * np.pi) ** 3 / self.vol) ** (1 / 4)
                plt.plot(new_rad,self.orig_radial[orbind]/other_const,color="black",linewidth=2.5,label="Pseudo orbital")
                #plt.scatter(min_rad[big_ang],test_new[big_ang],facecolors="None",edgecolors="red")
                just_one = func_for_rad(new_rad,a,b,c,d,e,f,g,h,l=l)
                just_one[new_rad>cutoff] = func_for_rad_exp(new_rad[new_rad>cutoff],a_e,b_e,c_e,d_e,e_e,f_e,l=l)
                #plt.ylim((0,0.0005))
                plt.xlim((0,5))
                plt.plot(new_rad,just_one,color="purple",linewidth=2.5,label="Gauss+exp fit")#c=c,d=d,*popt
                
                [oa, ob, oc, od, oe, of, og, oh, l] = self.real_orbcoeffs[orb]
                const = self.gridxyz[0]*self.gridxyz[1]*self.gridxyz[2]/self.vol**(1/2)
                other_const = ((2 * np.pi) ** 3 / self.vol) ** (1 / 4)
                print("other const:",other_const)
                plt.plot(new_rad,func_for_rad(new_rad,oa, ob, oc, od, oe, of,og,oh, l)/const/other_const,color="blue",label="previous fit")

                # plot the just guassian fit
                plt.plot(new_rad,func_for_rad(new_rad,r_a,r_b,r_c,r_d,r_e,r_f,r_g,r_h,l=l),color='#db61ed',linewidth = 2.5,label="Final gaus fit")
                #plt.ylim((-0.001,0.001))
                plt.xlim((0,4))
                #print("showing plot")
                
                plt.xlabel("Radius (Å)")
                plt.ylabel("Radial part of orbital")
                plt.legend()
                plt.show()
                plt.close()

            [a,b,c,d,e,f,g,h] = [r_a,r_b,r_c,r_d,r_e,r_f,r_g,r_h]
            #if no_fit:
            #    [a,b,c,d,e,f,g,h] = [pa,pb,pc,pd,pe,pf,pg,ph]
            #    [b,d,f,h] = np.array([b,d,f,h])*orbfactor

            # now normalize for the PAW treatment
            # find orbital projector overlap (overlap_OrbProj) and ae-psuedo diff (Qab)
            if self.haveQab == False:
                self.get_Qab()
                self.haveQab = True
            Qab = self.Qab

            # change the coeffs so they normalize correctly
            const = self.gridxyz[0]*self.gridxyz[1]*self.gridxyz[2]/self.vol**(1/2)
            if self.verbose > 1:
                print("volume:", self.vol)
                print("const",const)
            [a,c,e,g] = np.array([a,c,e,g])*const #analytical calc
            
            # switch to a*e^-bx^2 format form a/b*e^-1/2(x/b)^2 format
            old_params = copy.deepcopy([a,b,c,d,e,f,g,h,l])
            #print("check same:",[a,b,c,d,e,f,g,h,l])
            [a,c,e,g] = [a/b,c/d,e/f,g/h]
            [b,d,f,h] = 1/2/np.array([b,d,f,h])**2
            #print("as:", old_params,[a,b,c,d])

            # normalize the orbital just with the onsite overlap
            l= int(l)
            orbind = orbind
            real_proj_data = self.atmProjectordata[self.elements[atom]]
            proj_cutoff = real_proj_data[0]
            real_proj_grid = np.linspace(0, proj_cutoff, num=100,endpoint=False)
            gaus_orb = func_for_rad(real_proj_grid,*old_params)
            
            # analytical integral of Gaussian funcs
            #true_gauss_norm = [np.pi**(1/2)/4,1/2,3*np.pi**(1/2)/8][l] * (a*d**((3+l)/2)+c*b**((3+l)/2))/d**((3+l)/2)/b**((3+l)/2)
            #gaus_gaus_int = [1,3,15][l]*np.pi**(1/2)*(a*c/(2**(1+l)*(d+b)**(3/2+l))+(a**2*d**(3/2+l)+c**2*b**(3/2+l))/(2**(7/2+2*l)*b**(3/2+l)*d**(3/2+l))) #analytical calc
            fnt_coef = np.array([a,c,e,g])
            exp_coef =np.array([b,d,f,h])
            #gaus_norm = factorial2(1+2*l)*np.pi**(1/2)*(2**(-7/2-2*l)*np.sum(fnt_coef**2*exp_coef**(-3/2-l))+2**(-1-l)*np.prod(fnt_coef)*np.sum(exp_coef)**(-3/2-l))
            gaus_norm = factorial2(1+2*l)*np.pi**(1/2)*2**(-2-l)*np.sum(fnt_coef[:,None]*fnt_coef[None,:]*(exp_coef[:,None]+exp_coef[None,:])**(-3/2-l))
            #gaus_gaus_int = gaus_gaus_int *(self.gridxyz[0]*self.gridxyz[1]*self.gridxyz[2])**(2)/self.vol
            #true_gauss_norm = true_gauss_norm * self.gridxyz[0]* self.gridxyz[1]* self.gridxyz[2]

            # only do projectors that have the same center angular momentum as the orbital
            num_projs = self.num_orbs + self.num_exorbs
            all_sphharm = np.append(self.sph_harm_key,self.ex_sph_harm_key)
            all_orbtype = np.append(self.orbtype, self.ex_orbtype)
            all_orbatm = np.append(self.orbatomnum, self.exorbatomnum)
            orb1 = orbind # each orbital in the group should have the same norm
            sphharm1 = all_sphharm[orb1]
            atm = all_orbatm[orb1]
            nonzero = (all_sphharm == sphharm1) & (all_orbatm == atm) # only nonzero if the orbs are on the same atom and have the same l and ml quantum nums
            orbs = np.arange(num_projs,dtype=np.int_)[nonzero]
            
            real_proj_data = self.atmProjectordata[self.elements[atom]]
            proj_cutoff = real_proj_data[0]
            real_proj_grid = np.linspace(0, proj_cutoff, num=100,endpoint=False)
            gaus_orb = func_for_rad(real_proj_grid,*old_params)
            #gaus_gaus_orbs = gaus_orb * gaus_orb
            ae_overlap = gaus_norm
            for orb1 in orbs:
                type1 = all_orbtype[orb1]
                gaus_proj_orbs1 = gaus_orb * real_proj_data[1][type1]
                gaus_proj_norm1 = integrate.trapezoid((gaus_proj_orbs1) * real_proj_grid ** 2, x=real_proj_grid)
                for orb2 in orbs:
                    type2 = all_orbtype[orb2]
                    gaus_proj_orbs2 = gaus_orb * real_proj_data[1][type2]
                    gaus_proj_norm2 = integrate.trapezoid((gaus_proj_orbs2) * real_proj_grid ** 2, x=real_proj_grid)
                    ae_overlap += gaus_proj_norm1 * Qab[orb1,orb2] * gaus_proj_norm2

            psuedo_overlap = gaus_norm
            individ_norm =  ae_overlap

            if self.verbose > 0:
                print("pseudo norm and ae adjustments:",orbind, psuedo_overlap,individ_norm-psuedo_overlap)# sum_overi)
            [a,c,e,g] = [a,c,e,g] / individ_norm ** (1 / 2)

            #calculate the average (Bohr) radius
            fnt_coef = np.array([a, c, e, g])
            exp_coef = np.array([b, d, f, h])
            avg_rad = 1/2*math.factorial(1+l)*np.sum(fnt_coef[:,None]*fnt_coef[None,:]*(exp_coef[:,None]+exp_coef[None,:])**(-2-l)) #analytical calc
            #old_rad = 1/8*math.factorial(1+l)*(2**(-l)*a**2*b**(-2-l)+2**(-l)*c**2*d**(-2-l)+8*a*c*(b+d)**(-2-l))
            #[oa, ob, oc, od, oe, of, og, oh, l] = self.real_orbcoeffs[orb]
            #old_fcoeff = np.array([oa, oc, oe, og])
            #old_ecoeff = np.array([ob, od, of, oh])
            #old_rad = 1/2*math.factorial(1+l)*np.sum(old_fcoeff[:,None]*old_fcoeff[None,:]*(old_ecoeff[:,None]+old_ecoeff[None,:])**(-2-l))
            #if self.verbose > -1:
            #    print("average radius!",avg_rad,old_rad)
            newgaus_gaus_int = factorial2(1+2*l)*np.pi**(1/2)*2**(-2-l)*np.sum(fnt_coef[:,None]*fnt_coef[None,:]*(exp_coef[:,None]+exp_coef[None,:])**(-3/2-l))

            # convert to reciprocal space
            # g(k) = ∫j_l(kr)*g(r)*r^2*dr was done in mathematica to get the symbol conversion for analytical calc
            ak = np.pi**(1/2)*2**(-2-l)*a*b**(-3/2-l) * (2/np.pi)**(1/2) #*(self.num_kpts/self.vol)**(1/4)#* self.vol**(1/4)/np.pi**(3/4)
            bk = 1/4/b
            ck = np.pi**(1/2)*2**(-2-l)*c*d**(-3/2-l) * (2/np.pi)**(1/2) # *(self.num_kpts/self.vol)**(1/4)#* self.vol**(1/4)/np.pi**(3/4)
            dk = 1/4/d
            ek = np.pi**(1/2)*2**(-2-l)*e*f**(-3/2-l) * (2/np.pi)**(1/2)
            fk = 1/4/f
            gk = np.pi**(1/2)*2**(-2-l)*g*h**(-3/2-l) * (2/np.pi)**(1/2)
            hk = 1/4/h
            
            #newgaus_gaus_int = [1,3,15][l]*np.pi**(1/2)*(a*c/(2**(1+l)*(d+b)**(3/2+l))+(a**2*d**(3/2+l)+c**2*b**(3/2+l))/(2**(7/2+2*l)*b**(3/2+l)*d**(3/2+l)))
            fnt_coefk = np.array([ak, ck, ek, gk])
            exp_coefk = np.array([bk, dk, fk, hk])
            recipgaus_gaus_int = factorial2(1+2*l)*np.pi**(1/2)*2**(-2-l)*np.sum(fnt_coefk[:,None]*fnt_coefk[None,:]*(exp_coefk[:,None]+exp_coefk[None,:])**(-3/2-l))

            # convert back to a/b*e^-1/2(x/b)^2 format from a*e^-bx^2 format
            [b,d,f,h] =  1/(2*np.array([b,d,f,h]))**(1/2) # 1/2/np.array([b,d,f,h])**2
            [a,c,e,g] = [a*b,c*d,e*f,g*h]
            
            [bk,dk,fk,hk] =  1/(2*np.array([bk,dk,fk,hk]))**(1/2) # 1/2/np.array([b,d,f,h])**2
            [ak,ck,ek,gk] = [ak*bk,ck*dk,ek*fk,gk*hk]
            
            #save to orbital coeffs
            #for orb in orbgroup:
            avg_radius[orbind] = avg_rad
            orbital_coeffs[orbind] = np.array([a,b,c,d,e,f,g,h,l])
            recip_orbcoeffs[orbind] = np.array([ak,bk,ck,dk,ek,fk,gk,hk,l])

            #recipgaus_gaus_int = [1,3,15][l]*np.pi**(1/2)*(ak*ck/(2**(1+l)*(dk+bk)**(3/2+l))+(ak**2*dk**(3/2+l)+ck**2*bk**(3/2+l))/(2**(7/2+2*l)*bk**(3/2+l)*dk**(3/2+l)))
            if self.verbose > 0:
                print("new norms:",newgaus_gaus_int,recipgaus_gaus_int)
            #end orb dependant section

        # check error between the initial orbs and numerical wannier orb
        #                         fit wannier orb and numerical wannier orb
        #                         and initial orbs and fit wannier orbs
        # get radial coordinates for everything
        vec = cart_XYZ.transpose()  # point - np.array([atm]).transpose()
        rad = np.linalg.norm(vec, axis=0)
        # print("check recip rad",kpoint,recip_rad)
        rad[rad==0] = 0.0000000001
        theta = np.arccos(vec[2] / rad)
        theta[rad == 0.0000000001] = 0
        phi = np.arctan2(vec[1], vec[0])
        wan_plots = []
        all_data = [] # [wan,init,new_fit]
        for orb in range(self.num_orbs):
            const = ((2 * np.pi) ** 3 / self.vol) ** (1 / 4)
            wan_orb = centered_realwan[orb]*const

            # make the initial orbitals
            real_coeff = self.real_orbcoeffs[orb]
            [a, b, c, d, e, f, g, h, l] = real_coeff
            orb_peratom = self.sph_harm_key[orb]
            angular_part = complex128funs(phi, theta, self.orb_vec[orb])
            radial_part = func_for_rad(rad, a, b, c, d, e, f, g, h, l)
            initial_orb = radial_part * angular_part
            # structure constant
            # const = ((2 * np.pi) ** 3 / self.vol) ** (1 / 2)
            # recip_orbs[orb] = initial_orb * const

            # make the fitted wannier orb
            real_coeff = orbital_coeffs[orb]
            [a, b, c, d, e, f, g, h, l] = real_coeff
            orb_peratom = self.sph_harm_key[orb]
            angular_part = complex128funs(phi, theta, self.orb_vec[orb])
            radial_part = func_for_rad(rad, a, b, c, d, e, f, g, h, l)
            fit_wan_orb = radial_part * angular_part

            X = np.reshape(vec[0],cart_dim)
            Y = np.reshape(vec[1],cart_dim)
            Z = np.reshape(vec[2],cart_dim)
            wan_orb = np.reshape(wan_orb,cart_dim)
            initial_orb = np.reshape(initial_orb,cart_dim)
            fit_wan_orb = np.reshape(fit_wan_orb,cart_dim)

            wan_initial_error = wan_orb-initial_orb
            wan_fit_error = wan_orb-fit_wan_orb
            fit_initial_error = initial_orb-fit_wan_orb
            scaling = 1/(rad**2+0.05)*np.exp(-rad/4)
            scaling = np.reshape(scaling,cart_dim)
            wi_maxer = np.amax(wan_initial_error)
            wi_miner = np.amin(wan_initial_error)
            wf_maxer = np.amax(wan_fit_error)
            wf_miner = np.amin(wan_fit_error)
            fi_maxer = np.amax(fit_initial_error)
            fi_miner = np.amin(fit_initial_error)

            wi_error = np.average(np.abs(wan_initial_error)*scaling)
            wf_error = np.average(np.abs(wan_fit_error)*scaling)
            fi_error = np.average(np.abs(fit_initial_error)*scaling)
            print("max/min errors:",wi_maxer,wi_miner,wf_maxer,wf_miner,fi_maxer,fi_miner)
            print("average errors:",wi_error,wf_error,fi_error)
            norm_wan = np.average(np.abs(wan_orb*scaling)**2)**(1/2)
            norm_int = np.average(np.abs(initial_orb*scaling)**2)**(1/2)
            norm_fit = np.average(np.abs(fit_wan_orb*scaling)**2)**(1/2)
            nrms_wi_error = np.average((np.abs(wan_initial_error)*scaling)**2)**(1/2)#/norm
            nrms_wf_error = np.average((np.abs(wan_fit_error)*scaling)**2)**(1/2)#/norm_fit
            nrms_fi_error = np.average((np.abs(fit_initial_error)*scaling)**2)**(1/2)#/norm
            print("rms errors:",nrms_wi_error,nrms_wf_error,nrms_fi_error)
            print("norms:", norm_wan, norm_int, norm_fit)

            #plot_grid = True
            if self.plot_orbs == True:
                select = np.arange(-40,0)
                select = np.append(select,np.arange(0,40))
                data = []
                data.append(go.Surface(x=X[select][:,select,0], y=Y[select][:,select,0], z=wan_orb[select][:,select,0].real))
                #data.append(go.Surface(x=X[select][:,select,0], y=Y[select][:,select,0], z=wan_fit_error[select][:,select,0].real))
                #data.append(go.Surface(x=X[select][:,select,0], y=Y[select][:,select,0], z=fit_initial_error[select][:,select,0].real))
                fig = go.Figure(data=data)
                fig.update_layout(title='Wannier orbital', autosize=False,
                                  width=500, height=500,
                                  margin=dict(l=65, r=50, b=65, t=90))

                fig.show()
            
            # plot on 1D line to closest atom and save
            #for orb in range(self.num_orbs):
            atmnum = self.orbatomnum[orb]
            diff_atoms = np.array(self.cartAtoms)-np.array(self.cartAtoms[atmnum])[None,:]
            closed_atm = np.argsort(np.linalg.norm(diff_atoms,axis=1))[1]
            print(self.cartAtoms[atmnum],closed_atm,diff_atoms[closed_atm])
            #XYZ = np.array([X.flatten(),Y.flatten(),Z.flatten()])
            wan_norm = np.sum(np.conj(wan_orb)*wan_orb)
            init_norm = np.sum(np.conj(initial_orb)*initial_orb)
            new_norm = np.sum(np.conj(fit_wan_orb)*fit_wan_orb)
            wan_dists1, wan_vals1 = extract_line_data(wan_orb.flatten(), vec, np.array([0,0,0]), diff_atoms[closed_atm], tolerance=0.15)
            og_dists1, og_vals1 = extract_line_data(initial_orb.flatten()*(wan_norm/init_norm)**(1/2), vec, np.array([0,0,0]), diff_atoms[closed_atm], tolerance=0.15)
            new_dists1, new_vals1 = extract_line_data(fit_wan_orb.flatten()*(wan_norm/new_norm)**(1/2), vec, np.array([0,0,0]), diff_atoms[closed_atm], tolerance=0.15)
            #cog_dists2, cog_vals2 = extract_line_data(real_new[4].flatten(), self.cartXYZ, self.cartAtoms[1], self.cartAtoms[0], tolerance=0.1)
    
            fig, ax = plt.subplots()
            #plt.figure(figsize=(8, 5))
            ax.plot(wan_dists1, wan_vals1, "blue",label="wannier")
            ax.plot(new_dists1, new_vals1, "red",label="new atomic fit")
            ax.plot(og_dists1, og_vals1, "black",label="initial fit")
            #ax.set_xlim((np.amin(og_dists1),np.amax(og_dists1)))
            #ax.set_ylim((np.amin(og_vals1)-np.amax(og_vals1)*0.2,np.amax(og_vals1)*1.2))
            ax.set_xlabel("Distance along line (Å)")
            ax.set_ylabel("Orbital magnitude")
            ax.set_title("Orbital Magnitude vs. Distance")
            ax.legend()
            #plt.grid(True)
            wan_plots.append(fig)
            all_data.append([[list(wan_dists1.real), list(wan_vals1.real)],[list(og_dists1.real), list(og_vals1.real)],[list(new_dists1.real), list(new_vals1.real)]])
            plt.close()
            #plt.show()
        if self.save_orb_figs:
            combine_and_save_plots(wan_plots, filename=self.directory+"wannier_orbs"+self.tag+".png")
        if self.save_orb_data:
            import json
            with open(self.directory+"all_wannier_data"+self.tag+".json", 'w') as f:
                # indent=2 is not needed but makes the file human-readable 
                # if the data is nested
                json.dump(all_data, f, indent=2) 

        #with open("all_wannier_data", "wb") as fp:   #Pickling
        #    pickle.dump(all_data, fp)
        
        #rad_part = self
        #centered_realwan
        print("radius:",avg_radius)

        self.real_orbcoeffs = orbital_coeffs
        self.recip_orbcoeffs = recip_orbcoeffs

    def get_TBparameter(self,num_trans = 5):
        #print("I hope that you have provided a WAVECAR with uniform sampling and no symmetry!")
        # generate the correct amount of trans based on kpoints
        kprim1 = np.sort(np.around(np.abs(self.kpoints[:,0]),decimals=6))
        kprim2 = np.sort(np.around(np.abs(self.kpoints[:,1]),decimals=6))
        kprim3 = np.sort(np.around(np.abs(self.kpoints[:,2]),decimals=6))
        trans1 = int(np.around(1/kprim1[kprim1 != 0][0],decimals=0)) if len(kprim1[kprim1 != 0]) > 0 else 1 
        trans2 = int(np.around(1/kprim2[kprim2 != 0][0],decimals=0)) if len(kprim2[kprim2 != 0]) > 0 else 1 
        trans3 = int(np.around(1/kprim3[kprim3 != 0][0],decimals=0)) if len(kprim3[kprim3 != 0]) > 0 else 1 

        num_each_dir = np.array(np.floor((np.array([trans1,trans2,trans3]))/2), dtype=np.int_)
        self.num_trans = num_each_dir*2+1
        [trans1,trans2,trans3] = self.num_trans
        print("kpoint grid, now numtrans:",self.num_trans)
        
        tb_param = np.zeros((trans1,trans2,trans3,self.num_orbs,self.num_orbs),dtype=np.complex128)
        overlaps = np.zeros((trans1,trans2,trans3,self.num_orbs,self.num_orbs),dtype=np.complex128)
        vec_to_trans = np.zeros((trans1,trans2,trans3,3),dtype=np.int_)
        for x in range(trans1):
            for y in range(trans2):
                for z in range(trans3):
                    vec_to_trans[x,y,z] = [x-num_each_dir[0],y-num_each_dir[1],z-num_each_dir[2]]
        self.vec_to_trans = vec_to_trans
        self.num_each_dir = num_each_dir #np.floor(self.num_trans / 2,dtype=np.int_)

        vec_to_orbs = np.zeros(vec_to_trans.shape,dtype=np.float64)
        test_params = np.zeros((trans1,trans2,trans3,self.num_orbs,self.num_orbs),dtype=np.complex128)
        #holdham = self.hamilton.flatten()
        #iszero = np.abs(holdham) < 0.000001
        #holdham[iszero] = 0
        #self.hamilton = np.reshape(holdham,(self.hamilton.shape))
        #print("orbital center offsets:", self.orb_center_offset)
        for orb1 in range(self.num_orbs):
            for orb2 in range(self.num_orbs):
                vec_to_orbs[:,:,:] = vec_to_trans[:,:,:] + self.orbpos[orb2] - self.orbpos[orb1] #+ self.orb_center_offset[orb2] - self.orb_center_offset[orb1]
                hold_vecs = np.reshape(vec_to_orbs,(trans1*trans2*trans3,3))
                for kptind, kpt in enumerate(self.kpoints):
                    exp_fac = np.exp(-2j*np.pi * np.dot(kpt,hold_vecs.transpose()))
                    exp_fac = np.reshape(exp_fac,(trans1,trans2,trans3)) #np.around(,decimals=10)
                    #exp_factor = np.exp(2j * np.pi * np.dot(kpt, del_r))
                    tb_param[:, :, :, orb1, orb2] += self.hamilton[orb1, orb2, kptind] * exp_fac

                    overlaps[:, :, :, orb1, orb2] += self.Sij[kptind][orb1, orb2] * exp_fac
        #print("exp fac:",exp_fac)
        #print("ham:",self.hamilton.T)
        #print("coeff:",self.mnkcoefficients[:,:,:self.num_orbs])
        #normalize properly
        tb_param = tb_param/self.num_kpts
        overlaps = overlaps/self.num_kpts
        #if test_params.all() == tb_param.all():
        #    print("its the same!")
        #if test_params[2,2,2,0,0] == tb_param[2,2,2,0,0]:
        #    print("also the same!")
        if self.verbose > 2:
            print("tb params:",tb_param[0,0,0])
            print("kdep of Aij:",self.Aij[:,0,0])
        #print("test params:",tb_param[1,1,1])
        #print("test params:",overlaps[1,1,1])#,tb_param[1,1,1][:5],tb_param[1,1,1][21:])
        self.tb_param = tb_param
        self.overlaps = overlaps
        self.write_TBfiles()
        
    def get_interp_ham(self,kind,return_truevec = True,return_params=False):
        # kind - int: the index of the kpoint
        kpt = self.kpoints[kind]
        ham = np.zeros((self.num_orbs,self.num_orbs),dtype=np.complex128)
        kdep_overlap = np.zeros((self.num_orbs,self.num_orbs),dtype=np.complex128)
        #vec_to_orbs = np.zeros(self.vec_to_trans.shape)
        #print("check same:", self.orbitals[str(0)][12,12,:])
        if return_params == True:
            scaledTB = np.zeros((self.num_orbs, self.num_orbs, self.num_trans[0], self.num_trans[1], self.num_trans[2]),dtype=np.complex128)
            scaledOvlap = np.zeros((self.num_orbs, self.num_orbs, self.num_trans[0], self.num_trans[1], self.num_trans[2]), dtype=np.complex128)
        self.set_min = True
        self.min_hopping_dist = 10

        # vectorize above! with the indices [numTx,numTy,numTz,orb1,orb2]
        vecs = np.array(self.vec_to_trans)
        orb_pos = np.array(self.orbpos)
        vec_to_orbs = vecs[:,:,:,None,None] + orb_pos[None,None,None,None,:] - orb_pos[None,None,None,:,None]
        #print("shape of vec:",vec_to_orbs.shape)
        hold_vecs = np.reshape(vec_to_orbs,(self.num_trans[0]*self.num_trans[1]*self.num_trans[2]*self.num_orbs*self.num_orbs,3)).transpose()
        cart_hold_vecs = _red_to_cart((self._a[0], self._a[1], self._a[2]), hold_vecs.transpose())
        dist_to_orbs = np.linalg.norm(cart_hold_vecs,axis=1)
        exp_fac = np.exp(2j*np.pi*np.dot(kpt,hold_vecs))
        temp_fac = np.reshape(exp_fac,(self.num_trans[0],self.num_trans[1],self.num_trans[2],self.num_orbs,self.num_orbs))
        if self.set_min == True:
            exp_fac[dist_to_orbs > self.min_hopping_dist] = 0
        exp_fac = np.reshape(exp_fac,(self.num_trans[0],self.num_trans[1],self.num_trans[2],self.num_orbs,self.num_orbs))
        #ham[orb1,orb2] = np.sum(self.TB_params[orb1,orb2]*exp_fac)
        #kdep_overlap[orb1, orb2] = np.sum(self.overlaps_params[orb1, orb2] * exp_fac)
        if return_params == True:
            scaledTB = self.tb_param*exp_fac
            scaledOvlap = self.overlaps*exp_fac
            
            # convert to [orb1,orb2,numTx,numTy,numTz] dimension
            scaledTB = np.transpose(scaledTB,axes=(3,4,0,1,2))
            scaledOvlap = np.transpose(scaledOvlap,axes=(3,4,0,1,2))
        
        if return_params == True:
            return scaledTB,scaledOvlap

        
    def get_neighbors(self):
        vecs = np.array(self.vec_to_trans)
        orb_pos = np.array(self.orbpos)
        vec_to_orbs = vecs[:,:,:,None,None] + orb_pos[None,None,None,None,:] - orb_pos[None,None,None,:,None]
        #print("shape of vec:",vec_to_orbs.shape)
        hold_vecs = np.reshape(vec_to_orbs,(self.num_trans[0]*self.num_trans[1]*self.num_trans[2]*self.num_orbs*self.num_orbs,3)).transpose()
        cart_hold_vecs = _red_to_cart((self._a[0], self._a[1], self._a[2]), hold_vecs.transpose())
        dist_to_orbs = np.linalg.norm(cart_hold_vecs,axis=1)
        small_dist_bool = dist_to_orbs < 10
        small_dist_index = np.arange(len(dist_to_orbs))[small_dist_bool]
        small_dist_orbs = dist_to_orbs[small_dist_index]
        sorted_dist_orbs = np.around(np.sort(small_dist_orbs),decimals=4)
        NN_dists = np.unique(sorted_dist_orbs)
        NN_index = {}
        for NN in range(min(10,len(NN_dists))):
            small_indices = np.abs(small_dist_orbs - NN_dists[NN]) <= 0.001
            NN_index[NN] = small_dist_index[small_indices] #flattened indices
        self.NN_index = NN_index
        self.NN_dists = NN_dists[:10]
        
    
    def orthogonalize_basis(self):
        # this will orthogonalize the atomic orbital basis (not Bloch orbitals) such that overlaps.txt file would be 1's for 000ii and 0's for everything else
        #self.num_trans = [num_trans,num_trans,num_trans]
        num_trans = np.array(self.num_trans) # [ -1, 0, 1]
        vec_to_trans = np.zeros((num_trans[0],num_trans[1],num_trans[2],3))
        flat_vecs = []
        num_vecs = num_trans[0]*num_trans[1]*num_trans[2]
        print(np.floor(num_trans/2))
        num_each_dir = np.array(np.floor(num_trans/2), dtype=np.int_)
        for x in range(num_trans[0]):
            for y in range(num_trans[1]):
                for z in range(num_trans[2]):
                    x_dir = x if x <= num_each_dir[0] else x-num_trans[0]
                    y_dir = y if y <= num_each_dir[1] else y-num_trans[1]
                    z_dir = z if z <= num_each_dir[2] else z-num_trans[2]
                    vec_to_trans[x,y,z] = [x_dir,y_dir,z_dir]
                    flat_vecs.append([x_dir,y_dir,z_dir])
        
        flat_vecs = np.array(flat_vecs,dtype=np.int_)
        print("check vecs:",flat_vecs)
        num_orbs = self.num_orbs
        full_overlaps = np.zeros((num_orbs*num_vecs,num_orbs*num_vecs))
        full_hams = np.zeros((num_orbs*num_vecs,num_orbs*num_vecs))
        for vec1 in range(num_vecs):
            for vec2 in range(num_vecs):
                vec_diff = flat_vecs[vec1] - flat_vecs[vec2]
                [ind_x,ind_y,ind_z] = vec_diff+num_each_dir #2 from num_each_dir being 2 for the overlaps
                if ind_x < num_trans[0] and ind_y < num_trans[1] and ind_z < num_trans[2]:
                    full_overlaps[vec1*num_orbs:(vec1+1)*num_orbs,vec2*num_orbs:(vec2+1)*num_orbs] = self.overlaps[ind_x,ind_y,ind_z]
                    full_hams[vec1*num_orbs:(vec1+1)*num_orbs,vec2*num_orbs:(vec2+1)*num_orbs] = self.tb_param[ind_x,ind_y,ind_z]
        print("all overlaps:",full_overlaps[:num_orbs])
        
        # Computing diagonalization
        #evalues, evectors = np.linalg.eigh(coeff_overlap)
        #sqrt_matrix = evectors * np.sqrt(evalues) @ np.linalg.inv(evectors)
        sqrt_overlap =  linalg.sqrtm(full_overlaps) #sqrt_matrix
        sqrt_overlap = np.array(sqrt_overlap, dtype=np.float64)
        #print(coeff_overlap[:2])
        inv_sqrt_overlap = np.linalg.inv(sqrt_overlap)
        print("lowdin orth matrix:",inv_sqrt_overlap[:num_orbs,:4*num_orbs])
        
        # orthogonalize the TBparams
        orth_ham = np.matmul(inv_sqrt_overlap.T,np.matmul(full_hams,inv_sqrt_overlap))
        orth_laps = np.matmul(np.linalg.inv(full_overlaps),full_overlaps)
        shape = self.tb_param.shape
        print("shape",shape)
        orth_params = np.zeros(self.tb_param.shape,dtype=np.float64)
        orth_overlaps = np.zeros(self.tb_param.shape,dtype=np.float64)
        print(orth_params.shape)
        for vec1 in range(num_vecs):
            for vec2 in range(num_vecs):
                vec_diff = flat_vecs[vec1] - flat_vecs[vec2]
                [ind_x,ind_y,ind_z] = vec_diff+num_each_dir #2 from num_each_dir being 2 for the overlaps
                if ind_x < num_trans[0] and ind_y < num_trans[1] and ind_z < num_trans[2]:
                    orth_params[ind_x,ind_y,ind_z] = orth_ham[vec1*num_orbs:(vec1+1)*num_orbs,vec2*num_orbs:(vec2+1)*num_orbs]
                    orth_overlaps[ind_x,ind_y,ind_z] = orth_laps[vec1*num_orbs:(vec1+1)*num_orbs,vec2*num_orbs:(vec2+1)*num_orbs]
        self.orth_params = orth_params
        self.orth_overlaps = orth_overlaps
        self.tb_param = orth_params
        self.overlaps = orth_overlaps
        self.write_TBfiles()
        
    
    def get_offset(self,a=2):
        # this function tries to center the bands 
        num_each_dir = self.num_each_dir
        num_trans = self.vec_to_trans
        # this method find the offset that maximizes the similarity between the TB params and overlap params
        tbparams = copy.deepcopy(self.tb_param)
        overlaps = copy.deepcopy(self.overlaps)
        for orb in range(self.num_orbs):
                tbparams[num_each_dir[0],num_each_dir[1],num_each_dir[2],orb,orb] = 0
                overlaps[num_each_dir[0],num_each_dir[1],num_each_dir[2],orb,orb] = 0
        test_ratio = 1
        
        def min_offset(x,orb1,orb2):
            #orb1,orb2 = args
            cur_tb = tbparams[:,:,:,orb1,orb2].flatten()
            cur_overlap = overlaps[:,:,:,orb1,orb2].flatten()
            cur_tb = cur_tb[np.abs(cur_overlap)>0.01]
            cur_overlap = cur_overlap[np.abs(cur_overlap)>0.01]
            arg_max = np.argmax(np.abs(cur_overlap))
            test_ratio = cur_tb[arg_max]/cur_overlap[arg_max]
            summed_diff = np.sum(np.abs(cur_tb-x*cur_overlap+test_ratio*cur_overlap))
            return summed_diff
        
        from scipy.optimize import minimize
        for orb1 in range(self.num_orbs):
            for orb2 in range(self.num_orbs):
                result = minimize(min_offset,1,args=(orb1,orb2))
                print("check result:",orb1,orb2,result.x,test_ratio)
            
        
        
        '''
        #this function determines the energy offset required to achieve an integrated COHP of all states of 0
        num_each_dir = self.num_each_dir
        num_trans = self.vec_to_trans
        
        energysum = 0
        overlapsum = 0
        for kpt in range(self.num_kpts):
            scaledTB,scaledOvlap = self.get_interp_ham(kpt,return_params=True) #[orb1,orb2,trans1,trans2,trans3]
            #energyCont = np.zeros((self.numOrbs, self.numOrbs,num_trans[0], num_trans[1], num_trans[2],self.num_orbs), dtype=np.complex128) #[band,orb,orb,trans,trans,trans]
            coeff = self.mnkcoefficients[kpt,:self.num_orbs] #[band, orb]
            energyCont = scaledTB[None,:,:,:,:,:]*np.conj(coeff)[:,:,None,None,None,None]*coeff[:,None,:,None,None,None]
            overlapCont = scaledOvlap[None,:,:,:,:,:]*np.conj(coeff)[:,:,None,None,None,None]*coeff[:,None,:,None,None,None]
            #for orb1 in range(self.numOrbs):
            #    for orb2 in range(self.numOrbs):
            #        energyCont[orb1,orb2] = scaledTBparams[orb1,orb2]*np.conj(eigen_var[orb1])*eigen_var[orb2]
            #print(energyCont.shape,overlapCont.shape)
            #set onsite to zero
            for orb in range(self.num_orbs):
                #print(orb,orb,num_each_dir[0],num_each_dir[1],num_each_dir[2])
                #print("onsite term:",overlapCont[:,orb,orb,num_each_dir[0],num_each_dir[1],num_each_dir[2]],energyCont[:,orb,orb,num_each_dir[0],num_each_dir[1],num_each_dir[2]])
                energyCont[:,orb,orb,num_each_dir[0],num_each_dir[1],num_each_dir[2]] = 0
                overlapCont[:,orb,orb,num_each_dir[0],num_each_dir[1],num_each_dir[2]] = 0
            
            kpt_offset = np.sum(energyCont)/np.sum(overlapCont)
            print("kpt offset:",kpt,kpt_offset)
            energysum = energysum + np.sum(energyCont)
            overlapsum = overlapsum + np.sum(overlapCont)
        print("final offset:",energysum/overlapsum)
        '''

    def write_TBfiles(self):
        # get the maximum interatomic distance
        
        kprim1 = np.sort(np.around(np.abs(self.kpoints[:,0]),decimals=6))
        kprim2 = np.sort(np.around(np.abs(self.kpoints[:,1]),decimals=6))
        kprim3 = np.sort(np.around(np.abs(self.kpoints[:,2]),decimals=6))
        max1 = np.around(1/kprim1[kprim1 != 0][0],decimals=0) if len(kprim1[kprim1 != 0]) > 0 else 1 
        max2 = np.around(1/kprim2[kprim2 != 0][0],decimals=0) if len(kprim2[kprim2 != 0]) > 0 else 1 
        max3 = np.around(1/kprim3[kprim3 != 0][0],decimals=0) if len(kprim3[kprim3 != 0]) > 0 else 1 
        print("cutoff for params!",max1,max2,max3)
        
        #write the TB parameters
        wannier_hr = open(self.directory+"TBparams"+self.tag+".txt", "w")
        wannier_hr.write("File generated by COGITO\n")
        num_trans = np.array(self.num_trans)
        num_each_dir = np.floor(num_trans/2)
        wannier_hr.write(str(int(max1/2))+" "+str(int(max2/2))+" "+str(int(max3/2))+" "+str(self.num_orbs)+"\n")

        wannier_hr.write("start params\n")
        for a1 in range(num_trans[0]):
            trans1 = int(a1 - np.floor(num_trans[0]/2))
            for a2 in range(num_trans[1]):
                trans2 = int(a2 - np.floor(num_trans[1] / 2))
                for a3 in range(num_trans[2]):
                    trans3 = int(a3 - np.floor(num_trans[2] / 2))
                    for orb1 in range(self.num_orbs):
                        for orb2 in range(self.num_orbs):
                            orb_dist = np.array([trans1,trans2,trans3]) + self.orbpos[orb2] - self.orbpos[orb1]
                            if abs(orb_dist[0]) <= max1/2 and abs(orb_dist[1]) <= max2/2 and abs(orb_dist[2]) <= max3/2:
                                realpart = f"{self.tb_param[a1,a2,a3,orb1,orb2].real:.12f}"
                                imagpart = f"{self.tb_param[a1,a2,a3,orb1,orb2].imag:.12f}"
                                wannier_hr.write(
                                    '{:>5} {:>5} {:>5} {:>5} {:>5} {:>14} {:>14}'.format(str(trans1), str(trans2), str(trans3),
                                                                             str(orb1+1), str(orb2+1),realpart, imagpart) + "\n")
        #write the overlap parameters
        wannier_hr = open(self.directory+"overlaps"+self.tag+".txt", "w")
        wannier_hr.write("File generated by COGITO\n")
        wannier_hr.write(str(int(max1/2))+" "+str(int(max2/2))+" "+str(int(max3/2))+" "+str(self.num_orbs)+"\n")
        num_trans = self.num_trans

        wannier_hr.write("start params\n")
        for a1 in range(num_trans[0]):
            trans1 = int(a1 - np.floor(num_trans[0]/2))
            for a2 in range(num_trans[1]):
                trans2 = int(a2 - np.floor(num_trans[1] / 2))
                for a3 in range(num_trans[2]):
                    trans3 = int(a3 - np.floor(num_trans[2] / 2))
                    for orb1 in range(self.num_orbs):
                        for orb2 in range(self.num_orbs):
                            orb_dist = np.array(np.array([trans1,trans2,trans3]) + self.orbpos[orb2] - self.orbpos[orb1])
                            if abs(orb_dist[0]) <= max1/2 and abs(orb_dist[1]) <= max2/2 and abs(orb_dist[2]) <= max3/2:
                                realpart = f"{self.overlaps[a1,a2,a3,orb1,orb2].real:.12f}"
                                imagpart = f"{self.overlaps[a1,a2,a3,orb1,orb2].imag:.12f}"
                                wannier_hr.write(
                                    '{:>5} {:>5} {:>5} {:>5} {:>5} {:>14} {:>14}'.format(str(trans1), str(trans2), str(trans3),
                                                                             str(orb1+1), str(orb2+1),realpart, imagpart) + "\n")

    def write_recip_rad_orbs(self):
        recip_coeff = self.recip_orbcoeffs #[orb,9]
        #[ak, bk, ck, dk, ek, fk, gk, hk, l] = recip_coeff
        orb_peratom = np.array([self.sph_harm_key])
        all_orb_info = np.append(recip_coeff,orb_peratom.T,axis=1)
        np.save(self.directory + "orbitals"+self.tag, all_orb_info)

        # old way to save
        '''
        recip_rad_orbs = self.recip_rad_orbs

        recip_rad_max = self.rad_grid_max
        recip_grid_size = self.rad_grid_size
        num_orbs = self.num_orbs


        file = open(self.directory+"recip_rad_orbs.txt","w")
        file.write(str(num_orbs)+ " " + str(recip_grid_size) + " " + str(recip_rad_max)+"\n")

        orbkeys = list(recip_rad_orbs.keys())
        for oind,orb in enumerate(orbkeys):
            orbital = np.around(recip_rad_orbs[orb].real,decimals=6)
            orb_per_atom = self.sph_harm_key[oind]
            atomnum = self.orbatomnum[oind]
            center = np.around(self.primAtoms[atomnum],decimals=6)
            file.write("\nnew_orb"+" "+str(orb_per_atom)+"\n")
            file.write("center " + str(center[0])+" "+str(center[1])+" "+str(center[2])+"\n")
            num_lines = math.floor(len(orbital)/6)
            for line in range(num_lines):
                start = line*6
                file.write('{:>14} {:>14} {:>14} {:>14} {:>14} {:>14}'.
                           format(f"{orbital[start]:.8f}", f"{orbital[start+1]:.8f}", f"{orbital[start+2]:.8f}",
                                  f"{orbital[start+3]:.8f}", f"{orbital[start+4]:.8f}", f"{orbital[start+5]:.8f}") + "\n")
            for final_elements in range(start + 6, len(orbital)):
                file.write('{:>14} '.
                           format(f"{orbital[final_elements]:.8f}"))
        '''

    def read_recip_rad_orbs(self,tag=''):
        if not hasattr(self, 'tag'):
            self.tag = tag
        #recip_coeff = self.recip_orbcoeffs #[orb,9]
        #[ak, bk, ck, dk, ek, fk, gk, hk, l] = recip_coeff
        #orb_peratom = np.array([self.sph_harm_key])
        #all_orb_info = np.append(recip_coeff,orb_peratom.T,axis=1)
        #np.save(self.directory + "orbitals"+self.tag, all_orb_info)
        all_orb_info = np.load(self.directory + "orbitals"+self.tag+'.npy')
        recip_coeff = all_orb_info[:,:-1]
        self.recip_orbcoeffs = recip_coeff

    def get_proj_on_aeorb(self):
        # this function gets the pseudo orbitals projection on it's main ae orbital
        # this is different than obtaining the total ae overlap correction, which includes many more projectors
        ae_proj = np.zeros(self.num_orbs)
        for orb in range(self.num_orbs):
            #recip stuff
            atomnum = self.orbatomnum[orb]
            recip_proj_dat = self.atmRecipProjdata[self.elements[atomnum]]
            proj_gmax = recip_proj_dat[0]
            recip_proj_grid = np.arange(100) * proj_gmax / 100
            recip_proj_rad = recip_proj_dat[1][self.orbtype[orb]]
            
            # back to the fitted gaussian orbital
            old_params = self.recip_orbcoeffs[orb]
            gaus_orb = func_for_rad(recip_proj_grid,*old_params)
            
            #gaus_gaus_orbs = gaus_orb * gaus_orb
            #ae_overlap = gaus_norm
            #for orb1 in orbs:
            #    type1 = all_orbtype[orb1]
            gaus_proj_orbs1 = gaus_orb * recip_proj_rad
            gaus_proj_norm1 = integrate.trapezoid((gaus_proj_orbs1) * recip_proj_grid ** 2, x=recip_proj_grid)
            proj_proj_norm = integrate.trapezoid((recip_proj_rad*recip_proj_rad) * recip_proj_grid ** 2, x=recip_proj_grid)
            #    for orb2 in orbs:
            #        type2 = all_orbtype[orb2]
            #gaus_proj_orbs2 = gaus_orb * recip_proj_rad
            #gaus_proj_norm2 = integrate.trapezoid((gaus_proj_orbs2) * recip_proj_grid ** 2, x=recip_proj_grid)
            ae_proj[orb] = gaus_proj_norm1 #/ self.vol**(1/2) #* gaus_proj_norm1

            # also get the norm of the ae orbital
            ae_data = self.atmPsuedoAeData[self.elements[atomnum]]
            orbtype = self.orbtype[orb]
            ae_orb = np.array(ae_data[2][orbtype])
            ae_grid = np.array(ae_data[0])
            cutoff = ae_data[3]
            #print("cutoff:",cutoff,ae_grid[-1])
            ae_orb[ae_grid > cutoff] = 0
            ae_ae_norm = integrate.trapezoid((ae_orb*ae_orb) * ae_grid ** 2, x=ae_grid)
            
            print(ae_ae_norm)

        print("ae projections:",ae_proj)
        
    def write_input_file(self):
        # writes necessary information to file
        # includes: lattice vectors, atom type and position, orbital position and corresponding atom
        file = open(self.directory+"tb_input"+self.tag+".txt","w")
        file.write("lattice_vecs\n")
        a = self._a
        file.write('{:>14} {:>14} {:>14}'.format(f"{a[0][0]:.8f}", f"{a[0][1]:.8f}", f"{a[0][2]:.8f}") + "\n")
        file.write('{:>14} {:>14} {:>14}'.format(f"{a[1][0]:.8f}", f"{a[1][1]:.8f}", f"{a[1][2]:.8f}") + "\n")
        file.write('{:>14} {:>14} {:>14}'.format(f"{a[2][0]:.8f}", f"{a[2][1]:.8f}", f"{a[2][2]:.8f}") + "\n")

        file.write("atoms\n")
        for atom in range(self.numAtoms):
            atom_pos = self.primAtoms[atom]
            atom_type = self.elements[atom]
            elec = self.atm_elec[atom]
            file.write('{:>14} {:>14} {:>14} '.format(f"{atom_pos[0]:.8f}", f"{atom_pos[1]:.8f}", f"{atom_pos[2]:.8f}"))
            file.write(atom_type)
            file.write(' {:>7}'.format(f"{elec:.4f}"))
            file.write("\n")

        file.write("orbitals\n")
        for orb in range(self.num_orbs):
            orb_pos = self.orbpos[orb]
            orb_atom = self.orbatomnum[orb]
            orb_type = self.orbtype[orb]
            file.write('{:>14} {:>14} {:>14} '.format(f"{orb_pos[0]:.8f}", f"{orb_pos[1]:.8f}", f"{orb_pos[2]:.8f}"))
            file.write(str(orb_atom)+" ")
            file.write(orb_type+"\n")

        file.write("orbital spherical harmonics combo\n")
        for orb in range(self.num_orbs):
            orb_vec = self.orb_vec[orb]
            for val in orb_vec:
                file.write('{:>16} '.format(f"{val:.12f}"))
            file.write("\n")

        file.write("fermi energy " + '{:>14}'.format(f"{self.efermi:.8f}") + "\n")
        file.write("energy shift " + '{:>14}'.format(f"{self.energy_shift:.8f}") + "\n")
        file.write("spin polar " + str(self.spin_polar))

    def plot_BS(self):

        fig = plt.figure(figsize=(20,12))
        ax = fig.add_subplot(111)
        for band in range(self.max_band):
            ax.plot(range(self.num_kpts), self.eigval[:self.num_kpts,band,0].real)#, c='black')

        print("showing plot")
        plt.show()


    def plot_projectedBS(self,orbs):
        #orbs: either a 1D array of the actual positions (eg: [1,2,3])
        #       a 2D array where the first row is the elements and 2nd is the orbital type
        #       eg: [["Pb"],["s","p"]]
        if type(orbs[0]) is not int:
            neworbs = []
            #find orbs for the element
            for element in orbs[0]:
                for orbtype in orbs[1]:
                    for addorb in range(self.num_orbs):
                        if self.orbatomname[addorb] == element and orbtype in self.exactorbtype[addorb]:
                            neworbs.append(addorb)
            #print(neworbs)
            orbs = neworbs

        band_orb_weight = np.zeros(self.mnkcoefficients.shape, dtype=np.complex128)
        for kpt in range(self.num_kpts):
            coeffs = self.mnkcoefficients[kpt]
            orb_overlap = self.Sij[kpt]
            #reg_overlap = np.matmul(np.conj(coeffs),np.matmul(orb_overlap, coeffs.transpose())) # [band,orb]*([orb,orb]*[orb,band]) = [band,orb]*[orb,band] = [band,band]
            band_orb_weight[kpt] = np.conj(coeffs)*np.matmul(orb_overlap, coeffs.transpose()).T # [band,orb]. ([orb,orb]*[orb,band]).T = [band,orb].[band,orb] = [band,orb]
        
        projections = np.sum(band_orb_weight[:,:,orbs],axis=-1).real
        
        fig = plt.figure(figsize=(20,12))
        ax = fig.add_subplot(111)
        for band in range(self.max_band):
            for kpt in range(self.num_kpts):
                #if band < self.num_orbs:
                ax.plot(kpt, self.eigval[kpt][band][0].real, 'o',c='blue', markersize=projections[kpt,band]*6+0.5)
            ax.plot(range(self.num_kpts), self.eigval[:self.num_kpts,band,0].real)#, c='black')

        print("showing plot")
        plt.show()

    def plot_projectedDOS(self,orbs,xlim=None):
        # orbs: either a 1D array of the actual positions (eg: [1,2,3])
        #       a 2D array where the first row is the elements and 2nd is the orbital type
        #       eg: [["Pb"],["s","p"]]
        if type(orbs[0]) is not int:
            neworbs = []
            # find orbs for the element
            for element in orbs[0]:
                for orbtype in orbs[1]:
                    for addorb in range(self.num_orbs):
                        if self.orbatomname[addorb] == element and orbtype in self.exactorbtype[addorb]:
                            neworbs.append(addorb)
            print(neworbs)
            orbs = neworbs

        band_orb_weight = np.zeros(self.mnkcoefficients.shape, dtype=np.complex128)
        justkptweight = np.ones(self.mnkcoefficients[:, :, 0].shape)
        for kpt in range(self.num_kpts):
            coeffs = self.mnkcoefficients[kpt]
            orb_overlap = self.Sij[kpt]
            #reg_overlap = np.matmul(np.conj(coeffs),np.matmul(orb_overlap, coeffs.transpose())) # [band,orb]*([orb,orb]*[orb,band]) = [band,orb]*[orb,band] = [band,band]
            band_orb_weight[kpt] = np.conj(coeffs)*np.matmul(orb_overlap, coeffs.transpose()).T # [band,orb]. ([orb,orb]*[orb,band]).T = [band,orb].[band,orb] = [band,orb]
        
        totprojections = np.sum(band_orb_weight,axis=-1)
        projections = np.sum(band_orb_weight[:,:,orbs],axis=-1)
        
        totprojections = (totprojections.T*self.kpt_weights).T
        projections = (projections.T*self.kpt_weights).T

        justkptweight = (justkptweight.T*self.kpt_weights).T
        #print("proj shape:",totprojections.shape,self.eigval[:,:,0].shape)

        fig = plt.figure()
        fig.set_size_inches(15, 5)
        ax = fig.add_subplot(111)
        numbins = 151
        print("max band:",self.max_band)
        origdos,x,_=ax.hist(self.eigval[:,:self.max_band,0].flatten(), bins=numbins,weights=justkptweight.flatten(),color="white")
        totproj,_,_=ax.hist(self.eigval[:,:self.max_band,0].flatten(), bins=numbins,weights=totprojections.flatten(),color="white")
        myproj,_,_=ax.hist(self.eigval[:,:self.max_band,0].flatten(), bins=numbins,weights=projections.flatten(),color="white")
        cut = 0
        #origden = signal.savgol_filter(origdos[cut:],numbins-cut,numbins-cut-1)
        #totden = signal.savgol_filter(totproj[cut:],numbins-cut,numbins-cut-1)
        #myden = signal.savgol_filter(myproj[cut:],numbins-cut,numbins-cut-1)
        origden = smooth(origdos[cut:], 3)
        totden = smooth(totproj[cut:], 3)
        myden = smooth(myproj[cut:], 3)
        ax.plot(x[cut:-1],origden*2+1,color = "black")#-self.efermi
        ax.plot(x[cut:-1],totden*2+1,color = "blue")#-self.efermi
        ax.plot(x[cut:-1],myden*2+1,color = "black")#-self.efermi
        #pymatgen dos plots
        allprojected = list(self.spd_dos.values())[0].densities[Spin.up]+list(self.spd_dos.values())[1].densities[Spin.up]+list(self.spd_dos.values())[2].densities[Spin.up]
        allprojected = smooth(allprojected,2)
        ax.plot(self.pymatenergies,allprojected/4,color = "pink")#-self.efermi
        ax.plot(self.pymatenergies,self.total_dos.densities[Spin.up]/4,color = "red")#-self.efermi

        ax.plot()
        ax.set_ylim([0,2])
        if xlim != None:
            ax.set_xlim(xlim)

        print("showing plot")
        plt.show()

    def generate_gpnts(self,kpt):  # from pymatgen.io.vasp.outputs.Wavecar
        self._C = 0.262465831
        self.encut = 520

        self.b = self.b_vecs
        self._nbmax = [0, 0, 0]
        self._nbmax[0] = int(np.around(np.linalg.norm(self._a[0]) * 2, decimals=0))
        self._nbmax[1] = int(np.around(np.linalg.norm(self._a[1]) * 2, decimals=0))
        self._nbmax[2] = int(np.around(np.linalg.norm(self._a[2]) * 2, decimals=0))
        #print(self._nbmax)

        gpoints = []
        G_ind = 0
        #i3, j2, k1 = np.mgrid[-self._nbmax[2]:self._nbmax[2]:(2 * self._nbmax[2] + 1) * 1j,
        #             -self._nbmax[1]:self._nbmax[1]:(2 * self._nbmax[1] + 1) * 1j,
        #             -self._nbmax[0]:self._nbmax[0]:(2 * self._nbmax[0] + 1) * 1j]
        
        i3, j2, k1 = np.mgrid[0:2*self._nbmax[2]:(2 * self._nbmax[2] + 1) * 1j,
                     0:2*self._nbmax[1]:(2 * self._nbmax[1] + 1) * 1j,
                     0:2*self._nbmax[0]:(2 * self._nbmax[0] + 1) * 1j]
        i3 = i3.flatten()
        j2 = j2.flatten()
        k1 = k1.flatten()
        i3[i3 > self._nbmax[2]] = i3[i3 > self._nbmax[2]] - (2 * self._nbmax[2] + 1)
        j2[j2 > self._nbmax[1]] = j2[j2 > self._nbmax[1]] - (2 * self._nbmax[1] + 1)
        k1[k1 > self._nbmax[0]] = k1[k1 > self._nbmax[0]] - (2 * self._nbmax[0] + 1)
        # nonvectorized version that pymatgen uses
        # j2 = np.arange(2 * self._nbmax[1] + 1)- self._nbmax[1]
        # k1 = np.arange(2 * self._nbmax[0] + 1)- self._nbmax[0]

        # for i in range(2 * self._nbmax[2] + 1):
        #    i3 = i - self._nbmax[2] # - 2 * self._nbmax[2] - 1 if i > self._nbmax[2] else i
        #    for j in range(2 * self._nbmax[1] + 1):
        #        j2 = j - self._nbmax[1] # - 2 * self._nbmax[1] - 1 if j > self._nbmax[1] else j
        #        for k in range(2 * self._nbmax[0] + 1):
        #            k1 = k - self._nbmax[0] # - 2 * self._nbmax[0] - 1 if k > self._nbmax[0] else k
        G = np.array([k1, j2, i3])
        v = np.array(np.array([kpt]).T + G).T
        g = np.linalg.norm(np.dot(v, self.b), axis=1)
        E = np.array(g ** 2 / self._C)
        # if E < self.encut:
        #    gpoints.append(G)
        #    G_ind += 1
        gpoints1 = G[0][E < self.encut]
        gpoints2 = G[1][E < self.encut]
        gpoints3 = G[2][E < self.encut]
        gpoints = np.array([gpoints1, gpoints2, gpoints3],dtype=np.int_).T
        return gpoints


def func_for_rad(x, a,b,c,d,e,f,g,h,l):
    return x**l * (a/b*np.exp(-1/2*(x/b)**2)+c/d*np.exp(-1/2*(x/d)**2)+e/f*np.exp(-1/2*(x/f)**2)+g/h*np.exp(-1/2*(x/h)**2))

def func_for_rad_fit(x, a,b,con1,con2,con3,c,g,con4,l):
    # this function is for fitting the gaussian coefficients with constraints that disallow nodes
    d = b * con1 # limit con1 between 0 and 1
    f = b * con3 # limit con3 between 0 and 1 (except if node)
    h = b * con4 # limit con4 between 0 and 1
    e = (con2 - a/b) * f # limit con2 to be positive (except if node)
    return x**l * (a/b*np.exp(-1/2*(x/b)**2)+c/d*np.exp(-1/2*(x/d)**2)+e/f*np.exp(-1/2*(x/f)**2)+g/h*np.exp(-1/2*(x/h)**2))

def func_for_rad_exp(x, a,b,c,d,e,f,l):
    return x**l * (a/b*np.exp(-x/b)+c/d*np.exp(-x/d)+e/f*np.exp(-x/f))

def lowdin_orth_vectors(vectors):
    #coeff_overlap = []
    #for u in vectors:
    #    row = []
    #    for v in vectors:
    #        row.append(np.dot(np.conj(u), v))
    #    coeff_overlap.append(row)
    #coeff_overlap = np.array(coeff_overlap)
    coeff_overlap = np.matmul(np.conj(vectors), vectors.transpose())
    #print("check orth overlap:",coeff_overlap)
    # Computing diagonalization
    #evalues, evectors = np.linalg.eigh(coeff_overlap)
    #sqrt_matrix = evectors * np.sqrt(evalues) @ np.linalg.inv(evectors)
    sqrt_overlap =  linalg.sqrtm(coeff_overlap) #sqrt_matrix
    sqrt_overlap = np.array(sqrt_overlap, dtype=np.complex128)
    #print(coeff_overlap[:2])
    inv_sqrt_overlap = np.linalg.inv(sqrt_overlap)
    #print("coeff overlap:",coeff_overlap)
    #orth_vectors = []
    #for vec1 in range(len(vectors)):
    #    add_to_orb = np.zeros((vectors[0].shape), dtype=np.complex128)
    #    for vec2 in range(len(vectors)):
    #        add_to_orb += np.conj(inv_sqrt_overlap[vec1,vec2])*vectors[vec2]
    #    orth_vectors.append(add_to_orb)
    orth_vectors = np.matmul(np.conj(inv_sqrt_overlap), vectors)
    #print(orth_vectors)
    return orth_vectors

def lowdin_orth_vectors_orblap(vectors,orblap,energies=None,return_energy=False):
    #coeff_overlap = []
    #for u in vectors:
    #    row = []
    #    for v in vectors:
    #        #row.append(np.dot(np.conj(u), v))
    #        row.append(np.matmul(np.conj(u), np.matmul(orblap, v.transpose())))
    #    coeff_overlap.append(row)
    #coeff_overlap = np.array(coeff_overlap)
    coeff_overlap = np.matmul(np.conj(vectors), np.matmul(orblap, vectors.transpose()))

    # Computing diagonalization
    #evalues, evectors = np.linalg.eigh(coeff_overlap)
    #sqrt_matrix = evectors * np.sqrt(evalues) @ np.linalg.inv(evectors)
    sqrt_overlap =  linalg.sqrtm(coeff_overlap) #sqrt_matrix
    sqrt_overlap = np.array(sqrt_overlap, dtype=np.complex128)
    #print(coeff_overlap[:2])
    inv_sqrt_overlap = np.linalg.inv(sqrt_overlap)
    #print("coeff overlap:",coeff_overlap)
    #orth_vectors = []
    #for vec1 in range(len(vectors)):
    #    add_to_orb = np.zeros((vectors[0].shape), dtype=np.complex128)
    #    for vec2 in range(len(vectors)):
    #        add_to_orb += np.conj(inv_sqrt_overlap[vec1,vec2])*vectors[vec2]
    #    orth_vectors.append(add_to_orb)
    orth_vectors = np.matmul(np.conj(inv_sqrt_overlap), vectors)

    if return_energy==True:
        # most legit way
        new_energy = np.matmul(inv_sqrt_overlap.T,np.matmul(energies,np.conj(inv_sqrt_overlap)))*np.diag(coeff_overlap)
        # way which is conceptually simplier and only diagonal terms
        transfer = inv_sqrt_overlap * np.conj(inv_sqrt_overlap)*np.diag(coeff_overlap)[:,None]**(1/2)*np.diag(coeff_overlap)[None,:]**(1/2)
        #print("transfer:",transfer)
        new_energy = np.diag(np.sum(transfer[:,:]*np.diag(energies)[:,None],axis=0))
        
        #new_energy = np.zeros((energies.shape), dtype=np.complex128)
        #for vec1 in range(len(vectors)):
        #    add_to_orb = np.zeros((energies[0].shape), dtype=np.complex128)
        #    for vec2 in range(len(vectors)):
        #        add_to_orb += np.conj(inv_sqrt_overlap[vec1,vec2])*energies[vec2]
        #    new_energy[vec1] = add_to_orb
        
        return orth_vectors,new_energy
    else:
        #print(orth_vectors)
        return orth_vectors

def GS_orth_twoLoworthSets(vectors1,vectors2):
    num_bands1 = len(vectors1)
    num_bands2 = len(vectors2)
    #print("num both bands:",num_bands1,num_bands2)
    coeff_overlap = []
    for u in vectors1:
        row = []
        for v in vectors2:
            row.append(np.dot(np.conj(u), v))
        coeff_overlap.append(row)

    #print("GS check orth overlap:",coeff_overlap)
    #coeff_overlap = np.array(coeff_overlap)
    #sqrt_overlap = scipy.linalg.sqrtm(coeff_overlap)
    #print("in GS",coeff_overlap)
    #inv_sqrt_overlap = np.linalg.inv(sqrt_overlap)
    # print("coeff overlap:",coeff_overlap)
    orth_vectors2 = vectors2
    for vec2 in range(num_bands2):
        add_to_orb = np.zeros((vectors2[0].shape), dtype=np.complex128)
        for vec1 in range(num_bands1):
            add_to_orb += - coeff_overlap[vec1][vec2] * vectors1[vec1]
        orth_vectors2[vec2] = orth_vectors2[vec2] +add_to_orb
    # print(orth_vectors)
    return orth_vectors2

def GS_orth_twoLoworthSets_orblap(vectors1,vectors2,orblap,energy1=None,energy2=None,return_energy=False,energycut=0,ratio=[]):
    if len(ratio) == 0:
        ratio = np.ones(len(vectors1))
    num_bands1 = len(vectors1)
    num_bands2 = len(vectors2)
    #print("num both bands:",num_bands1,num_bands2)
    #coeff_overlap = []
    #for u in vectors1:
    #    row = []
    #    for v in vectors2:
    #        #row.append(np.dot(np.conj(u), v))
    #        row.append(np.matmul(np.conj(u), np.matmul(orblap, v.transpose())))
    #    coeff_overlap.append(row)
    #coeff_overlap = np.array(coeff_overlap,dtype=np.complex128)
    coeff_overlap = np.matmul(np.conj(vectors1), np.matmul(orblap, vectors2.transpose()))
    #print("overlap:",coeff_overlap)
    orth_vectors2 = copy.deepcopy(vectors2)
    #for vec2 in range(num_bands2):
    #    add_to_orb = np.zeros((vectors2[0].shape), dtype=np.complex128)
    #    for vec1 in range(num_bands1):
    #        add_to_orb = add_to_orb - coeff_overlap[vec1][vec2] * vectors1[vec1] * ratio[vec1]
    #    #print(add_to_orb)
    #    orth_vectors2[vec2] = orth_vectors2[vec2] + add_to_orb
    orth_vectors2 = orth_vectors2 - np.matmul(coeff_overlap.T, vectors1 * ratio[:,None])
        
    
    if return_energy == True:

        norm1 = []
        for v in vectors1:
            # row.append(np.dot(np.conj(u), v))
            norm1.append(np.matmul(np.conj(v), np.matmul(orblap, v.transpose())))
        norm1 = np.array(norm1)

        norm2 = []
        for v in vectors2:
            #row.append(np.dot(np.conj(u), v))
            norm2.append(np.matmul(np.conj(v), np.matmul(orblap, v.transpose())))
        norm2 = np.array(norm2)
        #print("norm:",norm)


        #print("overlap:",coeff_overlap)
        energy2 = np.diag(energy2)
        new_energy2 = energy2#*norm2
        transfer = ((np.conj(coeff_overlap)*coeff_overlap)**(1/2)/norm2[None,:]**(1/2)/norm1[:,None]**(1/2))
        ener_norm = np.ones(len(norm2))+np.sum(transfer,axis=0)
        #print("transfer:",transfer)
        #print(energy1)
        #print(energy2)
        #print(transfer*np.diag(energy1)[:,None])
        #print(np.sum(transfer*np.diag(energy1)[:,None],axis=0))
        #new_energy2 += np.matmul(-np.conj(coeff_overlap).T,np.matmul(energy1,-coeff_overlap))
        new_energy2 = (new_energy2 + np.sum(transfer*np.diag(energy1)[:,None],axis=0))/ener_norm
        # finally only change the energy when further away from energy cutoff to avoid sharp changes
        shift = (energy2-energycut)>0
        notshift = (energy2-energycut)<=0
        new_energy2[shift] = energy2[shift]*(1-np.tanh(np.abs(energy2[shift]-energycut)/5)) + new_energy2[shift]*np.tanh(np.abs(energy2[shift]-energycut)/5)# ~76% of changed energy when 2 eV away from energycut
        new_energy2[notshift] = energy2[notshift]
        #new_energy2 = energy2
        #for vec2 in range(num_bands2):
        #    add_to_orb = 0
        #    for vec1 in range(num_bands1):
        #        add_to_orb += -coeff_overlap[vec1][vec2] * energy1[vec1]
        #    new_energy2[vec2] = new_energy2[vec2] + add_to_orb
        #print(new_energy2)
        return orth_vectors2, np.diag(new_energy2)
    else:
        # print(orth_vectors)
        return orth_vectors2

def GS_combine_states_orblap(vectors1,vectors2,orblap,energy1=None,energy2=None,return_energy=False):
    num_bands1 = len(vectors1)
    num_bands2 = len(vectors2)
    #print("num both bands:",num_bands1,num_bands2)
    coeff_overlap = []
    for u in vectors1:
        row = []
        for v in vectors2:
            #row.append(np.dot(np.conj(u), v))
            row.append(np.matmul(np.conj(u), np.matmul(orblap, v.transpose())))
        coeff_overlap.append(row)
    coeff_overlap = np.array(coeff_overlap,dtype=np.complex128)
    #print("overlap:",coeff_overlap)
    orth_vectors2 = copy.deepcopy(vectors2)
    
    norm1 = [] 
    for v in vectors1:
        #row.append(np.dot(np.conj(u), v))
        norm1.append(np.matmul(np.conj(v), np.matmul(orblap, v.transpose())))
    norm1 = np.array(norm1)
    
    norm2 = [] 
    for v in vectors2:
        #row.append(np.dot(np.conj(u), v))
        norm2.append(np.matmul(np.conj(v), np.matmul(orblap, v.transpose())))
    norm2 = np.array(norm2)
    #print("norm:",norm)
    
    for vec2 in range(num_bands2):
        add_to_orb = np.zeros((vectors2[0].shape), dtype=np.complex128)
        for vec1 in range(num_bands1):
            add_to_orb = add_to_orb + coeff_overlap[vec1][vec2] * vectors1[vec1] #/ (norm1[vec1]*norm2[vec2])**(1/2)
        #print(add_to_orb)
        orth_vectors2[vec2] = orth_vectors2[vec2] + add_to_orb
        
    
    if return_energy == True:

        #print("overlap:",coeff_overlap)
        new_energy2 = energy2*norm2
        transfer = np.conj(coeff_overlap)*coeff_overlap # / (norm1[:,None]*norm2[None,:])**(1/2)
        ener_norm = norm2+np.sum(transfer,axis=0)
        #print("transfer:",transfer)
        #print(energy1)
        #print(energy2)
        #print(transfer*np.diag(energy1)[:,None])
        #print(np.sum(transfer*np.diag(energy1)[:,None],axis=0))
        #new_energy2 += np.matmul(-np.conj(coeff_overlap).T,np.matmul(energy1,-coeff_overlap))
        new_energy2 = (new_energy2 + np.diag(np.sum(transfer*np.diag(energy1)[:,None],axis=0)))/ener_norm
        #new_energy2 = energy2
        #for vec2 in range(num_bands2):
        #    add_to_orb = 0
        #    for vec1 in range(num_bands1):
        #        add_to_orb += -coeff_overlap[vec1][vec2] * energy1[vec1]
        #    new_energy2[vec2] = new_energy2[vec2] + add_to_orb
        #print(new_energy2)
        return orth_vectors2, new_energy2
    else:
        # print(orth_vectors)
        return orth_vectors2

def complex128funs(phi,theta,sphharm_key):
    # sphharmkey is now a 1D array of length 1 (for s), 3 (for p), 5 (for d), or 7 (for f).
    # The vector passed should be an correctly normalized eigenvector of a local environment tensor.
    # Or you could just do [0,0,0,1,0] if you want a specific d orbital, for example.
    key_set = sphharm_key
    if len(key_set) == 1:
        return np.ones(len(phi))*(1/4/np.pi)**(1/2)
    elif len(key_set) == 3: # new order is y z x # old order was x y z ; m = +1, -1, 0
        orb = (key_set[0] * np.sin(theta)*np.sin(phi)*(3/4/np.pi)**(1/2) + # m = -1
               key_set[1] * np.cos(theta)*(3/4/np.pi)**(1/2) + # m = 0
               key_set[2]*np.sin(theta)*np.cos(phi)*(3/4/np.pi)**(1/2)) # m = +1
        return orb
    elif len(key_set) == 5: # new order is dxy dyz dz2 dxz dx2y2 # old order was dz2 dxz dyz dx2y2 dxy ; m = 0, 1, -1, 2, -2
        orb = (key_set[0] * np.sin(theta) ** 2 *np.sin(2*phi)*(15/16/np.pi)**(1/2) +  # m = -2
               key_set[1] * np.sin(theta) * np.cos(theta)*np.sin(phi)*(15/4/np.pi)**(1/2) + # m = -1
               key_set[2] * (3 * np.cos(theta) ** 2 - 1)*(5/16/np.pi)**(1/2) +  # m = 0
               key_set[3] * np.sin(theta) * np.cos(theta)*np.cos(phi)*(15/4/np.pi)**(1/2) + # m = +1
               key_set[4] * np.sin(theta) ** 2 *np.cos(2*phi)*(15/16/np.pi)**(1/2)) # m = +2
        return orb
    else: # new order seen below. Old order was f_z3, f_xz2, f_yz2, f_xyz, f_z(x2-y2), f_x(x2-3y2), f_y(3x2-y2)
        orb = (
            # m = -3: f_y(3x2-y2) -> sin(3*phi)
            key_set[0] * (35 / 32 / np.pi)**(1 / 2) * np.sin(theta)**3 * np.sin(3*phi) +
            # m = -2: f_xyz -> sin(2*phi)
            key_set[1] * (105 / 16 / np.pi)**(1 / 2) * np.sin(theta)**2 * np.cos(theta) * np.sin(2*phi) +
            # m = -1: f_yz2 -> sin(phi)
            key_set[2] * (21 / 32 / np.pi)**(1 / 2) * np.sin(theta) * (5*np.cos(theta)**2 - 1) * np.sin(phi) +
            # m = 0: f_z3 -> z dependence only
            key_set[3] * (7 / 16 / np.pi)**(1 / 2) * (5*np.cos(theta)**3 - 3*np.cos(theta)) +
            # m = +1: f_xz2 -> cos(phi)
            key_set[4] * (21 / 32 / np.pi)**(1 / 2) * np.sin(theta) * (5*np.cos(theta)**2 - 1) * np.cos(phi) +
            # m = +2: f_z(x2-y2) -> cos(2*phi)
            key_set[5] * (105 / 16 / np.pi)**(1 / 2) * np.sin(theta)**2 * np.cos(theta) * np.cos(2*phi) +
            # m = +3: f_x(x2-3y2) -> cos(3*phi)
            key_set[6] * (35 / 32 / np.pi)**(1 / 2) * np.sin(theta)**3 * np.cos(3*phi))
        return orb

def normalize_wf(wavefunc, prim_vec, gridnum, return_integral=False,recip=False):
    # print("wavefunc:",wavefunc[5,5,:],np.conj(wavefunc)[5,5,:],(np.conj(wavefunc)*wavefunc)[5,5,:])

    if recip == False:
        integral = periodic_integral_3d(np.conj(wavefunc) * wavefunc, prim_vec, gridnum)
    else:
        integral = reciprocal_integral(np.conj(wavefunc) * wavefunc)#,prim_vec, gridnum)
    #print("integral:", (integral) ** (1 / 2))
    if return_integral == True:
        return wavefunc / ((integral) ** (1 / 2)), (integral) ** (1 / 2)
    else:
        return wavefunc / ((integral) ** (1 / 2))

def periodic_integral_3d(f,prim_vec,n,multiple_wfs=False):

    func = np.array(f).transpose() #3D float array
    grid = n #[24,24,33]
    #max_val = np.max(func)
    nz = grid[2]
    ny = grid[1]
    nx = grid[0]
    cross_ab = np.cross(prim_vec[0],prim_vec[1])
    supercell_volume = abs(np.dot(cross_ab,prim_vec[2]))
    indivcell_volume = supercell_volume/nz/ny/nx
    volume = indivcell_volume
    integral = (func[0,:,:]+func[nz-1,:,:])*0.5
    tosum = (func[:-1,:,:]+func[1:,:,:])*0.5
    integral = integral + np.sum(tosum,axis=0)
    #for k in range(nz-1):
    #    integral += (func[:,:, k] + func[:,:, k + 1]) * 0.5
    afterz_integral = integral
    integral = (afterz_integral[0,:] + afterz_integral[ny - 1,:]) * 0.5
    tosum = (afterz_integral[:-1,:]+afterz_integral[1:,:])*0.5
    integral = integral + np.sum(tosum,axis=0)
    #for k in range(ny - 1):
    #    integral += (afterz_integral[:, k] + afterz_integral[:, k + 1]) * 0.5
    aftery_integral = integral

    integral = (aftery_integral[0] + aftery_integral[nx - 1]) * 0.5
    tosum = (aftery_integral[:-1]+aftery_integral[1:])*0.5
    integral = integral + np.sum(tosum,axis=0)
    #for k in range(nx - 1):
    #   integral += (aftery_integral[k] + aftery_integral[k + 1]) * 0.5
    total_integral = integral *nz*ny*nx#* volume
    '''
    grid = n
    nz = grid[2]
    ny = grid[1]
    nx = grid[0]
    alt_int = np.sum(f)*nx*ny*nz
    '''
    return total_integral.transpose()

def reciprocal_integral(f):#,prim_vec,n):
    integral = np.sum(f)
    #scale so that is same as real space integral
    #grid = n
    #nz = grid[2]
    #ny = grid[1]
    #nx = grid[0]
    #cross_ab = np.cross(prim_vec[0], prim_vec[1])
    #supercell_volume = abs(np.dot(cross_ab, prim_vec[2]))
    #indivcell_volume = supercell_volume / nz / ny / nx
    #volume = indivcell_volume
    return integral #*volume /nz / ny / nx

def _cart_to_red(tmp,cart):
    """
    Convert cartesian vectors cart to reduced coordinates of a1,a2,a3 vectors.
    """
    #  ex: prim_coord = _cart_to_red((a1,a2,a3),cart_coord)
    (a1,a2,a3)=tmp
    # matrix with lattice vectors
    cnv=np.array([a1,a2,a3])
    # transpose a matrix
    cnv=cnv.T
    # invert a matrix
    cnv=np.linalg.inv(cnv)
    # reduced coordinates
    red=np.zeros_like(cart,dtype=float)
    for i in range(0,len(cart)):
        red[i]=np.dot(cnv,cart[i])
    return red

def _red_to_cart(prim_vec,prim_coord):
    """

    Args:    
        prim_vec (unknown): three float tuples representing the primitive vectors.
        prim_coord (unknown): list of float tuples for primitive coordinates.

    Returns:    
        unknown: list of float tuples for cartesian coordinates.
                        ex: cart_coord = _red_to_cart((a1,a2,a3),prim_coord)
    """
    (a1,a2,a3)=prim_vec
    prim = np.array(prim_coord)
    # cartesian coordinates
    cart=np.zeros_like(prim_coord,dtype=float)
    #for i in range(0,len(cart)):
    cart = [a1]*np.array([prim[:,0]]).T + [a2]*np.array([prim[:,1]]).T + [a3]*np.array([prim[:,2]]).T
    return cart

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def extract_line_data(grid, gridXYZ, p1, p2, tolerance=0.5): # started from ChatGPT
    """
    Extracts orbital magnitude along a line passing through p1 and p2 in a non-Cartesian 3D grid.

    Parameters:
        grid (numpy.ndarray): 3D array of orbital magnitudes.
        gridXYZ (numpy.ndarray): 2D array (3, num_points) of real-space coordinates.
        p1 (tuple): First point (x1, y1, z1) in real space.
        p2 (tuple): Second point (x2, y2, z2) in real space.
        tolerance (float): Distance threshold to include points near the line.


    Returns:    
        distances (numpy.ndarray): Distances along the line.
         magnitudes (numpy.ndarray): Orbital magnitudes at corresponding distances.
    """
    
    # Flatten coordinate and grid arrays for vectorized computation
    points = gridXYZ.T#.reshape(-1, 3)
    magnitudes = grid #.ravel()
    #print(points.shape,points)
    #print(grid.shape,grid)

    # Define line vector and normalize
    p1, p2 = np.array(p1), np.array(p2)
    line_vec = p2 - p1
    line_vec /= np.linalg.norm(line_vec)

    # Compute distance of each point to the line
    p1_to_points = points - p1  # Vector from p1 to all points
    proj_lengths = np.dot(p1_to_points, line_vec)  # Projected length on the line
    #print(proj_lengths)
    proj_points = p1 + np.outer(proj_lengths, line_vec)  # Closest points on the line
    
    # Distance from actual points to projected points (perpendicular distance)
    #print(points.shape,points-proj_points)
    distances_to_line = np.linalg.norm(points - proj_points, axis=1)
    #print(distances_to_line)

    # Mask points within tolerance
    mask = (np.abs(distances_to_line) <= tolerance) & (np.abs(proj_lengths) <= 6)
    selected_distances = proj_lengths[mask]
    selected_magnitudes = magnitudes[mask]

    # Sort by distance along the line
    sort_idx = np.argsort(selected_distances)
    return selected_distances[sort_idx], selected_magnitudes[sort_idx]


def combine_and_save_plots(plots, filename="combined_plot.png", layout=None):
    """
    Combines multiple Matplotlib plots into a single figure and saves to a file.

    Parameters:
        plots (list of matplotlib.figure.Figure): List of Matplotlib figures.
        filename (str): Output filename (supports .png, .pdf, .svg, etc.).
        layout (tuple or str): (rows, cols) for custom layout or "auto" for automatic grid.


    Returns:    
        None
    """
    num_plots = len(plots)
    # Determine layout if not provided
    if layout is None:
        cols = min(3, num_plots)  # Limit to 3 columns for readability
        rows = (num_plots + cols - 1) // cols  # Compute required rows
    else:
        rows, cols = layout
    
    # Create a new figure for combined plots
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axes = axes.flatten() if num_plots > 1 else [axes]  # Ensure axes is iterable
    
    for i, plot in enumerate(plots):
        for ax in plot.axes:  # Extract all axes from the figure
            
            # Replot line plots (including markers)
            for line in ax.get_lines():
                # Check if the line has markers
                if line.get_marker() != 'None':
                    axes[i].plot(line.get_xdata(), line.get_ydata(), 'o', 
                                 markersize=line.get_markersize(), color=line.get_color(), label=line.get_label())
                else:
                    axes[i].plot(line.get_xdata(), line.get_ydata(), label=line.get_label(),
                                 linestyle=line.get_linestyle(), color=line.get_color())
                    
            axes[i].set_xlim(ax.get_xlim())  # Preserve x limits
            axes[i].set_ylim(ax.get_ylim())  # Preserve y limits
            axes[i].set_xlabel(ax.get_xlabel())
            axes[i].set_ylabel(ax.get_ylabel())
            axes[i].set_title(ax.get_title())
            axes[i].legend()
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close(fig)


def plot_matrix(matrix, low_center=0.2, high_center=0.85, filename="matrix.png"):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    
    # Define two normalization schemes
    normlow = mcolors.TwoSlopeNorm(vmin=0, vcenter=low_center, vmax=0.6)
    normhigh = mcolors.TwoSlopeNorm(vmin=0, vcenter=high_center, vmax=1)

    # Create a masked array to separate diagonal and off-diagonal elements
    mask_diag = np.eye(matrix.shape[0], dtype=bool)
    mask_offdiag = ~mask_diag

    # Create the plot
    fig, ax = plt.subplots(figsize=(6, 6))

    # Plot off-diagonal elements
    # Define custom colormap from white to dark red
    cmap = mcolors.LinearSegmentedColormap.from_list("white_to_darkred", ["white", "#f50202"])

    im_offdiag = ax.imshow(np.ma.masked_where(mask_diag, matrix), cmap=cmap, norm=normlow, origin='upper')

    # Plot diagonal elements
    im_diag = ax.imshow(np.ma.masked_where(mask_offdiag, matrix), cmap='Greys', norm=normhigh, origin='upper')

    # Add colorbars with more tick markers
    cbar_low = plt.colorbar(im_offdiag, ax=ax, fraction=0.046, pad=0.20)
    cbar_high = plt.colorbar(im_diag, ax=ax, fraction=0.046, pad=0.12)

    # Customize colorbars
    cbar_low.set_label("Off-diagonal Magnitude")
    cbar_high.set_label("Diagonal Magnitude")

    # Add more tick markers to colorbars
    cbar_low.set_ticks(np.linspace(0, 0.6, 13))   # Adjust number of ticks as needed
    cbar_high.set_ticks(np.linspace(0, 1, 21))  # Adjust number of ticks as needed

    # Optional: Add grid lines
    ax.grid(False)  # Set to True if you want grid lines

    # Optional: Add labels
    
    # Apply tick marks with selective labeling
    
    # Get all indices
    all_ticks = np.arange(matrix.shape[1])
    labeled_ticks = all_ticks[all_ticks % 5 == 0]  # Keep labels only for multiples of 5
    ax.set_xticks(all_ticks)  
    ax.set_xticklabels(["" if i not in labeled_ticks else str(i) for i in all_ticks])  
    ax.set_yticks(all_ticks)  
    ax.set_yticklabels(["" if i not in labeled_ticks else str(i) for i in all_ticks])  

    #ax.set_xticks(np.arange(matrix.shape[1]))
    #ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_xlabel("Column index")
    ax.set_ylabel("Row index")
    ax.set_title("Matrix Heatmap")

    # Save and show the plot
    plt.savefig(filename, dpi=500, bbox_inches="tight")
    plt.show()


def D_real_l3_from_R(R):
    """
    Compute the Wigner D matrix for f orbitals (l=3) from rotation matrix R
    using properly symmetrized and trace-free rank-3 tensors.

    Parameters:
    R : 3x3 rotation matrix

    Returns:
    D_matrix : 7x7 matrix mapping old f orbitals to new f orbitals
    Order: f_z3, f_xz2, f_yz2, f_xyz, f_z(x2-y2), f_x(x2-3y2), f_y(3x2-y2)
    """

    A_f = np.zeros((7, 3, 3, 3))

    # f_z3: proportional to z^3
    # In the harmonic set, f_z3 = 5*z^3 - 3*z*r^2 = 2*z^3 - 3*z*x^2 - 3*z*y^2
    A_f[0, 2, 2, 2] = 2.0
    A_f[0, 2, 0, 0] = -1.0
    A_f[0, 0, 2, 0] = -1.0
    A_f[0, 0, 0, 2] = -1.0
    A_f[0, 2, 1, 1] = -1.0
    A_f[0, 1, 2, 1] = -1.0
    A_f[0, 1, 1, 2] = -1.0

    # f_xz2: proportional to x*z^2
    # In the harmonic set, f_xz2 = x*(5*z^2 - r^2) = x*(4*z^2 - x^2 - y^2)
    A_f[1, 0, 2, 2] = 4.0 / 3.0
    A_f[1, 2, 0, 2] = 4.0 / 3.0
    A_f[1, 2, 2, 0] = 4.0 / 3.0
    A_f[1, 0, 0, 0] = -1.0
    A_f[1, 0, 1, 1] = -1.0 / 3.0
    A_f[1, 1, 0, 1] = -1.0 / 3.0
    A_f[1, 1, 1, 0] = -1.0 / 3.0

    # f_yz2: proportional to y*z^2
    # In the harmonic set, f_yz2 = y*(5*z^2 - r^2) = y*(4*z^2 - x^2 - y^2)
    A_f[2, 1, 2, 2] = 4.0 / 3.0
    A_f[2, 2, 1, 2] = 4.0 / 3.0
    A_f[2, 2, 2, 1] = 4.0 / 3.0
    A_f[2, 1, 0, 0] = -1.0 / 3.0
    A_f[2, 0, 1, 0] = -1.0 / 3.0
    A_f[2, 0, 0, 1] = -1.0 / 3.0
    A_f[2, 1, 1, 1] = -1.0

    # f_xyz: proportional to x*y*z
    A_f[3, 0, 1, 2] = 1.0 / 6.0
    A_f[3, 0, 2, 1] = 1.0 / 6.0
    A_f[3, 1, 0, 2] = 1.0 / 6.0
    A_f[3, 1, 2, 0] = 1.0 / 6.0
    A_f[3, 2, 0, 1] = 1.0 / 6.0
    A_f[3, 2, 1, 0] = 1.0 / 6.0

    # f_z(x2-y2): proportional to z*(x^2-y^2)
    A_f[4, 0, 0, 2] = 1.0 / 3.0
    A_f[4, 0, 2, 0] = 1.0 / 3.0
    A_f[4, 2, 0, 0] = 1.0 / 3.0
    A_f[4, 1, 1, 2] = -1.0 / 3.0
    A_f[4, 1, 2, 1] = -1.0 / 3.0
    A_f[4, 2, 1, 1] = -1.0 / 3.0

    # f_x(x2-3y2): proportional to x*(x^2-3*y^2)
    A_f[5, 0, 0, 0] = 1.0
    A_f[5, 1, 1, 0] = -1.0
    A_f[5, 1, 0, 1] = -1.0
    A_f[5, 0, 1, 1] = -1.0

    # f_y(3x2-y2): proportional to y*(3*x^2-y^2)
    A_f[6, 0, 0, 1] = 1.0
    A_f[6, 0, 1, 0] = 1.0
    A_f[6, 1, 0, 0] = 1.0
    A_f[6, 1, 1, 1] = -1.0

    #print("=" * 80)
    #print("SYMMETRY CHECK")
    #print("=" * 80)
    for a in range(7):
        is_sym = True
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    val = A_f[a, i, j, k]
                    if not (np.isclose(val, A_f[a, i, k, j]) and
                            np.isclose(val, A_f[a, j, i, k]) and
                            np.isclose(val, A_f[a, j, k, i]) and
                            np.isclose(val, A_f[a, k, i, j]) and
                            np.isclose(val, A_f[a, k, j, i])):
                        is_sym = False
                        break
        #print(f"Orbital {a}: Symmetric = {is_sym}")

    # Project to trace-free subspace
    M_f = np.zeros((7, 3, 3, 3))

    for a in range(7):
        # For symmetric tensor, compute trace vector
        T = np.zeros(3)
        for k in range(3):
            T[k] = A_f[a, 0, 0, k] + A_f[a, 1, 1, k] + A_f[a, 2, 2, k]

        # Trace-free projection: M_ijk = A_ijk - (1/5)(δ_ij T_k + δ_ik T_j + δ_jk T_i)
        M_f[a] = A_f[a].copy()
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    correction = 0.0
                    if i == j:
                        correction += T[k]
                    if i == k:
                        correction += T[j]
                    if j == k:
                        correction += T[i]
                    M_f[a, i, j, k] -= correction / 5.0
                    #print("Was is trace-free?", correction) # Yes, correction always is zero now

        # Normalize to unit Frobenius norm
        norm = np.linalg.norm(M_f[a])
        M_f[a] /= norm

    # Compute the D matrix: D_ab = <M_a, R·M_b·R^T>
    D_matrix = np.zeros((7, 7))

    for a in range(7):
        for b in range(7):
            # Rotate M_b by applying R on each index
            M_b_rot = np.einsum('ia,jb,kc,abc->ijk', R, R, R, M_f[b], optimize=True)
            # Frobenius inner product
            D_matrix[a, b] = np.tensordot(M_f[a], M_b_rot, axes=3)

    vasp_map = [6, 3, 2, 0, 1, 4, 5]

    # Create the reordered matrix
    D_matrix_vasp = np.zeros((7, 7))
    for i in range(7):
        for j in range(7):
            # The element at [i, j] in VASP order comes from
            # [vasp_map[i], vasp_map[j]] in your internal order
            D_matrix_vasp[i, j] = D_matrix[vasp_map[i], vasp_map[j]]

    return D_matrix_vasp


'''
def plot_matrix(matrix,low_center=0.2,high_center=0.85,filename="matrix.png"):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    
    # Define two normalization schemes
    normlow = mcolors.TwoSlopeNorm(vmin=0, vcenter=low_center, vmax=1)
    normhigh = mcolors.TwoSlopeNorm(vmin=0, vcenter=high_center, vmax=1)
    
    # Create a masked array to separate diagonal and off-diagonal elements
    mask_diag = np.eye(matrix.shape[0], dtype=bool)
    mask_offdiag = ~mask_diag
    
    # Create the plot
    plt.figure(figsize=(6, 6))
    
    # Plot off-diagonal elements
    plt.imshow(np.ma.masked_where(mask_diag, matrix), cmap='viridis', norm=normlow, origin='upper')
    
    # Plot diagonal elements
    plt.imshow(np.ma.masked_where(mask_offdiag, matrix), cmap='viridis', norm=normhigh, origin='upper')
    
    plt.colorbar(label='Magnitude')  # Add color bar
    
    # Optional: Add grid lines
    plt.grid(False)  # Set to True if you want grid lines
    
    # Optional: Add labels
    plt.xticks(np.arange(matrix.shape[1]))
    plt.yticks(np.arange(matrix.shape[0]))
    plt.xlabel("Column index")
    plt.ylabel("Row index")
    plt.title("Matrix Heatmap")
    
    # Show the plot
    plt.savefig(filename, dpi=300)
    plt.show()
'''

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

def run_cogito(directory,save_metadata:bool=True, invariant=True, irreducible_grid=True, verbose=0, tag="", include_excited=1, calc_nrms=False, save_orb_converg_info:bool=True,
                         orbfactor=1.0, num_steps=50, num_outer=3, plot_orbs = False,
                         min_proj=0.01, band_opt=True,orb_opt=True,orb_orth=False,start_from_orbnpy = False,plot_projBS = False,plot_projDOS=False,orbs = None,save_orb_data=False,save_orb_figs=False,minimum_orb_energy: float=-60,min_duplicate_energy:float=-60):
    '''
    Runs COGITO. See generate_TBmodel() for a description of all the other arguments.

    Args:
        directory (str): The directory that has all the input files.

    Usage:
        from COGITO import run_cogito
        run_cogito("silicon")
    '''
    if save_metadata:
        import sys
        import json
        from datetime import datetime
        from contextlib import redirect_stdout, redirect_stderr

        input_params = {
            'directory': directory,
            'invariant': invariant,
            "irreducible_grid": irreducible_grid,
            "verbose": verbose,
            "tag": tag,
            "include_excited": include_excited,
            "calc_nrms": calc_nrms,
            "save_orb_converg_info": save_orb_converg_info,
            "orbfactor": orbfactor,
            "num_steps": num_steps,
            "num_outer": num_outer,
            "plot_orbs": plot_orbs,
            "min_proj": min_proj,
            "band_opt": band_opt,
            "orb_opt": orb_opt,
            "orb_orth": orb_orth,
            "start_from_orbnpy": start_from_orbnpy,
            "plot_projBS": plot_projBS,
            "plot_projDOS": plot_projDOS,
            "orbs": orbs,
            "save_orb_data": save_orb_data,
            "save_orb_figs": save_orb_figs,
            "minimum_orb_energy": minimum_orb_energy,
            "min_duplicate_energy": min_duplicate_energy,
        }

        # save metadata
        metadata = {'cli args':sys.argv,'final args':input_params,'dist_version':dist_version,'datetime':datetime.now().isoformat(sep=" ", timespec="seconds")}
        with open(directory+"metadata"+tag+".json","w") as f:
            json.dump(metadata,f,indent=2)

        # Open log file for terminal output
        log_file = open(directory+"COG_output"+tag+".txt", "w", buffering=1)
        tee_out = Tee(sys.stdout, log_file)
        tee_err = Tee(sys.stderr, log_file)

        with redirect_stdout(tee_out), redirect_stderr(tee_err):
            COGITOmodel = COGITO(directory)
            # do full algorithm and generate input files for TB model
            COGITOmodel.generate_TBmodel(irreducible_grid=irreducible_grid, verbose=verbose, tag=tag, include_excited=include_excited, calc_nrms=calc_nrms,
                            save_orb_converg_info = save_orb_converg_info,orbfactor = orbfactor, num_steps = num_steps, num_outer = num_outer,
                            plot_orbs = plot_orbs,band_opt = band_opt, orb_opt = orb_opt, min_proj = min_proj,
                            start_from_orbnpy = start_from_orbnpy,save_orb_figs = save_orb_figs,
                            minimum_orb_energy = minimum_orb_energy, min_duplicate_energy = min_duplicate_energy)

            if COGITOmodel.spin_polar:  # rerun with spin=1 for magnetic calculations
                COGITOmodel = COGITO(directory, spin=1)
                # do full algorithm and generate input files for TB model
                COGITOmodel.generate_TBmodel(irreducible_grid=irreducible_grid, verbose=verbose, tag=tag, include_excited=include_excited, calc_nrms=calc_nrms,
                            save_orb_converg_info = save_orb_converg_info,orbfactor = orbfactor, num_steps = num_steps, num_outer = num_outer,
                            plot_orbs = plot_orbs,band_opt = band_opt, orb_opt = orb_opt, min_proj = min_proj,
                            start_from_orbnpy = start_from_orbnpy,save_orb_figs = save_orb_figs,
                            minimum_orb_energy = minimum_orb_energy, min_duplicate_energy = min_duplicate_energy)

    else: # just run without saving any meta data
        COGITOmodel = COGITO(directory)
        # do full algorithm and generate input files for TB model
        COGITOmodel.generate_TBmodel(irreducible_grid=irreducible_grid, verbose=verbose, tag=tag,
                                     include_excited=include_excited, calc_nrms=calc_nrms,
                                     save_orb_converg_info=save_orb_converg_info, orbfactor=orbfactor, num_steps=num_steps,
                                     num_outer=num_outer,
                                     plot_orbs=plot_orbs, band_opt=band_opt, orb_opt=orb_opt, min_proj=min_proj,
                                     start_from_orbnpy=start_from_orbnpy, save_orb_figs=save_orb_figs,
                                     minimum_orb_energy=minimum_orb_energy, min_duplicate_energy=min_duplicate_energy)

        if COGITOmodel.spin_polar:  # rerun with spin=1 for magnetic calculations
            COGITOmodel = COGITO(directory, spin=1)
            # do full algorithm and generate input files for TB model
            COGITOmodel.generate_TBmodel(irreducible_grid=irreducible_grid, verbose=verbose, tag=tag,
                                         include_excited=include_excited, calc_nrms=calc_nrms,
                                         save_orb_converg_info=save_orb_converg_info, orbfactor=orbfactor, num_steps=num_steps,
                                         num_outer=num_outer,
                                         plot_orbs=plot_orbs, band_opt=band_opt, orb_opt=orb_opt, min_proj=min_proj,
                                         start_from_orbnpy=start_from_orbnpy, save_orb_figs=save_orb_figs,
                                         minimum_orb_energy=minimum_orb_energy, min_duplicate_energy=min_duplicate_energy)


def main(argv=None):
    import argparse
    cogito_args = argparse.ArgumentParser(description="This code runs the COGITO basis and TB model creation from DFT wavefunctions and energies. "
                                                        "As input, COGITO requires output files from a VASP calculation in the run directory or as specficied by '--dir'. "
                                                        "The VASP run must be a uniform kpt grid with no symmetry or full symmetry. "
                                                        "Spin-polarization and magnetization is hangled automatically. However, the calculation cannot be noncolinear (have spin-orbit coupling). "
                                                        "!!IMPORTANT!! To set any of the boolean variables to False do '--irred_grid ''', where the bool('') reads as False in python. Any other string will be True.")
    cogito_args.add_argument("--dir",type=str,help="The directory that contains the VASP output files.",default="")
    cogito_args.add_argument("--no_save_metadata",help="If called, does not save CLI, input parameters, and printed output from COGITO run to 'metadata'+tag+'.json' and 'COG_output'+tag+'.txt'.",action='store_true') #not #
    cogito_args.add_argument("--not_invariant",help="If called, does not construct the orbitals as eigenvectors of the local environment tensor.",action='store_true')  #not #
    cogito_args.add_argument("--full_kgrid",help="If called, the k-point grid from VASP must be full grid (not irreducible) (i.e. ISYM=-1).",action='store_true') #not #
    cogito_args.add_argument("--verb",type=int,help="Verbosity with which to print updates and data.",default=0)
    cogito_args.add_argument("--tag",type=str,help="Tag to be appended to the runs output files.",default="")
    cogito_args.add_argument("--include_ex",type=int,help="Integer that cues how many excited orbitals to include in the COGITO basis. Options are 0, 1, or 2.",default=1)
    cogito_args.add_argument("--calc_nrms",help="If called, does extra analysis to compare reconstructed COGITO wfs to DFT wfs. Prints nrmse to error_output.txt",action='store_true') #
    cogito_args.add_argument("--no_save_converg",help="If called, does not save statistics on the orbital convergence to orb_converg_info.json.",action='store_true') # not #
    cogito_args.add_argument("--orbfactor",type=float,help="The amount to scale the initial orbital basis by.",default=1.0)
    cogito_args.add_argument("--num_steps",type=int,help="Maximum number of steps in the first convergence section, which finds numerical Bloch orbitals.",default=50)
    cogito_args.add_argument("--num_outer",type=int,help="Number of steps in the full convergence loop. Do not exceed 10.",default=3)
    cogito_args.add_argument("--plot_orbs",help="If called, does plt.show() for COGITO radial fitting plots.",action='store_true') #
    cogito_args.add_argument("--no_band_opt",help="If called, does not to apply Lowdin+Gram+extras band optimization scheme for improved TB interpolation (scheme improves band error from ~40meV to ~2meV).",action='store_true') #not #
    cogito_args.add_argument("--no_orb_opt",help="If called, does not enforce zero orbital mixing for improved TB interpolation.",action='store_true') # not #
    cogito_args.add_argument("--save_orb_figs",help="If called, saves the COGITO radial fitting to fit_cogito.png (if --verb > 1, also saves init_cogito.png and all_converge_data.json).",action='store_true') #
    cogito_args.add_argument("--min_proj",type=float,help="Minimum projectability for wavefunction to be included.",default=0.01)
    cogito_args.add_argument("--start_orbnpy",help="If called, starts from previously output COGITO basis. Skips orbital convergence and goes to projection and TB model generation.",action='store_true') #
    cogito_args.add_argument("--minimum_orb_en",type=float,help="The lower limit to add semi-core states in the POTCAR into the COGITO basis.",default=-80)
    cogito_args.add_argument("--min_duplicate_en",type=float,help="The lower limit to add semi-core states when there is another valence state of the same l quantum number in the POTCAR into the COGITO basis.",default=-80)


    args = cogito_args.parse_args(argv) # if args_manual==None then should take from command line
    print("!!IMPORTANT!! To set any of the boolean variables to False do --irred_grid '', where the bool('') reads as False in python. Any other string will be True.")

    run_cogito(directory=args.dir, save_metadata = not args.no_save_metadata,invariant=not args.not_invariant, irreducible_grid=not args.full_kgrid, verbose=args.verb, tag=args.tag, include_excited=args.include_ex, calc_nrms=args.calc_nrms,
                        save_orb_converg_info =not args.no_save_converg,orbfactor = args.orbfactor, num_steps = args.num_steps, num_outer = args.num_outer,
                        plot_orbs = args.plot_orbs,band_opt =not args.no_band_opt, orb_opt =not args.no_orb_opt, min_proj = args.min_proj,
                        start_from_orbnpy = args.start_orbnpy,save_orb_figs = args.save_orb_figs,
                        minimum_orb_energy = args.minimum_orb_en, min_duplicate_energy = args.min_duplicate_en)

if __name__ == "__main__":
    raise SystemExit(main())
