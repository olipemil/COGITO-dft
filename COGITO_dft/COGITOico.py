import numpy as np
import numpy.typing as npt
#import pymatgen.core.structure as struc
#import pymatgen.symmetry.kpath as kpath
#from pymatgen.io.vasp.outputs import Eigenval
# import matplotlib
# matplotlib.use("Qt4Agg")
#import matplotlib.pyplot as plt
import copy
import math

class COGITO_ICO(object):
    def __init__(self, directory: str, verbose: int = 0, file_suffix: str = "", orbs_orth: bool = False,
                 spin_polar: bool = False) -> None:
        """
        Initializes the object from tb_input.txt, ICOHP.npy, and ICOOP.npy.

        Args:    
            directory (str): The path for the input files.
            verbose (int): How much will be printed (0 is least).
            file_suffix (str): The suffix to the TBparams and overlaps files.
            orbs_orth (bool): Whether the orbitals are orthogonal, if from COGITO this is always False.
        """
        self.directory = directory
        self.verbose = verbose
        self.spin_polar = spin_polar
        self.read_input()
        self.orbs_orth = orbs_orth
        self.file_suff = file_suffix
        self.show_figs = True
        if self.verbose > 1:
            print(self._a)
        self.cartAtoms = _red_to_cart((self._a[0], self._a[1], self._a[2]), self.primAtoms)
        if self.verbose > 1:
            print(self.cartAtoms)
        # generate reciprocal lattice
        self.vol = np.dot(self._a[0, :], np.cross(self._a[1, :], self._a[2, :]))
        b = np.array(
            [np.cross(self._a[1, :], self._a[2, :]),
             np.cross(self._a[2, :], self._a[0, :]),
             np.cross(self._a[0, :], self._a[1, :])])
        b = 2 * np.pi * b / self.vol
        self._b = b
        if self.verbose > 2:
            print("recip vecs:", b)

        cross_ab = np.cross(self._a[0], self._a[1])
        supercell_volume = abs(np.dot(cross_ab, self._a[2]))
        self.cell_vol = supercell_volume

        if self.spin_polar:
            keys = [0,1]
        else:
            keys = [0]
        self.keys = keys

        # read in params
        self.read_ICOHP(file="ICOHP" + file_suffix + ".txt")
        if orbs_orth == False:
            self.read_ICOOP(file="ICOOP" + file_suffix + ".txt")

        # set more parameters hehe
        vecs = np.array(self.vec_to_trans)
        orb_pos = np.array(self.orb_redcoords)
        vec_to_orbs = vecs[None, None, :, :, :] + orb_pos[None, :, None, None, None] - orb_pos[:, None, None, None,
                                                                                       None]  # [orb1,orb2,t1,t2,t3]
        # print("shape of vec:",vec_to_orbs.shape)
        hold_vecs = np.reshape(vec_to_orbs, (
        self.num_orbs * self.num_orbs * self.num_trans[0] * self.num_trans[1] * self.num_trans[2], 3)).transpose()
        cart_hold_vecs = _red_to_cart((self._a[0], self._a[1], self._a[2]), hold_vecs.transpose())
        dist_to_orbs = np.linalg.norm(cart_hold_vecs, axis=1)

        self.dist_to_orbs = dist_to_orbs
        self.flat_vec_to_orbs = hold_vecs
        #self.restrict_params(maximum_dist=50, minimum_value=0.00000001)
        if not hasattr(self, 'NN_index'):
            self.NN_index, self.NN_dists = self.get_neighbors(self)

    def read_input(self,file: str = "tb_input.txt") -> None:
        """
        Reads the tb_input.txt for information about the unit cell, atoms, orbitals, fermi energy, energy shift, and spin polarization.
        """
        filename = self.directory + file
        if self.verbose > 1:
            print(filename)
        filedata = open(filename)
        filelines = filedata.readlines()
        num_lines = len(filelines)
        spin_polar = self.spin_polar
        orbvec_set = False
        for lind,line in enumerate(filelines):
            if "lattice_vecs" in line:
                lattice_start = lind+1
            if "atoms" in line:
                atom_start = lind+1
            if "orbitals" in line:
                orb_start = lind+1
            if "orbital spherical" in line:
                orbvec_start = lind+1
                orbvec_set = True
            if "fermi" in line:
                efermi = float(line.strip().split()[2])
            if "shift" in line:
                energy_shift = float(line.strip().split()[2])
            if "spin polar" in line:
                polar = line.strip().split()[-1]
                if polar == "True":
                    spin_polar = True #bool(line.strip().split()[-1])
                else:
                    spin_polar = False
        self.efermi = efermi
        self.energy_shift = energy_shift
        self.spin_polar = spin_polar
        if self.verbose > 0:
            print("fermi energy:",self.efermi)
            print("energy shift:",self.energy_shift)
            print(self.spin_polar)
        a1 = [float(i) for i in filelines[lattice_start].strip().split()]
        a2 = [float(i) for i in filelines[lattice_start+1].strip().split()]
        a3 = [float(i) for i in filelines[lattice_start+2].strip().split()]
        self._a = np.array([a1,a2,a3])
        self.abc = [np.linalg.norm(a1),np.linalg.norm(a2),np.linalg.norm(a3)]
        if self.verbose > 1:
            print("prim vecs:",self._a)

        atoms_pos = []
        elems = []
        elec_count = []
        elec = 0
        for lind in range(atom_start,orb_start-1):
            vals = filelines[lind].strip().split()
            if len(vals) >= 4:
                atom_pos = [float(i) for i in vals[:3]]
                elem = vals[3]
                if len(vals) == 5:
                    elec = float(vals[4])
                atoms_pos.append(atom_pos)
                elems.append(elem)
                elec_count.append(elec)
        if self.verbose > 0:
            print("atoms:",atoms_pos,elems)
        self.elements = np.array(elems)
        self.primAtoms = np.array(atoms_pos)
        self.elec_count = elec_count
        self.numAtoms = len(self.elements)

        orbs_pos = []
        orbatomnum = []
        orbatomname = []
        exactorbtype = []
        for lind in range(orb_start,num_lines):
            vals = filelines[lind].strip().split()
            if len(vals) == 5:
                orb_pos = [float(i) for i in vals[:3]]
                num = int(vals[3])
                exact_orb = vals[4]
                orbs_pos.append(orb_pos)
                orbatomnum.append(num)
                orbatomname.append(self.elements[num])
                exactorbtype.append(exact_orb)
            else:
                break

        num_orbs = len(orbatomnum)
        # read in the orbital vec for sphharm combo
        allorb_vecs = []
        if orbvec_set:
            for lind in range(orbvec_start,orbvec_start+num_orbs):
                vals = filelines[lind].strip().split()
                orb_vec= [float(i) for i in vals]
                allorb_vecs.append(orb_vec)
        else: # will just be normal sph harm
            # make sphharm from exactorbtype
            # (this is definitely not fully correct and depends on the versions ordering of xyz, etc)
            sphharm = []
            countp = 0
            countd = 0
            for ot in exactorbtype:
                if 's' in ot:
                    sphharm.append(0)
                if 'p' in ot:
                    sphharm.append(int(1+countp))
                    countp = countp + 1
                    if countp > 2:
                        countp = 0
                if 'd' in ot:
                    sphharm.append(int(1+countd))
                    countd = countd + 1
                    if countd > 4:
                        countd = 0
            switch_to_vecs = [[1], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0],
                              [0, 0, 0, 1, 0], [0, 0, 0, 0, 1],
                              [1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0],
                              [0, 0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]]
            for key in sphharm:
                allorb_vecs.append(switch_to_vecs[key])
        self.orb_vec = allorb_vecs

        if self.verbose > 2:
            print("orbitals:",orbs_pos,orbatomnum)
        self.orbatomnum = np.array(orbatomnum)
        self.orbatomname = orbatomname
        self.orb_redcoords = np.array(orbs_pos)
        self.exactorbtype = np.array(exactorbtype)

    def read_overlaps(self,file: str = "overlaps.txt") -> None:
        """
        Reads in the orbital overlaps from a file formatted like wannier90_hr.dat. Only neccesary for plotting the bond densities.

        Args:
            file (str): The file name for the overlaps.
        """
        #get TB params from a file formatted like wannier90_hr.dat
        if self.orbs_orth == True:
            file = "orth_overlaps.txt"
        original_name = self.directory + file
        orig_prefix = original_name.split(".")[0]
        filename = original_name
        #print(filename)
        if self.spin_polar == True:
            filename = orig_prefix + "0" + ".txt"
        all_files = {}
        filedata = open(filename)
        filelines = filedata.readlines()
        all_files[0] = filelines
        if self.spin_polar == True: # read in spin up too
            filename1 = orig_prefix + "1" + ".txt"
            filedata1 = open(filename1)
            filelines1 = filedata1.readlines()
            all_files[1] = filelines1
        [trans1,trans2,trans3,num_orbs] = [int(i) for i in filelines[1].strip().split()]
        print(trans1,trans2,trans3,num_orbs)
        #num_orbs = int(filelines[1])
        num_each_dir = np.array([trans1,trans2,trans3])
        og_each_dir = num_each_dir
        #self.num_orbs = num_orbs
        #print(num_orbs)
        if self.orbs_orth == True:
            first_line = 100#3+num_orbs #12
        else:
            first_line = 3
        last_line = len(filelines)
        #print(filelines[first_line])
        count = 0
        #num_each_dir = np.array([2,2,3])# abs(int(filelines[first_line].split()[0]))

        num_trans = num_each_dir*2+1
        #self.num_trans = num_trans
        #print(num_trans)
        #generate list of the displacement between translations
        vec_to_trans = np.zeros((num_trans[0],num_trans[1],num_trans[2],3))
        for x in range(num_trans[0]):
            for y in range(num_trans[1]):
                for z in range(num_trans[2]):
                    vec_to_trans[x,y,z] = [x-num_each_dir[0],y-num_each_dir[1],z-num_each_dir[2]]
        #self.vec_to_trans = vec_to_trans
        atomic_overlaps = {}
        overlaps_params = {}
        for key in list(all_files.keys()):
            cur_lines = all_files[key]
            atomic_overlaps[key] = []
            #read in the TB parameters
            overlaps_params[key] = np.zeros((num_orbs,num_orbs,num_trans[0],num_trans[1],num_trans[2]), dtype=np.complex128)
            for line in cur_lines[first_line:]:
                info = line.split()
                if abs(int(info[0])) <=num_each_dir[0] and abs(int(info[1])) <=num_each_dir[1] and abs(int(info[2])) <=num_each_dir[2]:
                    trans1 = int(info[0])+num_each_dir[0]
                    trans2 = int(info[1])+num_each_dir[1]
                    trans3 = int(info[2])+num_each_dir[2]
                    orb1 = int(info[3])-1
                    orb2 = int(info[4])-1
                    value = float(info[5]) + float(info[6])*1.0j
                    if overlaps_params[key][orb1,orb2,trans1,trans2,trans3] != 0:
                        print("already set TB param")
                    #only set if orbitals are not on the same atom
                    same_atom = np.abs(np.array(self.orb_redcoords[orb1]) - np.array(self.orb_redcoords[orb2]))
                    #print(same_atom)

                    overlaps_params[key][orb1, orb2, trans1, trans2, trans3] = value
                    if (same_atom < 0.001).all() and (int(info[0])==0 and int(info[1])==0 and int(info[2])==0) and orb1!=orb2 and abs(value) > 0.0001:
                        #print("Same atom orbital hopping term that should be zero!", orb1, orb2, value)
                        #overlaps_params[key][orb1,orb2,trans1,trans2,trans3] = 0
                        pass
                    if (int(info[0])==0 and int(info[1])==0 and int(info[2])==0) and orb1==orb2:
                        #print("onsite term:",value)
                        atomic_overlaps[key].append(value)
        if self.verbose > 1:
            print("the atomic orbital overlaps!",atomic_overlaps)
        #if self.spin_polar:
        self.atomic_overlaps = atomic_overlaps
        self.overlaps_params = overlaps_params

        small_overlap = {}
        for key in self.keys: # reduce the size of ICOOP and overlap
            old_each_dir = og_each_dir # self.num_each_dir
            new_each_dir = self.num_each_dir # self.ICO_eachdir
            small_overlap[key] = self.overlaps_params[key][:,:, old_each_dir[0] - new_each_dir[0]: old_each_dir[0] + new_each_dir[0] + 1,
                                    old_each_dir[1] - new_each_dir[1]:old_each_dir[1] + new_each_dir[1] + 1,
                                    old_each_dir[2] - new_each_dir[2]:old_each_dir[2] + new_each_dir[2] + 1]

        self.small_overlap = small_overlap

    def read_ICOHP(self, file: str = "ICOHP.txt") -> None:
        import os
        # get TB params from a file formatted like wannier90_hr.dat
        if self.orbs_orth == True:
            file = "orth_tbparams.txt"
        original_name = self.directory + file
        print(original_name)
        orig_prefix = original_name.split(".")[-2]
        print(orig_prefix)
        filename = original_name
        # check if *.npy was made
        if os.path.exists(orig_prefix+"0"+".npy"):
            #print("get to load fast!")
            if os.path.exists(orig_prefix + "1" + ".npy"):
                print("Switching to spin polarized")
                self.spin_polar = True
                self.keys = [0,1]

            icohp = {}

            for key in list(self.keys):
                file = orig_prefix+str(key)+".npy"
                #print(file)
                icohp[key] = np.load(file)
                #print("cohp shape:",icohp[key].shape)
                num_orbs = icohp[key].shape[0]
                num_trans = np.array([icohp[key].shape[2],icohp[key].shape[3],icohp[key].shape[4]])
            num_each_dir = np.array(np.floor((num_trans-1)/2),dtype=np.int_)
            self.num_orbs = num_orbs
            self.num_trans = num_trans
            self.num_each_dir = num_each_dir
            # generate list of the displacement between translations
            vec_to_trans = np.zeros((num_trans[0], num_trans[1], num_trans[2], 3))
            for x in range(num_trans[0]):
                for y in range(num_trans[1]):
                    for z in range(num_trans[2]):
                        vec_to_trans[x, y, z] = [x - num_each_dir[0], y - num_each_dir[1], z - num_each_dir[2]]
            self.vec_to_trans = vec_to_trans
        else:
            # print(filename)
            #if self.spin_polar == True:
            filename = orig_prefix + "0" + ".txt"
            all_files = {}
            filedata = open(filename)
            filelines = filedata.readlines()
            all_files[0] = filelines
            if os.path.exists(orig_prefix + "1" + ".txt"):
                print("Switching to spin polarized")
                self.spin_polar = True
                self.keys = [0,1]
            if self.spin_polar == True:  # read in spin up too
                filename1 = orig_prefix + "1" + ".txt"
                filedata1 = open(filename1)
                filelines1 = filedata1.readlines()
                all_files[1] = filelines1
            [trans1, trans2, trans3, num_orbs] = [int(i) for i in filelines[1].strip().split()]
            #print(trans1, trans2, trans3, num_orbs)
            # num_orbs = int(filelines[1])
            num_each_dir = np.array([trans1, trans2, trans3])
            self.num_each_dir = num_each_dir
            self.num_orbs = num_orbs
            # print(num_orbs)
            if self.orbs_orth == True:
                first_line = 100  # 3+num_orbs #12
            else:
                first_line = 3
            last_line = len(filelines)
            # print(filelines[first_line])
            count = 0
            # num_each_dir = np.array([2,2,3])# abs(int(filelines[first_line].split()[0]))

            num_trans = num_each_dir * 2 + 1
            self.num_trans = num_trans
            # print(num_trans)
            # generate list of the displacement between translations
            vec_to_trans = np.zeros((num_trans[0], num_trans[1], num_trans[2], 3))
            for x in range(num_trans[0]):
                for y in range(num_trans[1]):
                    for z in range(num_trans[2]):
                        vec_to_trans[x, y, z] = [x - num_each_dir[0], y - num_each_dir[1], z - num_each_dir[2]]
            self.vec_to_trans = vec_to_trans
            atomic_energies = {}
            icohp = {}
            for key in list(all_files.keys()):
                cur_lines = all_files[key]
                atomic_energies[key] = []
                # read in the TB parameters
                icohp[key] = np.zeros((num_orbs, num_orbs, num_trans[0], num_trans[1], num_trans[2]),
                                          dtype=np.complex128)
                for line in cur_lines[first_line:]:
                    info = line.split()
                    if abs(int(info[0])) <= num_each_dir[0] and abs(int(info[1])) <= num_each_dir[1] and abs(
                            int(info[2])) <= num_each_dir[2]:
                        trans1 = int(info[0]) + num_each_dir[0]
                        trans2 = int(info[1]) + num_each_dir[1]
                        trans3 = int(info[2]) + num_each_dir[2]
                        orb1 = int(info[3]) - 1
                        orb2 = int(info[4]) - 1
                        value = float(info[5]) + float(info[6]) * 1.0j
                        if icohp[key][orb1, orb2, trans1, trans2, trans3] != 0:
                            print("already set TB param")
                        # only set if orbitals are not on the same atom
                        same_atom = np.abs(np.array(self.orb_redcoords[orb1]) - np.array(self.orb_redcoords[orb2]))
                        # print(same_atom)

                        icohp[key][orb1, orb2, trans1, trans2, trans3] = value
                        if (same_atom < 0.001).all() and (
                                int(info[0]) == 0 and int(info[1]) == 0 and int(info[2]) == 0) and orb1 != orb2 and abs(
                                value) > 0.0001:
                            # print("Same atom orbital hopping term that should be zero!", orb1, orb2, value)
                            # icohp[key][orb1,orb2,trans1,trans2,trans3] = 0
                            pass
                        if (int(info[0]) == 0 and int(info[1]) == 0 and int(info[2]) == 0) and orb1 == orb2:
                            # print("onsite term:",value)
                            atomic_energies[key].append(value)
            if self.verbose > 0:
                print("the atomic orbital energies!", atomic_energies)
            # if self.spin_polar:
            self.atomic_energies = atomic_energies
        self.ICOHP = icohp
        # else: # pass just the parameters instead of a dictionary with one key/item
        #    self.atomic_energies = atomic_energies[0]
        #    self.TB_params = TB_params[0]

    def read_ICOOP(self, file: str = "ICOOP.txt") -> None:
        import os
        # get TB params from a file formatted like wannier90_hr.dat
        original_name = self.directory + file
        filename = original_name
        orig_prefix = original_name.split(".")[0]
        #if self.spin_polar == True:

        # check if *.npy was made
        if os.path.exists(orig_prefix+"0"+".npy"):
            #print("get to load fast!")
            icoop = {}
            for key in list(self.keys):
                icoop[key] = np.load(orig_prefix+str(key)+".npy")
        else:
            filename = orig_prefix + "0" + ".txt"
            all_files = {}
            filedata = open(filename)
            filelines = filedata.readlines()
            all_files[0] = filelines
            if self.spin_polar == True:  # read in spin up too
                filename1 = orig_prefix + "1" + ".txt"
                filedata1 = open(filename1)
                filelines1 = filedata1.readlines()
                all_files[1] = filelines1

            if self.orbs_orth == True:
                first_line = 100  # 3+num_orbs #12
            else:
                first_line = 3
            num_orbs = self.num_orbs
            num_trans = self.num_trans
            num_each_dir = np.array((self.num_trans - 1) / 2, dtype=np.int_)
            # print(num_trans)
            # generate list of the displacement between translations
            atomic_overlaps = {}
            icoop = {}
            for key in list(all_files.keys()):
                cur_lines = all_files[key]
                atomic_overlaps[key] = []
                # read in the TB parameters
                icoop[key] = np.zeros((num_orbs, num_orbs, num_trans[0], num_trans[1], num_trans[2]),
                                                dtype=np.complex128)
                for line in cur_lines[first_line:]:
                    info = line.split()
                    if abs(int(info[0])) <= num_each_dir[0] and abs(int(info[1])) <= num_each_dir[1] and abs(
                            int(info[2])) <= num_each_dir[2]:
                        trans1 = int(info[0]) + num_each_dir[0]
                        trans2 = int(info[1]) + num_each_dir[1]
                        trans3 = int(info[2]) + num_each_dir[2]
                        orb1 = int(info[3]) - 1
                        orb2 = int(info[4]) - 1
                        value = float(info[5]) + float(info[6]) * 1.0j
                        if icoop[key][orb1, orb2, trans1, trans2, trans3] != 0:
                            print("already set TB param")
                        # only set if orbitals are not on the same atom
                        same_atom = np.abs(np.array(self.orb_redcoords[orb1]) - np.array(self.orb_redcoords[orb2]))
                        # print(same_atom)

                        icoop[key][orb1, orb2, trans1, trans2, trans3] = value
                        if (same_atom < 0.001).all() and (
                                int(info[0]) == 0 and int(info[1]) == 0 and int(info[2]) == 0) and orb1 != orb2 and abs(
                                value) > 0.0001:
                            # print("Same atom orbital overlap that should be zero!", orb1, orb2, value)
                            # icoop[orb1,orb2,trans1,trans2,trans3] = 0
                            pass
                        if (int(info[0]) == 0 and int(info[1]) == 0 and int(info[2]) == 0) and orb1 == orb2:
                            # print("onsite term:",value)
                            atomic_overlaps[key].append(value)
            if self.verbose > 1:
                print("the atomic orbital self overlaps!", atomic_overlaps)
            # if self.spin_polar:
        self.ICOOP = icoop
        # else:
        #    self.overlaps_params = overlaps_params[0]

    def read_bond_occup(self, file: str = "bond_occup.txt") -> None:
        import os
        # get bond_occup from a file numpy array file
        original_name = self.directory + file
        filename = original_name
        orig_prefix = original_name.split(".")[0]
        #if self.spin_polar == True:

        # check if *.npy was made
        if os.path.exists(orig_prefix+"0"+".npy"):
            #print("get to load fast!")
            bond_occup = {}
            for key in list(self.keys):
                bond_occup[key] = np.load(orig_prefix+str(key)+".npy")
            self.bond_occup = bond_occup
        else:
            print("ERROR: Didn't save bond occupation and can't load!!")

    def read_orbitals(self,file: str = "orbitals.npy") -> None:
        """
        This function reads in the orbitals as coefficents for a gaussian expansion.
        The information in 'orbitals.npy' is combined with the orbital data in 'tb_input.txt'.

        Args:    
            file (str): Orbital file.
        """

        orbital_coeffs = np.load(self.directory + file)[:,:-1]
        sph_harm_key = np.load(self.directory + file)[:,-1]
        self.recip_orbcoeffs = orbital_coeffs
        self.sph_harm_key = sph_harm_key

        real_orbcoeffs = np.zeros(orbital_coeffs.shape)
        orb_radii = np.zeros(self.num_orbs)
        for orb in range(self.num_orbs):
            [ak, bk, ck, dk, ek, fk, gk, hk, l] = self.recip_orbcoeffs[orb]

            # switch to a*e^-bx^2 format form a/b*e^-1/2(x/b)^2 format
            # print("check same:",[a,b,c,d,e,f,g,h,l])
            [oak, ock, oek, ogk] = [ak / bk, ck / dk, ek / fk, gk / hk]
            [obk, odk, ofk, ohk] = 1 / 2 / np.array([bk, dk, fk, hk]) ** 2

            # switch to real space coefficients
            [ob,od,of,oh] = 1/4/np.array([obk,odk,ofk,ohk])
            #ak = 2**(-3/2-l)* a* b**(-3/2-l)
            [oa, oc, oe, og] = np.array([oak,ock,oek,ogk]) * 2**(3/2+l) * np.array([ob,od,of,oh])**(3/2+l)

            fnt_coef = np.array([oa, oc, oe, og])
            exp_coef = np.array([ob, od, of, oh])
            avg_rad = 1 / 2 * math.factorial(1 + int(l)) * np.sum(fnt_coef[:, None] * fnt_coef[None, :] * (exp_coef[:, None] + exp_coef[None, :]) ** (-2 - l/2))  # analytical calc
            orb_radii[orb] = avg_rad

            # convert back to a/b*e^-1/2(x/b)^2 format from a*e^-bx^2 format
            [b, d, f, h] = 1 / (2 * np.array([ob, od, of, oh])) ** (1 / 2)  # 1/2/np.array([b,d,f,h])**2
            [a, c, e, g] = [oa * b, oc * d, oe * f, og * h]

            real_orbcoeffs[orb] = np.array([a, b, c, d, e, f, g, h, l])

        self.real_orbcoeffs = real_orbcoeffs
        self.orb_radii = orb_radii

    def save_ICOnpy(self):
        for key in self.keys:
            np.save(self.directory+"ICOHP"+str(key),self.ICOHP[key])
            np.save(self.directory+"ICOOP"+str(key),self.ICOOP[key])

    @staticmethod
    def get_neighbors(self) -> list:
        """
        This sorts the matrix of TB parameters into terms which are 1NN, 2NN, etc. Relevant for labeling bonds.
        """
        dist_to_orbs = self.dist_to_orbs
        small_dist_bool = dist_to_orbs < 10
        small_dist_index = np.arange(len(dist_to_orbs))[small_dist_bool]
        small_dist_orbs = dist_to_orbs[small_dist_index]
        sorted_dist_orbs = np.around(np.sort(small_dist_orbs),decimals=1)
        NN_dists = np.unique(sorted_dist_orbs)
        NN_index = {}
        for NN in range(min(50,len(NN_dists))):
            small_indices = np.abs(np.around(small_dist_orbs,decimals=1) - NN_dists[NN]) == 0
            NN_index[NN] = small_dist_index[small_indices] #flattened indices
        #self.NN_index = NN_index
        #self.NN_dists = NN_dists[:16]
        return NN_index, NN_dists[:50]

    def get_bonds_figure(self, energy_cutoff: float = 0.1, elem_colors: list = [], atom_colors: list = [], atom_labels: list = [], plot_atom: int = None,
                         one_atom: bool = False, bond_max: float = 3.0, fovy: float = 10,return_fig: bool = False) -> None:
        """
        This is deprecated.
        This will plot the crystal structure atoms with line weighted by iCOHP.
        Each line should also be hoverable to reveal the number and amounts that are s-s,s-p, and p-p.

        Args:    
            energy_cutoff (float): This is the minimum bond magnitude that will be plotted.
            elem_colors (list): Colors for the elements based on order in tb_input. Length of list should be the number of
                        unique elements. Can either be integer list to reference the default colors or list of
                        plotly compatable colors.
            atom_colors (list): Colors for the atoms based on order in tb_input. Length of list should be the number of
                        atoms in the primitive cell. Can either be integer list to reference the default colors or
                        list of plotly compatable colors. If not set defaults to elem_colors.
            atom_labels (list): List of atom labels as a string.
            plot_atom (int): Set with one_atom=True, plots only one atom and it's bonds, this passes the atom number to plot.
            one_atom (bool): Whether only the atom defined in plot_atom should be plotted; default is False.
            bond_max (float): The maximum bond distance that will be plotted outside the primitive cell.
            fovy (float): Field of view in the vertical direction. Use this tag to adjust depth perception in crystal.
                        Set between 3 (for close to orthographic) and 30 (for good perspective depth).

        Returns:    
            None: Nothing
        """
        import plotly.graph_objs as go
        if not hasattr(self, 'NN_index'):
            self.get_neighbors()
        # atom size based on element
        all_atom_rad = {"H": 0.53, "He": 0.31, "Li": 1.51, "Be": 1.12, "B": 0.87, "C": 0.67, "N": 0.56, "O": 0.48,
                        "F": 0.42, "Ne": 0.38,
                        "Na": 1.90, "Mg": 1.45, "Al": 1.18, "Si": 1.11, "P": 0.98, "S": 0.88, "Cl": 0.79, "Ar": 0.71,
                        "K": 2.43, "Ca": 1.94, "Sc": 1.84, "Ti": 1.76, "V": 1.71, "Cr": 1.66, "Mn": 1.61, "Fe": 1.56,
                        "Co": 1.52,
                        "Ni": 1.49, "Cu": 1.45, "Zn": 1.42, "Ga": 1.36, "Ge": 1.25, "As": 1.14, "Se": 1.03, "Br": 0.94,
                        "Kr": 0.88,
                        "Rb": 2.65, "Sr": 2.19, "Y": 2.12, "Zr": 2.06, "Nb": 1.98, "Mo": 1.90, "Tc": 1.83, "Ru": 1.78,
                        "Rh": 1.73,
                        "Pd": 1.69, "Ag": 1.65, "Cd": 1.61, "In": 1.56, "Sn": 1.45, "Sb": 1.33, "Te": 1.23, "I": 1.15,
                        "Xe": 1.08,
                        "Cs": 2.98, "Ba": 2.53, "Lu": 2.17, "Hf": 2.08, "Ta": 2.00, "W": 1.93, "Re": 1.88, "Os": 1.85,
                        "Ir": 1.80,
                        "Pt": 1.77, "Au": 1.75, "Hg": 1.71, "Tl": 1.56, "Pb": 1.54, "Bi": 1.43, "Po": 1.35, "At": 1.27,
                        "Rn": 1.20}

        # get the cohp
        #if not hasattr(self, 'ICOHP'):
        #    self.get_ICOHP()
        icohp_orb = self.ICOHP
        # each_dir = np.array([1, 1, 1]) #self.num_each_dir
        cell = self.num_trans  # np.array(self.num_trans) # each_dir * 2 + 1
        vec_to_trans = np.zeros((cell[0], cell[1], cell[2], 3))  # self.vec_to_trans
        each_dir = self.num_each_dir

        for x in range(cell[0]):
            for y in range(cell[1]):
                for z in range(cell[2]):
                    vec_to_trans[x, y, z] = [x - each_dir[0], y - each_dir[1], z - each_dir[2]]

        num_total_atoms = self.numAtoms * cell[0] * cell[1] * cell[2]
        vec_to_atoms = vec_to_trans[None, :, :, :] + np.array(self.primAtoms)[:, None, None, None]
        vec_to_atoms = np.reshape(vec_to_atoms, (num_total_atoms, 3)).T
        # print(vec_to_atoms)
        cart_to_atoms = _red_to_cart((self._a[0], self._a[1], self._a[2]), vec_to_atoms.transpose()).T

        # plot the atoms onto the crystal
        x_data = cart_to_atoms[0]
        y_data = cart_to_atoms[1]
        z_data = cart_to_atoms[2]
        abc_data = vec_to_atoms
        # crystal = go.Scatter3d(x=x_data,y=y_data,z=z_data,mode='markers',marker=dict(size=20,color='blue',opacity=0.8),hoverinfo='none')
        if self.spin_polar:
            keys = [0, 1]
        else:
            keys = [0]

        all_bonds_spin = {}
        bonds_spin = {}
        for key in keys:
            '''
            cohp[key] = np.zeros((self.num_kpts, self.num_orbs, self.num_orbs, cell[0], cell[1], cell[2]),
                            dtype=np.complex128)  # kpts,orb1,orb2,T1,T2,T3
            # get it from TB params so have all except orbital energies
            vecs = np.array(vec_to_trans)  # this will align different with the TB_params!!
            orb_pos = np.array(self.orb_redcoords)
            param_dir = self.num_each_dir
            energysum = 0
            overlapsum = 0
            for kind in range(self.num_kpts):
                kpoint = self.kpoints[kind]
                # now only use the indices that are short
                vec_to_orbs = vecs[None, None, :, :, :] + orb_pos[None, :, None, None, None] - orb_pos[:, None, None, None,
                                                                                               None]
                hold_vecs = np.reshape(vec_to_orbs, (self.num_orbs * self.num_orbs * cell[0] * cell[1] * cell[2], 3))
                exp_fac = np.exp(2j * np.pi * np.dot(kpoint, hold_vecs.T))
                exp_fac = np.reshape(exp_fac, (self.num_orbs, self.num_orbs, cell[0], cell[1], cell[2]))

                #scaledTB = self.TB_params[key][:, :, param_dir[0] - each_dir[0]:param_dir[0] + each_dir[0] + 1,
                #           param_dir[1] - each_dir[1]:param_dir[1] + each_dir[1] + 1,
                #           param_dir[2] - each_dir[2]:param_dir[2] + each_dir[2] + 1] * exp_fac
                scaledTB = self.TB_params[key] * exp_fac

                coeff = self.eigvecs[key][kind, :self.num_orbs].T  # [band, orb]
                include_band = self.eigvals[key][kind, :].flatten() < self.efermi + self.energy_shift
                count = np.zeros(self.num_orbs)
                count[include_band] = 1
                mult_coeff = np.conj(coeff)[:, :, None] * coeff[:, None, :] * count[:, None, None]  # band,orb1,orb2
                sum_coeff = np.sum(mult_coeff, axis=0)
                energyCont = scaledTB * sum_coeff[:, :, None, None,
                                        None]  # np.conj(coeff)[:,full_ind[0]]*coeff[:,full_ind[1]] #[orb1,orb2,T1,T2,T3]
                cohp[key][kind] = energyCont * self.kpt_weights[kind]


            icohp_orb[key] = np.sum(cohp[key], axis=0).real  # [orb1,orb2,T1,T2,T3]
            print(icohp_orb[key][:,:,each_dir[0],each_dir[1],each_dir[2]])
            '''
            icohp_orb[key] = icohp_orb[key].real  # self.TB_model.get_ICOHP(self,spin=key).real
            # [s,p]*[s,p] = [s-s,s-p,p-s,p-p]
            # [s,p] * [s,p,d] = [s-s,s-p,p-s,p-p,s-d,p-d]
            icohp_atom = np.zeros((self.numAtoms, self.numAtoms, cell[0], cell[1], cell[2]))
            ssbonds = np.zeros((self.numAtoms, self.numAtoms, cell[0], cell[1], cell[2]))
            spbonds = np.zeros((self.numAtoms, self.numAtoms, cell[0], cell[1], cell[2]))
            psbonds = np.zeros((self.numAtoms, self.numAtoms, cell[0], cell[1], cell[2]))
            ppbonds = np.zeros((self.numAtoms, self.numAtoms, cell[0], cell[1], cell[2]))
            sdbonds = np.zeros((self.numAtoms, self.numAtoms, cell[0], cell[1], cell[2]))
            dsbonds = np.zeros((self.numAtoms, self.numAtoms, cell[0], cell[1], cell[2]))
            pdbonds = np.zeros((self.numAtoms, self.numAtoms, cell[0], cell[1], cell[2]))
            dpbonds = np.zeros((self.numAtoms, self.numAtoms, cell[0], cell[1], cell[2]))
            ddbonds = np.zeros((self.numAtoms, self.numAtoms, cell[0], cell[1], cell[2]))
            # add f orbs
            sfbonds = np.zeros((self.numAtoms, self.numAtoms, cell[0], cell[1], cell[2]))
            fsbonds = np.zeros((self.numAtoms, self.numAtoms, cell[0], cell[1], cell[2]))
            pfbonds = np.zeros((self.numAtoms, self.numAtoms, cell[0], cell[1], cell[2]))
            fpbonds = np.zeros((self.numAtoms, self.numAtoms, cell[0], cell[1], cell[2]))
            dfbonds = np.zeros((self.numAtoms, self.numAtoms, cell[0], cell[1], cell[2]))
            fdbonds = np.zeros((self.numAtoms, self.numAtoms, cell[0], cell[1], cell[2]))
            ffbonds = np.zeros((self.numAtoms, self.numAtoms, cell[0], cell[1], cell[2]))
            for atm1 in range(self.numAtoms):
                for atm2 in range(self.numAtoms):
                    orbs1 = np.arange(self.num_orbs)[self.orbatomnum == atm1]
                    orbs2 = np.arange(self.num_orbs)[self.orbatomnum == atm2]
                    sum_cohp = icohp_orb[key][orbs1, :][:, orbs2]
                    orb_type1 = self.exactorbtype[orbs1]
                    orb_type2 = self.exactorbtype[orbs2]
                    # s-s
                    is_s1 = []
                    is_p1 = []
                    is_d1 = []
                    is_f1 = []
                    for i in orb_type1:
                        is_s1.append("s" in i)
                        is_p1.append("p" in i)
                        is_d1.append("d" in i)
                        is_f1.append("f" in i)
                    is_s2 = []
                    is_p2 = []
                    is_d2 = []
                    is_f2 = []
                    for i in orb_type2:
                        is_s2.append("s" in i)
                        is_p2.append("p" in i)
                        is_d2.append("d" in i)
                        is_f2.append("f" in i)
                    sorbs1 = np.arange(len(orb_type1))[is_s1]
                    porbs1 = np.arange(len(orb_type1))[is_p1]
                    dorbs1 = np.arange(len(orb_type1))[is_d1]
                    forbs1 = np.arange(len(orb_type1))[is_f1]
                    sorbs2 = np.arange(len(orb_type2))[is_s2]
                    porbs2 = np.arange(len(orb_type2))[is_p2]
                    dorbs2 = np.arange(len(orb_type2))[is_d2]
                    forbs2 = np.arange(len(orb_type2))[is_f2]


                    ssbond = np.sum(sum_cohp[sorbs1, :][:, sorbs2], axis=(0, 1))
                    spbond = np.sum(sum_cohp[sorbs1, :][:, porbs2], axis=(0, 1))
                    psbond = np.sum(sum_cohp[porbs1, :][:, sorbs2], axis=(0, 1))
                    ppbond = np.sum(sum_cohp[porbs1, :][:, porbs2], axis=(0, 1))
                    sdbond = np.sum(sum_cohp[sorbs1, :][:, dorbs2], axis=(0, 1))
                    dsbond = np.sum(sum_cohp[dorbs1, :][:, sorbs2], axis=(0, 1))
                    pdbond = np.sum(sum_cohp[porbs1, :][:, dorbs2], axis=(0, 1))
                    dpbond = np.sum(sum_cohp[dorbs1, :][:, porbs2], axis=(0, 1))
                    ddbond = np.sum(sum_cohp[dorbs1, :][:, dorbs2], axis=(0, 1))
                    # add f orbs
                    sfbond = np.sum(sum_cohp[sorbs1, :][:, forbs2], axis=(0, 1))
                    fsbond = np.sum(sum_cohp[forbs1, :][:, sorbs2], axis=(0, 1))
                    pfbond = np.sum(sum_cohp[porbs1, :][:, forbs2], axis=(0, 1))
                    fpbond = np.sum(sum_cohp[forbs1, :][:, porbs2], axis=(0, 1))
                    dfbond = np.sum(sum_cohp[dorbs1, :][:, forbs2], axis=(0, 1))
                    fdbond = np.sum(sum_cohp[forbs1, :][:, dorbs2], axis=(0, 1))
                    ffbond = np.sum(sum_cohp[forbs1, :][:, forbs2], axis=(0, 1))

                    icohp_atom[atm1, atm2] = np.sum(sum_cohp, axis=(0, 1))
                    ssbonds[atm1, atm2] = ssbond
                    spbonds[atm1, atm2] = spbond
                    psbonds[atm1, atm2] = psbond
                    ppbonds[atm1, atm2] = ppbond
                    sdbonds[atm1, atm2] = sdbond
                    dsbonds[atm1, atm2] = dsbond
                    pdbonds[atm1, atm2] = pdbond
                    dpbonds[atm1, atm2] = dpbond
                    ddbonds[atm1, atm2] = ddbond
                    # add f orbs
                    sfbonds[atm1, atm2] = sfbond
                    fsbonds[atm1, atm2] = fsbond
                    pfbonds[atm1, atm2] = pfbond
                    fpbonds[atm1, atm2] = fpbond
                    dfbonds[atm1, atm2] = dfbond
                    fdbonds[atm1, atm2] = fdbond
                    ffbonds[atm1, atm2] = ffbond
            #print(icohp_atom.shape)
            # print("cohp!",icohp_atom)
            bonds_spin[key] = icohp_atom.flatten()
            ssbonds = ssbonds.flatten()
            spbonds = spbonds.flatten()
            psbonds = psbonds.flatten()
            ppbonds = ppbonds.flatten()
            sdbonds = sdbonds.flatten()
            dsbonds = dsbonds.flatten()
            pdbonds = pdbonds.flatten()
            dpbonds = dpbonds.flatten()
            ddbonds = ddbonds.flatten()
            # add f orbs
            sfbonds = sfbonds.flatten()
            fsbonds = fsbonds.flatten()
            pfbonds = pfbonds.flatten()
            fpbonds = fpbonds.flatten()
            dfbonds = dfbonds.flatten()
            fdbonds = fdbonds.flatten()
            ffbonds = ffbonds.flatten()
            all_bonds_spin[key] = {}
            all_bonds_spin[key]["ss"] = ssbonds
            all_bonds_spin[key]["sp"] = spbonds
            all_bonds_spin[key]["ps"] = psbonds
            all_bonds_spin[key]["pp"] = ppbonds
            all_bonds_spin[key]["sd"] = sdbonds
            all_bonds_spin[key]["ds"] = dsbonds
            all_bonds_spin[key]["pd"] = pdbonds
            all_bonds_spin[key]["dp"] = dpbonds
            all_bonds_spin[key]["dd"] = ddbonds
            # add f orbs
            all_bonds_spin[key]["sf"] = sfbonds
            all_bonds_spin[key]["fs"] = fsbonds
            all_bonds_spin[key]["pf"] = pfbonds
            all_bonds_spin[key]["fp"] = fpbonds
            all_bonds_spin[key]["df"] = pfbonds
            all_bonds_spin[key]["fd"] = fpbonds
            all_bonds_spin[key]["ff"] = ffbonds


        # combine spin up and down or double energy
        all_bonds = {}
        if self.spin_polar:
            bonds = bonds_spin[0] + bonds_spin[1]
            for orbkey in ["ss", "sp", "ps", "pp", "sd", "ds", "pd", "dp", "dd","sf", "fs", "pf", "fp", "df", "fd", "ff"]: # add f orbs
                all_bonds[orbkey] = all_bonds_spin[0][orbkey] + all_bonds_spin[1][orbkey]
        else:
            bonds = bonds_spin[0] * 2
            for orbkey in ["ss", "sp", "ps", "pp", "sd", "ds", "pd", "dp", "dd","sf", "fs", "pf", "fp", "df", "fd", "ff"]: # add f orbs
                all_bonds[orbkey] = all_bonds_spin[0][orbkey] * 2

        # get vectors to bonds
        num_bonds = len(bonds)
        bond_indices = np.unravel_index(np.arange(num_bonds), (self.numAtoms, self.numAtoms, cell[0], cell[1], cell[2]))
        is_onsite = (bond_indices[2] == each_dir[0]) & (bond_indices[3] == each_dir[1]) & (
                bond_indices[4] == each_dir[2]) & (bond_indices[0] == bond_indices[1])
        bonds[is_onsite] = 0
        start_atom = [bond_indices[0], np.ones(num_bonds, dtype=np.int_) * each_dir[0],
                      np.ones(num_bonds, dtype=np.int_) * each_dir[1], np.ones(num_bonds, dtype=np.int_) * each_dir[2]]
        start_ind = np.ravel_multi_index(start_atom, (self.numAtoms, cell[0], cell[1], cell[2]))
        end_atom = [bond_indices[1], bond_indices[2], bond_indices[3], bond_indices[4]]
        end_ind = np.ravel_multi_index(end_atom, (self.numAtoms, cell[0], cell[1], cell[2]))
        atom_indices = np.unravel_index(np.arange(num_total_atoms), (self.numAtoms, cell[0], cell[1], cell[2]))[0]
        atom_elems = self.elements[atom_indices]
        og_atom_labels = copy.deepcopy(atom_labels)
        if atom_labels == []:
            atom_labels = atom_elems
        else:
            atom_labels = atom_labels[atom_indices]
        uniq_elem, atom_to_elem = np.unique(self.elements, return_inverse=True)
        # print("atm to elem:",atom_to_elem)
        # print(atom_to_elem[atom_indices])
        if atom_colors == []:
            if elem_colors == []:
                colors = np.array(
                    ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
                     '#17becf', '#564580', '#B87333', '#63c433', '#db61ed', '#298ec4', '#db61ed'])
                atm_colors = colors[atom_to_elem[atom_indices]]
            elif type(elem_colors[0]) == int:
                elem_colors = np.array(elem_colors)
                colors = np.array(
                    ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
                     '#17becf', '#564580', '#B87333', '#63c433', '#db61ed', '#298ec4', '#db61ed'])
                atm_colors = colors[elem_colors[atom_to_elem[atom_indices]]]
            else:

                elem_colors = np.array(elem_colors)
                atm_colors = elem_colors[atom_to_elem[atom_indices]]
        elif type(atom_colors[0]) == int:
            atom_colors = np.array(atom_colors)
            colors = np.array(
                ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
                 '#17becf', '#564580', '#B87333', '#63c433', '#db61ed', '#298ec4', '#db61ed'])
            atm_colors = colors[atom_colors[atom_indices]]
        else:
            atom_colors = np.array(atom_colors)
            atm_colors = atom_colors[atom_indices]
        atom_rad = []
        for elem in atom_elems:
            if elem in all_atom_rad.keys():
                atom_rad.append(all_atom_rad[elem])
            else:
                atom_rad.append(1.3)
        # colors = ["blue","green","orange","purple","red",
        def cylinder_mesh(point1, point2, radius, resolution=20):
            """
            Create the coordinates of a cylinder mesh between two points.
            Args:
                point1 (array-like): [x, y, z] coordinates of one end of the cylinder.
                point2 (array-like): [x, y, z] coordinates of the other end of the cylinder.
                radius (float): Radius of the cylinder.
                resolution (int): Number of segments along the circle.

            Returns:    
                (x, y, z): Mesh coordinates for the cylinder.
            """
            # Vector along the cylinder's axis
            v = np.array(point2) - np.array(point1)
            v = v / np.linalg.norm(v)  # Normalize the vector

            # Create a perpendicular vector

            # Choose an arbitrary vector not parallel to v
            if abs(v[0]) < 1e-8 and abs(v[1]) < 1e-8:
                # v is approximately along z-axis
                not_v = np.array([1, 0, 0])
            else:
                not_v = np.array([0, 0, 1])
            n1 = np.cross(v, not_v)
            n1 /= np.linalg.norm(n1)
            n2 = np.cross(v, n1)

            # Generate points around a circle at the base of the cylinder
            t = np.linspace(0, 2 * np.pi, resolution)
            circle = np.array([radius * np.cos(t), radius * np.sin(t)])  # Circle coordinates

            # Mesh points along the cylinder's axis
            X, Y, Z = [], [], []
            for i in [0, 1]:  # Iterate over start and end of the cylinder
                point = point1 if i == 0 else point2
                x = point[0] + circle[0] * n1[0] + circle[1] * n2[0]
                y = point[1] + circle[0] * n1[1] + circle[1] * n2[1]
                z = point[2] + circle[0] * n1[2] + circle[1] * n2[2]
                X.append(x), Y.append(y), Z.append(z)

            return np.concatenate(X), np.concatenate(Y), np.concatenate(Z)

        def midsection_hover_points(point1, point2, radius, resolution=20):
            """
            Generate hover points along the midsection of the cylinder.
            Args:
                point1, point2: Start and end points of the cylinder.
                radius: Radius of the cylinder.
                resolution: Number of points along the circle.

            Returns:    
                (x, y, z, text): Coordinates and hover labels for the midsection points.
            """
            # Calculate the midpoint of the cylinder
            midpoint = (np.array(point1) + np.array(point2)) / 2

            # Vector along the cylinder's axis
            v = np.array(point2) - np.array(point1)
            v = v / np.linalg.norm(v)  # Normalize the vector

            # Create two perpendicular vectors
            if abs(v[0]) < 1e-8 and abs(v[1]) < 1e-8:
                # v is approximately along z-axis
                not_v = np.array([1, 0, 0])
            else:
                not_v = np.array([0, 0, 1])
            n1 = np.cross(v, not_v) / np.linalg.norm(np.cross(v, not_v))
            n2 = np.cross(v, n1)

            # Generate points around the circle at the midpoint
            t = np.linspace(0, 2 * np.pi, resolution)
            x = midpoint[0] + radius * np.cos(t) * n1[0] + radius * np.sin(t) * n2[0]
            y = midpoint[1] + radius * np.cos(t) * n1[1] + radius * np.sin(t) * n2[1]
            z = midpoint[2] + radius * np.cos(t) * n1[2] + radius * np.sin(t) * n2[2]

            return x, y, z

        # plot the bond lines
        # Create connecting lines trace
        data = []
        all_start_atms = np.array([[10,10,10]])
        all_end_atms = np.array([[10,10,10]])
        all_centers = np.array([[10,10,10]])
        all_text = []
        all_colors = []
        all_energies = []
        all_styles = []
        all_atmrads = []
        all_atm_elems = []
        all_atm_labels = []
        for b_ind in range(num_bonds):
            bond_energy = bonds[b_ind]
            if bond_energy < 0:
                style = "solid"
            else:
                style = "dash"
            if abs(bond_energy) > energy_cutoff:
                start = start_ind[b_ind]
                end = end_ind[b_ind]
                ogstart_atm = np.array([x_data[start], y_data[start], z_data[start]])
                ogend_atm = np.array([x_data[end], y_data[end], z_data[end]])
                # place atoms also on 1 when start atm is at 0
                trans = np.array([[0, 0, 0]])
                hold_atm = np.around(abc_data[:, start], decimals=3)  # start atm in primitive coordinates
                abc = np.linalg.norm(self._a, axis=1)
                # print(hold_atm,hold_atm==0)
                if (hold_atm == 0).any():
                    for vectoswitch in np.arange(3)[hold_atm == 0]:  # (hold_atm==0).any():
                        if abc[vectoswitch] < 4 * (
                                np.sum(abc) - abc[vectoswitch]) / 2:  # aviod 2D cases with lots of vacuum
                            # vectoswitch = np.arange(3)[hold_atm==0][0]
                            switch = np.zeros(3)
                            switch[vectoswitch] = 1
                            new_trans = trans[:, :] + switch[None, :]
                            trans = np.append(trans, new_trans, axis=0)
                cart_trans = _red_to_cart((self._a[0], self._a[1], self._a[2]), trans)
                # print(cart_trans)
                for ti, tran in enumerate(cart_trans):
                    start_atm = ogstart_atm + tran
                    end_atm = ogend_atm + tran
                    prim_startatm = abc_data[:, start] + trans[ti]
                    plot = True
                    if one_atom == True:
                        if (np.abs((prim_startatm - self.primAtoms[plot_atom])) > 0.01).any():
                            plot = False
                    if plot == True:
                        prim_endatm = abc_data[:, end] + trans[ti]
                        abc = np.linalg.norm(self._a, axis=1)
                        dist = copy.deepcopy(prim_endatm)
                        dist[prim_endatm >= 0] = 0  # only keep negative values
                        dist[prim_endatm > 1] = prim_endatm[prim_endatm > 1] - 1
                        weighted_dist = np.abs(dist) * abc
                        bond_dist = np.linalg.norm(end_atm - start_atm)
                        # print(start_atm,end_atm, (weighted_dist < 0.1).all(), bond_dist < bond_max)
                        # first check
                        is_new_bond = False
                        if (weighted_dist < 0.1).all() or bond_dist < bond_max:
                            is_new_bond = True
                            # check whether another bond already exists with the same end points
                            same_start = (all_start_atms[:,0]==start_atm[0]) & (all_start_atms[:,1]==start_atm[1]) & (all_start_atms[:,2]==start_atm[2])
                            check_end_atm = all_end_atms[same_start]
                            same_end = ((check_end_atm[:,0]==end_atm[0]) & (check_end_atm[:,1]==end_atm[1]) & (check_end_atm[:,2]==end_atm[2])).any()
                            if same_end == False: # also check for opposite
                                same_start = (all_start_atms[:, 0] == end_atm[0]) & (
                                            all_start_atms[:, 1] == end_atm[1]) & (
                                                         all_start_atms[:, 2] == end_atm[2])
                                check_end_atm = all_end_atms[same_start]
                                same_end = ((check_end_atm[:, 0] == start_atm[0]) & (
                                            check_end_atm[:, 1] == start_atm[1]) & (
                                                        check_end_atm[:, 2] == start_atm[2])).any()
                            if same_end:
                                if self.verbose > 1:
                                    print("avoided duplicate!")
                            is_new_bond = not same_end

                        if is_new_bond:
                            all_start_atms = np.append(all_start_atms,np.array([start_atm]),axis=0)
                            all_end_atms = np.append(all_end_atms,np.array([end_atm]),axis=0)
                            # print("got new trans:",trans)
                            NN = np.arange(len(self.NN_dists))[
                                np.argmin(np.abs(self.NN_dists - bond_dist))]  # <= 0.0001][0]
                            centerx = (start_atm[0] + end_atm[0]) / 2
                            centery = (start_atm[1] + end_atm[1]) / 2
                            centerz = (start_atm[2] + end_atm[2]) / 2
                            orbs = ["s", "p", "d", "f"]
                            bond_label = ""
                            for orb1 in orbs:
                                for orb2 in orbs:
                                    bond_eng = all_bonds[orb1 + orb2][b_ind]
                                    if np.abs(bond_eng) > np.abs(bond_energy) / 10:
                                        bond_str = atom_elems[start] + " " + orb1 + " - " + atom_elems[
                                            end] + " " + orb2 + ": " + str(
                                            np.around(bond_eng, decimals=3)) + " eV" + "<br>"
                                        bond_label += bond_str
                            spin_label = ""
                            if self.spin_polar:
                                spin_label += "<br>Energy from spin up: " + str(
                                    np.around(bonds_spin[0][b_ind], decimals=3)) + " eV"
                                spin_label += "<br>Energy from spin down: " + str(
                                    np.around(bonds_spin[1][b_ind], decimals=3)) + " eV"
                            full_text = ("Bond energy: " + str(
                                np.around(bond_energy, decimals=3)) + " eV" + "<br>Bond length: "
                                         + str(np.around(bond_dist, decimals=3)) + " Å" + " (" + str(NN) + "NN)" +
                                         spin_label + "<br>Breakdown into orbitals: <br>" + bond_label)
                            all_atm_elems.append([atom_elems[start],atom_elems[end]])
                            all_atm_labels.append([atom_labels[start],atom_labels[end]])
                            '''
                            data.append(go.Scatter3d(
                                x=[centerx],
                                y=[centery],
                                z=[centerz],
                                mode='markers',
                                marker=dict(size=20, color='rgba(0,0,0,0)', opacity=0.),
                                hoverinfo="text",
                                hoverlabel=dict(bgcolor='ghostwhite',
                                                font=dict(family='Arial', size=12, color='black')),
                                text=full_text, showlegend=False
                            ))
                            '''
                            start_rad = atom_rad[start]
                            end_rad = atom_rad[end]
                            all_colors.append([atm_colors[start], atm_colors[end]])
                            all_energies.append(bond_energy)
                            all_styles.append(style)
                            all_atmrads.append([start_rad, end_rad])
                            if style == 'dash':
                                all_centers = np.append(all_centers,np.array([[centerx,centery,centerz]]),axis=0)
                                all_text.append(full_text)
                                data.append(go.Scatter3d(
                                    x=[start_atm[0], end_atm[0]],
                                    y=[start_atm[1], end_atm[1]],
                                    z=[start_atm[2], end_atm[2]],
                                    mode='lines',#'markers+lines',
                                    #marker=dict(size=[start_rad * 40, end_rad * 40],
                                    #            color=[atm_colors[start], atm_colors[end]], opacity=1.0,
                                    #            line=dict(color='black', width=20)),
                                    line=dict(
                                        color='gray',
                                        width=abs(bond_energy/2) ** (1 / 2) * 20,
                                        dash=style
                                    ),
                                    #hoverinfo="text",
                                    #hoverlabel=dict(bgcolor='ghostwhite',
                                    #                font=dict(family='Arial', size=12, color='black')),
                                    #text=full_text,
                                    hoverinfo='none',
                                    showlegend=False))
                            else:
                                # Get the mesh coordinates for the cylinder
                                radius = abs(bond_energy/2) ** (1 / 2) / 8
                                x, y, z = cylinder_mesh(start_atm, end_atm, radius,resolution=15)

                                # Plot the cylinder using Plotly Mesh3d
                                data.append(go.Mesh3d(
                                    x=x, y=y, z=z,
                                    opacity=1.0,
                                    color='gray',
                                    flatshading=False,  # Smooth shading
                                    lighting=dict(
                                        ambient=0.6,  # Ambient light (general light)
                                        diffuse=0.7,  # Diffuse light (soft shadows)
                                        fresnel=0.1,  # Reflective edges
                                        specular=0.9,  # Specular reflection (sharp highlights)
                                        roughness=0.55,  # Surface roughness
                                    ),
                                    lightposition=dict(
                                        x=10, y=10, z=10  # Light source position
                                    ),
                                    hoverinfo="none",
                                    #hoverlabel=dict(bgcolor='ghostwhite',
                                    #                font=dict(family='Arial', size=12, color='black')),
                                    #text=full_text,
                                    alphahull=0
                                ))

                                # plot center cross-section for hover labels
                                # Get hover points along the midsection circle
                                hover_x, hover_y, hover_z = midsection_hover_points(start_atm, end_atm, radius)
                                # Create hover points using Scatter3d
                                data.append(go.Scatter3d(
                                    x=hover_x, y=hover_y, z=hover_z,
                                    mode='markers',
                                    marker=dict(size=20, color='rgba(0,0,0,0)', opacity=0.),
                                    hoverinfo="text",
                                    hoverlabel=dict(bgcolor='ghostwhite',
                                                    font=dict(family='Arial', size=12, color='black')),
                                    text=full_text, showlegend=False
                                ))

        #print(all_end_atms.shape,all_start_atms.shape)
        # find all unique atoms
        combnd_atms = np.append(all_start_atms[1:],all_end_atms[1:],axis=0)
        all_colors = np.array(all_colors)
        all_atmrads = np.array(all_atmrads)
        all_atm_elems = np.array(all_atm_elems)
        all_atm_labels = np.array(all_atm_labels)
        combnd_colors = np.append(all_colors[:,0],all_colors[:,1])
        combnd_atmrad = np.append(all_atmrads[:,0],all_atmrads[:,1])
        combnd_elems = np.append(all_atm_elems[:,0],all_atm_elems[:,1])
        combnd_labels = np.append(all_atm_labels[:,0],all_atm_labels[:,1])
        uniq_atms,indices = np.unique(combnd_atms,axis=0,return_index=True)
        uniq_colors = combnd_colors[indices]
        uniq_rad = combnd_atmrad[indices]
        uniq_elems = combnd_elems[indices]
        uniq_labels = combnd_labels[indices]
        if self.verbose > 1:
            print("all atms:",uniq_atms)
            print(uniq_rad)
            print(uniq_colors)

        # append the bond labels
        all_centers = all_centers[1:]
        data.append(go.Scatter3d(
            x=all_centers[:,0],
            y=all_centers[:,1],
            z=all_centers[:,2],
            mode='markers',
            marker=dict(size=20, color='rgba(0,0,0,0)', opacity=0.),
            hoverinfo="text",
            hoverlabel=dict(bgcolor='ghostwhite',
                            font=dict(family='Arial', size=12, color='black')),
            text=all_text, showlegend=False))


        def create_sphere(center, radius, resolution=15):
            """
            Generate the mesh coordinates for a sphere.
            """
            u = np.linspace(0, 2 * np.pi, resolution)
            v = np.linspace(0, np.pi, resolution)
            x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
            y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
            z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
            return x.flatten(), y.flatten(), z.flatten()

        # Loop over atoms to create a sphere for each atom
        for ai,atm in enumerate(uniq_atms):
            x, y, z = create_sphere(atm, uniq_rad[ai]**(0.8)/2)
            trace = go.Mesh3d(
                x=x, y=y, z=z,
                color=uniq_colors[ai],
                opacity=1.0,
                lighting=dict(
                    ambient=0.6,  # Ambient light (general light)
                    diffuse=0.7,  # Diffuse light (soft shadows)
                    fresnel=0.1,  # Reflective edges
                    specular=0.9,  # Specular reflection (sharp highlights)
                    roughness=0.55,  # Surface roughness
                ),
                lightposition=dict(
                    x=10, y=10, z=10  # Light source position
                ),
                alphahull=0,  # Proper enclosure for the mesh
                hoverinfo = "text",
                #hoverlabel = dict(bgcolor='ghostwhite',
                #                  font=dict(family='Arial', size=12, color='black')),
                text = uniq_labels[ai], showlegend = False)

            data.append(trace)

        if one_atom == False:
            # draw primitive cell # place atoms also on 1 when start atm is at 0
            trans = np.array([[0, 0, 0]])
            for vectoswitch in np.arange(3):  # (hold_atm==0).any():
                if abc[vectoswitch] < 4 * (np.sum(abc) - abc[vectoswitch]) / 2:  # aviod 2D cases with lots of vacuum
                    # vectoswitch = np.arange(3)[hold_atm==0][0]
                    switch = np.zeros(3)
                    switch[vectoswitch] = 1
                    new_trans = trans[:, :] + switch[None, :]
                    trans = np.append(trans, new_trans, axis=0)
            num = len(trans)
            start_points = trans[:, None, :] * np.ones((num, num, 3))
            end_points = trans[None, :, :] * np.ones((num, num, 3))
            all_vecs = end_points - start_points
            norm = np.linalg.norm(all_vecs, axis=2)
            start_points = np.reshape(start_points, (num ** 2, 3))[norm.flatten() == 1]
            end_points = np.reshape(end_points, (num ** 2, 3))[norm.flatten() == 1]
            final_ind = []
            for veci in range(len(start_points)):
                diff = end_points[veci] - start_points[veci]
                if (diff >= 0).all():
                    final_ind.append(veci)
            final_ind = np.array(final_ind)

            start_points = start_points[final_ind]
            end_points = end_points[final_ind]
            cart_start = _red_to_cart((self._a[0], self._a[1], self._a[2]), start_points)
            cart_end = _red_to_cart((self._a[0], self._a[1], self._a[2]), end_points)

            for i in range(len(start_points)):
                start_atm = cart_start[i]
                end_atm = cart_end[i]
                data.append(go.Scatter3d(
                    x=[start_atm[0], end_atm[0]],
                    y=[start_atm[1], end_atm[1]],
                    z=[start_atm[2], end_atm[2]],
                    mode='lines',
                    line=dict(
                        color='black',
                        width=2,
                        dash="solid"
                    ), hoverinfo='none', showlegend=False))


        # Create figure
        fig = go.Figure(data=data)

        # update figure design
        # fig.update_traces(hovertemplate=None)
        fig.update_layout(hoverdistance=100, margin=dict(l=0, r=0, b=0, t=0),
                          paper_bgcolor='rgba(0,0,0,0)',  # Transparent background for the whole figure
                          plot_bgcolor='rgba(0,0,0,0)')  # ,hovermode="x unified")
        fig.update_layout(
            scene=dict(xaxis=dict(showbackground=False, showspikes=False, showticklabels=False, visible=False),
                       yaxis=dict(showbackground=False, showspikes=False, showticklabels=False, visible=False),
                       zaxis=dict(showbackground=False, showspikes=False, showticklabels=False, visible=False),
                       aspectmode='data'))

        #fig.layout.scene.camera.projection.type = "orthographic"
        # Set layout with camera and FOV (fovy)
        fig.update_layout(scene_camera=dict(eye=dict(x=1.25,y=-0.05,z=0.05)))

        fov = fovy
        dist = np.around(75/fov,decimals=2)
        post_script = ("const plot = document.querySelector('.js-plotly-plot'); const scene = plot._fullLayout.scene._scene;"
                   " const camera = scene.glplot; camera.fovy ="+str(fov)+" * Math.PI / 180;  scene.camera.distance ="+str(dist)+";  "
                       "console.log(`Updated FOV to ${camera.fovy} radians.`);")

        if return_fig:
            return fig, post_script

        else:
            if one_atom:
                fig.write_html(self.directory + "crystal_bonds" + str(plot_atom) + ".html", post_script=post_script,
                               div_id="plotDiv")
            else:
                fig.write_html(self.directory + "crystal_bonds.html", post_script=post_script, div_id="plotDiv")

            fig.show(post_script=post_script, div_id="plotDiv")

    def get_bonds_charge_figure(self, energy_cutoff: float = 0.1, bond_max: float = 3.0, elem_colors: list = [], atom_colors: list = [], atom_labels: list = [],
                                auto_label: str = "", plot_atom: int = None,
                         one_atom: bool = False, fovy: float = 10,return_fig: bool = False) -> None:
        """
        This will plot the crystal structure atoms with line weighted by iCOHP.
        Each line should also be hoverable to reveal the number and amounts that are s-s,s-p, and p-p.

        Args:    
            energy_cutoff (float): This is the minimum bond magnitude that will be plotted
            bond_max (float): Atoms are plotted outside the primitive cell only if they have a bond with
                        a primitive cell atom that is shorter than this amount. This does not filter bonds between atoms that are already plotted.
            elem_colors (list): Colors for the elements based on order in tb_input. Length of list should be the number of
                        unique elements. Can either be integer list to reference the default colors or list of
                        plotly compatable colors.
            atom_colors (list): Colors for the atoms based on order in tb_input. Length of list should be the number of
                        atoms in the primitive cell. Can either be integer list to reference the default colors or
                        list of plotly compatable colors. If not set defaults to elem_colors.
            atom_labels (list): List of atom labels as a string.
            auto_label (str): Different options for plotting includes: (can include multiple in the string)
                        "mulliken" - Plots the onsite charge and mag (if spin_polar) on atoms by mulliken population (overrides atom_labels).
                        "full" - Plots the charge and magnetics moments (if spin_polar) on atoms and bonds (overrides atom_labels).
                        "color" - Colors the atoms and bonds based on their charge (overrides atom_colors or elem_colors).
                        "color mag" - Colors the atoms and bonds based on their magnetic moments (overrides atom_colors or elem_colors).
                        NOTE: Only use "mulliken" OR "full", NOT both.
            plot_atom (int): Set with one_atom=True, plots only one atom and it's bonds, this passes the atom number to plot.
            one_atom (bool): Whether only the atom defined in plot_atom should be plotted; default is False.
            fovy (float): Field of view in the vertical direction. Use this tag to adjust depth perception in crystal.
                        Set between 3 (for close to orthographic) and 30 (for good perspective depth).
            return_fig (bool): If False, this function saves figure to crystal_bonds.html. If True, this function will return the plotly figure object

        Returns:    
            None: Depend on return_fig parameter.
        """
        import plotly.graph_objs as go
        if not hasattr(self, 'NN_index'):
            self.get_neighbors()
        # atom size based on element
        all_atom_rad = {"H": 0.53, "He": 0.31, "Li": 1.51, "Be": 1.12, "B": 0.87, "C": 0.67, "N": 0.56, "O": 0.48,
                        "F": 0.42, "Ne": 0.38,
                        "Na": 1.90, "Mg": 1.45, "Al": 1.18, "Si": 1.11, "P": 0.98, "S": 0.88, "Cl": 0.79, "Ar": 0.71,
                        "K": 2.43, "Ca": 1.94, "Sc": 1.84, "Ti": 1.76, "V": 1.71, "Cr": 1.66, "Mn": 1.61, "Fe": 1.56,
                        "Co": 1.52,
                        "Ni": 1.49, "Cu": 1.45, "Zn": 1.42, "Ga": 1.36, "Ge": 1.25, "As": 1.14, "Se": 1.03, "Br": 0.94,
                        "Kr": 0.88,
                        "Rb": 2.65, "Sr": 2.19, "Y": 2.12, "Zr": 2.06, "Nb": 1.98, "Mo": 1.90, "Tc": 1.83, "Ru": 1.78,
                        "Rh": 1.73,
                        "Pd": 1.69, "Ag": 1.65, "Cd": 1.61, "In": 1.56, "Sn": 1.45, "Sb": 1.33, "Te": 1.23, "I": 1.15,
                        "Xe": 1.08,
                        "Cs": 2.98, "Ba": 2.53, "Lu": 2.17, "Hf": 2.08, "Ta": 2.00, "W": 1.93, "Re": 1.88, "Os": 1.85,
                        "Ir": 1.80,
                        "Pt": 1.77, "Au": 1.75, "Hg": 1.71, "Tl": 1.56, "Pb": 1.54, "Bi": 1.43, "Po": 1.35, "At": 1.27,
                        "Rn": 1.20,
                        "La": 1.70,
                        "Ce": 1.70,}
        do_coop = True
        if "full" in auto_label:
            do_coop = True
        # get the cohp
        #if not hasattr(self, 'ICOHP'):
        #    self.get_ICOHP()
        if do_coop:
            #if not hasattr(self, 'ICOOP'):
            #    self.get_ICOOP()
            icoop_orb = self.ICOOP
        icohp_orb = self.ICOHP

        # each_dir = np.array([1, 1, 1]) #self.num_each_dir
        cell = self.num_trans  # np.array(self.num_trans) # each_dir * 2 + 1
        vec_to_trans = np.zeros((cell[0], cell[1], cell[2], 3))  # self.vec_to_trans
        each_dir = self.num_each_dir


        for x in range(cell[0]):
            for y in range(cell[1]):
                for z in range(cell[2]):
                    vec_to_trans[x, y, z] = [x - each_dir[0], y - each_dir[1], z - each_dir[2]]

        num_total_atoms = self.numAtoms * cell[0] * cell[1] * cell[2]
        vec_to_atoms = vec_to_trans[None, :, :, :] + np.array(self.primAtoms)[:, None, None, None]
        vec_to_atoms = np.reshape(vec_to_atoms, (num_total_atoms, 3)).T
        # print(vec_to_atoms)
        cart_to_atoms = _red_to_cart((self._a[0], self._a[1], self._a[2]), vec_to_atoms.transpose()).T

        # plot the atoms onto the crystal
        x_data = cart_to_atoms[0]
        y_data = cart_to_atoms[1]
        z_data = cart_to_atoms[2]
        abc_data = vec_to_atoms
        # crystal = go.Scatter3d(x=x_data,y=y_data,z=z_data,mode='markers',marker=dict(size=20,color='blue',opacity=0.8),hoverinfo='none')
        if self.spin_polar:
            keys = [0, 1]
        else:
            keys = [0]

        all_bonds_spin = {}
        bonds_spin = {}
        charge_spin = {}
        for key in keys:
            icohp_orb[key] = icohp_orb[key].real  # self.TB_model.get_ICOHP(self,spin=key).real
            # [s,p]*[s,p] = [s-s,s-p,p-s,p-p]
            # [s,p] * [s,p,d] = [s-s,s-p,p-s,p-p,s-d,p-d]
            icohp_atom = np.zeros((self.numAtoms, self.numAtoms, cell[0], cell[1], cell[2]))
            if do_coop:
                icoop_orb[key] = icoop_orb[key].real
                icoop_atom = np.zeros((self.numAtoms, self.numAtoms, cell[0], cell[1], cell[2]))
            ssbonds = np.zeros((self.numAtoms, self.numAtoms, cell[0], cell[1], cell[2]))
            spbonds = np.zeros((self.numAtoms, self.numAtoms, cell[0], cell[1], cell[2]))
            psbonds = np.zeros((self.numAtoms, self.numAtoms, cell[0], cell[1], cell[2]))
            ppbonds = np.zeros((self.numAtoms, self.numAtoms, cell[0], cell[1], cell[2]))
            sdbonds = np.zeros((self.numAtoms, self.numAtoms, cell[0], cell[1], cell[2]))
            dsbonds = np.zeros((self.numAtoms, self.numAtoms, cell[0], cell[1], cell[2]))
            pdbonds = np.zeros((self.numAtoms, self.numAtoms, cell[0], cell[1], cell[2]))
            dpbonds = np.zeros((self.numAtoms, self.numAtoms, cell[0], cell[1], cell[2]))
            ddbonds = np.zeros((self.numAtoms, self.numAtoms, cell[0], cell[1], cell[2]))
            # add f orbs
            sfbonds = np.zeros((self.numAtoms, self.numAtoms, cell[0], cell[1], cell[2]))
            fsbonds = np.zeros((self.numAtoms, self.numAtoms, cell[0], cell[1], cell[2]))
            pfbonds = np.zeros((self.numAtoms, self.numAtoms, cell[0], cell[1], cell[2]))
            fpbonds = np.zeros((self.numAtoms, self.numAtoms, cell[0], cell[1], cell[2]))
            dfbonds = np.zeros((self.numAtoms, self.numAtoms, cell[0], cell[1], cell[2]))
            fdbonds = np.zeros((self.numAtoms, self.numAtoms, cell[0], cell[1], cell[2]))
            ffbonds = np.zeros((self.numAtoms, self.numAtoms, cell[0], cell[1], cell[2]))
            for atm1 in range(self.numAtoms):
                for atm2 in range(self.numAtoms):
                    orbs1 = np.arange(self.num_orbs)[self.orbatomnum == atm1]
                    orbs2 = np.arange(self.num_orbs)[self.orbatomnum == atm2]
                    sum_cohp = icohp_orb[key][orbs1, :][:, orbs2]
                    if do_coop:
                        sum_coop = icoop_orb[key][orbs1, :][:, orbs2]
                    orb_type1 = self.exactorbtype[orbs1]
                    orb_type2 = self.exactorbtype[orbs2]
                    # s-s
                    is_s1 = []
                    is_p1 = []
                    is_d1 = []
                    is_f1 = []
                    for i in orb_type1:
                        is_s1.append("s" in i)
                        is_p1.append("p" in i)
                        is_d1.append("d" in i)
                        is_f1.append("f" in i)
                    is_s2 = []
                    is_p2 = []
                    is_d2 = []
                    is_f2 = []
                    for i in orb_type2:
                        is_s2.append("s" in i)
                        is_p2.append("p" in i)
                        is_d2.append("d" in i)
                        is_f2.append("f" in i)
                    sorbs1 = np.arange(len(orb_type1))[is_s1]
                    porbs1 = np.arange(len(orb_type1))[is_p1]
                    dorbs1 = np.arange(len(orb_type1))[is_d1]
                    forbs1 = np.arange(len(orb_type1))[is_f1]
                    sorbs2 = np.arange(len(orb_type2))[is_s2]
                    porbs2 = np.arange(len(orb_type2))[is_p2]
                    dorbs2 = np.arange(len(orb_type2))[is_d2]
                    forbs2 = np.arange(len(orb_type2))[is_f2]

                    ssbond = np.sum(sum_cohp[sorbs1, :][:, sorbs2], axis=(0, 1))
                    spbond = np.sum(sum_cohp[sorbs1, :][:, porbs2], axis=(0, 1))
                    psbond = np.sum(sum_cohp[porbs1, :][:, sorbs2], axis=(0, 1))
                    ppbond = np.sum(sum_cohp[porbs1, :][:, porbs2], axis=(0, 1))
                    sdbond = np.sum(sum_cohp[sorbs1, :][:, dorbs2], axis=(0, 1))
                    dsbond = np.sum(sum_cohp[dorbs1, :][:, sorbs2], axis=(0, 1))
                    pdbond = np.sum(sum_cohp[porbs1, :][:, dorbs2], axis=(0, 1))
                    dpbond = np.sum(sum_cohp[dorbs1, :][:, porbs2], axis=(0, 1))
                    ddbond = np.sum(sum_cohp[dorbs1, :][:, dorbs2], axis=(0, 1))
                    # add f orbs
                    sfbond = np.sum(sum_cohp[sorbs1, :][:, forbs2], axis=(0, 1))
                    fsbond = np.sum(sum_cohp[forbs1, :][:, sorbs2], axis=(0, 1))
                    pfbond = np.sum(sum_cohp[porbs1, :][:, forbs2], axis=(0, 1))
                    fpbond = np.sum(sum_cohp[forbs1, :][:, porbs2], axis=(0, 1))
                    dfbond = np.sum(sum_cohp[dorbs1, :][:, forbs2], axis=(0, 1))
                    fdbond = np.sum(sum_cohp[forbs1, :][:, dorbs2], axis=(0, 1))
                    ffbond = np.sum(sum_cohp[forbs1, :][:, forbs2], axis=(0, 1))

                    icohp_atom[atm1, atm2] = np.sum(sum_cohp, axis=(0, 1))
                    if do_coop:
                        icoop_atom[atm1, atm2] = np.sum(sum_coop, axis=(0, 1))
                    ssbonds[atm1, atm2] = ssbond
                    spbonds[atm1, atm2] = spbond
                    psbonds[atm1, atm2] = psbond
                    ppbonds[atm1, atm2] = ppbond
                    sdbonds[atm1, atm2] = sdbond
                    dsbonds[atm1, atm2] = dsbond
                    pdbonds[atm1, atm2] = pdbond
                    dpbonds[atm1, atm2] = dpbond
                    ddbonds[atm1, atm2] = ddbond
                    # add f orbs
                    sfbonds[atm1, atm2] = sfbond
                    fsbonds[atm1, atm2] = fsbond
                    pfbonds[atm1, atm2] = pfbond
                    fpbonds[atm1, atm2] = fpbond
                    dfbonds[atm1, atm2] = dfbond
                    fdbonds[atm1, atm2] = fdbond
                    ffbonds[atm1, atm2] = ffbond
            #print(icohp_atom.shape)
            # print("cohp!",icohp_atom)
            bonds_spin[key] = icohp_atom.flatten()
            if do_coop:
                charge_spin[key] = icoop_atom.flatten()
            ssbonds = ssbonds.flatten()
            spbonds = spbonds.flatten()
            psbonds = psbonds.flatten()
            ppbonds = ppbonds.flatten()
            sdbonds = sdbonds.flatten()
            dsbonds = dsbonds.flatten()
            pdbonds = pdbonds.flatten()
            dpbonds = dpbonds.flatten()
            ddbonds = ddbonds.flatten()
            # add f orbs
            sfbonds = sfbonds.flatten()
            fsbonds = fsbonds.flatten()
            pfbonds = pfbonds.flatten()
            fpbonds = fpbonds.flatten()
            dfbonds = dfbonds.flatten()
            fdbonds = fdbonds.flatten()
            ffbonds = ffbonds.flatten()
            all_bonds_spin[key] = {}
            all_bonds_spin[key]["ss"] = ssbonds
            all_bonds_spin[key]["sp"] = spbonds
            all_bonds_spin[key]["ps"] = psbonds
            all_bonds_spin[key]["pp"] = ppbonds
            all_bonds_spin[key]["sd"] = sdbonds
            all_bonds_spin[key]["ds"] = dsbonds
            all_bonds_spin[key]["pd"] = pdbonds
            all_bonds_spin[key]["dp"] = dpbonds
            all_bonds_spin[key]["dd"] = ddbonds
            # add f orbs
            all_bonds_spin[key]["sf"] = sfbonds
            all_bonds_spin[key]["fs"] = fsbonds
            all_bonds_spin[key]["pf"] = pfbonds
            all_bonds_spin[key]["fp"] = fpbonds
            all_bonds_spin[key]["df"] = pfbonds
            all_bonds_spin[key]["fd"] = fpbonds
            all_bonds_spin[key]["ff"] = ffbonds


        # combine spin up and down or double energy
        all_bonds = {}
        if self.spin_polar:
            bonds = bonds_spin[0] + bonds_spin[1] * 2 # x2 for bond degeneracy (<i|j+R> and <j|i-R>)
            for orbkey in ["ss", "sp", "ps", "pp", "sd", "ds", "pd", "dp", "dd","sf", "fs", "pf", "fp", "df", "fd", "ff"]: # add f orbs
                all_bonds[orbkey] = all_bonds_spin[0][orbkey] + all_bonds_spin[1][orbkey] * 2 # x2 for bond degeneracy (<i|j+R> and <j|i-R>)
            if do_coop:
                bond_charges = (- charge_spin[0] - charge_spin[1]) * 2 # x2 for bond degeneracy (<i|j+R> and <j|i-R>)
                bond_magmoms = (charge_spin[0] - charge_spin[1]) * 2 # x2 for bond degeneracy (<i|j+R> and <j|i-R>)
        else:
            bonds = bonds_spin[0] * 2 * 2 # x2 for bond degeneracy (<i|j+R> and <j|i-R>)
            if do_coop:
                bond_charges = - charge_spin[0] * 2 * 2 # consider fact that electrons are negative charge # extra x2 for fact that bond charge comes from <i|j+R> and <j|i-R>
            for orbkey in ["ss", "sp", "ps", "pp", "sd", "ds", "pd", "dp", "dd","sf", "fs", "pf", "fp", "df", "fd", "ff"]: # add f orbs
                all_bonds[orbkey] = all_bonds_spin[0][orbkey] * 2 * 2 # x2 for bond degeneracy (<i|j+R> and <j|i-R>)

        # get vectors to bonds
        num_bonds = len(bonds)
        bond_indices = np.unravel_index(np.arange(num_bonds), (self.numAtoms, self.numAtoms, cell[0], cell[1], cell[2]))
        is_onsite = (bond_indices[2] == each_dir[0]) & (bond_indices[3] == each_dir[1]) & (
                bond_indices[4] == each_dir[2]) & (bond_indices[0] == bond_indices[1])
        bonds[is_onsite] = 0

        # get atomic charge and mag
        import plotly.colors as pc
        def get_rgb_values(colorscale_name):
            """
            Retrieve RGB values and their positions from a Plotly colorscale.
            """
            colorscale = pc.get_colorscale(colorscale_name)
            # Convert the rgb strings to tuples of integers
            rgb_values = np.array([
                (point[0], np.array([int(c) for c in point[1][4:-1].split(',')]))
                for point in colorscale
            ], dtype=object)
            return rgb_values

        def interpolate_colors(values, vmin, vmax, rgb_values):
            """
            Vectorized interpolation of colors for a NumPy array of values.
            """
            # Normalize values to the [0, 1] range
            scaled_values = np.clip((values - vmin) / (vmax - vmin), 0, 1)
            # Interpolate colors using the color scale
            interpolated_colors = np.zeros((len(values), 3), dtype=int)
            for i in range(len(rgb_values) - 1):
                t1, color1 = rgb_values[i]
                t2, color2 = rgb_values[i + 1]
                # Find values that belong to the current color interval
                mask = (scaled_values >= t1) & (scaled_values <= t2)
                factor = (scaled_values[mask] - t1) / (t2 - t1)
                # Linear interpolation between the two colors
                interpolated_colors[mask] = (1 - factor)[:, None] * color1 + factor[:, None] * color2
            # Convert RGB to hex
            hex_colors = ['#{:02x}{:02x}{:02x}'.format(*color) for color in interpolated_colors]
            return hex_colors

        if do_coop:
            atm_charges = []
            atm_magmoms = []
            for atm in range(self.numAtoms):
                is_val = (bond_indices[2] == each_dir[0]) & (bond_indices[3] == each_dir[1]) & (
                bond_indices[4] == each_dir[2]) & (bond_indices[0] == atm) & (atm == bond_indices[1])
                #print(bond_charges[is_val][0])
                charge = self.elec_count[atm] + bond_charges[is_val][0] / 2 # divide out factor of two for bond degeneracy
                atm_charges.append(charge)
                bond_charges[is_val] = charge
                if self.spin_polar:
                    magmom = bond_magmoms[is_val][0] / 2 # divide out factor of two for bond degeneracy for onsite term
                    atm_magmoms.append(magmom)
                    bond_magmoms[is_val] = magmom
            if self.spin_polar:
                atm_magmoms = np.array(atm_magmoms)
            atm_charges = np.array(atm_charges)
            if self.verbose > 0:
                print("atom charge!",atm_charges)

            # get the color values for atom and bonds
            if ("color" in auto_label):
                if "mag" in auto_label:
                    vmax = np.amax(np.abs(bond_magmoms))/4 + 0.05 #np.log10(+1)
                    vals = bond_magmoms #np.log10(np.abs(bond_magmoms)+1)*np.sign(bond_magmoms)
                else:
                    vmax = np.amax(np.abs(bond_charges)) + 0.05
                    vals = bond_charges
                rgb_values = get_rgb_values('RdBu')
                vmin = -vmax
                bond_colors = interpolate_colors(vals, vmin, vmax, rgb_values)

        start_atom = [bond_indices[0], np.ones(num_bonds, dtype=np.int_) * each_dir[0],
                      np.ones(num_bonds, dtype=np.int_) * each_dir[1], np.ones(num_bonds, dtype=np.int_) * each_dir[2]]
        start_ind = np.ravel_multi_index(start_atom, (self.numAtoms, cell[0], cell[1], cell[2]))
        end_atom = [bond_indices[1], bond_indices[2], bond_indices[3], bond_indices[4]]
        end_ind = np.ravel_multi_index(end_atom, (self.numAtoms, cell[0], cell[1], cell[2]))
        atom_indices = np.unravel_index(np.arange(num_total_atoms), (self.numAtoms, cell[0], cell[1], cell[2]))[0]
        atom_elems = self.elements[atom_indices]
        if "mulliken" in auto_label:
            atom_labels = []
            atm_charges = []
            atm_magmoms = []
            for atmnum in range(self.numAtoms):
                vals = self.get_mulliken_charge(self,atmnum)
                #print("check type",type(vals),vals,atmnum)
                if type(vals) is float:
                    occ = str(np.around(vals,decimals=2))
                    elec = vals
                    if vals > 0:
                        occ = "+"+occ
                    label = self.elements[atmnum] + ": " + occ
                else: # has occ and mag
                    occ = str(np.around(vals[0],decimals=2))
                    elec = vals[0]
                    mom = vals[1]
                    mag = str(np.around(vals[1],decimals=2))
                    if vals[0] > 0:
                        occ = "+" + occ
                    if vals[1] > 0:
                        mag = "+" + mag
                    label = self.elements[atmnum] + ": " + occ + "\n" + "magmom: " + mag
                    atm_magmoms.append(mom)
                #print(label)
                atom_labels.append(label)
                atm_charges.append(elec)
            atm_charges = np.array(atm_charges)
            atm_magmoms = np.array(atm_magmoms)
        elif do_coop:
            atom_labels = []
            for atmnum in range(self.numAtoms):
                occ = np.around(atm_charges[atmnum], decimals=2)
                if occ > 0:
                    occ = "+" + str(occ)
                else:
                    occ = str(occ)
                if self.spin_polar:
                    mag = np.around(atm_magmoms[atmnum], decimals=2)
                    if mag > 0:
                        mag = "+" + str(mag)
                    else:
                        mag = str(mag)
                    label = self.elements[atmnum] + ": " + occ + "\n" + "magmom: " + mag
                else:
                    label = self.elements[atmnum] + ": " + occ

                #print(label)
                atom_labels.append(label)

        atom_labels = np.array(atom_labels)
        og_atom_labels = copy.deepcopy(atom_labels)
        if len(atom_labels) == 0:
            atom_labels = atom_elems
        else:
            atom_labels = atom_labels[atom_indices]
        if self.verbose > 1:
            print("atom labels:",atom_labels)
        uniq_elem, atom_to_elem = np.unique(self.elements, return_inverse=True)
        # print("atm to elem:",atom_to_elem)
        # print(atom_to_elem[atom_indices])

        # get the color values for atoms
        if ("color" in auto_label):
            if ("mag" in auto_label) and self.spin_polar:
                vmax = np.amax(np.abs(atm_magmoms))/4 + 0.05 #np.log10(+1)
                vals = atm_magmoms #np.log10(np.abs(atm_magmoms)+1)*np.sign(atm_magmoms)
            else:
                vmax = np.amax(np.abs(atm_charges)) + 0.05
                vals = atm_charges
            rgb_values = get_rgb_values('RdBu')
            vmin = -vmax
            atom_colors = interpolate_colors(vals, vmin, vmax, rgb_values)
            atom_colors = np.array(atom_colors)

        if len(atom_colors) == 0:
            if len(elem_colors) == 0:
                colors = np.array(
                    ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
                     '#17becf', '#564580', '#B87333', '#63c433', '#db61ed', '#298ec4', '#db61ed'])
                atm_colors = colors[atom_to_elem[atom_indices]]
            elif type(elem_colors[0]) == int:
                elem_colors = np.array(elem_colors)
                colors = np.array(
                    ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
                     '#17becf', '#564580', '#B87333', '#63c433', '#db61ed', '#298ec4', '#db61ed'])
                atm_colors = colors[elem_colors[atom_to_elem[atom_indices]]]
            else:

                elem_colors = np.array(elem_colors)
                atm_colors = elem_colors[atom_to_elem[atom_indices]]
        elif type(atom_colors[0]) == int:
            atom_colors = np.array(atom_colors)
            colors = np.array(
                ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
                 '#17becf', '#564580', '#B87333', '#63c433', '#db61ed', '#298ec4', '#db61ed'])
            atm_colors = colors[atom_colors[atom_indices]]
        else:
            atom_colors = np.array(atom_colors)
            atm_colors = atom_colors[atom_indices]
        atom_rad = []
        for elem in atom_elems:
            if elem in all_atom_rad.keys():
                atom_rad.append(all_atom_rad[elem])
            else:
                atom_rad.append(1.3)
        # colors = ["blue","green","orange","purple","red",
        def cylinder_mesh(point1, point2, radius, resolution=15):
            """
            Create the coordinates of a cylinder mesh between two points.
            Args:
                point1 (array-like): [x, y, z] coordinates of one end of the cylinder.
                point2 (array-like): [x, y, z] coordinates of the other end of the cylinder.
                radius (float): Radius of the cylinder.
                resolution (int): Number of segments along the circle.

            Returns:
                (x, y, z): Mesh coordinates for the cylinder.
            """
            # Vector along the cylinder's axis
            v = np.array(point2) - np.array(point1)
            v = v / np.linalg.norm(v)  # Normalize the vector

            # Create a perpendicular vector
            if abs(v[0]) < 1e-8 and abs(v[1]) < 1e-8:
                # v is approximately along z-axis
                not_v = np.array([1, 0, 0])
            else:
                not_v = np.array([0, 0, 1])
            n1 = np.cross(v, not_v)
            n1 /= np.linalg.norm(n1)
            n2 = np.cross(v, n1)

            # Generate points around a circle at the base of the cylinder
            t = np.linspace(0, 2 * np.pi, resolution)
            circle = np.array([radius * np.cos(t), radius * np.sin(t)])  # Circle coordinates

            # Mesh points along the cylinder's axis
            X, Y, Z = [], [], []
            for i in [0, 1]:  # Iterate over start and end of the cylinder
                point = point1 if i == 0 else point2
                x = point[0] + circle[0] * n1[0] + circle[1] * n2[0]
                y = point[1] + circle[0] * n1[1] + circle[1] * n2[1]
                z = point[2] + circle[0] * n1[2] + circle[1] * n2[2]
                X.append(x), Y.append(y), Z.append(z)

            return np.concatenate(X), np.concatenate(Y), np.concatenate(Z)

        def midsection_hover_points(point1, point2, radius, resolution=20):
            """
            Generate hover points along the midsection of the cylinder.
            Args:
                point1, point2: Start and end points of the cylinder.
                radius: Radius of the cylinder.
                resolution: Number of points along the circle.

            Returns:
                (x, y, z, text): Coordinates and hover labels for the midsection points.
            """
            # Calculate the midpoint of the cylinder
            midpoint = (np.array(point1) + np.array(point2)) / 2

            # Vector along the cylinder's axis
            v = np.array(point2) - np.array(point1)
            v = v / np.linalg.norm(v)  # Normalize the vector

            # Create two perpendicular vectors
            if abs(v[0]) < 1e-8 and abs(v[1]) < 1e-8:
                # v is approximately along z-axis
                not_v = np.array([1, 0, 0])
            else:
                not_v = np.array([0, 0, 1])
            n1 = np.cross(v, not_v) / np.linalg.norm(np.cross(v, not_v))
            n2 = np.cross(v, n1)

            # Generate points around the circle at the midpoint
            t = np.linspace(0, 2 * np.pi, resolution)
            x = midpoint[0] + radius * np.cos(t) * n1[0] + radius * np.sin(t) * n2[0]
            y = midpoint[1] + radius * np.cos(t) * n1[1] + radius * np.sin(t) * n2[1]
            z = midpoint[2] + radius * np.cos(t) * n1[2] + radius * np.sin(t) * n2[2]

            return x, y, z

        # plot the bond lines
        # Create connecting lines trace
        data = []
        all_start_atms = np.array([[10,10,10]])
        all_end_atms = np.array([[10,10,10]])
        all_centers = np.array([[10,10,10]])
        all_text = []
        all_customdata = []
        all_colors = []
        all_energies = []
        all_styles = []
        all_atmrads = []
        all_atm_primind = []
        all_atm_elems = []
        all_atm_labels = []
        atom_bond_info = []
        atom_bond_vecs = []
        for atm in range(self.numAtoms):
            atom_bond_info.append(np.array([['','','','','']]))
            atom_bond_vecs.append(np.array([[0,0,0]]))
        for b_ind in range(num_bonds):
            bond_energy = bonds[b_ind]
            if bond_energy < 0:
                style = "solid"
            else:
                style = "dash"
            if abs(bond_energy) > energy_cutoff:
                start = start_ind[b_ind]
                end = end_ind[b_ind]
                ogstart_atm = np.array([x_data[start], y_data[start], z_data[start]])
                ogend_atm = np.array([x_data[end], y_data[end], z_data[end]])
                # place atoms also on 1 when start atm is at 0
                trans = np.array([[0, 0, 0]])
                hold_atm = np.around(abc_data[:, start], decimals=3)  # start atm in primitive coordinates
                abc = np.linalg.norm(self._a, axis=1)
                # print(hold_atm,hold_atm==0)
                if (np.abs(hold_atm) <= 0.01).any() or (np.abs(hold_atm-1) <= 0.01).any():
                    for vectoswitch in np.arange(3)[np.abs(hold_atm) <= 0.01]:  # (hold_atm==0).any():
                        if abc[vectoswitch] < 4 * (
                                np.sum(abc) - abc[vectoswitch]) / 2:  # aviod 2D cases with lots of vacuum
                            # vectoswitch = np.arange(3)[hold_atm==0][0]
                            switch = np.zeros(3)
                            switch[vectoswitch] = 1
                            new_trans = trans[:, :] + switch[None, :]
                            trans = np.append(trans, new_trans, axis=0)
                    for vectoswitch in np.arange(3)[np.abs(hold_atm-1) <= 0.01]:  # (hold_atm==0).any():
                        if abc[vectoswitch] < 4 * (
                                np.sum(abc) - abc[vectoswitch]) / 2:  # aviod 2D cases with lots of vacuum
                            # vectoswitch = np.arange(3)[hold_atm==0][0]
                            switch = np.zeros(3)
                            switch[vectoswitch] = -1
                            new_trans = trans[:, :] + switch[None, :]
                            trans = np.append(trans, new_trans, axis=0)
                cart_trans = _red_to_cart((self._a[0], self._a[1], self._a[2]), trans)
                # print(cart_trans)
                for ti, tran in enumerate(cart_trans):
                    start_atm = ogstart_atm + tran
                    end_atm = ogend_atm + tran
                    bond_dist = np.linalg.norm(end_atm - start_atm)
                    prim_startatm = abc_data[:, start] + trans[ti]
                    plot = True
                    if one_atom == True:
                        if (np.abs((prim_startatm - self.primAtoms[plot_atom])) > 0.01).any():
                            plot = False
                    if plot == True: # only plot when one of the atoms is in the primitive cell

                        all_start_atms = np.append(all_start_atms,np.array([start_atm]),axis=0)
                        all_end_atms = np.append(all_end_atms,np.array([end_atm]),axis=0)
                        # print("got new trans:",trans)
                        NN = np.arange(len(self.NN_dists))[
                            np.argmin(np.abs(self.NN_dists - bond_dist))]  # <= 0.0001][0]
                        centerx = (start_atm[0] + end_atm[0]) / 2
                        centery = (start_atm[1] + end_atm[1]) / 2
                        centerz = (start_atm[2] + end_atm[2]) / 2
                        orbs = ["s", "p", "d", "f"]
                        bond_label = ""
                        for orb1 in orbs:
                            for orb2 in orbs:
                                bond_eng = all_bonds[orb1 + orb2][b_ind]
                                if np.abs(bond_eng) > np.abs(bond_energy) / 10:
                                    bond_str = atom_elems[start] + " " + orb1 + " - " + atom_elems[
                                        end] + " " + orb2 + ": " + str(
                                        np.around(bond_eng, decimals=3)) + " eV" + "<br>"
                                    bond_label += bond_str
                        spin_label = ""
                        if self.spin_polar:
                            spin_label += "<br>Energy from spin up: " + str(
                                np.around(bonds_spin[0][b_ind], decimals=3)) + " eV"
                            spin_label += "<br>Energy from spin down: " + str(
                                np.around(bonds_spin[1][b_ind], decimals=3)) + " eV"
                        full_text = ("Bond energy: " + str(
                            np.around(bond_energy, decimals=3)) + " eV" + "<br>Bond length: "
                                     + str(np.around(bond_dist, decimals=3)) + " Å" + " (" + str(NN) + "NN)" +
                                     spin_label + "<br>Breakdown into orbitals: <br>" + bond_label)
                        all_atm_elems.append([atom_elems[start],atom_elems[end]])
                        all_atm_labels.append([atom_labels[start],atom_labels[end]])
                        start_rad = atom_rad[start]
                        end_rad = atom_rad[end]
                        all_colors.append([atm_colors[start], atm_colors[end]])
                        all_energies.append(bond_energy)
                        all_styles.append(style)
                        all_atmrads.append([start_rad, end_rad])
                        all_atm_primind.append([atom_indices[start],atom_indices[end]])

                        # Plot the cylinder using Plotly Mesh3d
                        color = 'gray'
                        if do_coop:
                            full_text = full_text + "Bond charge: " + str(np.around(bond_charges[b_ind],decimals=3))
                            if ("color" in auto_label):
                                color = bond_colors[b_ind]
                            if self.spin_polar:
                                full_text = full_text + "<br>" + "Bond magmom: " + str(np.around(bond_magmoms[b_ind],decimals=3))

                        sorted_elems = np.sort([atom_elems[start],atom_elems[end]])
                        bond_label = np.char.add(np.char.add(sorted_elems[0],sorted_elems[1]),str(NN))

                        # now save bond to the first atom to construct an atom center picture of bonding
                        atm_ind = atom_indices[start]
                        atom_bond_vecs[atm_ind] = np.append(atom_bond_vecs[atm_ind],np.array([end_atm-start_atm]), axis=0)
                        atom_bond_info[atm_ind] = np.append(atom_bond_info[atm_ind], np.array([[str(bond_energy), full_text, bond_label, style, color]]),axis=0)  # [bond_vec, bond_energy, full_text, bond_label, style, color]
        #print(all_end_atms.shape,all_start_atms.shape)
        # first append all atoms in the primitive cell to all_start_atms
        abc = np.linalg.norm(self._a, axis=1)
        for atmnum in range(self.numAtoms):
            prim_atm = np.array(self.primAtoms[atmnum])
            trans = np.array([[0, 0, 0]])
            if (np.abs(prim_atm) <= 0.01).any() or (np.abs(prim_atm-1) <= 0.01).any():
                for vectoswitch in np.arange(3)[(np.abs(prim_atm) <= 0.01)]:  # (prim_atm==0).any():
                    if abc[vectoswitch] < 4 * (np.sum(abc) - abc[vectoswitch]) / 2:  # aviod 2D cases with lots of vacuum
                        # vectoswitch = np.arange(3)[prim_atm==0][0]
                        switch = np.zeros(3)
                        switch[vectoswitch] = 1
                        new_trans = trans[:, :] + switch[None, :]
                        trans = np.append(trans, new_trans, axis=0)
                for vectoswitch in np.arange(3)[(np.abs(prim_atm-1) <= 0.01)]:  # (prim_atm==0).any():
                    if abc[vectoswitch] < 4 * (np.sum(abc) - abc[vectoswitch]) / 2:  # aviod 2D cases with lots of vacuum
                        # vectoswitch = np.arange(3)[prim_atm==0][0]
                        switch = np.zeros(3)
                        switch[vectoswitch] = -1
                        new_trans = trans[:, :] + switch[None, :]
                        trans = np.append(trans, new_trans, axis=0)
            cart_trans = _red_to_cart((self._a[0], self._a[1], self._a[2]), trans)
            for t in cart_trans:
                all_start_atms = np.append(all_start_atms, np.array([self.cartAtoms[atmnum]+t]), axis=0)
                all_atm_elems.append([self.elements[atmnum],''])
                all_atm_labels.append([atom_labels[atmnum],''])
                all_colors.append([atm_colors[atmnum], ''])
                all_atmrads.append([atom_rad[atmnum], 0])
                all_atm_primind.append([atmnum,-1])

        # find all unique atoms
        combnd_atms = np.append(all_start_atms[1:],all_end_atms[1:],axis=0)
        all_colors = np.array(all_colors)
        all_atmrads = np.array(all_atmrads)
        all_atm_primind = np.array(all_atm_primind)
        all_atm_elems = np.array(all_atm_elems)
        all_atm_labels = np.array(all_atm_labels)
        combnd_colors = np.append(all_colors[:,0],all_colors[:,1][all_colors[:,1]!=''])
        combnd_atmrad = np.append(all_atmrads[:,0],all_atmrads[:,1][all_atmrads[:,1]!=0])
        combnd_primind = np.append(all_atm_primind[:,0],all_atm_primind[:,1][all_atm_primind[:,1]!=-1])
        combnd_elems = np.append(all_atm_elems[:,0],all_atm_elems[:,1][all_atm_elems[:,1]!=''])
        combnd_labels = np.append(all_atm_labels[:,0],all_atm_labels[:,1][all_atm_labels[:,1]!=''])
        uniq_atms,indices = np.unique(combnd_atms,axis=0,return_index=True)
        uniq_colors = combnd_colors[indices]
        uniq_rad = combnd_atmrad[indices]
        uniq_elems = combnd_elems[indices]
        uniq_labels = combnd_labels[indices]
        uniq_primind = combnd_primind[indices]

        if bond_max: # contributed by Ayesha
            #print("Checking for atoms outside primitive cell with distance:",  bond_max)
            # filter atoms that are outside the primitive cell
            prim_atms = _cart_to_red((self._a[0], self._a[1], self._a[2]), uniq_atms)
            inside_cell = ((prim_atms >= -0.01) & (prim_atms <= 1.01)).all(axis=1)
            indices = np.ones(len(uniq_atms))
            for i, atom in enumerate(uniq_atms):
                if inside_cell[i] == False:
                    distances_to_prim = np.linalg.norm(atom - uniq_atms[inside_cell], axis=1)
                    #print("Distances to primitive cell for atom:", atom, "are", distances_to_prim)
                    #print("Minimum distance to primitive cell:", np.min(distances_to_prim))
                    if np.min(distances_to_prim) >  bond_max:
                        #print("Removing atom outside primitive cell:", atom, "with distance:",np.min(distances_to_prim))
                        indices[i] = 0
            uniq_atms = uniq_atms[indices.astype(bool)]
            uniq_colors = uniq_colors[indices.astype(bool)]
            uniq_rad = uniq_rad[indices.astype(bool)]
            uniq_elems = uniq_elems[indices.astype(bool)]
            uniq_labels = uniq_labels[indices.astype(bool)]
            uniq_primind = uniq_primind[indices.astype(bool)]

        if self.verbose > 1:
            print("all atms:",uniq_atms)
            print(uniq_rad)
            print(uniq_colors)

        def create_sphere(center, radius, resolution=15):
            """
            Generate the mesh coordinates for a sphere.
            """
            u = np.linspace(0, 2 * np.pi, resolution)
            v = np.linspace(0, np.pi, resolution)
            x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
            y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
            z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
            return x.flatten(), y.flatten(), z.flatten()

        # Loop over atoms to create a sphere for each atom
        for ai,atm in enumerate(uniq_atms):
            x, y, z = create_sphere(atm, uniq_rad[ai]**(0.8)/2)
            trace = go.Mesh3d(
                x=x, y=y, z=z,
                color=uniq_colors[ai],
                opacity=1.0,
                lighting=dict(
                    ambient=0.6,  # Ambient light (general light)
                    diffuse=0.7,  # Diffuse light (soft shadows)
                    fresnel=0.1,  # Reflective edges
                    specular=0.9,  # Specular reflection (sharp highlights)
                    roughness=0.55,  # Surface roughness
                ),
                lightposition=dict(
                    x=10, y=10, z=10  # Light source position
                ),
                alphahull=0,  # Proper enclosure for the mesh
                hoverinfo = "text",
                #hoverlabel = dict(bgcolor='ghostwhite',
                #                  font=dict(family='Arial', size=12, color='black')),
                text = uniq_labels[ai], showlegend = False)

            data.append(trace)

        # add all bonds for each atom if both atoms are plotted
        for atm in range(self.numAtoms):
            uniq_bond_vecs,indices = np.unique(atom_bond_vecs[atm][1:],axis=0,return_index=True)
            bond_info = atom_bond_info[atm]
            uniq_bond_info = bond_info[indices+1]
            #print("unique atom bonds:",uniq_bond_vecs,uniq_bond_info)
            atom_bond_vecs[atm] = uniq_bond_vecs
            atom_bond_info[atm] = uniq_bond_info


        all_start_atms = np.array([[10,10,10]])
        all_end_atms = np.array([[10,10,10]])
        all_centers = np.array([[10,10,10]])
        for ai,atm in enumerate(uniq_atms):
            vec_to_atm = uniq_atms - atm[None,:]
            bond_vecs = atom_bond_vecs[uniq_primind[ai]]
            bond_info = atom_bond_info[uniq_primind[ai]]
            if len(bond_vecs) != 0:
                # Create a (len(arr1), len(arr2), 3) array of differences
                matches = (np.around(vec_to_atm[:, None, :],decimals=2) == np.around(bond_vecs[None, :, :],decimals=2)).all(axis=2)
                #matches_opposite = (np.around(vec_to_atm[:, None, :],decimals=2) == np.around(-bond_vecs[None, :, :],decimals=2)).all(axis=2)
                # Combine both conditions
                #matches = matches_direct | matches_opposite
                # matches[i, j] is True if arr1[i] == arr2[j]
                #print(matches)
                uniq_idx, bond_idx = np.where(matches)
                #print(bond_info[bond_idx[0]],uniq_idx,bond_idx)

                for bi in range(len(uniq_idx)):
                    start_atm = atm
                    end_atm = uniq_atms[uniq_idx[bi]]
                    [bond_energy, full_text, bond_label, style, color] = bond_info[bond_idx[bi]]
                    bond_energy = float(bond_energy)

                    # first check if the bond is already created
                    is_new_bond = True
                    # check whether another bond already exists with the same end points
                    same_start = (all_start_atms[:, 0] == start_atm[0]) & (all_start_atms[:, 1] == start_atm[1]) & (
                                all_start_atms[:, 2] == start_atm[2])
                    check_end_atm = all_end_atms[same_start]
                    same_end = ((check_end_atm[:, 0] == end_atm[0]) & (check_end_atm[:, 1] == end_atm[1]) & (
                                check_end_atm[:, 2] == end_atm[2])).any()
                    if same_end == False:  # also check for opposite
                        same_start = (all_start_atms[:, 0] == end_atm[0]) & (
                                all_start_atms[:, 1] == end_atm[1]) & (
                                             all_start_atms[:, 2] == end_atm[2])
                        check_end_atm = all_end_atms[same_start]
                        same_end = ((check_end_atm[:, 0] == start_atm[0]) & (
                                check_end_atm[:, 1] == start_atm[1]) & (
                                            check_end_atm[:, 2] == start_atm[2])).any()
                    if same_end:
                        if self.verbose > 1:
                            print("avoided duplicate!")
                    is_new_bond = not same_end

                    if is_new_bond == True:
                        all_start_atms = np.append(all_start_atms, np.array([start_atm]), axis=0)
                        all_end_atms = np.append(all_end_atms, np.array([end_atm]), axis=0)
                        if style == 'dash':
                            centerx = (start_atm[0] + end_atm[0]) / 2
                            centery = (start_atm[1] + end_atm[1]) / 2
                            centerz = (start_atm[2] + end_atm[2]) / 2
                            closer_atom1 = (start_atm * 5 + end_atm) / 6
                            closer_atom2 = (start_atm + end_atm * 5) / 6
                            all_centers = np.append(all_centers, np.array([[centerx, centery, centerz], closer_atom1, closer_atom2]),
                                                    axis=0)
                            all_text.append(full_text)
                            all_text.append(full_text)
                            all_text.append(full_text)
                            all_customdata.append(bond_label)  # np.around(bond_energy,decimals=1))
                            all_customdata.append(bond_label)
                            all_customdata.append(bond_label)
                            data.append(go.Scatter3d(
                                x=[start_atm[0], end_atm[0]],
                                y=[start_atm[1], end_atm[1]],
                                z=[start_atm[2], end_atm[2]],
                                mode='lines',  # 'markers+lines',
                                # marker=dict(size=[start_rad * 40, end_rad * 40],
                                #            color=[atm_colors[start], atm_colors[end]], opacity=1.0,
                                #            line=dict(color='black', width=20)),
                                line=dict(
                                    color=color,
                                    width=abs(bond_energy/2) ** (1 / 2) * 20,
                                    dash=style
                                ),
                                # hoverinfo="text",
                                # hoverlabel=dict(bgcolor='ghostwhite',
                                #                font=dict(family='Arial', size=12, color='black')),
                                # text=full_text,
                                hoverinfo='none',
                                showlegend=False))
                        else:
                            # Get the mesh coordinates for the cylinder
                            radius = abs(bond_energy/2) ** (1 / 2) / 8
                            x, y, z = cylinder_mesh(start_atm, end_atm, radius, resolution=15)

                            data.append(go.Mesh3d(
                                x=x, y=y, z=z,
                                opacity=1.0,
                                color=color,
                                flatshading=False,  # Smooth shading
                                lighting=dict(
                                    ambient=0.6,  # Ambient light (general light)
                                    diffuse=0.7,  # Diffuse light (soft shadows)
                                    fresnel=0.1,  # Reflective edges
                                    specular=0.9,  # Specular reflection (sharp highlights)
                                    roughness=0.55,  # Surface roughness
                                ),
                                lightposition=dict(
                                    x=10, y=10, z=10  # Light source position
                                ),
                                hoverinfo="none",
                                # hoverlabel=dict(bgcolor='ghostwhite',
                                #                font=dict(family='Arial', size=12, color='black')),
                                # text=full_text,
                                alphahull=0
                            ))

                            # plot center cross-section for hover labels
                            # Get hover points along the midsection circle
                            hover_x, hover_y, hover_z = midsection_hover_points(start_atm, end_atm, radius)
                            # Create hover points using Scatter3d
                            data.append(go.Scatter3d(
                                x=hover_x, y=hover_y, z=hover_z,
                                mode='markers',
                                marker=dict(size=20, color='rgba(0,0,0,0)', opacity=0.),
                                hoverinfo="text",
                                hoverlabel=dict(bgcolor='ghostwhite',
                                                font=dict(family='Arial', size=12, color='black')),
                                text=full_text, showlegend=False, customdata=[bond_label] * len(hover_x)
                                # np.around(bond_energy,decimals=1)
                            ))

        # append the antibond labels
        marker_dict = dict(size=20, color='rgba(0,0,0,0)', opacity=0.)
        if "mag" in auto_label: # include stuff just to plot the colorbar for magnetism
            marker_dict = dict(size=20, color='rgba(0,0,0,0)', opacity=0.,colorscale='RdBu',cmin=vmin,cmax=vmax,
                                    colorbar=dict(title="Magnetic moment",#'log(Magnetization+1)',
                                            #tickvals=np.linspace(vmin, vmax, 5),  # Set ticks for the log scale
                                            #ticktext=[f"{np.around(np.sign(tick)*(10**np.abs(tick)-1),decimals=2)}" for tick in np.linspace(vmin, vmax, 5)]  # Custom tick labels
                                        ))
        elif "color" in auto_label:
            marker_dict = dict(size=20, color='rgba(0,0,0,0)', opacity=0.,colorscale='RdBu',cmin=vmin,cmax=vmax,
                                    colorbar=dict(title='Charge',
                                            #tickvals=np.linspace(vmin, vmax, 5),  # Set ticks for the log scale
                                            #ticktext=[f"{np.around(np.sign(tick)*(10**np.abs(tick)-1),decimals=2)}" for tick in np.linspace(vmin, vmax, 5)]  # Custom tick labels
                                        ))
        # add hover points for antibonds
        all_centers = all_centers[1:]
        data.append(go.Scatter3d(
            x=all_centers[:,0],
            y=all_centers[:,1],
            z=all_centers[:,2],
            mode='markers',
            marker=marker_dict,
            hoverinfo="text",
            hoverlabel=dict(bgcolor='ghostwhite',
                            font=dict(family='Arial', size=12, color='black')),
            text=all_text, showlegend=False,customdata=all_customdata))

        if one_atom == False:
            # draw primitive cell # place atoms also on 1 when start atm is at 0
            trans = np.array([[0, 0, 0]])
            for vectoswitch in np.arange(3):  # (hold_atm==0).any():
                if abc[vectoswitch] < 4 * (np.sum(abc) - abc[vectoswitch]) / 2:  # aviod 2D cases with lots of vacuum
                    # vectoswitch = np.arange(3)[hold_atm==0][0]
                    switch = np.zeros(3)
                    switch[vectoswitch] = 1
                    new_trans = trans[:, :] + switch[None, :]
                    trans = np.append(trans, new_trans, axis=0)
            num = len(trans)
            start_points = trans[:, None, :] * np.ones((num, num, 3))
            end_points = trans[None, :, :] * np.ones((num, num, 3))
            all_vecs = end_points - start_points
            norm = np.linalg.norm(all_vecs, axis=2)
            start_points = np.reshape(start_points, (num ** 2, 3))[norm.flatten() == 1]
            end_points = np.reshape(end_points, (num ** 2, 3))[norm.flatten() == 1]
            final_ind = []
            for veci in range(len(start_points)):
                diff = end_points[veci] - start_points[veci]
                if (diff >= 0).all():
                    final_ind.append(veci)
            final_ind = np.array(final_ind)

            start_points = start_points[final_ind]
            end_points = end_points[final_ind]
            cart_start = _red_to_cart((self._a[0], self._a[1], self._a[2]), start_points)
            cart_end = _red_to_cart((self._a[0], self._a[1], self._a[2]), end_points)

            for i in range(len(start_points)):
                start_atm = cart_start[i]
                end_atm = cart_end[i]
                data.append(go.Scatter3d(
                    x=[start_atm[0], end_atm[0]],
                    y=[start_atm[1], end_atm[1]],
                    z=[start_atm[2], end_atm[2]],
                    mode='lines',
                    line=dict(
                        color='black',
                        width=2,
                        dash="solid"
                    ), hoverinfo='none', showlegend=False))


        # Create figure
        fig = go.Figure(data=data)

        # update figure design
        # fig.update_traces(hovertemplate=None)
        fig.update_layout(hoverdistance=100, margin=dict(l=0, r=0, b=0, t=0),
                          paper_bgcolor='rgba(0,0,0,0)',  # Transparent background for the whole figure
                          plot_bgcolor='rgba(0,0,0,0)')  # ,hovermode="x unified")
        fig.update_layout(
            scene=dict(xaxis=dict(showbackground=False, showspikes=False, showticklabels=False, visible=False),
                       yaxis=dict(showbackground=False, showspikes=False, showticklabels=False, visible=False),
                       zaxis=dict(showbackground=False, showspikes=False, showticklabels=False, visible=False),
                       aspectmode='data'))

        #fig.layout.scene.camera.projection.type = "orthographic"
        # Set layout with camera and FOV (fovy)
        fig.update_layout(scene_camera=dict(eye=dict(x=1.25,y=-0.05,z=0.05)))

        fov = fovy
        dist = np.around(75/fov,decimals=2)
        post_script = ("const plot = document.querySelector('.js-plotly-plot'); const scene = plot._fullLayout.scene._scene;"
                   " const camera = scene.glplot; camera.fovy ="+str(fov)+" * Math.PI / 180;  scene.camera.distance ="+str(dist)+";  "
                       "console.log(`Updated FOV to ${camera.fovy} radians.`);")
        if return_fig:
            return fig, post_script

        else:
            if one_atom:
                fig.write_html(self.directory + "crystal_bonds"+str(plot_atom)+".html",post_script=post_script,div_id = "plotDiv")
            else:
                fig.write_html(self.directory + "crystal_bonds.html", post_script=post_script, div_id="plotDiv")

            if self.show_figs:
                fig.show(post_script=post_script,div_id = "plotDiv")


    def make_bond(self,atmind1,atmind2,center1,center2,orbCOOP,cartXYZ):
        """
        This is a function which will generate populate the cartXYZ grid with values for the bond density between
        the atoms given using the orbCOOP provided.

        Args:    
            atmind1 (unknown): The atom number for the first atom.
            atmind2 (unknown): The atom number for the second atom.
            center1 (unknown): The center of the first atom (not using self.primATOMs).
            center2 (unknown): The center of the second atom (not using self.primATOMs).
            orbCOOP (unknown): The orbCOOP which reveals how much of each orbital combo that is included in the bond.
                        Dimension nxm where n is the # of orbitals for atom 1 and m is # of orbitals for atom 2.
            cartXYZ (unknown): The 3D flattened grid that the bond density is calculated on.

        Returns:    
            unknown: A 1D array  (3D flattened) of the bond density.
        """

        if not hasattr(self, 'real_orbcoeffs'):
            self.read_orbitals()

        #print("orbital radii:",self.orb_radii)
        cartXYZ = np.array(cartXYZ)

        # calculate the orbitals for the first atom
        orbinds1 = np.arange(self.num_orbs)[np.array(self.orbatomnum) == atmind1]
        orb_XYZ = cartXYZ - np.array(center1)[:,None]
        # try out the recip orbital coeffs
        rad = np.linalg.norm(orb_XYZ, axis=0)
        rad[rad == 0] = 0.0000000001
        theta = np.arccos(orb_XYZ[2] / rad)
        theta[rad == 0.0000000001] = 0
        phi = np.arctan2(orb_XYZ[1], orb_XYZ[0])

        orbitals1 = np.zeros((len(orbinds1), len(rad)))#, dtype=np.complex128)  # {}
        for ind, orb in enumerate(orbinds1):
            real_coeff = self.real_orbcoeffs[orb]
            [a, b, c, d, e, f, g, h, l] = real_coeff
            orb_peratom = self.sph_harm_key[orb]
            #print(orb_peratom)
            angular_part = complex128funs(phi, theta, self.orb_vec[orb])
            radial_part = func_for_rad(rad, a, b, c, d, e, f, g, h, l)
            flat_real_orb = radial_part * angular_part
            #print("orb min max:",np.amin(flat_real_orb),np.amax(flat_real_orb))
            # structure constant
            #const = ((2 * np.pi) ** 3 / self.vol) ** (1 / 2)
            orbitals1[ind] = flat_real_orb #* const

        # calculate the orbitals for the second atom
        orbinds2 = np.arange(self.num_orbs)[np.array(self.orbatomnum) == atmind2]
        orb_XYZ = cartXYZ - np.array(center2)[:,None]
        # try out the recip orbital coeffs
        rad = np.linalg.norm(orb_XYZ, axis=0)
        rad[rad == 0] = 0.0000000001
        theta = np.arccos(orb_XYZ[2] / rad)
        theta[rad == 0.0000000001] = 0
        phi = np.arctan2(orb_XYZ[1], orb_XYZ[0])

        orbitals2 = np.zeros((len(orbinds2), len(rad)))#, dtype=np.complex128)  # {}
        for ind, orb in enumerate(orbinds2):
            real_coeff = self.real_orbcoeffs[orb]
            [a, b, c, d, e, f, g, h, l] = real_coeff
            orb_peratom = self.sph_harm_key[orb]
            #print(orb_peratom)
            angular_part = complex128funs(phi, theta, self.orb_vec[orb])
            radial_part = func_for_rad(rad, a, b, c, d, e, f, g, h, l)
            flat_real_orb = radial_part * angular_part
            #print("orb min max:",np.amin(flat_real_orb),np.amax(flat_real_orb))
            # structure constant
            #const = ((2 * np.pi) ** 3 / self.vol) ** (1 / 2)
            orbitals2[ind] = flat_real_orb #* const

        bond_density = np.zeros(len(rad))
        for orb1 in range(len(orbinds1)):
            for orb2 in range(len(orbinds2)):
                if orbCOOP[orb1,orb2] != 0:
                    #print("bond min max:",orb1, orb2, np.amin(orbitals1[orb1] * orbitals2[orb2]),np.amax(orbitals1[orb1] * orbitals2[orb2]))
                    bond_density = bond_density + orbitals1[orb1] * orbitals2[orb2] * orbCOOP[orb1,orb2]
                    #print("bond min max:",np.amin(bond_density),np.amax(bond_density))

        return bond_density

    def get_bond_density_figure(self, energy_cutoff: float = 0.1, iso_max: float = 0.03, iso_min: float = 0.003, elem_colors: list = [], atom_colors: list = [], atom_labels: list = [],
                                auto_label: str = "", plot_atom: int = None,
                         one_atom: bool = False, bond_max: float = 3.0, fovy: float = 10,return_fig: bool = False) -> None:
        """
        Plots the crystal structure atoms with bonds plotted based on iCOHP.
        Each line should also be hoverable to reveal the number and amounts that are s-s,s-p, and p-p.
        The charge and magnetic moment will also be plotted acording to auto_label.

        Args:    
            energy_cutoff (float): This is the minimum bond magnitude that will be plotted.
            iso_max (float): The positive isosurface for plotting the bonds.
            iso_min (float): The negative isosurface for plotting the bonds.
            elem_colors (list): Colors for the elements based on order in tb_input. Length of list should be the number of
                        unique elements. Can either be integer list to reference the default colors or list of
                        plotly compatable colors.
            atom_colors (list): Colors for the atoms based on order in tb_input. Length of list should be the number of
                        atoms in the primitive cell. Can either be integer list to reference the default colors or
                        list of plotly compatable colors. If not set defaults to elem_colors.
            atom_labels (list): List of atom labels as a string.
            auto_label (str): Different options for plotting includes: (can include multiple in the string)
                        "mulliken" - Plots the onsite charge and mag (if spin_polar) on atoms by mulliken population (overrides atom_labels).
                        "full" - Plots the charge and magnetics moments (if spin_polar) on atoms and bonds (overrides atom_labels).
                        "color" - Colors the atoms and bonds based on their charge (overrides atom_colors or elem_colors).
                        "color mag" - Colors the atoms and bonds based on their magnetic moments (overrides atom_colors or elem_colors).
                        NOTE: Only use "mulliken" OR "full", NOT both.
            plot_atom (int): Set with one_atom=True, plots only one atom and it's bonds, this passes the atom number to plot.
            one_atom (bool): Whether only the atom defined in plot_atom should be plotted; default is False.
            bond_max (float): The maximum bond distance that will be plotted outside the primitive cell.
            fovy (float): Field of view in the vertical direction. Use this tag to adjust depth perception in crystal.
                        Set between 3 (for close to orthographic) and 30 (for good perspective depth).

        Returns:    
            None: Nothing, but saves plotly figure to 'crystal_bonds.html'.
        """
        import plotly.graph_objs as go
        from skimage.measure import marching_cubes
        if not hasattr(self, 'NN_index'):
            self.get_neighbors()
        if not hasattr(self, 'real_orbcoeffs'):
            self.read_orbitals()
        if not hasattr(self, 'small_overlap'):
            self.read_overlaps()
        # atom size based on element
        all_atom_rad = {"H": 0.53, "He": 0.31, "Li": 1.51, "Be": 1.12, "B": 0.87, "C": 0.67, "N": 0.56, "O": 0.48,
                        "F": 0.42, "Ne": 0.38,
                        "Na": 1.90, "Mg": 1.45, "Al": 1.18, "Si": 1.11, "P": 0.98, "S": 0.88, "Cl": 0.79, "Ar": 0.71,
                        "K": 2.43, "Ca": 1.94, "Sc": 1.84, "Ti": 1.76, "V": 1.71, "Cr": 1.66, "Mn": 1.61, "Fe": 1.56,
                        "Co": 1.52,
                        "Ni": 1.49, "Cu": 1.45, "Zn": 1.42, "Ga": 1.36, "Ge": 1.25, "As": 1.14, "Se": 1.03, "Br": 0.94,
                        "Kr": 0.88,
                        "Rb": 2.65, "Sr": 2.19, "Y": 2.12, "Zr": 2.06, "Nb": 1.98, "Mo": 1.90, "Tc": 1.83, "Ru": 1.78,
                        "Rh": 1.73,
                        "Pd": 1.69, "Ag": 1.65, "Cd": 1.61, "In": 1.56, "Sn": 1.45, "Sb": 1.33, "Te": 1.23, "I": 1.15,
                        "Xe": 1.08,
                        "Cs": 2.98, "Ba": 2.53, "Lu": 2.17, "Hf": 2.08, "Ta": 2.00, "W": 1.93, "Re": 1.88, "Os": 1.85,
                        "Ir": 1.80,
                        "Pt": 1.77, "Au": 1.75, "Hg": 1.71, "Tl": 1.56, "Pb": 1.54, "Bi": 1.43, "Po": 1.35, "At": 1.27,
                        "Rn": 1.20}
        do_coop = False
        if "full" in auto_label:
            do_coop = True
        # get the cohp
        #if not hasattr(self, 'ICOHP'):
        #    self.get_ICOHP()
        if do_coop:
            #if not hasattr(self, 'ICOOP'):
            #    self.get_ICOOP()
            icoop_orb = self.ICOOP
        icohp_orb = self.ICOHP
        # each_dir = np.array([1, 1, 1]) #self.num_each_dir
        cell = self.num_trans  # np.array(self.num_trans) # each_dir * 2 + 1
        vec_to_trans = np.zeros((cell[0], cell[1], cell[2], 3))  # self.vec_to_trans
        each_dir = self.num_each_dir

        for x in range(cell[0]):
            for y in range(cell[1]):
                for z in range(cell[2]):
                    vec_to_trans[x, y, z] = [x - each_dir[0], y - each_dir[1], z - each_dir[2]]

        num_total_atoms = self.numAtoms * cell[0] * cell[1] * cell[2]
        vec_to_atoms = vec_to_trans[None, :, :, :] + np.array(self.primAtoms)[:, None, None, None]
        vec_to_atoms = np.reshape(vec_to_atoms, (num_total_atoms, 3)).T
        # print(vec_to_atoms)
        cart_to_atoms = _red_to_cart((self._a[0], self._a[1], self._a[2]), vec_to_atoms.transpose()).T

        # plot the atoms onto the crystal
        x_data = cart_to_atoms[0]
        y_data = cart_to_atoms[1]
        z_data = cart_to_atoms[2]
        abc_data = vec_to_atoms
        # crystal = go.Scatter3d(x=x_data,y=y_data,z=z_data,mode='markers',marker=dict(size=20,color='blue',opacity=0.8),hoverinfo='none')
        if self.spin_polar:
            keys = [0, 1]
        else:
            keys = [0]

        all_bonds_spin = {}
        bonds_spin = {}
        charge_spin = {}
        for key in keys:
            icohp_orb[key] = icohp_orb[key].real  # self.TB_model.get_ICOHP(self,spin=key).real
            # [s,p]*[s,p] = [s-s,s-p,p-s,p-p]
            # [s,p] * [s,p,d] = [s-s,s-p,p-s,p-p,s-d,p-d]
            icohp_atom = np.zeros((self.numAtoms, self.numAtoms, cell[0], cell[1], cell[2]))
            if do_coop:
                icoop_orb[key] = icoop_orb[key].real
                icoop_atom = np.zeros((self.numAtoms, self.numAtoms, cell[0], cell[1], cell[2]))
            ssbonds = np.zeros((self.numAtoms, self.numAtoms, cell[0], cell[1], cell[2]))
            spbonds = np.zeros((self.numAtoms, self.numAtoms, cell[0], cell[1], cell[2]))
            psbonds = np.zeros((self.numAtoms, self.numAtoms, cell[0], cell[1], cell[2]))
            ppbonds = np.zeros((self.numAtoms, self.numAtoms, cell[0], cell[1], cell[2]))
            sdbonds = np.zeros((self.numAtoms, self.numAtoms, cell[0], cell[1], cell[2]))
            dsbonds = np.zeros((self.numAtoms, self.numAtoms, cell[0], cell[1], cell[2]))
            pdbonds = np.zeros((self.numAtoms, self.numAtoms, cell[0], cell[1], cell[2]))
            dpbonds = np.zeros((self.numAtoms, self.numAtoms, cell[0], cell[1], cell[2]))
            ddbonds = np.zeros((self.numAtoms, self.numAtoms, cell[0], cell[1], cell[2]))
            # add f orbs
            sfbonds = np.zeros((self.numAtoms, self.numAtoms, cell[0], cell[1], cell[2]))
            fsbonds = np.zeros((self.numAtoms, self.numAtoms, cell[0], cell[1], cell[2]))
            pfbonds = np.zeros((self.numAtoms, self.numAtoms, cell[0], cell[1], cell[2]))
            fpbonds = np.zeros((self.numAtoms, self.numAtoms, cell[0], cell[1], cell[2]))
            dfbonds = np.zeros((self.numAtoms, self.numAtoms, cell[0], cell[1], cell[2]))
            fdbonds = np.zeros((self.numAtoms, self.numAtoms, cell[0], cell[1], cell[2]))
            ffbonds = np.zeros((self.numAtoms, self.numAtoms, cell[0], cell[1], cell[2]))
            for atm1 in range(self.numAtoms):
                for atm2 in range(self.numAtoms):
                    orbs1 = np.arange(self.num_orbs)[self.orbatomnum == atm1]
                    orbs2 = np.arange(self.num_orbs)[self.orbatomnum == atm2]
                    sum_cohp = icohp_orb[key][orbs1, :][:, orbs2]
                    if do_coop:
                        sum_coop = icoop_orb[key][orbs1, :][:, orbs2]
                    orb_type1 = self.exactorbtype[orbs1]
                    orb_type2 = self.exactorbtype[orbs2]
                    # s-s
                    is_s1 = []
                    is_p1 = []
                    is_d1 = []
                    is_f1 = []
                    for i in orb_type1:
                        is_s1.append("s" in i)
                        is_p1.append("p" in i)
                        is_d1.append("d" in i)
                        is_f1.append("f" in i)
                    is_s2 = []
                    is_p2 = []
                    is_d2 = []
                    is_f2 = []
                    for i in orb_type2:
                        is_s2.append("s" in i)
                        is_p2.append("p" in i)
                        is_d2.append("d" in i)
                        is_f2.append("f" in i)
                    sorbs1 = np.arange(len(orb_type1))[is_s1]
                    porbs1 = np.arange(len(orb_type1))[is_p1]
                    dorbs1 = np.arange(len(orb_type1))[is_d1]
                    forbs1 = np.arange(len(orb_type1))[is_f1]
                    sorbs2 = np.arange(len(orb_type2))[is_s2]
                    porbs2 = np.arange(len(orb_type2))[is_p2]
                    dorbs2 = np.arange(len(orb_type2))[is_d2]
                    forbs2 = np.arange(len(orb_type2))[is_f2]

                    ssbond = np.sum(sum_cohp[sorbs1, :][:, sorbs2], axis=(0, 1))
                    spbond = np.sum(sum_cohp[sorbs1, :][:, porbs2], axis=(0, 1))
                    psbond = np.sum(sum_cohp[porbs1, :][:, sorbs2], axis=(0, 1))
                    ppbond = np.sum(sum_cohp[porbs1, :][:, porbs2], axis=(0, 1))
                    sdbond = np.sum(sum_cohp[sorbs1, :][:, dorbs2], axis=(0, 1))
                    dsbond = np.sum(sum_cohp[dorbs1, :][:, sorbs2], axis=(0, 1))
                    pdbond = np.sum(sum_cohp[porbs1, :][:, dorbs2], axis=(0, 1))
                    dpbond = np.sum(sum_cohp[dorbs1, :][:, porbs2], axis=(0, 1))
                    ddbond = np.sum(sum_cohp[dorbs1, :][:, dorbs2], axis=(0, 1))
                    # add f orbs
                    sfbond = np.sum(sum_cohp[sorbs1, :][:, forbs2], axis=(0, 1))
                    fsbond = np.sum(sum_cohp[forbs1, :][:, sorbs2], axis=(0, 1))
                    pfbond = np.sum(sum_cohp[porbs1, :][:, forbs2], axis=(0, 1))
                    fpbond = np.sum(sum_cohp[forbs1, :][:, porbs2], axis=(0, 1))
                    dfbond = np.sum(sum_cohp[dorbs1, :][:, forbs2], axis=(0, 1))
                    fdbond = np.sum(sum_cohp[forbs1, :][:, dorbs2], axis=(0, 1))
                    ffbond = np.sum(sum_cohp[forbs1, :][:, forbs2], axis=(0, 1))

                    icohp_atom[atm1, atm2] = np.sum(sum_cohp, axis=(0, 1))
                    if do_coop:
                        icoop_atom[atm1, atm2] = np.sum(sum_coop, axis=(0, 1))
                    ssbonds[atm1, atm2] = ssbond
                    spbonds[atm1, atm2] = spbond
                    psbonds[atm1, atm2] = psbond
                    ppbonds[atm1, atm2] = ppbond
                    sdbonds[atm1, atm2] = sdbond
                    dsbonds[atm1, atm2] = dsbond
                    pdbonds[atm1, atm2] = pdbond
                    dpbonds[atm1, atm2] = dpbond
                    ddbonds[atm1, atm2] = ddbond
                    # add f orbs
                    sfbonds[atm1, atm2] = sfbond
                    fsbonds[atm1, atm2] = fsbond
                    pfbonds[atm1, atm2] = pfbond
                    fpbonds[atm1, atm2] = fpbond
                    dfbonds[atm1, atm2] = dfbond
                    fdbonds[atm1, atm2] = fdbond
                    ffbonds[atm1, atm2] = ffbond
            # print(icohp_atom.shape)
            # print("cohp!",icohp_atom)
            bonds_spin[key] = icohp_atom.flatten()
            if do_coop:
                charge_spin[key] = icoop_atom.flatten()
            ssbonds = ssbonds.flatten()
            spbonds = spbonds.flatten()
            psbonds = psbonds.flatten()
            ppbonds = ppbonds.flatten()
            sdbonds = sdbonds.flatten()
            dsbonds = dsbonds.flatten()
            pdbonds = pdbonds.flatten()
            dpbonds = dpbonds.flatten()
            ddbonds = ddbonds.flatten()
            # add f orbs
            sfbonds = sfbonds.flatten()
            fsbonds = fsbonds.flatten()
            pfbonds = pfbonds.flatten()
            fpbonds = fpbonds.flatten()
            dfbonds = dfbonds.flatten()
            fdbonds = fdbonds.flatten()
            ffbonds = ffbonds.flatten()
            all_bonds_spin[key] = {}
            all_bonds_spin[key]["ss"] = ssbonds
            all_bonds_spin[key]["sp"] = spbonds
            all_bonds_spin[key]["ps"] = psbonds
            all_bonds_spin[key]["pp"] = ppbonds
            all_bonds_spin[key]["sd"] = sdbonds
            all_bonds_spin[key]["ds"] = dsbonds
            all_bonds_spin[key]["pd"] = pdbonds
            all_bonds_spin[key]["dp"] = dpbonds
            all_bonds_spin[key]["dd"] = ddbonds
            # add f orbs
            all_bonds_spin[key]["sf"] = sfbonds
            all_bonds_spin[key]["fs"] = fsbonds
            all_bonds_spin[key]["pf"] = pfbonds
            all_bonds_spin[key]["fp"] = fpbonds
            all_bonds_spin[key]["df"] = pfbonds
            all_bonds_spin[key]["fd"] = fpbonds
            all_bonds_spin[key]["ff"] = ffbonds

        # combine spin up and down or double energy
        all_bonds = {}
        if self.spin_polar:
            bonds = bonds_spin[0] + bonds_spin[1] * 2  # x2 for bond degeneracy (<i|j+R> and <j|i-R>)
            for orbkey in ["ss", "sp", "ps", "pp", "sd", "ds", "pd", "dp", "dd", "sf", "fs", "pf", "fp", "df", "fd",
                           "ff"]:  # add f orbs
                all_bonds[orbkey] = all_bonds_spin[0][orbkey] + all_bonds_spin[1][
                    orbkey] * 2  # x2 for bond degeneracy (<i|j+R> and <j|i-R>)
            if do_coop:
                bond_charges = (- charge_spin[0] - charge_spin[1]) * 2  # x2 for bond degeneracy (<i|j+R> and <j|i-R>)
                bond_magmoms = (charge_spin[0] - charge_spin[1]) * 2  # x2 for bond degeneracy (<i|j+R> and <j|i-R>)
        else:
            bonds = bonds_spin[0] * 2 * 2  # x2 for bond degeneracy (<i|j+R> and <j|i-R>)
            if do_coop:
                bond_charges = - charge_spin[
                    0] * 2 * 2  # consider fact that electrons are negative charge # extra x2 for fact that bond charge comes from <i|j+R> and <j|i-R>
            for orbkey in ["ss", "sp", "ps", "pp", "sd", "ds", "pd", "dp", "dd", "sf", "fs", "pf", "fp", "df", "fd",
                           "ff"]:  # add f orbs
                all_bonds[orbkey] = all_bonds_spin[0][orbkey] * 2 * 2  # x2 for bond degeneracy (<i|j+R> and <j|i-R>)

        # get vectors to bonds
        num_bonds = len(bonds)
        bond_indices = np.unravel_index(np.arange(num_bonds), (self.numAtoms, self.numAtoms, cell[0], cell[1], cell[2]))
        is_onsite = (bond_indices[2] == each_dir[0]) & (bond_indices[3] == each_dir[1]) & (
                bond_indices[4] == each_dir[2]) & (bond_indices[0] == bond_indices[1])
        bonds[is_onsite] = 0

        # get atomic charge and mag
        import plotly.colors as pc
        def get_rgb_values(colorscale_name):
            """
            Retrieve RGB values and their positions from a Plotly colorscale.
            """
            colorscale = pc.get_colorscale(colorscale_name)
            # Convert the rgb strings to tuples of integers
            rgb_values = np.array([
                (point[0], np.array([int(c) for c in point[1][4:-1].split(',')]))
                for point in colorscale
            ], dtype=object)
            return rgb_values

        def interpolate_colors(values, vmin, vmax, rgb_values):
            """
            Vectorized interpolation of colors for a NumPy array of values.
            """
            # Normalize values to the [0, 1] range
            scaled_values = np.clip((values - vmin) / (vmax - vmin), 0, 1)
            # Interpolate colors using the color scale
            interpolated_colors = np.zeros((len(values), 3), dtype=int)
            for i in range(len(rgb_values) - 1):
                t1, color1 = rgb_values[i]
                t2, color2 = rgb_values[i + 1]
                # Find values that belong to the current color interval
                mask = (scaled_values >= t1) & (scaled_values <= t2)
                factor = (scaled_values[mask] - t1) / (t2 - t1)
                # Linear interpolation between the two colors
                interpolated_colors[mask] = (1 - factor)[:, None] * color1 + factor[:, None] * color2
            # Convert RGB to hex
            hex_colors = ['#{:02x}{:02x}{:02x}'.format(*color) for color in interpolated_colors]
            return hex_colors

        if do_coop:
            atm_charges = []
            atm_magmoms = []
            for atm in range(self.numAtoms):
                is_val = (bond_indices[2] == each_dir[0]) & (bond_indices[3] == each_dir[1]) & (
                        bond_indices[4] == each_dir[2]) & (bond_indices[0] == atm) & (atm == bond_indices[1])
                # print(bond_charges[is_val][0])
                charge = self.elec_count[atm] + bond_charges[is_val][
                    0] / 2  # divide out factor of two for bond degeneracy
                atm_charges.append(charge)
                bond_charges[is_val] = charge
                if self.spin_polar:
                    magmom = bond_magmoms[is_val][0] / 2  # divide out factor of two for bond degeneracy for onsite term
                    atm_magmoms.append(magmom)
                    bond_magmoms[is_val] = magmom
            if self.spin_polar:
                atm_magmoms = np.array(atm_magmoms)
            atm_charges = np.array(atm_charges)
            if self.verbose > 1:
                print("atom charge!", atm_charges)

            # get the color values for atom and bonds
            if ("color" in auto_label):
                if "mag" in auto_label:
                    vmax = np.amax(np.abs(bond_magmoms)) / 4 + 0.05  # np.log10(+1)
                    vals = bond_magmoms  # np.log10(np.abs(bond_magmoms)+1)*np.sign(bond_magmoms)
                else:
                    vmax = np.amax(np.abs(bond_charges)) + 0.05
                    vals = bond_charges
                rgb_values = get_rgb_values('RdBu')
                vmin = -vmax
                bond_colors = interpolate_colors(vals, vmin, vmax, rgb_values)

        start_atom = [bond_indices[0], np.ones(num_bonds, dtype=np.int_) * each_dir[0],
                      np.ones(num_bonds, dtype=np.int_) * each_dir[1], np.ones(num_bonds, dtype=np.int_) * each_dir[2]]
        start_ind = np.ravel_multi_index(start_atom, (self.numAtoms, cell[0], cell[1], cell[2]))
        end_atom = [bond_indices[1], bond_indices[2], bond_indices[3], bond_indices[4]]
        end_ind = np.ravel_multi_index(end_atom, (self.numAtoms, cell[0], cell[1], cell[2]))
        atom_indices = np.unravel_index(np.arange(num_total_atoms), (self.numAtoms, cell[0], cell[1], cell[2]))[0]
        atom_elems = self.elements[atom_indices]
        if "mulliken" in auto_label:
            atom_labels = []
            atm_charges = []
            atm_magmoms = []
            for atmnum in range(self.numAtoms):
                vals = self.get_mulliken_charge(self, atmnum)
                # print("check type",type(vals),vals,atmnum)
                if type(vals) is float:
                    occ = str(np.around(vals, decimals=2))
                    elec = vals
                    if vals > 0:
                        occ = "+" + occ
                    label = self.elements[atmnum] + ": " + occ
                else:  # has occ and mag
                    occ = str(np.around(vals[0], decimals=2))
                    elec = vals[0]
                    mom = vals[1]
                    mag = str(np.around(vals[1], decimals=2))
                    if vals[0] > 0:
                        occ = "+" + occ
                    if vals[1] > 0:
                        mag = "+" + mag
                    label = self.elements[atmnum] + ": " + occ + "\n" + "magmom: " + mag
                    atm_magmoms.append(mom)
                # print(label)
                atom_labels.append(label)
                atm_charges.append(elec)
            atm_charges = np.array(atm_charges)
            atm_magmoms = np.array(atm_magmoms)
        elif do_coop:
            atom_labels = []
            for atmnum in range(self.numAtoms):
                occ = np.around(atm_charges[atmnum], decimals=2)
                if occ > 0:
                    occ = "+" + str(occ)
                else:
                    occ = str(occ)
                if self.spin_polar:
                    mag = np.around(atm_magmoms[atmnum], decimals=2)
                    if mag > 0:
                        mag = "+" + str(mag)
                    else:
                        mag = str(mag)
                    label = self.elements[atmnum] + ": " + occ + "\n" + "magmom: " + mag
                else:
                    label = self.elements[atmnum] + ": " + occ

                # print(label)
                atom_labels.append(label)

        atom_labels = np.array(atom_labels)
        og_atom_labels = copy.deepcopy(atom_labels)
        if len(atom_labels) == 0:
            atom_labels = atom_elems
        else:
            atom_labels = atom_labels[atom_indices]
        if self.verbose > 1:
            print("atom labels:", atom_labels)
        uniq_elem, atom_to_elem = np.unique(self.elements, return_inverse=True)
        # print("atm to elem:",atom_to_elem)
        # print(atom_to_elem[atom_indices])

        # get the color values for atoms
        if ("color" in auto_label):
            if ("mag" in auto_label) and self.spin_polar:
                vmax = np.amax(np.abs(atm_magmoms)) / 4 + 0.05  # np.log10(+1)
                vals = atm_magmoms  # np.log10(np.abs(atm_magmoms)+1)*np.sign(atm_magmoms)
            else:
                vmax = np.amax(np.abs(atm_charges)) + 0.05
                vals = atm_charges
            rgb_values = get_rgb_values('RdBu')
            vmin = -vmax
            atom_colors = interpolate_colors(vals, vmin, vmax, rgb_values)
            atom_colors = np.array(atom_colors)

        if len(atom_colors) == 0:
            if len(elem_colors) == 0:
                colors = np.array(
                    ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
                     '#17becf', '#564580', '#B87333', '#63c433', '#db61ed', '#298ec4', '#db61ed'])
                atm_colors = colors[atom_to_elem[atom_indices]]
            elif type(elem_colors[0]) == int:
                elem_colors = np.array(elem_colors)
                colors = np.array(
                    ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
                     '#17becf', '#564580', '#B87333', '#63c433', '#db61ed', '#298ec4', '#db61ed'])
                atm_colors = colors[elem_colors[atom_to_elem[atom_indices]]]
            else:

                elem_colors = np.array(elem_colors)
                atm_colors = elem_colors[atom_to_elem[atom_indices]]
        elif type(atom_colors[0]) == int:
            atom_colors = np.array(atom_colors)
            colors = np.array(
                ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
                 '#17becf', '#564580', '#B87333', '#63c433', '#db61ed', '#298ec4', '#db61ed'])
            atm_colors = colors[atom_colors[atom_indices]]
        else:
            atom_colors = np.array(atom_colors)
            atm_colors = atom_colors[atom_indices]
        atom_rad = []
        for elem in atom_elems:
            if elem in all_atom_rad.keys():
                atom_rad.append(all_atom_rad[elem])
            else:
                atom_rad.append(1.3)

        # colors = ["blue","green","orange","purple","red",
        def cylinder_mesh(point1, point2, radius, resolution=15):
            """
            Create the coordinates of a cylinder mesh between two points.
            Args:
                point1 (array-like): [x, y, z] coordinates of one end of the cylinder.
                point2 (array-like): [x, y, z] coordinates of the other end of the cylinder.
                radius (float): Radius of the cylinder.
                resolution (int): Number of segments along the circle.

            Returns:
                (x, y, z): Mesh coordinates for the cylinder.
            """
            # Vector along the cylinder's axis
            v = np.array(point2) - np.array(point1)
            v = v / np.linalg.norm(v)  # Normalize the vector

            # Create a perpendicular vector
            if abs(v[0]) < 1e-8 and abs(v[1]) < 1e-8:
                # v is approximately along z-axis
                not_v = np.array([1, 0, 0])
            else:
                not_v = np.array([0, 0, 1])
            n1 = np.cross(v, not_v)
            n1 /= np.linalg.norm(n1)
            n2 = np.cross(v, n1)

            # Generate points around a circle at the base of the cylinder
            t = np.linspace(0, 2 * np.pi, resolution)
            circle = np.array([radius * np.cos(t), radius * np.sin(t)])  # Circle coordinates

            # Mesh points along the cylinder's axis
            X, Y, Z = [], [], []
            for i in [0, 1]:  # Iterate over start and end of the cylinder
                point = point1 if i == 0 else point2
                x = point[0] + circle[0] * n1[0] + circle[1] * n2[0]
                y = point[1] + circle[0] * n1[1] + circle[1] * n2[1]
                z = point[2] + circle[0] * n1[2] + circle[1] * n2[2]
                X.append(x), Y.append(y), Z.append(z)

            return np.concatenate(X), np.concatenate(Y), np.concatenate(Z)

        def midsection_hover_points(point1, point2, radius, resolution=20):
            """
            Generate hover points along the midsection of the cylinder.
            Args:
                point1, point2: Start and end points of the cylinder.
                radius: Radius of the cylinder.
                resolution: Number of points along the circle.

            Returns:
                (x, y, z, text): Coordinates and hover labels for the midsection points.
            """
            # Calculate the midpoint of the cylinder
            midpoint = (np.array(point1) + np.array(point2)) / 2

            # Vector along the cylinder's axis
            v = np.array(point2) - np.array(point1)
            v = v / np.linalg.norm(v)  # Normalize the vector

            # Create two perpendicular vectors
            if abs(v[0]) < 1e-8 and abs(v[1]) < 1e-8:
                # v is approximately along z-axis
                not_v = np.array([1, 0, 0])
            else:
                not_v = np.array([0, 0, 1])
            n1 = np.cross(v, not_v) / np.linalg.norm(np.cross(v, not_v))
            n2 = np.cross(v, n1)

            # Generate points around the circle at the midpoint
            t = np.linspace(0, 2 * np.pi, resolution)
            x = midpoint[0] + radius * np.cos(t) * n1[0] + radius * np.sin(t) * n2[0]
            y = midpoint[1] + radius * np.cos(t) * n1[1] + radius * np.sin(t) * n2[1]
            z = midpoint[2] + radius * np.cos(t) * n1[2] + radius * np.sin(t) * n2[2]

            return x, y, z

        # plot the bond lines
        # Create connecting lines trace
        data = []
        all_start_atms = np.array([[10, 10, 10]])
        all_end_atms = np.array([[10, 10, 10]])
        all_centers = np.array([[10, 10, 10]])
        all_text = []
        all_customdata = []
        all_colors = []
        all_energies = []
        all_styles = []
        all_atmrads = []
        all_atm_elems = []
        all_atm_labels = []
        all_atm_inds = []
        for b_ind in range(num_bonds):
            bond_energy = bonds[b_ind]
            if bond_energy < 0:
                style = "solid"
            else:
                style = "dash"
            if abs(bond_energy) > energy_cutoff:
                start = start_ind[b_ind]
                end = end_ind[b_ind]
                ogstart_atm = np.array([x_data[start], y_data[start], z_data[start]])
                ogend_atm = np.array([x_data[end], y_data[end], z_data[end]])
                # place atoms also on 1 when start atm is at 0
                trans = np.array([[0, 0, 0]])
                hold_atm = np.around(abc_data[:, start], decimals=3)  # start atm in primitive coordinates
                abc = np.linalg.norm(self._a, axis=1)
                # print(hold_atm,hold_atm==0)
                if (hold_atm == 0).any():
                    for vectoswitch in np.arange(3)[hold_atm == 0]:  # (hold_atm==0).any():
                        if abc[vectoswitch] < 4 * (
                                np.sum(abc) - abc[vectoswitch]) / 2:  # aviod 2D cases with lots of vacuum
                            # vectoswitch = np.arange(3)[hold_atm==0][0]
                            switch = np.zeros(3)
                            switch[vectoswitch] = 1
                            new_trans = trans[:, :] + switch[None, :]
                            trans = np.append(trans, new_trans, axis=0)
                cart_trans = _red_to_cart((self._a[0], self._a[1], self._a[2]), trans)
                # print(cart_trans)
                for ti, tran in enumerate(cart_trans):
                    start_atm = ogstart_atm + tran
                    end_atm = ogend_atm + tran
                    prim_startatm = abc_data[:, start] + trans[ti]

                    plot = True
                    if one_atom == True:
                        if (np.abs((prim_startatm - self.primAtoms[plot_atom])) > 0.01).any():
                            plot = False
                    if plot == True:
                        prim_endatm = abc_data[:, end] + trans[ti]
                        abc = np.linalg.norm(self._a, axis=1)
                        dist = copy.deepcopy(prim_endatm)
                        dist[prim_endatm >= 0] = 0  # only keep negative values
                        dist[prim_endatm > 1] = prim_endatm[prim_endatm > 1] - 1
                        weighted_dist = np.abs(dist) * abc
                        bond_dist = np.linalg.norm(end_atm - start_atm)
                        # print(weighted_dist)
                        # first check
                        is_new_bond = False
                        if (weighted_dist < 0.1).all() or bond_dist < bond_max:
                            is_new_bond == True
                            # check whether another bond already exists with the same end points
                            same_start = (all_start_atms[:, 0] == start_atm[0]) & (
                                        all_start_atms[:, 1] == start_atm[1]) & (all_start_atms[:, 2] == start_atm[2])
                            check_end_atm = all_end_atms[same_start]
                            same_end = ((check_end_atm[:, 0] == end_atm[0]) & (check_end_atm[:, 1] == end_atm[1]) & (
                                        check_end_atm[:, 2] == end_atm[2])).any()
                            if same_end == False:  # also check for opposite
                                same_start = (all_start_atms[:, 0] == end_atm[0]) & (
                                        all_start_atms[:, 1] == end_atm[1]) & (
                                                     all_start_atms[:, 2] == end_atm[2])
                                check_end_atm = all_end_atms[same_start]
                                same_end = ((check_end_atm[:, 0] == start_atm[0]) & (
                                        check_end_atm[:, 1] == start_atm[1]) & (
                                                    check_end_atm[:, 2] == start_atm[2])).any()
                            if same_end:
                                if self.verbose > 1:
                                    print("avoided duplicate!")
                            is_new_bond = not same_end

                        if is_new_bond == True:
                            # all_start_atms = np.append(all_start_atms,np.array([start_atm]),axis=0)
                            # all_end_atms = np.append(all_end_atms,np.array([end_atm]),axis=0)
                            # print("got new trans:",trans)
                            NN = np.arange(len(self.NN_dists))[
                                np.argmin(np.abs(self.NN_dists - bond_dist))]  # <= 0.0001][0]
                            centerx = (start_atm[0] + end_atm[0]) / 2
                            centery = (start_atm[1] + end_atm[1]) / 2
                            centerz = (start_atm[2] + end_atm[2]) / 2
                            orbs = ["s", "p", "d", "f"]
                            bond_label = ""
                            for orb1 in orbs:
                                for orb2 in orbs:
                                    bond_eng = all_bonds[orb1 + orb2][b_ind]
                                    if np.abs(bond_eng) > np.abs(bond_energy) / 10:
                                        bond_str = atom_elems[start] + " " + orb1 + " - " + atom_elems[
                                            end] + " " + orb2 + ": " + str(
                                            np.around(bond_eng, decimals=3)) + " eV" + "<br>"
                                        bond_label += bond_str
                            spin_label = ""
                            if self.spin_polar:
                                spin_label += "<br>Energy from spin up: " + str(
                                    np.around(bonds_spin[0][b_ind], decimals=3)) + " eV"
                                spin_label += "<br>Energy from spin down: " + str(
                                    np.around(bonds_spin[1][b_ind], decimals=3)) + " eV"
                            full_text = ("Bond energy: " + str(
                                np.around(bond_energy, decimals=3)) + " eV" + "<br>Bond length: "
                                         + str(np.around(bond_dist, decimals=3)) + " Å" + " (" + str(NN) + "NN)" +
                                         spin_label + "<br>Breakdown into orbitals: <br>" + bond_label)
                            # all_atm_elems.append([atom_elems[start],atom_elems[end]])
                            # all_atm_labels.append([atom_labels[start],atom_labels[end]])
                            # atm1, atm2, cell1, cell2, cell3 = np.unravel_index(b_ind, (
                            #    self.numAtoms, self.numAtoms, cell[0], cell[1], cell[2]))
                            # all_atm_inds.append([atm1,atm2])
                            '''
                            data.append(go.Scatter3d(
                                x=[centerx],
                                y=[centery],
                                z=[centerz],
                                mode='markers',
                                marker=dict(size=20, color='rgba(0,0,0,0)', opacity=0.),
                                hoverinfo="text",
                                hoverlabel=dict(bgcolor='ghostwhite',
                                                font=dict(family='Arial', size=12, color='black')),
                                text=full_text, showlegend=False
                            ))
                            '''
                            start_rad = atom_rad[start]
                            end_rad = atom_rad[end]
                            # all_colors.append([atm_colors[start], atm_colors[end]])
                            # all_energies.append(bond_energy)
                            # all_styles.append(style)
                            # all_atmrads.append([start_rad, end_rad])

                            # Plot the cylinder using Plotly Mesh3d
                            color = 'gray'
                            if do_coop:
                                full_text = full_text + "Bond charge: " + str(
                                    np.around(bond_charges[b_ind], decimals=3))
                                if ("color" in auto_label):
                                    color = bond_colors[b_ind]
                                if self.spin_polar:
                                    full_text = full_text + "<br>" + "Bond magmom: " + str(
                                        np.around(bond_magmoms[b_ind], decimals=3))

                            sorted_elems = np.sort([atom_elems[start], atom_elems[end]])
                            bond_label = np.char.add(np.char.add(sorted_elems[0], sorted_elems[1]), str(NN))
                            if False:  # style == 'dash':
                                closer_atom1 = (start_atm * 5 + end_atm) / 6
                                closer_atom2 = (start_atm + end_atm * 5) / 6
                                all_centers = np.append(all_centers, np.array(
                                    [[centerx, centery, centerz], closer_atom1, closer_atom2]), axis=0)
                                all_text.append(full_text)
                                all_text.append(full_text)
                                all_text.append(full_text)
                                all_customdata.append(bond_label)  # np.around(bond_energy,decimals=1))
                                all_customdata.append(bond_label)
                                all_customdata.append(bond_label)
                                data.append(go.Scatter3d(
                                    x=[start_atm[0], end_atm[0]],
                                    y=[start_atm[1], end_atm[1]],
                                    z=[start_atm[2], end_atm[2]],
                                    mode='lines',  # 'markers+lines',
                                    # marker=dict(size=[start_rad * 40, end_rad * 40],
                                    #            color=[atm_colors[start], atm_colors[end]], opacity=1.0,
                                    #            line=dict(color='black', width=20)),
                                    line=dict(
                                        color=color,
                                        width=abs(bond_energy / 2) ** (1 / 2) * 20,
                                        dash=style
                                    ),
                                    # hoverinfo="text",
                                    # hoverlabel=dict(bgcolor='ghostwhite',
                                    #                font=dict(family='Arial', size=12, color='black')),
                                    # text=full_text,
                                    hoverinfo='none',
                                    showlegend=False))
                            else:
                                # instead generate the bond mesh with actually bond density
                                atm1, atm2, cell1, cell2, cell3 = np.unravel_index(b_ind, (
                                    self.numAtoms, self.numAtoms, cell[0], cell[1], cell[2]))
                                orbs1 = np.arange(self.num_orbs)[self.orbatomnum == atm1]
                                orbs2 = np.arange(self.num_orbs)[self.orbatomnum == atm2]
                                bond_coop = icoop_orb[0][orbs1, :][:, orbs2][:, :, cell1, cell2, cell3]
                                # print(bond_coop)
                                orb_lap = self.small_overlap[0][orbs1, :][:, orbs2][:, :, cell1, cell2, cell3].flatten()
                                bond_coop = bond_coop.flatten()
                                bond_coop[orb_lap != 0] = bond_coop[orb_lap != 0] / orb_lap[orb_lap != 0]
                                bond_coop = np.reshape(bond_coop, (len(orbs1), len(orbs2)))
                                lowxyz = [np.amin([start_atm[0], end_atm[0]]), np.amin([start_atm[1], end_atm[1]]),
                                          np.amin([start_atm[2], end_atm[2]])]
                                highxyz = [np.amax([start_atm[0], end_atm[0]]), np.amax([start_atm[1], end_atm[1]]),
                                           np.amax([start_atm[2], end_atm[2]])]
                                res = 20
                                X, Y, Z = np.mgrid[lowxyz[0] - 1:highxyz[0] + 1:res * 1j,
                                          lowxyz[1] - 1:highxyz[1] + 1:res * 1j, lowxyz[2] - 1:highxyz[2] + 1:res * 1j]
                                cartXYZ = np.array([X.flatten(), Y.flatten(), Z.flatten()])
                                bond_density = self.make_bond(atm1, atm2, start_atm, end_atm, bond_coop, cartXYZ)
                                print("bond density min and max:", np.amin(bond_density), np.amax(bond_density))
                                # print(bond_coop.shape,atm1,atm2,cell1, cell2, cell3)
                                # print(start_atm,end_atm)
                                # print(self.sph_harm_key[orbs1],self.sph_harm_key[orbs2])
                                # print(bond_coop)
                                dense_min = np.amin(bond_density)
                                dense_max = np.amax(bond_density)
                                isoval = np.array([dense_min, dense_max])[np.argmax([-dense_min, dense_max])] / 2
                                if dense_max > iso_max:
                                    all_start_atms = np.append(all_start_atms, np.array([start_atm]), axis=0)
                                    all_end_atms = np.append(all_end_atms, np.array([end_atm]), axis=0)
                                    all_atm_elems.append([atom_elems[start], atom_elems[end]])
                                    all_atm_labels.append([atom_labels[start], atom_labels[end]])
                                    all_colors.append([atm_colors[start], atm_colors[end]])
                                    all_energies.append(bond_energy)
                                    all_styles.append(style)
                                    all_atmrads.append([start_rad, end_rad])
                                    atm1, atm2, cell1, cell2, cell3 = np.unravel_index(b_ind, (
                                        self.numAtoms, self.numAtoms, cell[0], cell[1], cell[2]))
                                    all_atm_inds.append([atm1, atm2])
                                    isoval = iso_max
                                    spacing = (np.array(highxyz) - np.array(lowxyz) + 2) / (res - 1)
                                    verts, faces, _, _ = marching_cubes(np.reshape(bond_density, (res, res, res)),
                                                                        isoval,
                                                                        spacing=(spacing[0], spacing[1], spacing[2]))
                                    verts = verts + (np.array(lowxyz) - 1)[None, :]
                                    data.append(go.Mesh3d(
                                        x=verts[:, 0],
                                        y=verts[:, 1],
                                        z=verts[:, 2],
                                        i=faces[:, 0],
                                        j=faces[:, 1],
                                        k=faces[:, 2],
                                        opacity=1.0,
                                        # color=color,
                                        color=color,
                                        flatshading=False,  # Smooth shading
                                        lighting=dict(
                                            ambient=0.6,  # Ambient light (general light)
                                            diffuse=0.7,  # Diffuse light (soft shadows)
                                            fresnel=0.1,  # Reflective edges
                                            specular=0.9,  # Specular reflection (sharp highlights)
                                            roughness=0.55,  # Surface roughness
                                        ),
                                        lightposition=dict(
                                            x=10, y=10, z=10  # Light source position
                                        ),
                                        hoverinfo="text",
                                        hoverlabel=dict(bgcolor='ghostwhite',
                                                        font=dict(family='Arial', size=12, color='black')),
                                        text=full_text, showlegend=False, customdata=[bond_label] * len(verts),
                                        # alphahull=0
                                    ))
                                if dense_min < iso_min:
                                    all_start_atms = np.append(all_start_atms, np.array([start_atm]), axis=0)
                                    all_end_atms = np.append(all_end_atms, np.array([end_atm]), axis=0)
                                    all_atm_elems.append([atom_elems[start], atom_elems[end]])
                                    all_atm_labels.append([atom_labels[start], atom_labels[end]])
                                    atm1, atm2, cell1, cell2, cell3 = np.unravel_index(b_ind, (
                                        self.numAtoms, self.numAtoms, cell[0], cell[1], cell[2]))
                                    all_atm_inds.append([atm1, atm2])
                                    all_colors.append([atm_colors[start], atm_colors[end]])
                                    all_energies.append(bond_energy)
                                    all_styles.append(style)
                                    all_atmrads.append([start_rad, end_rad])
                                    isoval = iso_min
                                    spacing = (np.array(highxyz) - np.array(lowxyz) + 2) / (res - 1)
                                    verts, faces, _, _ = marching_cubes(np.reshape(bond_density, (res, res, res)),
                                                                        isoval,
                                                                        spacing=(spacing[0], spacing[1], spacing[2]))
                                    verts = verts + (np.array(lowxyz) - 1)[None, :]
                                    data.append(go.Mesh3d(
                                        x=verts[:, 0],
                                        y=verts[:, 1],
                                        z=verts[:, 2],
                                        i=faces[:, 0],
                                        j=faces[:, 1],
                                        k=faces[:, 2],
                                        opacity=1.0,
                                        # color=color,
                                        color="black",
                                        flatshading=False,  # Smooth shading
                                        lighting=dict(
                                            ambient=0.6,  # Ambient light (general light)
                                            diffuse=0.7,  # Diffuse light (soft shadows)
                                            fresnel=0.1,  # Reflective edges
                                            specular=0.9,  # Specular reflection (sharp highlights)
                                            roughness=0.55,  # Surface roughness
                                        ),
                                        lightposition=dict(
                                            x=10, y=10, z=10  # Light source position
                                        ),
                                        hoverinfo="text",
                                        hoverlabel=dict(bgcolor='ghostwhite',
                                                        font=dict(family='Arial', size=12, color='black')),
                                        text=full_text, showlegend=False, customdata=[bond_label] * len(verts),
                                        # alphahull=0
                                    ))

        # print(all_end_atms.shape,all_start_atms.shape)
        # find all unique atoms
        combnd_atms = np.append(all_start_atms[1:], all_end_atms[1:], axis=0)
        all_colors = np.array(all_colors)
        all_atmrads = np.array(all_atmrads)
        all_atm_elems = np.array(all_atm_elems)
        all_atm_labels = np.array(all_atm_labels)
        all_atm_inds = np.array(all_atm_inds)
        combnd_colors = np.append(all_colors[:, 0], all_colors[:, 1])
        combnd_atmrad = np.append(all_atmrads[:, 0], all_atmrads[:, 1])
        combnd_elems = np.append(all_atm_elems[:, 0], all_atm_elems[:, 1])
        combnd_labels = np.append(all_atm_labels[:, 0], all_atm_labels[:, 1])
        combnd_atminds = np.append(all_atm_inds[:, 0], all_atm_inds[:, 1])
        uniq_atms, indices = np.unique(combnd_atms, axis=0, return_index=True)
        uniq_colors = combnd_colors[indices]
        uniq_rad = combnd_atmrad[indices]
        uniq_elems = combnd_elems[indices]
        uniq_labels = combnd_labels[indices]
        uniq_atminds = combnd_atminds[indices]

        if self.verbose > 1:
            print("all atms:", uniq_atms)
            print(uniq_rad)
            print(uniq_colors)

        # append the bond labels
        marker_dict = dict(size=20, color='rgba(0,0,0,0)', opacity=0.)
        if "mag" in auto_label:  # include stuff just to plot the colorbar for magnetism
            marker_dict = dict(size=20, color='rgba(0,0,0,0)', opacity=0., colorscale='RdBu', cmin=vmin, cmax=vmax,
                               colorbar=dict(title="Magnetic moment",  # 'log(Magnetization+1)',
                                             # tickvals=np.linspace(vmin, vmax, 5),  # Set ticks for the log scale
                                             # ticktext=[f"{np.around(np.sign(tick)*(10**np.abs(tick)-1),decimals=2)}" for tick in np.linspace(vmin, vmax, 5)]  # Custom tick labels
                                             ))
        elif "color" in auto_label:
            marker_dict = dict(size=20, color='rgba(0,0,0,0)', opacity=0., colorscale='RdBu', cmin=vmin, cmax=vmax,
                               colorbar=dict(title='Charge',
                                             # tickvals=np.linspace(vmin, vmax, 5),  # Set ticks for the log scale
                                             # ticktext=[f"{np.around(np.sign(tick)*(10**np.abs(tick)-1),decimals=2)}" for tick in np.linspace(vmin, vmax, 5)]  # Custom tick labels
                                             ))
        all_centers = all_centers[1:]
        data.append(go.Scatter3d(
            x=all_centers[:, 0],
            y=all_centers[:, 1],
            z=all_centers[:, 2],
            mode='markers',
            marker=marker_dict,
            hoverinfo="text",
            hoverlabel=dict(bgcolor='ghostwhite',
                            font=dict(family='Arial', size=12, color='black')),
            text=all_text, showlegend=False, customdata=all_customdata))

        def create_sphere(center, radius, resolution=15):
            """
            Generate the mesh coordinates for a sphere.
            """
            u = np.linspace(0, 2 * np.pi, resolution)
            v = np.linspace(0, np.pi, resolution)
            x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
            y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
            z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
            return x.flatten(), y.flatten(), z.flatten()

        # Loop over atoms to create a sphere for each atom
        for ai, atm in enumerate(uniq_atms):
            x, y, z = create_sphere(atm, uniq_rad[ai] ** (0.8) / 2)
            trace = go.Mesh3d(
                x=x, y=y, z=z,
                color=uniq_colors[ai],
                opacity=1.0,
                lighting=dict(
                    ambient=0.6,  # Ambient light (general light)
                    diffuse=0.7,  # Diffuse light (soft shadows)
                    fresnel=0.1,  # Reflective edges
                    specular=0.9,  # Specular reflection (sharp highlights)
                    roughness=0.55,  # Surface roughness
                ),
                lightposition=dict(
                    x=10, y=10, z=10  # Light source position
                ),
                alphahull=0,  # Proper enclosure for the mesh
                hoverinfo="text",
                # hoverlabel = dict(bgcolor='ghostwhite',
                #                  font=dict(family='Arial', size=12, color='black')),
                text=uniq_labels[ai], showlegend=False)

            data.append(trace)

        # instead generate the bond mesh with actually bond density
        # Loop over atoms to create a sphere for each atom
        for ai, atm in enumerate(uniq_atms):
            atm1 = uniq_atminds[ai]
            orbs1 = np.arange(self.num_orbs)[self.orbatomnum == atm1]
            center = atm
            bond_coop = icoop_orb[0][orbs1, :][:, orbs1][:, :, each_dir[0], each_dir[1], each_dir[2]]
            # print(bond_coop)
            orb_lap = self.small_overlap[0][orbs1, :][:, orbs1][:, :, each_dir[0], each_dir[1], each_dir[2]].flatten()
            bond_coop = bond_coop.flatten()
            bond_coop[orb_lap != 0] = bond_coop[orb_lap != 0] / orb_lap[orb_lap != 0]
            bond_coop = np.reshape(bond_coop, (len(orbs1), len(orbs1)))
            bond_diag = np.diag(bond_coop)
            is_p = []
            for oind, orb in enumerate(orbs1):
                # print(self.exactorbtype[orb])
                if "p" in self.exactorbtype[orb]:
                    is_p.append(oind)
                if "s" in self.exactorbtype[orb]:
                    s_ind = oind
            # is_p = np.array(is_p)
            # print(is_p)
            total_p = np.sum(bond_diag[is_p])
            s_val = bond_diag[s_ind]
            hold_zeros = np.zeros(len(bond_diag))
            bond_diag = hold_zeros
            bond_diag[s_ind] = np.sum(np.abs(bond_coop)) / 20  # min(s_val,total_p)
            # print("orb vals:",total_p,min(s_val,total_p))
            np.fill_diagonal(bond_coop, 0)
            combo_electrons = np.sum(np.abs(bond_coop))
            if np.sum(np.abs(bond_coop)) > 0.1:  # only continue with decent amount of lone pair character
                bond_coop = bond_coop + np.diag(bond_diag)
                res = 30
                X, Y, Z = np.mgrid[center[0] - 1.5:center[0] + 1.5:res * 1j,
                          center[1] - 1.5:center[1] + 1.5:res * 1j, center[2] - 1.5:center[2] + 1.5:res * 1j]

                cartXYZ = np.array([X.flatten(), Y.flatten(), Z.flatten()])
                atom_density = self.make_bond(atm1, atm1, center, center, bond_coop, cartXYZ)
                label_text = uniq_labels[ai] + "<br>Lone pair from " + str(
                    np.around(combo_electrons, decimals=3)) + " sp overlap"
                # print("atom density min and max:", np.amin(atom_density), np.amax(atom_density))
                # print(bond_coop.shape, atm1, atm2)
                # print(center)
                # print(self.sph_harm_key[orbs1])
                # print(bond_coop)
                isoval = np.amax(np.abs(atom_density)) / 2
                spacing = 3 / (res - 1)
                verts, faces, _, _ = marching_cubes(np.reshape(atom_density, (res, res, res)), isoval,
                                                    spacing=(spacing, spacing, spacing))
                verts = verts + (np.array(center) - 1.5)[None, :]
                data.append(go.Mesh3d(
                    x=verts[:, 0],
                    y=verts[:, 1],
                    z=verts[:, 2],
                    i=faces[:, 0],
                    j=faces[:, 1],
                    k=faces[:, 2],
                    opacity=1.0,
                    # color=color,
                    color=uniq_colors[ai],
                    flatshading=False,  # Smooth shading
                    lighting=dict(
                        ambient=0.6,  # Ambient light (general light)
                        diffuse=0.7,  # Diffuse light (soft shadows)
                        fresnel=0.1,  # Reflective edges
                        specular=0.9,  # Specular reflection (sharp highlights)
                        roughness=0.55,  # Surface roughness
                    ),
                    lightposition=dict(
                        x=10, y=10, z=10  # Light source position
                    ),
                    hoverinfo="text",
                    hoverlabel=dict(bgcolor='ghostwhite',
                                    font=dict(family='Arial', size=12, color='black')),
                    text=label_text, showlegend=False
                    # alphahull=0
                ))

        if one_atom == False:
            # draw primitive cell # place atoms also on 1 when start atm is at 0
            trans = np.array([[0, 0, 0]])
            for vectoswitch in np.arange(3):  # (hold_atm==0).any():
                if abc[vectoswitch] < 4 * (np.sum(abc) - abc[vectoswitch]) / 2:  # aviod 2D cases with lots of vacuum
                    # vectoswitch = np.arange(3)[hold_atm==0][0]
                    switch = np.zeros(3)
                    switch[vectoswitch] = 1
                    new_trans = trans[:, :] + switch[None, :]
                    trans = np.append(trans, new_trans, axis=0)
            num = len(trans)
            start_points = trans[:, None, :] * np.ones((num, num, 3))
            end_points = trans[None, :, :] * np.ones((num, num, 3))
            all_vecs = end_points - start_points
            norm = np.linalg.norm(all_vecs, axis=2)
            start_points = np.reshape(start_points, (num ** 2, 3))[norm.flatten() == 1]
            end_points = np.reshape(end_points, (num ** 2, 3))[norm.flatten() == 1]
            final_ind = []
            for veci in range(len(start_points)):
                diff = end_points[veci] - start_points[veci]
                if (diff >= 0).all():
                    final_ind.append(veci)
            final_ind = np.array(final_ind)

            start_points = start_points[final_ind]
            end_points = end_points[final_ind]
            cart_start = _red_to_cart((self._a[0], self._a[1], self._a[2]), start_points)
            cart_end = _red_to_cart((self._a[0], self._a[1], self._a[2]), end_points)

            for i in range(len(start_points)):
                start_atm = cart_start[i]
                end_atm = cart_end[i]
                data.append(go.Scatter3d(
                    x=[start_atm[0], end_atm[0]],
                    y=[start_atm[1], end_atm[1]],
                    z=[start_atm[2], end_atm[2]],
                    mode='lines',
                    line=dict(
                        color='black',
                        width=2,
                        dash="solid"
                    ), hoverinfo='none', showlegend=False))

        # Create figure
        fig = go.Figure(data=data)

        # update figure design
        # fig.update_traces(hovertemplate=None)
        fig.update_layout(hoverdistance=100, margin=dict(l=0, r=0, b=0, t=0),
                          paper_bgcolor='rgba(0,0,0,0)',  # Transparent background for the whole figure
                          plot_bgcolor='rgba(0,0,0,0)')  # ,hovermode="x unified")
        fig.update_layout(
            scene=dict(xaxis=dict(showbackground=False, showspikes=False, showticklabels=False, visible=False),
                       yaxis=dict(showbackground=False, showspikes=False, showticklabels=False, visible=False),
                       zaxis=dict(showbackground=False, showspikes=False, showticklabels=False, visible=False),
                       aspectmode='data'))

        # fig.layout.scene.camera.projection.type = "orthographic"
        # Set layout with camera and FOV (fovy)
        fig.update_layout(scene_camera=dict(eye=dict(x=1.25, y=-0.05, z=0.05)))

        fov = fovy
        dist = np.around(75 / fov, decimals=2)
        post_script = (
                    "const plot = document.querySelector('.js-plotly-plot'); const scene = plot._fullLayout.scene._scene;"
                    " const camera = scene.glplot; camera.fovy =" + str(
                fov) + " * Math.PI / 180;  scene.camera.distance =" + str(dist) + ";  "
                                                                                  "console.log(`Updated FOV to ${camera.fovy} radians.`);")
        if return_fig:
            return fig, post_script

        else:
            fig.write_html(self.directory + "crystal_bond_density.html", post_script=post_script, div_id="plotDiv")

            if self.show_figs:
                fig.show(post_script=post_script, div_id="plotDiv")

    @staticmethod
    def get_COHP(self: object, orbs: dict, NN: int=None, include_onsite: bool=False, just_one: bool = False) -> npt.NDArray:
        """
        Calculates the COHP for the given orbitals and nearest neighbors.

        Args:    
            self (object): An object of the class COGITO_BAND or COGITO_UNIFORM.
            orbs (dict): Either a list of two dictionaries giving elements as keys and orbital types as items (eg [{"Pb":["s","d"],"O":["s","p"]},{"Pb":["s"]"O":["p"]}]) or give list of orb numbers [[1,2,3,5,6,7],[1,2,3,4,5,6,7,8]].
            NN (int): An integer for which nearest neighbor number to include (eg 1 for 1NN) or None or "All" for all nearest neighbors.
            include_onsite (bool): Includes atomic orbital energy terms (H_ab(R) where R=0 and a=b) instead of just bonding terms.

        Returns:    
            npt.NDArray: Returns COHP values in a [kpt,band] dimension.
        """

        if type(orbs[0]) is dict:
            firstorbs = []
            secorbs = []

            if type(list(orbs[0].keys())[0]) is str:
                # format of orbs should be {"Si":["s","p"],"Pb":["s"]}
                # find orbs for the element
                for element in orbs[0].keys():
                    #print(element)
                    for orbtype in orbs[0][element]:
                        for addorb in range(self.num_orbs):
                            if self.orbatomname[addorb] == element and orbtype in self.exactorbtype[addorb]:
                                firstorbs.append(addorb)

                for element in orbs[1].keys():
                    for orbtype in orbs[1][element]:
                        for addorb in range(self.num_orbs):
                            if self.orbatomname[addorb] == element and orbtype in self.exactorbtype[addorb]:
                                secorbs.append(addorb)
            else:  # type is int and should be referring to the specific atom number in prim cell
                for atm in orbs[0].keys():
                    print(atm)
                    for orbtype in orbs[0][atm]:
                        for addorb in range(self.num_orbs):
                            if self.orbatomnum[addorb] == atm and orbtype in self.exactorbtype[addorb]:
                                firstorbs.append(addorb)

                for atm in orbs[1].keys():
                    for orbtype in orbs[1][atm]:
                        for addorb in range(self.num_orbs):
                            if self.orbatomnum[addorb] == atm and orbtype in self.exactorbtype[addorb]:
                                secorbs.append(addorb)
            orbs = [firstorbs, secorbs]
            #print(orbs)

        # reference indices based on the nearest neighbors, much faster and can look at specific NN contributions
        num_each_dir = self.num_each_dir
        #if not hasattr(self, 'NN_index'):
        #    self.get_neighbors()
        if NN == None or NN == "All":
            indices = np.array([], dtype=np.int_)
            for nn in range(len(self.NN_dists)):
                indices = np.append(indices, self.NN_index[nn])
            total_index = indices
            if self.verbose > 0:
                print("NN distances that are included:", self.NN_dists)
        else:
            if int(NN) > len(self.NN_dists):
                print("give a lower NN integer!")
            NN_dist = self.NN_dists[int(NN)]
            total_index = self.NN_index[int(NN)]
            if self.verbose > 0:
                print("NN number, distance, count, and indices:", int(NN), NN_dist, len(total_index))  # ,NN_index)

        # convert the flattens indices into the full matrix indices
        full_indices = np.unravel_index(total_index, (
        self.num_orbs, self.num_orbs, self.num_trans[0], self.num_trans[1], self.num_trans[2]))
        # only select terms which have the orbitals the user is seeking
        orb1_in_full = full_indices[0][None, :] == np.array([orbs[0]]).T
        orb2_in_full = full_indices[1][None, :] == np.array([orbs[1]]).T
        in_full1 = np.full(len(orb1_in_full[0]), False)
        in_full2 = np.full(len(orb2_in_full[0]), False)
        for bool_list in orb1_in_full:
            in_full1 = in_full1 | bool_list
        for bool_list in orb2_in_full:
            in_full2 = in_full2 | bool_list
        in_full = in_full1 & in_full2
        # remove onsite terms
        is_onsite = (full_indices[2] == num_each_dir[0]) & (full_indices[3] == num_each_dir[1]) & (
                    full_indices[4] == num_each_dir[2]) & (full_indices[0] == full_indices[1])
        # print(in_full)
        if include_onsite == False:
            in_full = in_full & np.logical_not(is_onsite)
        # print(in_full)
        full_ind = [full_indices[0][in_full], full_indices[1][in_full], full_indices[2][in_full],
                    full_indices[3][in_full], full_indices[4][in_full]]
        if self.verbose > 0:
            print("num of params:", len(full_ind[0]))
        spec_icohp = 0
        for key in self.keys: # plot bands for each spin
            spec_icohp += np.sum(self.ICOHP[key][full_ind[0], full_ind[1], full_ind[2], full_ind[3], full_ind[4]])
        if len(self.keys) == 1: # multiply by two for spin
            spec_icohp = spec_icohp*2
        print("COHP!",spec_icohp)

        return spec_icohp.real # should be real already but stored in complex array

    @staticmethod
    def get_COOP(self: object, orbs: dict, NN: int=None, include_onsite: bool=False, spin: int=0) -> npt.NDArray:
        """
        Calculates the COHP for the given orbitals and nearest neighbors.

        Args:    
            self (object): An object of the class COGITO_BAND or COGITO_UNIFORM.
            orbs (dict): Either a list of two dictionaries giving elements as keys and orbital types as items (eg [{"Pb":["s","d"],"O":["s","p"]},{"Pb":["s"]"O":["p"]}]) or give list of orb numbers [[1,2,3,5,6,7],[1,2,3,4,5,6,7,8]].
            NN (int): An integer for which nearest neighbor number to include (eg 1 for 1NN) or None or "All" for all nearest neighbors.
            include_onsite (bool): Includes atomic orbital energy terms (H_ab(R) where R=0 and a=b) instead of just bonding terms.

        Returns:    
            npt.NDArray: Returns COHP values in a [kpt,band] dimension.
        """

        if type(orbs[0]) is dict:
            firstorbs = []
            secorbs = []
            if type(list(orbs[0].keys())[0]) is str:
                # format of orbs should be {"Si":["s","p"],"Pb":["s"]}
                # find orbs for the element
                for element in orbs[0].keys():
                    #print(element)
                    for orbtype in orbs[0][element]:
                        for addorb in range(self.num_orbs):
                            if self.orbatomname[addorb] == element and orbtype in self.exactorbtype[addorb]:
                                firstorbs.append(addorb)

                for element in orbs[1].keys():
                    for orbtype in orbs[1][element]:
                        for addorb in range(self.num_orbs):
                            if self.orbatomname[addorb] == element and orbtype in self.exactorbtype[addorb]:
                                secorbs.append(addorb)
            else:  # type is int and should be referring to the specific atom number in prim cell
                for atm in orbs[0].keys():
                    print(atm)
                    for orbtype in orbs[0][atm]:
                        for addorb in range(self.num_orbs):
                            if self.orbatomnum[addorb] == atm and orbtype in self.exactorbtype[addorb]:
                                firstorbs.append(addorb)

                for atm in orbs[1].keys():
                    for orbtype in orbs[1][atm]:
                        for addorb in range(self.num_orbs):
                            if self.orbatomnum[addorb] == atm and orbtype in self.exactorbtype[addorb]:
                                secorbs.append(addorb)

            orbs = [firstorbs, secorbs]
            #print(orbs)

        # reference indices based on the nearest neighbors, much faster and can look at specific NN contributions
        num_each_dir = self.num_each_dir
        #if not hasattr(self, 'NN_index'):
        #    self.get_neighbors()
        if NN == None or NN == "All":
            indices = np.array([], dtype=np.int_)
            for nn in range(len(self.NN_dists)):
                indices = np.append(indices, self.NN_index[nn])
            total_index = indices
            if self.verbose > 0:
                print("NN distances that are included:", self.NN_dists)
        else:
            if int(NN) > len(self.NN_dists):
                print("give a lower NN integer!")
            NN_dist = self.NN_dists[int(NN)]
            total_index = self.NN_index[int(NN)]
            if self.verbose > 0:
                print("NN number, distance, count, and indices:", int(NN), NN_dist, len(total_index))  # ,NN_index)

        # convert the flattens indices into the full matrix indices
        full_indices = np.unravel_index(total_index, (
        self.num_orbs, self.num_orbs, self.num_trans[0], self.num_trans[1], self.num_trans[2]))
        # only select terms which have the orbitals the user is seeking
        orb1_in_full = full_indices[0][None, :] == np.array([orbs[0]]).T
        orb2_in_full = full_indices[1][None, :] == np.array([orbs[1]]).T
        in_full1 = np.full(len(orb1_in_full[0]), False)
        in_full2 = np.full(len(orb2_in_full[0]), False)
        for bool_list in orb1_in_full:
            in_full1 = in_full1 | bool_list
        for bool_list in orb2_in_full:
            in_full2 = in_full2 | bool_list
        in_full = in_full1 & in_full2
        # remove onsite terms
        is_onsite = (full_indices[2] == num_each_dir[0]) & (full_indices[3] == num_each_dir[1]) & (
                    full_indices[4] == num_each_dir[2]) & (full_indices[0] == full_indices[1])
        # print(in_full)
        if include_onsite == False:
            in_full = in_full & np.logical_not(is_onsite)
        # print(in_full)
        full_ind = [full_indices[0][in_full], full_indices[1][in_full], full_indices[2][in_full],
                    full_indices[3][in_full], full_indices[4][in_full]]
        if self.verbose > 0:
            print("num of params:", len(full_ind[0]))
        #spec_icoop = 0
        #for key in self.keys: # plot bands for each spin
        key = spin
        spec_icoop = np.sum(self.ICOOP[key][full_ind[0], full_ind[1], full_ind[2], full_ind[3], full_ind[4]])
        #if len(self.keys) == 1: # multiply by two for spin
        #    spec_icoop = spec_icoop*2
        print("COOP!",spec_icoop)

        return spec_icoop.real # should be real already but stored in complex array

    @staticmethod
    def get_mulliken_charge(self: object,elem: str, only_onsite=False) -> float:
        """
        Returns the atomic partial charge by Mulliken partitioning. If the calculation is spin polarized, also returns the atomic magnetization by Mulliken.
        If the element label (str) is used, the code returns the average charge/magnetization for the element type.

        Args:
            elem (str or int): Either the element label (e.g. "Pb") or the index of the atom in the list of atoms (as seen in tb_input.txt).
                                When is the element label, the function will return a value for each atom of that type.
            only_onsite (bool): Default False. If True, don't actually do the Mulliken repartitioning from bonds, only include onsite charges.
                                Note if True: Summing only atoms with onsite charge will not be charge neutral, need to include bond charge for neutrality.

        Returns:
            (float or tuple): Atomic partial charge or if spin polarized, (partial charge, atomic magnetic moment).
                              If elem is element label (e.g. "Pb"), returns a list for charge and magmom.
        """
        if not hasattr(self, 'ICOOP'):
            self.get_ICOOP()
        icoop_orb = self.ICOOP
        each_dir = self.ICO_eachdir

        # find the orbitals to use
        atm_orbs = []
        if type(elem) is str:
            atm_nums = np.arange(len(self.elements))[self.elements == elem]
            for atmnum in atm_nums:
                numatmorbs = len(np.arange(self.num_orbs)[self.orbatomnum == atmnum])
                atm_orbs.append(np.arange(self.num_orbs)[self.orbatomnum == atmnum])
        else: # type is int
            atm_nums = [elem]
            for atmnum in atm_nums:
                numatmorbs = len(np.arange(self.num_orbs)[self.orbatomnum == atmnum])
                atm_orbs.append(np.arange(self.num_orbs)[self.orbatomnum == atmnum])
        atm_orbs = np.array(atm_orbs)
        icoop_orb = self.ICOOP
        onsite_occup = np.zeros((len(self.keys),len(atm_nums), numatmorbs))
        mulli_occup = np.zeros((len(self.keys),len(atm_nums), numatmorbs))
        for key in self.keys:
            for i in range(len(atm_nums)):
                orbs = atm_orbs[i]
                onsite_occup[key][i] = np.sum(icoop_orb[key][orbs][:,orbs,each_dir[0],each_dir[1],each_dir[2]], axis=1)
                mulli_occup[key][i] = np.sum(icoop_orb[key][orbs], axis=(1, 2, 3, 4))
        if only_onsite:
            mulli_occup = onsite_occup

        electron_occ = np.zeros(len(atm_nums))
        for key in self.keys:
            electron_mag = electron_occ - np.sum(mulli_occup[key],axis=1) # difference between spin = 0 and spin = 1
            electron_occ = electron_occ +  np.sum(mulli_occup[key],axis=1)

        if len(self.keys) == 1: # multiply by two for spin degeneracy
            electron_occ = electron_occ*2
            electron_mag[:] = 0 # no magnetism

        if not (type(elem) is str): # not return a single value instead of a length 1 list
            electron_occ = electron_occ[0]
            electron_mag = electron_mag[0]

        mulk_charge = np.around(self.elec_count[atm_nums[0]] - electron_occ,decimals=6).tolist()
        electron_mag = np.around(electron_mag,decimals=6).tolist()

        if len(self.keys) == 1:
            return mulk_charge
        else: # has spin and mag
            return mulk_charge, electron_mag

def _cart_to_red(tmp,cart):
    """
    Convert cartesian vectors cart to reduced coordinates of a1,a2,a3 vectors
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
        prim_vec (unknown): three float tuples representing the primitive vectors
        prim_coord (unknown): list of float tuples for primitive coordinates

    Returns:    
        unknown: list of float tuples for cartesian coordinates
                        ex: cart_coord = _red_to_cart((a1,a2,a3),prim_coord)
    """
    (a1,a2,a3)=prim_vec
    prim = np.array(prim_coord)
    # cartesian coordinates
    cart=np.zeros_like(prim_coord,dtype=float)
    #for i in range(0,len(cart)):
    cart = [a1]*np.array([prim[:,0]]).T + [a2]*np.array([prim[:,1]]).T + [a3]*np.array([prim[:,2]]).T
    return cart

def func_for_rad(x, a,b,c,d,e,f,g,h,l):
    return x**l * (a/b*np.exp(-1/2*(x/b)**2)+c/d*np.exp(-1/2*(x/d)**2)+e/f*np.exp(-1/2*(x/f)**2)+g/h*np.exp(-1/2*(x/h)**2))

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
