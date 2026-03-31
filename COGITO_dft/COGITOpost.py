#!/usr/bin/env python3

import numpy as np
import numpy.typing as npt
import pymatgen.core.structure as struc
import pymatgen.symmetry.kpath as kpath
from pymatgen.io.vasp.outputs import Eigenval
#import matplotlib
#matplotlib.use("Qt4Agg")
import matplotlib.pyplot as plt
import copy
import math

class COGITO_TB_Model(object):
    def __init__(self, directory: str, verbose: int = 0, file_suffix: str = "", orbs_orth: bool = False,spin_polar: bool = False) -> None:
        """
        Initializes the tight binding model from the tb_input.txt, TBparams.txt, and overlaps.txt created by COGITO.

        Args:    
            directory (str): The path for the input files.
            verbose (int): How much will be printed (0 is least).
            file_suffix (str): The suffix to the TBparams and overlaps files.
            orbs_orth (bool): Whether the orbitals are orthogonal, if from COGITO this is always False.
        """
        self.directory = directory
        self.verbose = verbose
        self.spin_polar = spin_polar
        self.read_input(file = "tb_input"+file_suffix+".txt")
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
            print("recip vecs:",b)

        cross_ab = np.cross(self._a[0], self._a[1])
        supercell_volume = abs(np.dot(cross_ab, self._a[2]))
        self.cell_vol = supercell_volume

        #read in params
        self.read_TBparams(file= "TBparams"+file_suffix+".txt")
        if orbs_orth == False:
            self.read_overlaps(file = "overlaps"+file_suffix+".txt")

        self.restrict_params(maximum_dist=50,minimum_value=0.00000001)

    def read_input(self,file: str = "tb_input.txt") -> None:
        """
        Reads the tb_input.txt for information about the unit cell, atoms, orbitals, fermi energy, energy shift, and spin polarization.
        """
        filename = self.directory + file
        # check if file exists, if not switch to untagged tb_input.txt
        from pathlib import Path
        if not Path(filename).is_file():
            print("The tb_input with specified tag does not exist, switching to untagged.")
            filename = self.directory + "tb_input.txt"
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

    def read_TBparams(self,file: str = "TBparams.txt") -> None:
        """
        Reads in the tight binding (hopping) parameters from a file formatted like wannier90_hr.dat.

        Args:
            file (str): The file name for the TB parameters.
        """
        if self.orbs_orth == True:
            file = "orth_tbparams.txt"
        original_name = self.directory + file
        orig_prefix = original_name.split(".t")[0]
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
        self.num_each_dir = num_each_dir
        self.num_orbs = num_orbs
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
        self.num_trans = num_trans
        #print(num_trans)
        #generate list of the displacement between translations
        vec_to_trans = np.zeros((num_trans[0],num_trans[1],num_trans[2],3))
        for x in range(num_trans[0]):
            for y in range(num_trans[1]):
                for z in range(num_trans[2]):
                    vec_to_trans[x,y,z] = [x-num_each_dir[0],y-num_each_dir[1],z-num_each_dir[2]]
        self.vec_to_trans = vec_to_trans
        atomic_energies = {}
        TB_params = {}
        for key in list(all_files.keys()):
            cur_lines = all_files[key]
            atomic_energies[key] = []
            #read in the TB parameters
            TB_params[key] = np.zeros((num_orbs,num_orbs,num_trans[0],num_trans[1],num_trans[2]), dtype=np.complex128)
            for line in cur_lines[first_line:]:
                info = line.split()
                if abs(int(info[0])) <=num_each_dir[0] and abs(int(info[1])) <=num_each_dir[1] and abs(int(info[2])) <=num_each_dir[2]:
                    trans1 = int(info[0])+num_each_dir[0]
                    trans2 = int(info[1])+num_each_dir[1]
                    trans3 = int(info[2])+num_each_dir[2]
                    orb1 = int(info[3])-1
                    orb2 = int(info[4])-1
                    value = float(info[5]) + float(info[6])*1.0j
                    if TB_params[key][orb1,orb2,trans1,trans2,trans3] != 0:
                        print("already set TB param")
                    #only set if orbitals are not on the same atom
                    same_atom = np.abs(np.array(self.orb_redcoords[orb1]) - np.array(self.orb_redcoords[orb2]))
                    #print(same_atom)

                    TB_params[key][orb1, orb2, trans1, trans2, trans3] = value
                    if (same_atom < 0.001).all() and (int(info[0])==0 and int(info[1])==0 and int(info[2])==0) and orb1!=orb2 and abs(value) > 0.0001:
                        #print("Same atom orbital hopping term that should be zero!", orb1, orb2, value)
                        #TB_params[key][orb1,orb2,trans1,trans2,trans3] = 0
                        pass
                    if (int(info[0])==0 and int(info[1])==0 and int(info[2])==0) and orb1==orb2:
                        #print("onsite term:",value)
                        atomic_energies[key].append(value)
        if self.verbose > -1:
            print("the atomic orbital energies!",atomic_energies)
        #if self.spin_polar:
        self.atomic_energies = atomic_energies
        self.TB_params = TB_params
        #else: # pass just the parameters instead of a dictionary with one key/item
        #    self.atomic_energies = atomic_energies[0]
        #    self.TB_params = TB_params[0]

    def read_overlaps(self,file: str = "overlaps.txt") -> None:
        """
        Reads in the orbital overlaps from a file formatted like wannier90_hr.dat.

        Args:
            file (str): The file name for the overlaps.
        """
        original_name = self.directory + file
        filename = original_name
        orig_prefix = original_name.split(".t")[0]
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

        if self.orbs_orth == True:
            first_line = 100 #3+num_orbs #12
        else:
            first_line = 3
        num_orbs = self.num_orbs
        num_trans = self.num_trans
        num_each_dir = np.array((self.num_trans-1)/2,dtype=np.int_)
        #print(num_trans)
        #generate list of the displacement between translations
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
                        #print("Same atom orbital overlap that should be zero!", orb1, orb2, value)
                        #TB_params[orb1,orb2,trans1,trans2,trans3] = 0
                        pass
                    if (int(info[0])==0 and int(info[1])==0 and int(info[2])==0) and orb1==orb2:
                        #print("onsite term:",value)
                        atomic_overlaps[key].append(value)
        if self.verbose > 1:
            print("the atomic orbital self overlaps!",atomic_overlaps)
        #if self.spin_polar:
        self.overlaps_params = overlaps_params
        #else:
        #    self.overlaps_params = overlaps_params[0]

    def read_orbitals(self,file: str = "orbitals.npy") -> None:
        """
        This function reads in the orbitals as coefficents for a gaussian expansion.
        The information in 'orbitals.npy' is combined with the orbital data in 'tb_input.txt'.

        Args:    
            file (str): Orbital file.
        """

        orbital_coeffs = np.load(self.directory + "orbitals"+self.file_suff+".npy")[:,:-1]
        sph_harm_key = np.load(self.directory + "orbitals"+self.file_suff+".npy")[:,-1]
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
            #gaus_norm = factorial2(1+2*l)*np.pi**(1/2)*2**(-2-l)*np.sum(fnt_coef[:,None]*fnt_coef[None,:]*(exp_coef[:,None]+exp_coef[None,:])**(-3/2-l))
            #fnt_coef = np.array([oa, oc, oe, og])/gaus_norm**(1/2)
            #exp_coef = np.array([ob, od, of, oh])
            avg_rad = 1 / 2 * math.factorial(1 + int(l)) * np.sum(fnt_coef[:, None] * fnt_coef[None, :] * (exp_coef[:, None] + exp_coef[None, :]) ** (-2 - l/2))  # analytical calc
            orb_radii[orb] = avg_rad

            # convert back to a/b*e^-1/2(x/b)^2 format from a*e^-bx^2 format
            [b, d, f, h] = 1 / (2 * np.array([ob, od, of, oh])) ** (1 / 2)  # 1/2/np.array([b,d,f,h])**2
            [a, c, e, g] = [oa * b, oc * d, oe * f, og * h]

            real_orbcoeffs[orb] = np.array([a, b, c, d, e, f, g, h, l])

        self.real_orbcoeffs = real_orbcoeffs
        self.orb_radii = orb_radii

    def normalize_params(self):
        """
        This function correctly normalized the overlap and hopping parameters such that the diagonal of overlap is one.
        i.e. overlap[orb1=orb2,trans123=0]=1.
        While the COGITO process gets nearly correctly normalized orbitals (0.99-1.01), generated the TB model from VASP psuedo is not correctly normalized.
        Running this function ensures that comparison between models of different types and normalizations is correct.
        """
        self.overlaps_params #[spin][orb1, orb2, trans1, trans2, trans3]
        if self.spin_polar:
            keys = [0,1]
        else:
            keys = [0]
        norms_overlaps = {}
        norms_tbparams = {}
        for key in keys:
            #norms_overlaps[key] = np.zeros((num_orbs,num_orbs,num_trans[0],num_trans[1],num_trans[2]), dtype=np.complex128)
            #norms_tbparams[key] = np.zeros((num_orbs,num_orbs,num_trans[0],num_trans[1],num_trans[2]), dtype=np.complex128)
            orb_norms = np.diag(self.overlaps_params[key][:,:,self.num_each_dir[0],self.num_each_dir[1],self.num_each_dir[2]])
            print("orbital norms:",orb_norms)
            print("setting orbital norms to 1")
            norms_overlaps[key] = self.overlaps_params[key] / orb_norms[:,None,None,None,None]**(1/2) / orb_norms[None,:,None,None,None]**(1/2)
            norms_tbparams[key] = self.TB_params[key] / orb_norms[:,None,None,None,None]**(1/2) / orb_norms[None,:,None,None,None]**(1/2)
            new_norms = np.diag(norms_overlaps[key][:,:,self.num_each_dir[0],self.num_each_dir[1],self.num_each_dir[2]])
            #atomic_energies = np.diag(self.TB_params[key][:,:,self.num_each_dir[0],self.num_each_dir[1],self.num_each_dir[2]])
            print("new norms:",new_norms)
            #print("new atomic energies:",atomic_energies)

        self.overlaps_params = norms_overlaps
        self.TB_params = norms_tbparams

        for key in keys:
            atomic_energies = np.diag(self.TB_params[key][:,:,self.num_each_dir[0],self.num_each_dir[1],self.num_each_dir[2]])
            #print("new norms:",new_norms)
            print("new atomic energies:",atomic_energies)
            self.atomic_energies[key] = atomic_energies


    def set_hoppings(self, value: float ,orb1: int, orb2: int, trans: tuple,spin: int = 0) -> None:
        """
        Change a TB parameter.

        Args:    
            value (float): The new parameter.
            orb1 (int): The first orbital index of the parameter.
            orb2 (int): The second orbital index of the parameter.
            trans (tuple): The tuple of translation indices.

        Returns:    
            None: 
        """
        (trans1,trans2,trans3) = trans
        self.TB_params[spin][orb1,orb2,trans1,trans2,trans3] = value

    def restrict_params(self,maximum_dist: float = 12.0, minimum_value: float = 0.0001) -> None:
        """
        Generates self.use_tbparams, self.use_overlaps, and self.use_vecs_to_orbs which are used in the gen_ham() funciton.
        With this, the calculation of hamiltonians by gen_ham() is both sparse and vectorized.

        Args:    
            maximum_dist (float): The maximmum distance between hopping parameters which should be included.
            minimum_value (float): The minimum magnitude of hopping parameter which should be included.
        """

        vecs = np.array(self.vec_to_trans)
        orb_pos = np.array(self.orb_redcoords)
        #orb_pos[:] = 0
        #print("orb pos", orb_pos)
        vec_to_orbs = vecs[None, None, :, :, :] + orb_pos[None, :, None, None, None] - orb_pos[:, None, None, None, None]  # [orb1,orb2,t1,t2,t3]
        # print("shape of vec:",vec_to_orbs.shape)
        hold_vecs = np.reshape(vec_to_orbs, (self.num_orbs * self.num_orbs * self.num_trans[0] * self.num_trans[1] * self.num_trans[2], 3)).transpose()
        cart_hold_vecs = _red_to_cart((self._a[0], self._a[1], self._a[2]), hold_vecs.transpose())
        dist_to_orbs = np.linalg.norm(cart_hold_vecs, axis=1)
        total_trans = self.num_trans[0] * self.num_trans[1] * self.num_trans[2]
        orb_orb_distance = np.reshape(dist_to_orbs,(self.num_orbs, self.num_orbs,total_trans))
        absTB = np.reshape(np.abs(self.TB_params[0]),(self.num_orbs, self.num_orbs,total_trans))

        self.dist_to_orbs = dist_to_orbs
        self.flat_vec_to_orbs = hold_vecs
        self.vecs_to_orbs = vec_to_orbs

        orb_orb_indices = np.zeros((self.num_orbs,self.num_orbs),dtype=np.object_)
        max_num = 0
        for orb1 in range(self.num_orbs):
            for orb2 in range(self.num_orbs):
                indices = np.arange(total_trans)[ (orb_orb_distance[orb1,orb2]<=maximum_dist) & (absTB[orb1,orb2] >= minimum_value)]
                tran123_indices =  np.array(np.unravel_index(indices,(self.num_trans[0], self.num_trans[1], self.num_trans[2])))
                orb_orb_indices[orb1,orb2] = tran123_indices
                #print("num params included:",len(tran123_indices[0]))
                num_inc = len(tran123_indices[0])
                if num_inc > max_num:
                    max_num = num_inc

        if self.spin_polar:
            keys = [0,1]
        else:
            keys = [0]
        new_tbparams = {}
        new_overlapparams = {}
        for key in keys:

            new_tbparams[key] =  np.zeros((self.num_orbs,self.num_orbs,max_num),dtype=np.complex128)
            new_overlapparams[key] =  np.zeros((self.num_orbs,self.num_orbs,max_num),dtype=np.complex128)
            new_vecs_to_orbs = np.zeros((self.num_orbs,self.num_orbs,max_num,3),dtype=np.complex128)
            for orb1 in range(self.num_orbs):
                for orb2 in range(self.num_orbs):
                    old_ind = orb_orb_indices[orb1,orb2]
                    new_tbparams[key][orb1,orb2][:len(old_ind[0])] = self.TB_params[key][orb1,orb2][old_ind[0],old_ind[1],old_ind[2]]
                    new_overlapparams[key][orb1,orb2][:len(old_ind[0])] = self.overlaps_params[key][orb1,orb2][old_ind[0],old_ind[1],old_ind[2]]
                    new_vecs_to_orbs[orb1,orb2][:len(old_ind[0])] = vec_to_orbs[orb1,orb2][old_ind[0],old_ind[1],old_ind[2]]

        self.use_tbparams = new_tbparams
        self.use_overlaps = new_overlapparams
        self.use_vecs_to_orbs = new_vecs_to_orbs

    @staticmethod
    def make_orbitals(self, cartXYZ):
        if not hasattr(self, 'real_orbcoeffs'):
            self.read_orbitals()

        #print("orbital radii:",self.orb_radii)
        cartXYZ = np.array(cartXYZ)
        old_center = np.array([0,0,0])
        orb_XYZ = copy.deepcopy(cartXYZ)
        # try out the recip orbital coeffs
        rad = np.linalg.norm(orb_XYZ, axis=0)
        rad[rad == 0] = 0.0000000001
        theta = np.arccos(orb_XYZ[2] / rad)
        theta[rad == 0.0000000001] = 0
        phi = np.arctan2(orb_XYZ[1], orb_XYZ[0])

        real_orbs = np.zeros((self.num_orbs, len(rad)))#, dtype=np.complex128)  # {}
        for orb in range(self.num_orbs):
            atomnum = self.orbatomnum[orb]
            center = np.array(self.cartAtoms[atomnum])
            if center[0] != old_center[0] or center[1] != old_center[1] or center[2] != old_center[2]:
                orb_XYZ = cartXYZ - center[:,None]
                # try out the recip orbital coeffs
                rad = np.linalg.norm(orb_XYZ, axis=0)
                rad[rad == 0] = 0.0000000001
                theta = np.arccos(orb_XYZ[2] / rad)
                theta[rad == 0.0000000001] = 0
                phi = np.arctan2(orb_XYZ[1], orb_XYZ[0])
                old_center = center

            real_coeff = self.real_orbcoeffs[orb]
            [a, b, c, d, e, f, g, h, l] = real_coeff
            orb_peratom = self.sph_harm_key[orb]
            angular_part = complex128funs(phi, theta, self.orb_vec[orb])
            radial_part = func_for_rad(rad, a, b, c, d, e, f, g, h, l)

            flat_real_orb = radial_part * angular_part

            # check what value is at avg rad
            slice_rad_value = self.orb_radii[orb] # The radius where the slice is taken
            slice_tolerance = 0.1  # Tolerance for the slice

            # Filter points close to the slicing plane
            slice_indices = np.abs(rad - slice_rad_value) <= slice_tolerance
            iso_cut = np.amax(np.abs(flat_real_orb))/2
            print("iso cut",iso_cut)

            # structure constant
            #const = ((2 * np.pi) ** 3 / self.vol) ** (1 / 2)
            real_orbs[orb] = flat_real_orb #* const

        return real_orbs

    @staticmethod
    def plot_orbitals(self):
        gridnum = np.around(np.array(self.abc)*10*2,decimals=0)
        gridnum = [int(gridnum[0]),int(gridnum[1]),int(gridnum[2])]
        print("grid num",gridnum)
        X, Y, Z = np.mgrid[-0.5:(1.5 - 2 / gridnum[0]):gridnum[0] * 1j, -0.5:(1.5 - 2 / gridnum[1]):gridnum[1] * 1j, -0.5:(1.5 - 2 / gridnum[2]):gridnum[2] * 1j]
        self.X = X
        self.Y = Y
        self.Z = Z

        print(X.flatten(), Y.flatten(), Z.flatten())
        primXYZ = np.array([X.flatten(), Y.flatten(), Z.flatten()])
        print(primXYZ[0], primXYZ[1], primXYZ[2])
        print(self._a)
        cartXYZ = _red_to_cart((self._a[0], self._a[1], self._a[2]),
                                    np.array([X.flatten(), Y.flatten(), Z.flatten()]).T).T
        print(cartXYZ[0], cartXYZ[1], cartXYZ[2])


        orbitals = self.make_orbitals(self,cartXYZ)
        print(self.orb_radii)
        print(orbitals)

        import matplotlib.pyplot as plt
        import plotly.graph_objects as go
        import plotly
        from skimage.measure import marching_cubes
        print("plotly version:",plotly.__version__)

        for orb in range(self.num_orbs):
            orbital = orbitals[orb].flatten()

            '''
            atomnum = self.orbatomnum[orb]
            center = np.array(self.cartAtoms[atomnum])
            # Define the slicing plane
            slice_x_value = center[0]  # The x-coordinate where the slice is taken
            slice_tolerance = 0.1  # Tolerance for the slice

            # Filter points close to the slicing plane
            slice_indices = np.abs(cartXYZ[0, :] - slice_x_value) <= slice_tolerance
            slice_data = [cartXYZ[1,slice_indices],cartXYZ[2,slice_indices]]
            slice_orbital = orbital[slice_indices]

            # Plot the slice
            plt.figure(figsize=(8, 6))
            scatter = plt.scatter(slice_data[0], slice_data[1], c=slice_orbital, cmap='viridis', s=5)
            plt.colorbar(scatter, label='Orbital Value')
            plt.xlabel('Y')
            plt.ylabel('Z')
            plt.title(f"Slice of Orbital Data at X = {slice_x_value}")
            #plt.show()
            '''
            # Plot the isosurface
            isoval = np.amax(np.abs(orbital))/2
            print("plotting vals:",isoval)
            print("Orbital range:", orbital.min(), orbital.max())
            data = []
            x,y,z = [cartXYZ[0].flatten(),cartXYZ[1].flatten(),cartXYZ[2].flatten()]

            verts, faces, _, _ = marching_cubes(np.reshape(orbital,(gridnum[0],gridnum[1],gridnum[2])), isoval, spacing=(1 / gridnum[0], 1 / gridnum[1], 1 / gridnum[2]))
            new_verts = np.array([verts[:, 0], verts[:, 1], verts[:, 2]]).transpose()
            cart_verts = _red_to_cart((self._a[0], self._a[1], self._a[2]), new_verts)

            # Step 4: Plot the isosurface using Plotly
            data = []
            data.append(go.Mesh3d(
                x=cart_verts[:, 0],
                y=cart_verts[:, 1],
                z=cart_verts[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                opacity=0.5,
                color='green',
                colorscale='Viridis'
            ))

            if np.amin(orbital) < -isoval:
                neg_verts, neg_faces, _, _ = marching_cubes(np.reshape(orbital,(gridnum[0],gridnum[1],gridnum[2])), -isoval, spacing=(1 / gridnum[0], 1 / gridnum[1], 1 / gridnum[2]))
                new_verts = np.array([neg_verts[:, 0], neg_verts[:, 1], neg_verts[:, 2]]).transpose()
                neg_cart_verts = _red_to_cart((self._a[0], self._a[1], self._a[2]), new_verts)

                data.append(go.Mesh3d(
                    x=neg_cart_verts[:, 0],
                    y=neg_cart_verts[:, 1],
                    z=neg_cart_verts[:, 2],
                    i=neg_faces[:, 0],
                    j=neg_faces[:, 1],
                    k=neg_faces[:, 2],
                    opacity=0.5,
                    color='pink',
                    colorscale='Viridis'
                ))

            fig = go.Figure(data=data)
            fig.update_layout(margin=dict(l=0, r=0, b=0, t=0),
                              paper_bgcolor='rgba(0,0,0,0)',  # Transparent background for the whole figure
                              plot_bgcolor='rgba(0,0,0,0)')  # ,hovermode="x unified")
            # Define the ranges for X, Y, Z axes
            x_range = [cartXYZ[0].min(), cartXYZ[0].max()]
            y_range = [cartXYZ[1].min(), cartXYZ[1].max()]
            z_range = [cartXYZ[2].min(), cartXYZ[2].max()]
            #fig.update_layout(
            #    scene=dict(xaxis=dict(showbackground=False, showspikes=False, showticklabels=False, visible=False),
            #               yaxis=dict(showbackground=False, showspikes=False, showticklabels=False, visible=False),
            #               zaxis=dict(showbackground=False, showspikes=False, showticklabels=False, visible=False),
            #               aspectmode='data'))
            fig.update_layout(title="Isosurface of Orbital Data",scene=dict(aspectmode='data' ))
            if self.show_figs:
                fig.show()

    @staticmethod
    def generate_gpnts(self,kpt: list) -> npt.NDArray[np.int_]:
        """
        Similar to from pymatgen.io.vasp.outputs.Wavecar but is vectorized.

        Args:    
            kpt (list): The k-point in reduced coordinates.

        Returns:    
            npt.NDArray[np.int_]: The gpoints
        """
        _C = 0.262465831
        encut = 520
        # calculate reciprocal lattice

        _nbmax = [0,0,0]
        _nbmax[0] = int(np.around(np.linalg.norm(self._a[0])*2,decimals=0))
        _nbmax[1] = int(np.around(np.linalg.norm(self._a[1])*2,decimals=0))
        _nbmax[2] = int(np.around(np.linalg.norm(self._a[2])*2,decimals=0))
        #print(_nbmax)

        i3,j2,k1 = np.mgrid[-_nbmax[2]:_nbmax[2]:(2 * _nbmax[2] + 1)*1j,-_nbmax[1]:_nbmax[1]:(2 * _nbmax[1] + 1)*1j,
                   -_nbmax[0]:_nbmax[0]:(2 * _nbmax[0] + 1) * 1j]
        i3 = i3.flatten()
        j2 = j2.flatten()
        k1 = k1.flatten()
        G = np.array([k1, j2, i3])
        v = np.array(np.array([kpt]).T + G).T
        g = np.linalg.norm(np.dot(v, self._b),axis=1)
        E = np.array( g ** 2 / _C )
        gpoints1 = G[0][E<encut]
        gpoints2 = G[1][E<encut]
        gpoints3 = G[2][E<encut]
        gpoints = np.array([gpoints1,gpoints2,gpoints3],dtype=np.int_).T
        return gpoints

    @staticmethod
    def get_ham(self: object,kpt: list,return_overlap: bool = False,return_truevec: bool = True,spin: int = 0) -> list:
        """
        This function generates the hamiltonian and overlap matrices for a given kpt.
        Then it solves the generalized eigenvalue problem to return the eigvalues and vectors.

        Args:    
            self (object): An object with the attributes of the COGITO_TB_Model class.
            kpt (list): The kpoint to regenerate at in reduced coordinates.
            return_overlap (bool): If True, the function will also return the overlap matrix at the kpt. Default is False.
            return_truevec (bool): If True, the function will return the eigenvectors in the original nonorthogonal basis.
                        Default if True.
            spin (int): The spin of the parameters for a spin-polarized calculation. Default is 0--for non spin-polarized.

        Returns:    
            list: Returns a list of the eigenvalues and eigenvectors (and overlap if return_overlap=True).
        """

        # get the overlap and hamiltonian matrices using arrays defined in the function restrict_params()
        exp_fac = np.exp(2j * np.pi * np.matmul(self.use_vecs_to_orbs,kpt))
        kdep_overlap = np.sum(self.use_overlaps[spin] * exp_fac,axis=2)
        ham = np.sum(self.use_tbparams[spin] * exp_fac,axis=2)

        # diagonalize the hamiltonian
        if self.orbs_orth == True:
            (eigval, eigvec) = np.linalg.eigh(ham)
            true_eigvec = eigvec

        # get eigenvalues and vectors from generalized eigenvalue problem
        else:
            kdep_Sij = kdep_overlap
            #self.Sij[kpt] = kdep_Sij
            #print("check overlap:",kdep_Sij)

            eigenvalj, kdep_Dij = np.linalg.eigh(kdep_Sij)
            # check correctness of eigen
            # construct Aij
            kdep_Aij = np.zeros((self.num_orbs, self.num_orbs), dtype=np.complex128)
            #for j in range(self.num_orbs):
            kdep_Aij[:, :] = kdep_Dij[:, :] / (eigenvalj[None,:]) ** (1 / 2)
            #get ham for orthogonal eigenvectors
            conj_Aij = np.conj(kdep_Aij).transpose()
            AtHA = np.matmul(conj_Aij,np.matmul(ham,kdep_Aij))
            AtHA = (AtHA + np.conj(AtHA).T)/2 # ensure that the hamiltonian is perfectly Hermitian
            #ham = np.matmul(np.linalg.inv(conj_Aij),np.matmul(AtHA,np.linalg.inv(kdep_Aij)))
            eigval,eigvec = np.linalg.eigh(AtHA)
            if return_truevec == True:
                true_eigvec = np.matmul(kdep_Aij,eigvec)
            else:
                true_eigvec = eigvec
                
            # test that you can reconstruct the hamiltonian from eigenvecs
            test_AtHA = np.matmul(eigvec,np.matmul(np.diag(eigval), np.linalg.inv(eigvec)))
            #newAtHA = (AtHA + np.conj(AtHA).T)/2
            diff_AtHA = AtHA - test_AtHA
            if (diff_AtHA.flatten() > 0.00001).any():
                print("couldn't construct the same ham!",diff_AtHA.flatten()[diff_AtHA.flatten() > 0.00001])
            #else:
            #    print("success")

        if return_overlap == True and self.orbs_orth == False:
            return eigval, true_eigvec, kdep_Sij#Aij, ham
        else:
            return eigval,true_eigvec
    @staticmethod
    def get_fullHam(self: object, kpt: list, spin: int = 0):
        """
        Find and return the full overlap and hamiltonian matrices (orb1,orb2,T1,T2,T3) but scaled with the phase of a specific k-point.

        Args:
            self (object): An object with the attributes of the COGITO_TB_Model class.
            kpt (list): The kpoint to regenerate at in reduced coordinates.
            spin (int): The spin of the parameters for a spin-polarized calculation. Default is 0--for non spin-polarized.
        @return:
        """

        exp_fac = np.exp(2j * np.pi * np.matmul(self.vecs_to_orbs,kpt))
        print("shape exp_fac:",exp_fac.shape)
        scaledTB = self.TB_params[spin] * exp_fac
        return scaledTB


    @staticmethod
    def get_neighbors(self) -> list:
        """
        This sorts the matrix of TB parameters into terms which are 1NN, 2NN, etc.
        Relevant for labeling equivalent bonds and creates NN numbers which is used in COOP and COHP plots.
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

    @staticmethod
    def plot_crystal_field(self, atomnum: int = 0, orbitals: str = "", ylim: list = (-10,0),spin: int = 0) -> None:
        """
        Plots the crystal field splitting diagram for the orbitals and atom given.
        Add stuff to plot the orbital type and orbital radius on the energy line (in pretty small font)
        Also have default to plot all the orbital on the atom.

        Args:    
            atomnum (int): Which atom to plot for.
            orbitals (str): Which orbitals to plot. If not set shows all the orbitals.
            ylim (list): The limits of the y-axis in the plot, will default to good value if left (-10,0).
        """
        if self.spin_polar:
            keys = [0,1]
        else:
            keys = [0]


        orb_energies = {}
        spec_orb_rad = {}
        for key in keys:

            # get the orbital radii
            import json
            spin_tag = ""
            if self.spin_polar:
                spin_tag = str(key)
            with open(self.directory + "orb_converg_info" + self.file_suff + "" + spin_tag + ".json", "r") as f:
                orb_info = json.load(f)
            all_orb_rad = np.array(orb_info["orb_rad"])

            on_atm = self.orbatomnum == atomnum
            atom_energy = np.around(self.atomic_energies[key],decimals=4).real[on_atm]
            atom_orbtype = np.array(self.exactorbtype)[on_atm]
            orb_energy = []
            spec_type = []
            spec_orb_rad[key] = []
            for i in range(len(atom_orbtype)):
                if orbitals in atom_orbtype[i]:
                    orb_energy.append(atom_energy[i])
                    print(atom_orbtype[i])
                    spec_type.append(atom_orbtype[i][0]+" ")
                    spec_orb_rad[key].append(str(np.around(all_orb_rad[on_atm][i],decimals=2)))
            orb_energies[key] = orb_energy
        print(orb_energies)
        if self.spin_polar:
            energies = np.append(orb_energies[0],orb_energies[1])
            types = np.array(spec_type,dtype='<U4')
            rads_up = np.array(spec_orb_rad[0],dtype='<U4')
            rads_down = np.array(spec_orb_rad[1],dtype='<U4')
            up = np.array("↑ ",dtype='<U4')
            down = np.array("↓ ",dtype='<U4')
            orbital_types = np.append(np.char.add(np.char.add(types,up),rads_up),np.char.add(np.char.add(types,down),rads_down))
        else:
            energies = np.array(orb_energies[0])
            rads_up = np.array(spec_orb_rad[0],dtype='<U4')
            orbital_types = np.char.add(np.array(spec_type),rads_up)
        print(energies)
        print(orbital_types)

        import re
        if self.spin_polar:
            fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(10, 8))
            # Unique orbital types to set y positions
            num_o = int(len(energies)/2)

            # do for spin up
            unique_energies,unique_ind,unique_count = np.unique(np.around(energies[:num_o],decimals=1),return_inverse=True,return_counts=True)
            print(unique_energies,unique_ind,unique_count)
            for dset in range(len(unique_energies)):
                set_ener = energies[:num_o][unique_ind==dset]
                unique_energies[dset] = energies[:num_o][unique_ind==dset][0]
                set_type = orbital_types[:num_o][unique_ind==dset]
                start = -0.2 - 0.22*(len(set_ener)-1)
                for dorb in range(len(set_ener)):
                    ax1.plot([start, start + 0.4], [set_ener[dorb], set_ener[dorb]], 'b-',color="black",lw=4)
                    label = set_type[dorb]
                    shift = 0.1*np.sign(start)
                    if start == -0.2:
                        shift = 0
                    ax1.text(start+0.2+shift,set_ener[dorb], label, ha='center', va='bottom',size=16)  # Label each line with orbital type
                    start = start + 0.44
            # plot splitting arrows
            low_en = unique_energies[0]
            add = 0.08
            for en in unique_energies[1:]:
                en_diff = en - low_en
                ax1.arrow(0.1, low_en, 0, en_diff, length_includes_head=True, width=0.01, head_width=0.035,
                          head_length=0.05, color="black")
                label = r"$" + str(np.around(np.abs(en_diff), decimals=2)) + "$" + "eV"
                ax1.text(0.12, low_en + en_diff / 2 - add, label, ha='left', va='center',
                         size=22)  # Label each line with orbital type
                low_en = en
                add = +0.08

            # do for spin down
            unique_energies,unique_ind,unique_count = np.unique(np.around(energies[num_o:],decimals=1),return_inverse=True,return_counts=True)
            print(unique_energies,unique_ind,unique_count)
            for dset in range(len(unique_energies)):
                set_ener = energies[num_o:][unique_ind==dset]
                unique_energies[dset] = energies[num_o:][unique_ind==dset][0]
                set_type = orbital_types[num_o:][unique_ind==dset]
                start = -0.2 - 0.22*(len(set_ener)-1)
                for dorb in range(len(set_ener)):
                    ax2.plot([start, start + 0.4], [set_ener[dorb], set_ener[dorb]], 'b-',color="black",lw=4)
                    label = set_type[dorb]
                    shift = 0.1*np.sign(start)
                    if start == -0.2:
                        shift = 0
                    ax2.text(start+0.2+shift,set_ener[dorb], label, ha='center', va='bottom',size=16)  # Label each line with orbital type
                    start = start + 0.44
            # plot splitting arrows
            low_en = unique_energies[0]
            add = 0.08
            for en in unique_energies[1:]:
                en_diff = en - low_en
                ax2.arrow(0.1, low_en, 0, en_diff, length_includes_head=True, width=0.01, head_width=0.035,
                          head_length=0.05, color="black")
                label = r"$" + str(np.around(np.abs(en_diff), decimals=2)) + "$" + "eV"
                ax2.text(0.12, low_en + en_diff / 2 - add, label, ha='left', va='center',
                         size=22)  # Label each line with orbital type
                low_en = en
                add = +0.08


            if ylim == (-10, 0):
                amin = np.amin(energies[energies > -20]) - 0.5
                amax = np.amax(energies) + 0.5
                ylim = (amin, amax)
            ax1.set_ylim(ylim)
            ax2.set_ylim(ylim)
            # Save the figure
            plt.savefig(self.directory + 'field_splitting' + str(atomnum) + orbitals + '.pdf')
            if self.show_figs:
                plt.show()

            plt.close()

        else:
            plt.figure(figsize=(5, 8))
            # Unique orbital types to set y positions
            unique_energies,unique_ind,unique_count = np.unique(np.around(energies,decimals=1),return_inverse=True,return_counts=True)
            print(unique_energies,unique_ind,unique_count)

            for dset in range(len(unique_energies)):
                set_ener = energies[unique_ind==dset]
                unique_energies[dset] = energies[unique_ind==dset][0]
                set_type = orbital_types[unique_ind==dset]
                start = -0.2 - 0.22*(len(set_ener)-1)
                for dorb in range(len(set_ener)):
                    plt.plot([start, start + 0.4], [set_ener[dorb], set_ener[dorb]], 'b-',color="black",lw=4)
                    label = set_type[dorb]

                    shift = 0.1*np.sign(start)
                    if start == -0.2:
                        shift = 0
                    plt.text(start+0.2+shift,set_ener[dorb], label, ha='center', va='bottom',size=16)  # Label each line with orbital type
                    start = start + 0.44

            # plot splitting arrows
            low_en = unique_energies[0]
            add= 0.08
            for en in unique_energies[1:]:
                en_diff = en-low_en
                plt.arrow(0.1,low_en,0,en_diff,length_includes_head=True,width=0.01,head_width=0.035,head_length=0.05,color="black")
                label = r"$" + str(np.around(np.abs(en_diff),decimals=2))+"$"+"eV"
                plt.text(0.12,low_en + en_diff/2-add, label, ha='left', va='center',size=22)  # Label each line with orbital type
                low_en = en
                add= +0.08
            if ylim == (-10,0):
                amin = np.amin(energies[energies>-20]) - 0.5
                amax = np.amax(energies) + 0.5
                ylim = (amin,amax)
            plt.ylim(ylim)
            # Save the figure
            plt.savefig(self.directory + 'field_splitting'+str(atomnum)+orbitals+'.pdf')
            if self.show_figs:
                plt.show()

            plt.close()

    @staticmethod
    def plot_hopping(self, spin: int = 0) -> None:
        """
        Plots the log(hopping) versus distance between parameters. Used to check quality of tight binding model.
        Parameters should show a rough linear decay on plot. Although, it is common for them to flatten out after ~10Ang.
        """
        #fig = plt.figure()
        actual_hoppings = copy.deepcopy(self.TB_params[spin])

        for orb in range(self.num_orbs):
            center = ((self.num_trans-1)/2).astype(int)
            actual_hoppings[orb][orb][center[0]][center[1]][center[2]] = 0
        hopping_strength = actual_hoppings.flatten().real
        hopping_distance = np.zeros((self.num_orbs,self.num_orbs,self.num_trans[0],self.num_trans[1],self.num_trans[2]))
        for orb1 in range(self.num_orbs):
            for orb2 in range(self.num_orbs):
                vec_to_orbs = self.vec_to_trans[:,:,:] + self.orb_redcoords[orb2] - self.orb_redcoords[orb1]
                hold_vecs = np.reshape(vec_to_orbs,(self.num_trans[0]*self.num_trans[1]*self.num_trans[2],3)).transpose()
                cart_hold_vecs = _red_to_cart((self._a[0], self._a[1], self._a[2]), hold_vecs.transpose())
                dist_to_orbs = np.linalg.norm(cart_hold_vecs,axis=1)
                #print(vec_to_orbs.shape,hold_vecs.shape,cart_hold_vecs.shape,dist_to_orbs.shape)
                dist_to_orbs = np.reshape(dist_to_orbs,(self.num_trans[0],self.num_trans[1],self.num_trans[2]))
                hopping_distance[orb1][orb2] = dist_to_orbs
        hopping_distance = hopping_distance.flatten()
        num_hops = len(hopping_distance)
        #print("hopping dist:",hopping_distance)
        #print("num hops:",num_hops)
        print("")
        print("Should see linear decay in plot below, indicating exponential decay of hoppings with distance.")
        print("The TB parameters shown below should also seem roughly proportional to the overlaps.")
        plot = hopping_strength > 10**(-6)
        #hopping_strength[hopping_strength <= 10**(-6)] = 10**(-6)

        #plt.get_current_fig_manager().window.raise_()
        plt.scatter(hopping_distance[plot],np.log10(np.abs(hopping_strength[plot])),c='steelblue')
        plt.xlim((0,20))
        plt.xlabel("Distance between orbitals (Å)")
        plt.ylabel("log(|TB parameter|)")
        plt.title("Log decay of hopping parameters")
        plt.tight_layout()
        plt.savefig(self.directory+'tbparams_decay.png',transparent=True,dpi=150,format='png')
        if self.show_figs:
            plt.show()
        plt.close()

    @staticmethod
    def plot_overlaps(self, spin: int = 0) -> None:
        """
        Plots the log(overlaps) versus distance between parameters. Used to check quality of tight binding model.
        Parameters should show a rough linear decay on plot. Although, it is common for them to flatten out after ~10Ang.
        """
        #fig = plt.figure()
        actual_hoppings = copy.deepcopy(self.overlaps_params[spin])

        for orb in range(self.num_orbs):
            center = ((self.num_trans-1)/2).astype(int)
            actual_hoppings[orb][orb][center[0]][center[1]][center[2]] = 0
        hopping_strength = actual_hoppings.flatten().real
        hopping_distance = np.zeros((self.num_orbs,self.num_orbs,self.num_trans[0],self.num_trans[1],self.num_trans[2]))
        for orb1 in range(self.num_orbs):
            for orb2 in range(self.num_orbs):
                vec_to_orbs = self.vec_to_trans[:,:,:] + self.orb_redcoords[orb2] - self.orb_redcoords[orb1]
                hold_vecs = np.reshape(vec_to_orbs,(self.num_trans[0]*self.num_trans[1]*self.num_trans[2],3)).transpose()
                cart_hold_vecs = _red_to_cart((self._a[0], self._a[1], self._a[2]), hold_vecs.transpose())
                dist_to_orbs = np.linalg.norm(cart_hold_vecs,axis=1)
                #print(vec_to_orbs.shape,hold_vecs.shape,cart_hold_vecs.shape,dist_to_orbs.shape)
                dist_to_orbs = np.reshape(dist_to_orbs,(self.num_trans[0],self.num_trans[1],self.num_trans[2]))
                hopping_distance[orb1][orb2] = dist_to_orbs
        hopping_distance = hopping_distance.flatten()
        num_hops = len(hopping_distance)
        #print("hopping dist:",hopping_distance)
        #print("num hops:",num_hops)
        print("")
        print("Should see linear decay in plot below, indicating exponential decay of overlaps with distance.")
        #hopping_strength[hopping_strength <= 10**(-6)] = 10**(-6)
        plot = np.abs(hopping_strength) > 10 ** (-6)
        #plt.get_current_fig_manager().window.raise_()
        plt.scatter(hopping_distance[plot],np.log10(np.abs(hopping_strength[plot])))
        plt.xlim((0,20))
        plt.xlabel("Distance between orbitals (Å)")
        plt.ylabel("log(|overlap parameter|)")
        plt.title("Log decay of overlap parameters")
        plt.tight_layout()
        plt.savefig(self.directory+'overlaps_decay.png',transparent=True,dpi=150,format='png')
        if self.show_figs:
            plt.show()
        plt.close()

    @staticmethod
    def compare_to_DFT(self, directory: str,extra_tag='') -> list:
        """
        This function reads the EIGENVAL from a DFT run, generates the energies from the TB model for the kpt grid.
        Then it plots and compares the error between the TB model energies and DFT energies.

        Args:    
            self (unknown): An object of the class COGITO_TB_Model (can not be the BAND or UNIFORM classes!).
            directory (str): The directory where the EIGENVAL file is.

        Returns:
            png: Saves a png of the DFT and COGITO bands at (self.directory+'compareDFT' + self.file_suff + extra_tag + '.png').
            txt: Saves a error values at self.directory + "DFT_band_error" + self.file_suff + extra_tag + ".txt"
            list: Returns a list of the (averaged over the valance bands) band distance (as defined by Marzari),
                        average maximum error, and average band error.
        """
        if self.spin_polar:
            keys = [0,1]
        else:
            keys = [0]
        #print("is polar?",self.spin_polar)
        bandstrucKPT = Eigenval(directory)
        band_kpts = bandstrucKPT.kpoints
        num_kpts = bandstrucKPT.nkpt
        bs_bands = bandstrucKPT.nbands
        # get DFT eigenvalues
        limited_kpts = num_kpts

        #for key in keys:
        all_values = np.array(list(bandstrucKPT.eigenvalues.values()))
        if not self.spin_polar:
            if self.file_suff == "1" and len(all_values) == 2:
                DFTeigvals =  all_values[1:,:,:,0][:limited_kpts]
                occupations = all_values[1:,:,:,1][:limited_kpts]
                print("! 1")
            else:
                DFTeigvals = all_values[:1, :, :, 0][:limited_kpts]
                occupations = all_values[:1,:,:,1][:limited_kpts]
                print("! 0")
        else:
            DFTeigvals = all_values[:, :, :, 0][:limited_kpts]
            occupations = all_values[:,:,:,1][:limited_kpts]
            print("! 01")
        #print(DFTeigvals.shape)
        #DFTeigvals = all_values[:,:,:,0][:limited_kpts]
        #occupations = all_values[:,:,:,1]
        #print("kpts:",num_kpts,band_kpts)
        #print("evals:",DFTeigvals.shape,DFTeigvals)
        # get TB model eigenvalues
        num_kpts = limited_kpts #num_kpts
        kpoints = band_kpts[:limited_kpts]

        TBeigvals = np.zeros((len(keys),num_kpts,self.num_orbs))
        TBeigvecs= np.zeros((len(keys),num_kpts,self.num_orbs,self.num_orbs),dtype=np.complex128)
        #TBeigvals = np.zeros((num_kpts,self.num_orbs))
        #TBeigvecs = np.zeros((num_kpts,self.num_orbs,self.num_orbs),dtype=np.complex128)

        for key in keys:
            print("solving for", num_kpts,"k-points")
            for kpt in range(num_kpts):
                    print("-",end="")
            print("")
            for kpt in range(num_kpts):
                print("-",end="")
                TBeigvals[key][kpt],TBeigvecs[key][kpt] = self.get_ham(self,kpoints[kpt],return_truevec = True,spin=key)
            print("")

        #print(TBeigvals.shape)
        TBeigvals = TBeigvals-self.energy_shift
        # error data #in the future use occupations
        num_bands = min(bs_bands,len(TBeigvals[0][0]))
        fnk = (DFTeigvals[ :, :, :num_bands].flatten()<=self.efermi+0)#+self.energy_shift # as True for 1 and False for 0
        bottomcb = (DFTeigvals[ :, :, :num_bands].flatten()<self.efermi+3)#+self.energy_shift) & (self.eigvals[:, :].flatten()>self.efermi+self.energy_shift)
        cb = DFTeigvals[ :, :, :num_bands].flatten()>self.efermi#+self.energy_shift
        fnk_sum = np.sum(np.ones((len(keys),num_kpts,num_bands)).flatten()[fnk])
        diffEig = np.abs(DFTeigvals[ :, :,:num_bands] - TBeigvals[ :, :,:num_bands])
        sqDiff = np.square(diffEig)
        band_dist = (np.sum(sqDiff.flatten()[fnk])/fnk_sum)**(1/2)
        band_error = np.average(diffEig.flatten()[fnk])
        avgMCBerr = np.average(diffEig.flatten()[bottomcb])
        avgCBerr = np.average(diffEig.flatten()[cb])
        max_error = np.amax(diffEig.flatten()[fnk])
        print("average error in Valence Bands:", band_error)
        print("band distance in Valence Bands:", band_dist)
        print("max error in valence band:",max_error)
        print("average error in Bottom CB:", avgMCBerr)
        print("average error in Conduc Bands:", avgCBerr)
        # now save all of this into a file instead
        # write the TB parameters
        error_file = open(self.directory + "DFT_band_error" + self.file_suff + extra_tag + ".txt", "w")
        error_file.write("File generated by COGITO TB\n")
        error_file.write('average error in Valence Bands + 0eV: '+'{:>8}'.format(f"{band_error:.6f}") + " eV" + "\n")
        error_file.write('band distance in Valence Bands + 0eV: '+'{:>8}'.format(f"{band_dist:.6f}") + " eV" + "\n")
        error_file.write('max error in valence band + 0eV: '+'{:>8}'.format(f"{max_error:.6f}") + " eV" + "\n")
        error_file.write('average error in Bottom 3 eV of CB: '+'{:>8}'.format(f"{avgMCBerr:.6f}") + " eV" + "\n")
        error_file.write('average error in Conduc Bands: '+'{:>8}'.format(f"{avgCBerr:.6f}") + " eV" + "\n")

        #num_bands = len(DFTeigvals[0])
        fig = plt.figure()
        ax = fig.add_subplot(111)
        #plt.get_current_fig_manager().window.raise_()
        linestyle = ["","--"]
        for key in keys:
            for band in range(bs_bands):
                if band == 0:
                    ax.plot(range(num_kpts), DFTeigvals[key][:, band], linestyle[key], c='black',label='DFT energies')
                else:
                    ax.plot(range(num_kpts), DFTeigvals[key][:, band], linestyle[key], c='black')

            for band in range(len(TBeigvals[key][0])):
                #for kpt in range(num_kpts):
                #    ax.plot(kpt, eigvals[kpt][band], 'o', c='blue', markersize=projections[kpt, band] * 6 + 0.5)
                if band == 0:
                    ax.plot(range(num_kpts), TBeigvals[key][:, band], linestyle[key], c='red',label='COGITO energies')
                else:
                    ax.plot(range(num_kpts), TBeigvals[key][:, band], linestyle[key], c='red')

        #ax.set_ylim(5,8)
        plt.xlabel("k-point number")
        plt.ylabel("Band energy (eV)")
        plt.legend()
        plt.title("Compare COGITO TB bands with VASP bands")
        plt.tight_layout()
        plt.savefig(self.directory+'compareDFT' + self.file_suff + extra_tag + '.png',transparent=True,dpi=150,format='png')
        if self.show_figs:
            plt.show()
        plt.close()
        return band_dist, max_error, band_error

    @staticmethod
    def get_COHP(self: object, orbs: dict, NN: int=None, include_onsite: bool=False,spin: int = 0) -> npt.NDArray:
        """
        Calculates the COHP for the given orbitals and nearest neighbors.

        Args:    
            self (object): An object of the class COGITO_BAND or COGITO_UNIFORM.
            orbs (dict): Either a list of two dictionaries giving elements as keys and orbital types as items (eg [{"Pb":["s","d"],"O":["s","p"]},{"Pb":["s"]"O":["p"]}]) or give list of orb numbers [[1,2,3,5,6,7],[1,2,3,4,5,6,7,8]].
            NN (int): An integer for which nearest neighbor number to include (eg 1 for 1NN) or None or "All" for all nearest neighbors.
            include_onsite (bool): Includes atomic orbital energy terms (H_ab(R) where R=0 and a=b) instead of just bonding terms.
            spin (int): The spin of the tight binding parameters; default is 0 works for nonspin-polarized.

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
                    if self.verbose > 1:
                        print(element)
                    for orbtype in orbs[0][element]:
                        for addorb in range(self.num_orbs):
                            if self.orbatomname[addorb] == element and orbtype in self.exactorbtype[addorb]:
                                firstorbs.append(addorb)

                for element in orbs[1].keys():
                    for orbtype in orbs[1][element]:
                        for addorb in range(self.num_orbs):
                            if self.orbatomname[addorb] == element and orbtype in self.exactorbtype[addorb]:
                                secorbs.append(addorb)
            else: # type is int and should be referring to the specific atom number in prim cell
                for atm in orbs[0].keys():
                    if self.verbose > 1:
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
            if self.verbose > 1:
                print(orbs)

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
        if not include_onsite:
            in_full = in_full & np.logical_not(is_onsite)
        # print(in_full)
        full_ind = [full_indices[0][in_full], full_indices[1][in_full], full_indices[2][in_full],
                    full_indices[3][in_full], full_indices[4][in_full]]
        if self.verbose > 0:
            print("num of params:", len(full_ind[0]))

        # generate the hamiltonian matrix for neccessary kpoints
        cohp = np.zeros(self.eigvecs[spin][:, :self.num_orbs, 0].shape, dtype=np.complex128)  # kpts,bands

        # get it from TB params so have all except orbital energies
        vecs = np.array(self.vec_to_trans)
        orb_pos = np.array(self.orb_redcoords)

        for kind in range(self.num_kpts):
            kpoint = self.kpoints[kind]
            '''
            # use all the parameters (much slower)
            exp_fac = np.exp(2j*np.pi*np.dot(kpoint,self.flat_vec_to_orbs))
            #if self.set_min == True:
            #    exp_fac[self.dist_to_orbs > self.min_hopping_dist] = 0
            exp_fac = np.reshape(exp_fac,(self.num_orbs,self.num_orbs,self.num_trans[0],self.num_trans[1],self.num_trans[2]))
            ham = np.sum(self.TB_params*exp_fac,axis=(2,3,4)) # [orb1,orb2]
            coeff = self.eigvecs[kind,:self.num_orbs].T #[band, orb]
            energyCont = ham[None,:,:]*np.conj(coeff[:,:,None])*coeff[:,None,:] # [band,orb1,orb2]
            cohp[kind] = np.sum(energyCont,axis=(1,2)) * self.kpt_weights[kind]
            '''
            # now only use the indices that are short
            vec_to_orbs = vecs[full_ind[2], full_ind[3], full_ind[4]] + orb_pos[full_ind[1]] - orb_pos[full_ind[0]]
            exp_fac = np.exp(2j * np.pi * np.matmul(vec_to_orbs,kpoint))
            scaledTB = self.TB_params[spin][full_ind[0], full_ind[1], full_ind[2], full_ind[3], full_ind[4]] * exp_fac
            # scaledOvlap = self.overlaps_params[full_ind[0],full_ind[1],full_ind[2],full_ind[3],full_ind[4]]*exp_fac
            coeff = self.eigvecs[spin][kind, :self.num_orbs].T  # [band, orb]
            # coeff = np.transpose(coeff,axes = (0,2,1))
            energyCont = scaledTB[None, :] * np.conj(coeff)[:, full_ind[0]] * coeff[:, full_ind[1]]
            partial_enCont = energyCont
            cohp[kind] = np.sum(partial_enCont, axis=1) * self.kpt_weights[kind]
            #print(kind, cohp[kind])
            #print(self.eigvals[kind])

        return cohp.real # should be real already but stored in complex array

    @staticmethod
    def get_ICOHP(self,spin: int = 0) -> npt.NDArray:
        """
        Generates an ICOHP matrix with same dimension as TB parameters (orb1,orb2,T1,T2,T3). Used for fast calculation of energies in crystal bond plot or other integrated energy quantities.
        """
        icohp = np.zeros((self.num_orbs,self.num_orbs,self.num_trans[0],self.num_trans[1],self.num_trans[2]), dtype=np.complex128)  # kpts,bands

        # get it from TB params so have all except orbital energies
        vecs = np.array(self.vec_to_trans)
        orb_pos = np.array(self.orb_redcoords)

        for kind in range(self.num_kpts):
            kpoint = self.kpoints[kind]

            # use all the parameters (much slower)
            exp_fac = np.exp(2j*np.pi*np.dot(kpoint,self.flat_vec_to_orbs))
            #if self.set_min == True:
            #    exp_fac[self.dist_to_orbs > self.min_hopping_dist] = 0
            exp_fac = np.reshape(exp_fac,(self.num_orbs,self.num_orbs,self.num_trans[0],self.num_trans[1],self.num_trans[2]))
            ham = self.TB_params[spin]*exp_fac #np.sum(,axis=(2,3,4)) # [orb1,orb2]

            coeff = self.eigvecs[spin][kind, :self.num_orbs].T  # [band, orb]
            include_band = self.eigvals[spin][kind, :].flatten() < self.efermi + self.energy_shift
            count = np.zeros(self.num_orbs)
            count[include_band] = 1
            mult_coeff = np.conj(coeff)[:, :, None] * coeff[:, None, :] * count[:, None, None]  # band,orb1,orb2
            sum_coeff = np.sum(mult_coeff, axis=0)

            energyCont = ham * sum_coeff[:, :, None, None, None]

            #coeff = self.eigvecs[spin][kind,:self.num_orbs].T #[band, orb]
            #energyCont = ham[None,:,:,:,:,:]*np.conj(coeff[:,:,None,None,None,None])*coeff[:,None,:,None,None,None] # [band,orb1,orb2,tran1,tran2,tran3]
            icohp += energyCont * self.kpt_weights[kind] #np.sum(,axis=(0))  #[orb1,orb2,tran1,tran2,tran3]

        return icohp

    @staticmethod
    def get_COOP(self: object, orbs: dict, NN: int=None, include_onsite: bool=False,spin: int = 0) -> npt.NDArray:
        """
        Calculates the COOP for the given orbitals and nearest neighbors.

        Args:    
            self (object): An object of the class COGITO_BAND or COGITO_UNIFORM.
            orbs (dict): Either a list of two dictionaries giving elements as keys and orbital types as items (eg [{"Pb":["s","d"],"O":["s","p"]},{"Pb":["s"]"O":["p"]}]) or give list of orb numbers [[1,2,3,5,6,7],[1,2,3,4,5,6,7,8]].
            NN (int): An integer for which nearest neighbor number to include (eg 1 for 1NN) or None or "All" for all nearest neighbors.
            include_onsite (bool): Includes atomic overlap terms (S_ab(R) where R=0 and a=b) instead of just bonding terms, this can be used with NN=0 to give a projected bandstructure.
            spin (int): The spin of the tight binding parameters; default is 0 works for nonspin-polarized.

        Returns:    
            npt.NDArray: Returns COOP values in a [kpt,band] dimension.
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
                    #print(atm)
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
        #print("num of params:", len(full_ind[0]))

        # generate the hamiltonian matrix for neccessary kpoints
        coop = np.zeros(self.eigvecs[spin][:, :, 0].shape, dtype=np.complex128)  # kpts,bands

        # get it from TB params so have all except orbital energies
        vecs = np.array(self.vec_to_trans)
        orb_pos = np.array(self.orb_redcoords)

        for kind in range(self.num_kpts):
            kpoint = self.kpoints[kind]
            '''
            # use all the parameters (much slower)
            exp_fac = np.exp(2j*np.pi*np.dot(kpoint,self.flat_vec_to_orbs))
            if self.set_min == True:
                exp_fac[self.dist_to_orbs > self.min_hopping_dist] = 0
            exp_fac = np.reshape(exp_fac,(self.num_orbs,self.num_orbs,self.num_trans[0],self.num_trans[1],self.num_trans[2]))
            ham = np.sum(self.overlaps_params*exp_fac,axis=(2,3,4)) # [orb1,orb2]
            coeff = self.eigvecs[kind,:self.num_orbs].T #[band, orb]
            energyCont = ham[None,:,:]*np.conj(coeff[:,:,None])*coeff[:,None,:] # [band,orb1,orb2]
            coop[kind] = np.sum(energyCont,axis=(1,2)) * self.kpt_weights[kind]
            '''
            # now only use the indices that are short
            vec_to_orbs = vecs[full_ind[2], full_ind[3], full_ind[4]] + orb_pos[full_ind[1]] - orb_pos[full_ind[0]]
            exp_fac = np.exp(2j * np.pi * np.matmul(vec_to_orbs,kpoint))
            scaled_olap = self.overlaps_params[spin][full_ind[0], full_ind[1], full_ind[2], full_ind[3], full_ind[4]] * exp_fac
            # scaledOvlap = self.overlaps_params[full_ind[0],full_ind[1],full_ind[2],full_ind[3],full_ind[4]]*exp_fac
            coeff = self.eigvecs[spin][kind, :].T  # [band, orb]
            # coeff = np.transpose(coeff,axes = (0,2,1))
            energyCont = scaled_olap[None, :] * np.conj(coeff)[:, full_ind[0]] * coeff[:, full_ind[1]]
            partial_enCont = energyCont
            coop[kind] = np.sum(partial_enCont, axis=1) * self.kpt_weights[kind]
        #print(coop)

        return coop.real # should be real already but stored in complex array

    @staticmethod
    def get_ICOOP(self,spin: int = 0) -> npt.NDArray:
        """
        Generates an ICOOP matrix with same dimension as TB parameters (orb1,orb2,T1,T2,T3). Used for fast calculation of bond charges in crystal bond plot or other integrated charge quantities.
        """
        icoop = np.zeros((self.num_orbs,self.num_orbs,self.num_trans[0],self.num_trans[1],self.num_trans[2]), dtype=np.complex128)  # kpts,bands

        # get it from TB params so have all except orbital energies
        vecs = np.array(self.vec_to_trans)
        orb_pos = np.array(self.orb_redcoords)

        for kind in range(self.num_kpts):
            kpoint = self.kpoints[kind]

            # use all the parameters (much slower)
            exp_fac = np.exp(2j*np.pi*np.dot(kpoint,self.flat_vec_to_orbs))
            #if self.set_min == True:
            #    exp_fac[self.dist_to_orbs > self.min_hopping_dist] = 0
            exp_fac = np.reshape(exp_fac,(self.num_orbs,self.num_orbs,self.num_trans[0],self.num_trans[1],self.num_trans[2]))
            scaled_olap = self.overlaps_params[spin]*exp_fac #np.sum(,axis=(2,3,4)) # [orb1,orb2]

            coeff = self.eigvecs[spin][kind, :self.num_orbs].T  # [band, orb]
            include_band = self.eigvals[spin][kind, :].flatten() < self.efermi + self.energy_shift
            count = np.zeros(self.num_orbs)
            count[include_band] = 1
            mult_coeff = np.conj(coeff)[:, :, None] * coeff[:, None, :] * count[:, None, None]  # band,orb1,orb2
            sum_coeff = np.sum(mult_coeff, axis=0)

            energyCont = scaled_olap * sum_coeff[:, :, None, None, None]

            #coeff = self.eigvecs[spin][kind,:self.num_orbs].T #[band, orb]
            #energyCont = ham[None,:,:,:,:,:]*np.conj(coeff[:,:,None,None,None,None])*coeff[:,None,:,None,None,None] # [band,orb1,orb2,tran1,tran2,tran3]
            icoop += energyCont * self.kpt_weights[kind] #np.sum(,axis=(0))  #[orb1,orb2,tran1,tran2,tran3]

        return icoop


    @staticmethod
    def get_bond_occup(self,spin: int = 0) -> npt.NDArray:
        """
        Generates an bond occupancy matrix with same dimension as TB parameters (orb1,orb2,T1,T2,T3).
        Used for fast calculation of bond charges in crystal bond plot or other integrated charge quantities.
        """
        bond_occup = np.zeros((self.num_orbs,self.num_orbs,self.num_trans[0],self.num_trans[1],self.num_trans[2]), dtype=np.complex128)  # kpts,bands

        # get it from TB params so have all except orbital energies
        vecs = np.array(self.vec_to_trans)
        orb_pos = np.array(self.orb_redcoords)

        for kind in range(self.num_kpts):
            kpoint = self.kpoints[kind]

            # use all the parameters (much slower)
            exp_fac = np.exp(2j*np.pi*np.dot(kpoint,self.flat_vec_to_orbs))
            #if self.set_min == True:
            #    exp_fac[self.dist_to_orbs > self.min_hopping_dist] = 0
            exp_fac = np.reshape(exp_fac,(self.num_orbs,self.num_orbs,self.num_trans[0],self.num_trans[1],self.num_trans[2]))

            coeff = self.eigvecs[spin][kind, :self.num_orbs].T  # [band, orb]
            include_band = self.eigvals[spin][kind, :].flatten() < self.efermi + self.energy_shift
            count = np.zeros(self.num_orbs)
            count[include_band] = 1
            mult_coeff = np.conj(coeff)[:, :, None] * coeff[:, None, :] * count[:, None, None]  # band,orb1,orb2
            sum_coeff = np.sum(mult_coeff, axis=0)

            energyCont = exp_fac * sum_coeff[:, :, None, None, None]
            bond_occup += energyCont * self.kpt_weights[kind] #np.sum(,axis=(0))  #[orb1,orb2,tran1,tran2,tran3]
        return bond_occup


class COGITO_BAND(object): # ought to pass the class COGITO_TB_Model
    def __init__(self,TB_model: object, num_kpts: int=100):
        """
        This class deals with all post-processing band structure analysis.

        Args:    
            TB_model (object): Requires an object of the class COGITO_TB_Model.
        """
        # copy all attributes of the TB_model
        attributes = vars(TB_model)
        for attr, value in attributes.items():
            setattr(self, attr, value)
        # save TB_model to self to use TB_model functions
        self.TB_model = TB_model
        #print("is polarized?",self.spin_polar)
        # calculate the bandstructure
        self.get_bandstructure(num_kpts)
        self.NN_index, self.NN_dists = self.TB_model.get_neighbors(self)

    def get_bandstructure(self, num_kpts: int = 100) -> None:
        """
        The function automatically generates a kpath using pymatgen and kpathseek.
        Then calculates the band energies and vectors for the kpath.

        Args:    
            num_kpts (int): The number of kpoints between EACH kpath.

        Returns:    
            None: Nothing
        """

        pymat_struc = struc.Structure(self._a, self.elements, self.primAtoms)
        print(pymat_struc)
        kpts_obj = kpath.KPathSeek(pymat_struc)
        high_sym = kpts_obj.get_kpoints(1, coords_are_cartesian=False)

        num_each = int(num_kpts / (len(high_sym) / 2))
        print("num kpts per path:", num_each)
        kpoints = kpts_obj.get_kpoints(num_each, coords_are_cartesian=False)
        self.num_kpts = len(kpoints[0])
        print(self.num_kpts)
        self.kpoints = np.array(kpoints[0])
        self.kpoint_labels = np.array(kpoints[1])

        if self.spin_polar:
            keys = [0,1]
        else:
            keys = [0]
        self.keys = keys

        eigvals = np.zeros((len(keys), self.num_kpts, self.num_orbs))
        eigvecs = np.zeros((len(keys), self.num_kpts, self.num_orbs, self.num_orbs), dtype=np.complex128)

        for key in keys:
            for kpt in range(self.num_kpts):
                print("-", end="")
            print("")
            for kpt in range(self.num_kpts):
                print("-", end="")
                eigvals[key][kpt], eigvecs[key][kpt] = self.TB_model.get_ham(self, self.kpoints[kpt], return_truevec=True,spin=key)
            print("")
        self.eigvals = eigvals
        self.eigvecs = eigvecs
        self.kpt_weights = np.ones(self.num_kpts)

    def plotBS(self, ax: object=None, ylim: list = (-10, 10), color_label: str = "", colors: npt.NDArray = np.array([]), colorhalf: float = 10) -> object:
        """

        Args:    
            ax (object): (matplotlib.pyplot axis): Axis to save the bandstructure to, otherwise generate new axis.
            ylim (list): Limits on the y-axis of plot.
            color_label (str): When being plotted from another function, this passes "COHP" or "COOP".
            colors (npt.NDArray): Magnitude for each point to use in color plotting, passed by get_COHP() function.
            colorhalf (float): Sets scale of color bar.

        Returns:    
            object: matplotlib.pyplot axis with bandstructure plotted.
        """
        # reformat data to find nodes and stop repeating them
        num_nodes = 1
        nodes = [0]
        labels = [self.kpoint_labels[0]]
        kpt_line = np.arange(self.num_kpts)
        next_label = ''
        for kpt_ind, name in enumerate(self.kpoint_labels[1:-1]):
            if name != '' and name != next_label:
                nodes.append(kpt_ind - num_nodes + 2)
                next_label = self.kpoint_labels[kpt_ind + 2]
                if name == next_label:
                    labels.append(name)
                else:
                    labels.append(name + "|" + next_label)
                kpt_line[kpt_ind + 2:] = kpt_line[kpt_ind + 2:] - 1
                num_nodes += 1

        # initialize figure and axis
        fig = plt.figure()
        if ax == None:
            ax = fig.add_subplot(111)
        else:
            ax = ax

        # plot the bands, i loops over each band

        linestyle = ["","--"]
        color = ["dimgray","lightseagreen"]
        markers = ["o","X"]
        for key in self.keys: # plot bands for each spin
            if color_label == "":
                for i in range(self.num_orbs):
                    ax.plot(kpt_line, self.eigvals[key][:, i] - (self.efermi + self.energy_shift),linestyle[key], c=color[key], linewidth=1)
            else:
                from matplotlib import colors as plt_colors
                if colorhalf == 10: # default value
                    halfrange = np.average(np.abs(colors[0]))
                else:
                    halfrange = colorhalf
                for i in range(self.num_orbs):
                    scatter = ax.scatter(kpt_line, self.eigvals[key][:, i] - (self.efermi + self.energy_shift),marker = markers[key],c=colors[key][:, i],
                                         cmap='PiYG_r', linewidth=0.5, norm=plt_colors.CenteredNorm(0, halfrange))
                if key == 0:
                    plt.colorbar(scatter, label=color_label)

        # plot the vertical lines for high symmetry points
        for n in range(num_nodes):
            ax.axvline(x=nodes[n], linewidth=0.5, color='k')

        # plot the orbital energies
        # for orb in range(self.num_orbs):
        #    ax.plot(kpt_line, np.ones(len(kpt_line))*self.atomic_energies[orb] - (self.efermi+self.energy_shift),c="black")

        # make plot look nice
        ax.set_xlabel("Path in k-space")  # ,fontsize=20)
        ax.set_ylabel("Energy (eV)")  # ,fontsize=20)
        # ax.set_xlim(self.k_dist[0], self.k_dist[-1])
        ax.set_xticks(nodes)
        ax.set_xticklabels(labels)  # ,fontsize=20)
        # autoset axes if ylim is default
        if ylim == (-10,10):
            amin = np.amin(self.eigvals[0][:, :]) - (self.efermi+self.energy_shift)
            amax = np.amax(self.eigvals[0][:, :]) - (self.efermi+self.energy_shift)
            ylim = (amin-1,amax+1)
        ax.set_ylim(ylim)
        # fig.tight_layout()
        # plt.close()
        plt.savefig(self.directory+'bandstruc.png',transparent=True,dpi=150,format='png')
        if self.show_figs:
            plt.show()
        plt.close()
        return ax

    def plotlyBS(self,ylim=(-10,10),color_label="",colors=None,colorhalf=None, orbProj: bool=False) -> object:
        """
        Plots bandstructure (or projected bandstructure) using plotly graph_objects.

        Args:    
            ylim (unknown): Limits on the y-axis of plot.
            color_label (unknown): When being plotted from another function, this passes "COHP" or "COOP".
            colors (unknown): Magnitude for each point to use in color plotting, passed by get_COHP() function.
            colorhalf (unknown): Sets scale of color bar.

        Returns:    
            object: plotly figure with bandstructure plotted.
        """

        #reformat data to find nodes and stop repeating them
        num_nodes = 1
        nodes = [0]
        labels = [self.kpoint_labels[0]]
        kpt_line = np.arange(self.num_kpts)
        next_label = ''
        for kpt_ind,name in enumerate(self.kpoint_labels[1:-1]):
            if name != '' and name != next_label:
                nodes.append(kpt_ind-num_nodes+2)
                next_label = self.kpoint_labels[kpt_ind+2]
                if name == next_label:
                    labels.append(name)
                else:
                    labels.append(name+"|"+next_label)
                kpt_line[kpt_ind+2:] = kpt_line[kpt_ind+2:]-1
                num_nodes += 1

        import plotly.graph_objects as go
        bs_figure = go.Figure()


        #plot the bands, i loops over each band
        markers = ["circle","x"]
        for key in self.keys: # plot bands for each spin
            if color_label == "":
                for i in range(self.num_orbs):
                    bs_figure.add_trace(go.Scatter(x=kpt_line,
                                       y= self.eigvals[key][:,i] - (self.efermi+self.energy_shift),
                                       marker=dict(color="black", size=2),showlegend=False))
            else: #passed from COHP and color points based on COHP
                if colorhalf == 10:
                    halfrange = np.average(np.abs(colors[0]))
                else:
                    halfrange = colorhalf
                colorscale = "PiYG_r"
                minv = -halfrange
                maxv = halfrange
                linecolor = "white"

                if orbProj:
                    minv = 0
                    maxv = 1
                    colorscale="Blues"
                    linecolor="gray"

                for i in range(self.num_orbs):
                    hoverlabels = np.array(np.around(colors[key][:,i],decimals=2),dtype=np.str_)
                    hoverlabels = np.char.add(np.array([color_label+': '],dtype=np.str_),hoverlabels)
                    #print(labels)
                    bs_figure.add_trace(go.Scatter(x=kpt_line,
                                       y= self.eigvals[key][:,i] - (self.efermi+self.energy_shift),mode='markers',
                                                   marker_symbol = markers[key],hovertext=hoverlabels,hoverinfo='text',
                                                    marker=dict(color=colors[key][:,i],colorscale=colorscale,cmin=minv,
                                                    cmax=maxv,colorbar=dict(title=color_label),size=8,line=dict(
                                                    color=linecolor, width=0.3)),showlegend=False))

        if ylim == None:
            minen = np.amin(self.eigvals.flatten() - (self.efermi+self.energy_shift))
            maxen = np.amax(self.eigvals.flatten() - (self.efermi+self.energy_shift))
            ylim = (minen-0.5,maxen+0.5)

        if ylim == (-10,10):
            amin = np.amin(self.eigvals[0][:, :]) - (self.efermi+self.energy_shift)
            amax = np.amax(self.eigvals[0][:, :]) - (self.efermi+self.energy_shift)
            ylim = (amin-1,amax+1)

        # Add custom grid lines
        for x_val in nodes:
            bs_figure.add_shape(type='line',
                                   x0=x_val, x1=x_val, y0=ylim[0], y1=ylim[1],
                                   line=dict(color='black', width=1))

        bs_figure.update_layout(xaxis={'title': 'Path in k-space'},
                           yaxis={'title': 'Energy (eV)'})

        # Update layout to set axis limits and ticks
        bs_figure.update_layout(margin=dict(l=20, r=20, t=50, b=0),
            xaxis=dict(showgrid=False,range=[kpt_line[0], kpt_line[-1]], tickvals=nodes,ticktext=labels,
                       showline=True, linewidth=1.5, linecolor='black', mirror=True,tickfont=dict(size=16), titlefont=dict(size=18)),
            yaxis=dict(showgrid=False,range=[ylim[0], ylim[1]],showline=True, linewidth=2, linecolor='black', mirror=True,
                       tickfont=dict(size=16), titlefont=dict(size=18)),
            title={'text':'<b>Bandstructure</b>', 'font':dict(size=26, color='black',family='Times')},
            #plot_bgcolor='rgba(0, 0, 0, 0)',
            title_x = 0.5,  # Center the title
            paper_bgcolor='rgba(0,0,0,0)',  # Transparent background for the whole figure
            plot_bgcolor='rgba(0,0,0,0)'
        )

        return bs_figure

    def get_COHP(self, orbs: dict, NN: int = None, ylim: list = (-10, 10), colorhalf: float = 10, include_onsite: bool = False,from_dash: bool=False) -> None:
        """
        Calculates and plots the projected COHP values for each band and k-point on band structure.

        Args:    
            orbs (dict): Either a list of two dictionaries giving elements as keys and orbital types as items (eg [{"Pb":["s","d"],"O":["s","p"]},{"Pb":["s"]"O":["p"]}]) or give list of orb numbers [[1,2,3,5,6,7],[1,2,3,4,5,6,7,8]].
            NN (int): An integer for which nearest neighbor number to include (eg 1 for 1NN) or None or "All" for all nearest neighbors.
            ylim (list): The limits of the y-axis (energy) of the band structure plot.
            colorhalf (float): Set the magnitude for the color bar; For default value of 10, it is set automatically to average(COHP)*3.
            include_onsite (bool): Includes atomic orbital energy terms (H_ab(R) where R=0 and a=b) instead of just bonding terms.
            from_dash (bool): Flag to set when calling from the dash app, sets plotting backend to plotly and returns the axis.

        Returns:    
            None: Nothing
        """
        self.NN_index, self.NN_dists = self.TB_model.get_neighbors(self)
        cohp = {}
        for key in self.keys: # plot bands for each spin
            cohp[key] = self.TB_model.get_COHP(self, orbs, NN=NN, include_onsite=include_onsite, spin = key)

        if from_dash == True:
            return self.plotlyBS(color_label="COHP", colors=cohp, colorhalf=colorhalf, ylim = ylim)
        else:
            fig = self.plotlyBS(color_label="COHP", colors=cohp, colorhalf=colorhalf, ylim = ylim)
            #self.plotBS(color_label="COHP", colors=cohp, colorhalf=colorhalf, ylim=ylim)
            atms1 = ''.join(list(orbs[0].keys()))
            atms2 = ''.join(list(orbs[1].keys()))
            fig.write_html(self.directory+"COHP_BS"+atms1+"-"+atms2+".html")

    def get_COOP(self, orbs: dict, NN: int = None, ylim: list = (-10, 10), colorhalf: float = 10, include_onsite: bool = False,from_dash=False, color_label: str = "COOP", orbProj: bool = False) -> None:
        """
        Calculates and plots the projected COOP values for each band and k-point on band structure.

        Args:    
            orbs (dict): either a list of two dictionaries giving elements as keys and orbital types as items (eg [{"Pb":["s","d"],"O":["s","p"]},{"Pb":["s"]"O":["p"]}]) or give list of orb numbers [[1,2,3,5,6,7],[1,2,3,4,5,6,7,8]].
            NN (int): An integer for which nearest neighbor number to include (eg 1 for 1NN) or None or "All" for all nearest neighbors.
            ylim (list): The limits of the y-axis (energy) of the band structure plot.
            colorhalf (float): Set the magnitude for the color bar; For default value of 10, it is set automatically to average(COOP)*3.
            include_onsite (bool): Includes atomic overlap terms (S_ab(R) where R=0 and a=b) instead of just bonding terms, this can be used with NN=0 to give a projected bandstructure.
            from_dash (unknown): Flag to set when calling from the dash app, sets plotting backend to plotly and returns the axis.

        Returns:    
            None: Nothing
        """
        self.NN_index, self.NN_dists = self.TB_model.get_neighbors(self)
        coop = {}
        for key in self.keys: # plot bands for each spin
            coop[key] = self.TB_model.get_COOP(self, orbs, NN=NN, include_onsite=include_onsite, spin = key)

        if from_dash == True:
            return self.plotlyBS(color_label=color_label, colors=coop, colorhalf=colorhalf, ylim = ylim, orbProj = orbProj)
        else:
            fig = self.plotlyBS(color_label=color_label, colors=coop, colorhalf=colorhalf, ylim = ylim, orbProj = orbProj)
            #self.plotBS(color_label="COOP", colors=coop, colorhalf=colorhalf, ylim=ylim))

            fig.write_html(self.directory+"COOP_BS.html")

    def get_projectedBS(self,orbdict : dict, ylim: list = (-10, 10), colorhalf: float = 10) -> None:
        """
        Returns the COGITO-orbital projected band structure. By construction, this performs the projection by Mulliken-style analysis. 
        Projection without Mulliken is obfuscated by the nonorthogonal nature of the orbitals, which causes much larger projection at high energy states due to antibonding. 
        
        Args:    
            orbdict (dict): Single dictionary of orbitals to include in projection. For example, {"Pb":["s","d"],"O":["s","p"]}.
            ylim (tuple): The y-axis limits.
            colorhalf (float): The upper and lower value for the projection. The code will find a reasonable value. For this, colorhalf=1 would be good. (Default is only 10 for signaling auto-update in later code.)
        
        Returns: 
            html: Saves the projection plot at self.directory+"projectedBS"+elem+".html".
        """
        allorbs = {}
        for elem in self.elements:
            allorbs[elem] = ["s","p","d","f"]
        elem = list(orbdict.keys())[0]
        orbs = list(orbdict.values())[0]
        all_orb = ''
        for orb in orbs:
            all_orb = all_orb+orb
        full_label = "% "+elem+" "+all_orb
        fig = self.get_COOP([orbdict,allorbs], include_onsite=True,from_dash=True,color_label=full_label, orbProj=True,ylim=ylim)
        fig.write_html(self.directory+"projectedBS"+elem+".html")

    def make_COHP_dashapp(self,pathname: str = "/COGITO_COHP/") -> None:
        """
        This function generate a dash app which allows the user to interactively select orbitals and nearest neighbors
        to examine their project COHP band structure quickly.

        Args:    
            pathname (str): Name appended to the default pathname for the html.

        Returns:    
            None: Nothing
        """
        import dash
        from dash import dcc, html
        from dash.dependencies import Input, Output, State

        # Sample data
        keys = np.unique(self.elements)
        data = {}
        for elem in keys:
            elemorbs = np.array(self.exactorbtype, dtype='<U1')[np.array(self.orbatomname) == elem]
            data[elem] = list(np.unique(elemorbs)[::-1])
        print(data)
        #data = {"Si": ["s", "p"], "Pb": ["s", "p", "d"]}

        # Initialize Dash app
        app = dash.Dash(__name__, requests_pathname_prefix=pathname,routes_pathname_prefix=pathname)

        # Define a recursive function to create nested checklist items
        def create_nested_checklist(data, level=0,orb=0):
            checklist_items = []
            for atom, orbitals in data.items():
                label = atom if level == 0 else f"{atom} ({', '.join(orbitals)})"
                checklist_items.append(html.Div([
                                                   label,
                                                   dcc.Checklist(
                                                       id={'type': 'orbitals-checklist'+str(orb), 'index': atom},
                                                       options=[{'label':"   "+orbital, 'value': f"{atom}-{orbital}"} for
                                                                orbital in orbitals],
                                                       value=[f"{atom}-{orbital}" for orbital in orbitals],
                                                       inline=True,labelStyle= {"width":"3rem"}
                                                   ),
                                               ] + create_nested_checklist({}, level + 1)))
            return checklist_items

        # App layout
        NNs = np.append(["All"],range(1,10))
        print(NNs)
        app.layout = html.Div([html.Div([html.H1("Orbital \u03B1", style={'margin': '5px', 'text-align': 'center', 'font-size': 22,'height':20}),
                        html.Ul(create_nested_checklist(data,orb=0),style={'font-size':20})],style={"width":200,'display': 'inline-block'}),
                html.Div([html.H1("Orbital \u03B2", style={'margin': '5px', 'text-align': 'center', 'font-size': 22,'height':20}),
                      html.Ul(create_nested_checklist(data,orb=1),style={'font-size':20})],style={"width":200,'display': 'inline-block'}),
                html.Div([html.H1("Nearest Neighbor", style={'margin': '5px', 'text-align': 'center', 'font-size': 20,'height':20,'width':100,'display': 'inline-block'}),
                          dcc.Dropdown(id='nearest-neighbor', options=[{'label': str(option), 'value': option} for option in NNs],value=NNs[0],style={'margin':'5px',"width":60,'display': 'inline-block'})],
                         style={"width":180,'display': 'inline-block'}),
            html.Button('Calculate new COHP', id='print-button', n_clicks=0),
            html.Div(dcc.Graph(id='COHP-figure')),
            dcc.Store(id='selected-items-store', data=[data,data])
            ],style={'height': 500,'width':600})

        # Callbacks to dynamically generate the selected items callback for each atom
        for atom in list(data.keys()):
            @app.callback(
                [Output({'type': 'orbitals-checklist0', 'index': atom}, 'value'),
                 Output('selected-items-store', 'data', allow_duplicate=True)],
                [Input({'type': 'orbitals-checklist0', 'index': atom}, 'value')],
                [State('selected-items-store', 'data')],
                prevent_initial_call=True
            )
            def update_selected_items(selected_items, stored_data, atom=atom):
                stored_data[0][atom] = [item.split('-')[1] for item in selected_items]
                return selected_items, stored_data

        # Callbacks to dynamically generate the selected items callback for each atom
        for atom in list(data.keys()):
            @app.callback(
                [Output({'type': 'orbitals-checklist1', 'index': atom}, 'value'),
                 Output('selected-items-store', 'data', allow_duplicate=True)],
                [Input({'type': 'orbitals-checklist1', 'index': atom}, 'value')],
                [State('selected-items-store', 'data')],
                prevent_initial_call=True
            )
            def update_selected_items(selected_items, stored_data, atom=atom):
                stored_data[1][atom] = [item.split('-')[1] for item in selected_items]
                return selected_items, stored_data

        # Callback to handle print button click and print selected items
        @app.callback(
            Output('COHP-figure', 'figure'),
            [Input('print-button', 'n_clicks')],
            [State('selected-items-store', 'data'),State('nearest-neighbor','value')],

        )
        def print_selected_items(n_clicks, stored_data,NN):
            print("orb for COHP",stored_data)
            fig = self.get_COHP(orbs=stored_data,colorhalf=10,NN=NN,from_dash=True)
            return fig

        # Run the app
        #if __name__ == '__main__':
        app.run_server(debug=True)

# also make an orbital projected bandstrucutre plotter by passed {orb} from user to COOP with [{orb}, {all_orbs}]
class COGITO_UNIFORM(object):
    def __init__(self,TB_model: object,grid: tuple):
        """
        This class deals with all post-processing uniform grid analysis.

        Args:    
            TB_model (object): Requires an object of the class COGITO_TB_Model.
        """
        # copy all attributes of the TB_model
        attributes = vars(TB_model)
        for attr, value in attributes.items():
            setattr(self, attr, value)
        # save TB_model to self to use TB_model functions
        self.TB_model = TB_model
        # calculate the uniform grid
        self.get_uniform(grid)
        self.NN_index, self.NN_dists = self.TB_model.get_neighbors(self)

    def get_uniform(self,grid: tuple)->None:
        """
        Sets up the uniform grid, calculating all eigenenergies and eigenvectors, and adjust fermi energy to ensure charge conservation.

        Args:    
            grid (tuple): The kpoint grid to use for the uniform sampling.
        """
        #print(grid)
        #from pymatgen.core import Structure
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
        # creating the k point grid - no weights b/c it is not the reduced form of the kpoint grid
        structure = struc.Structure(self._a, self.elements, self.primAtoms) # Structure.from_file(self.directory + 'POSCAR')
        # pass "structure" to define class into "space"
        space = SpacegroupAnalyzer(structure)
        # call the function within the class
        mesh,maps = space.get_ir_reciprocal_mesh_map(grid)
        # should at some point include symmetry so not all kpoints need to be calculated

        kptvecs = mesh
        kptvecs = np.array(kptvecs, dtype=np.float64)
        #print(kptvecs)
        num_kpts = len(kptvecs)
        self.num_kpts = num_kpts
        self.kpoints = kptvecs
        self.kpt_weights = np.ones(num_kpts)/num_kpts

        if self.spin_polar:
            keys = [0,1]
        else:
            keys = [0]
        self.keys = keys

        eigvals = np.zeros((len(self.keys), num_kpts,self.num_orbs))
        eigvecs = np.zeros((len(self.keys), num_kpts,self.num_orbs,self.num_orbs),dtype=np.complex128)
        kdep_Sij = np.zeros((len(self.keys), num_kpts, self.num_orbs, self.num_orbs), dtype=np.complex128)  # from get_ham

        all_eigvals = []
        for key in self.keys:
            for kpt in range(num_kpts):
                print("-",end="")
            print("")
            for kpt in range(num_kpts):
                print("-",end="")
                eigvals[key][kpt],eigvecs[key][kpt],kdep_Sij[key][kpt] = self.TB_model.get_ham(self,kptvecs[kpt],return_overlap=True, spin = key)
                all_eigvals.append(eigvals[key][kpt])
        print("")
        self.eigvals = eigvals
        self.eigvecs = eigvecs
        self.Sij = kdep_Sij

        # get new fermi energy 
        all_eigvals = np.array(all_eigvals).flatten()
        actual_elecs = np.sum(self.elec_count)
        num_elecs = np.sum((np.ones(len(all_eigvals))/num_kpts)[all_eigvals <= (self.efermi+self.energy_shift)] * 2 / len(self.keys))
        print("check num elecs:",actual_elecs,num_elecs)
        sorted_evals  = np.sort(all_eigvals)
        print(actual_elecs/2*len(self.keys)*num_kpts,int(actual_elecs/2*len(self.keys)*num_kpts))
        new_fermi = sorted_evals[int(actual_elecs/2*len(self.keys)*num_kpts)-1] + 0.000001
        print("new and old fermi:",new_fermi, self.efermi+self.energy_shift,self.efermi)
        print(sorted_evals[int(actual_elecs/2*len(self.keys)*num_kpts)-1:int(actual_elecs/2*len(self.keys)*num_kpts)+2])
        energ_bool = sorted_evals <= new_fermi
        print(energ_bool[int(actual_elecs/2*len(self.keys)*num_kpts)-1:int(actual_elecs/2*len(self.keys)*num_kpts)+2])

        check_elecs = np.sum((np.ones(len(all_eigvals))/num_kpts)[all_eigvals <= (new_fermi)] * 2 / len(self.keys))
        print("right num elecs?",check_elecs,1/num_kpts)
        self.old_efermi = self.efermi
        self.efermi = new_fermi - self.energy_shift
        tot_band_energy = np.sum(all_eigvals[all_eigvals <= (self.efermi+self.energy_shift)] * 2 / len(self.keys))/num_kpts
        print("band energy:", tot_band_energy)

    def get_occupation(self, spin: int = 0):
        """
        This function prints information about electron occupation/partitioning. It prints a version using Mulliken to
        assign charge to orbitals and one that just uses the onsite orbital terms. The onsite one will not be charge balanced as some charge is in bonds.
        It also prints the atomic partial charge by Mulliken population.
        """
        # now get and partition the electron charge to atoms or atoms and bond centers
        #def get_partial_charge(self):
        N_i = np.zeros(self.num_orbs)
        mnk2 = np.transpose(self.eigvecs[spin],axes=(0,2,1)) #[kpt,band,orb]
        cj = mnk2[:, :, None, :]  # self.mnkcoefficients[:,:,None,:]
        ci2 = np.conj(mnk2[:, :, :, None])  # self.mnkcoefficients[:,:,:,None])
        Sij = self.Sij[spin][:, None, :, :]

        fnk = np.zeros((self.num_kpts, self.num_orbs), dtype=np.float64)
        sum_n = np.zeros((self.num_orbs,self.num_orbs), dtype=np.complex128)
        for k in range(self.num_kpts):
            for i in range(self.num_orbs):
                if self.eigvals[spin][k, i] <= (self.efermi+self.energy_shift):
                    fnk[k, i] = 1
                    sum_n += np.conj(mnk2[k, i, :, None])*mnk2[k, i, None, :] * self.Sij[spin][k, :, :] / self.num_kpts
                elif self.eigvals[spin][k, i] > (self.efermi+self.energy_shift):
                    fnk[k, i] = 0

        #new_fnk = fnk[:, :, None, None]

        #Ni_value = ci2 * cj * new_fnk * Sij / self.num_kpts
        #sum_k = np.sum(Ni_value, axis=0)
        #sum_n = np.sum(sum_k, axis=0)
        sum_j = np.sum(sum_n, axis=0)
        N_i = sum_j
        realN_i = np.real(N_i)
        #print('realNi: ', realN_i)
        sum_Ni = np.sum(realN_i)
        #print('sum: ', sum_Ni)

        # get just the atomic portion (no charge from the overlap density)
        diag_Sij = self.Sij[spin].diagonal(0,1,2) # [kpt,orb]
        #print("diag Sij:",diag_Sij)
        ci_2 = np.conj(mnk2)*mnk2 #[kpt,band,orb]
        get_atomic_Ni = ci_2*diag_Sij[:,None,:]*fnk[:,:,None] / self.num_kpts
        sum_k = np.sum(get_atomic_Ni, axis=0)
        atomic_Ni = np.sum(sum_k, axis=0).real
        #print("atomic Ni:",atomic_Ni)

        # output charge on each atom
        # write num protons into tb_input.txt
        N_e = []
        for each_atom in range(max(self.orbatomnum) + 1):
            num_e_each = 0  # starts at 0, resets for every atom
            for i in range(len(self.orbatomnum)):
                if self.orbatomnum[i] == each_atom:
                    num_e_each = realN_i[i] + num_e_each
            N_e.append(num_e_each)

        #print("num_pro: ", self.num_pro)
        #print('unique elements: ', self.unique_elems)

        #N_p = np.zeros(len(self.elements))
        # print('beginingn Ne array: ', N_p_current)
        #for i in range(len(self.elements)):
        #    one_atom = self.elements[i]
        #    # print(one_atom)
        #    ind = np.where(np.array(self.unique_elems) == one_atom)[0][0]
        #    # print(ind)
        #    N_p[i] = self.num_pro[ind]
        #print('corrected Np array based on atoms in the system:', N_p)

        # num_e_each = 2 * num_e_each # multiply by 2, since Ni doesn't account for degeneracy
        N_e = np.array(N_e) * 2
        print("")
        print("Where are the electrons?")
        print('orbital + overlap occupation:', realN_i*2)
        print('sum: ', sum_Ni*2)
        print("orbital occupation without bonds:",atomic_Ni*2)
        print('The electron occupation for the atoms ', self.elements, ' is ', N_e)

        #oxid_state = N_p - N_e
        #print('the oxidation state for the atoms ', self.elements, ' is ', oxid_state)

    def get_COHP(self, orbs: dict, NN: int = None, ylim: list = (-10, 10),sigma=0.1, include_onsite: bool = False):
        """
        Calculates and plots the projected COHP values on the density of states.

        Args:
            orbs (dict): Either a list of two dictionaries giving elements as keys and orbital types as items (eg [{"Pb":["s","d"],"O":["s","p"]},{"Pb":["s"]"O":["p"]}]) or give list of orb numbers [[1,2,3,5,6,7],[1,2,3,4,5,6,7,8]].
            NN (int): An integer for which nearest neighbor number to include (eg 1 for 1NN) or None or "All" for all nearest neighbors.
            ylim (list): The limits of the y-axis (energy) of the DOS plot.
            sigma (float): Sets the spread (in energy) of the gaussian for the each wavefunction COHP. The full pCOHP is created by summing the gaussians.
                        This should trade off with k-point resolution. Low resolution requires high sigma (0.2) to smear between points. High resoltion can have low sigma (0.02).
            include_onsite (bool): Includes atomic orbital energy terms (H_ab(R) where R=0 and a=b) instead of just bonding terms.

        Returns:
            png: Saves projected COHP DOS to self.directory+'COHP_DOS.png'.
        """
        self.NN_index, self.NN_dists = self.TB_model.get_neighbors(self)
        # make energies
        shifted_energies = {}
        all_shifted_ener = self.eigvals[:,:,:].flatten() - (self.efermi + self.energy_shift)
        for key in self.keys:
            shifted_energies[key] = self.eigvals[key,:,:].flatten() - (self.efermi + self.energy_shift)
        all_shifted_ener = all_shifted_ener[all_shifted_ener < ylim[1]]
        minx = np.min(all_shifted_ener) - sigma * 2
        maxx = np.max(all_shifted_ener)  # +sigma*2
        points = 200
        energies = np.linspace(minx, maxx, points)
        cohp = {}
        flat_cohp = {}
        integrated_cohp = {}
        total_cohp = {}
        for key in self.keys: # plot bands for each spin
            cohp[key] = self.TB_model.get_COHP(self, orbs, NN=NN, include_onsite=include_onsite, spin = key)

            # now plot
            integrated_cohp[key] = np.sum(cohp[key].flatten()[self.eigvals[key][:, :].flatten() <= self.efermi + self.energy_shift])
            print("integrated occupied cohp:",[key], integrated_cohp[key])
            # print("max eigval:",self.eigval[:, self.num_orbs-1, 0])
            flat_cohp[key] = cohp[key].flatten()

            # create the DOS by histograms
            # myproj, x, _ = ax.hist(shifted_energies[shifted_energies < 1], bins=numbins,weights=flat_cohp[shifted_energies < 1],color="white", orientation='horizontal')#- self.efermi
            # smooth_cohp = smooth(myproj, 2)
            # smooth_cohp = myproj

            # create the DOS by sum of gaussians
            flat_cohp[key] = flat_cohp[key][shifted_energies[key] < ylim[1]]
            shifted_energies[key] = shifted_energies[key][shifted_energies[key] < ylim[1]]
            sig = sigma  # 0.15#5/self.num_kpts**(1/2)
            #print(key,flat_cohp[key].shape,shifted_energies[key].shape,energies.shape)
            all_cohp = 1 / (sig * (2 * np.pi) ** (1 / 2)) * np.exp(
                -1 / 2 * ((energies[:, None] - shifted_energies[key][None, :]) / sig) ** 2) * flat_cohp[key][None, :]
            total_cohp[key] = np.sum(all_cohp, axis=1)

        fig = plt.figure()
        fig.set_size_inches(3, 5)
        ax = fig.add_subplot(111)
        # plot vertically
        ax.plot(np.zeros(points), energies,'--', color="grey",linewidth=0.8)

        linestyle = ["","--"]
        color = ["dimgray","lightseagreen"]
        mini = []
        maxi = []
        for key in self.keys:
            ax.plot(total_cohp[key], energies,linestyle[key], color=color[key])
            mini.append(np.min(total_cohp[key]) - 0.2)
            maxi.append(np.max(total_cohp[key]) + 0.2)
        mini = min(mini)
        maxi = max(maxi)
        # mini = -1.2
        # maxi = 0.5
        ax.plot([mini, maxi], [0, 0],"--", color="grey",linewidth=0.8)
        ax.set_ylabel("Energy (eV)")
        ax.set_xlabel("COHP")
        plt.ylim(ylim)
        plt.xlim((mini, maxi))
        print("showing plot")
        # plt.xlim((mini,maxi))
        plt.gca().invert_xaxis()
        fig.tight_layout()
        plt.savefig(self.directory+'COHP_DOS.png',transparent=True,dpi=150,format='png')
        if self.show_figs:
            plt.show()
        plt.close()

    def get_ICOHP(self):
        """
        Generates an ICOHP matrix with from the TB_Model static function. Then reduces the matrix size. Used for fast calculation of energies in crystal bond plot or other integrated energy quantities.
        """
        icohp = {}
        for key in self.keys: # plot bands for each spin
            icohp[key] = self.TB_model.get_ICOHP(self, spin = key)

        self.ICOHP = icohp

        if not hasattr(self, 'ICO_trans'):
            # make the size of icohp smaller because most is zeros
            trans = self.num_trans
            old_each_dir = self.num_each_dir
            norbs = self.num_orbs
            round_icohp = np.around(self.ICOHP[0],decimals=4).flatten()
            non_zero_icohp = np.arange(norbs*norbs*trans[0]*trans[1]*trans[2])[round_icohp!=0]
            icohp_indices = np.unravel_index(non_zero_icohp, (norbs, norbs, trans[0], trans[1], trans[2]))
            eachdir1 = np.max(icohp_indices[2]-old_each_dir[0])
            eachdir2 = np.max(icohp_indices[3]-old_each_dir[1])
            eachdir3 = np.max(icohp_indices[4]-old_each_dir[2])
            new_each_dir = np.array([eachdir1,eachdir2,eachdir3],dtype=np.int_)
            print(new_each_dir)
            self.ICO_eachdir = new_each_dir
            self.ICO_trans = new_each_dir*2 + 1
            print("trans for ICOHP:",self.ICO_trans)

        self.full_ICOHP = icohp

        for key in self.keys:  # reduce the size of ICOHP
            old_each_dir = self.num_each_dir
            new_each_dir = self.ICO_eachdir
            icohp[key] = icohp[key][:,:, old_each_dir[0] - new_each_dir[0]: old_each_dir[0] + new_each_dir[0] + 1,
                                    old_each_dir[1] - new_each_dir[1]:old_each_dir[1] + new_each_dir[1] + 1,
                                    old_each_dir[2] - new_each_dir[2]:old_each_dir[2] + new_each_dir[2] + 1]

        self.ICOHP = icohp

        return icohp

    def save_ICOHP(self):
        """
        Saves the smaller ICOHP matrix to (self.directory+"ICOHP"+str(key)+".npy") for reading in later with COGITOico.
        """
        if not hasattr(self, 'ICOHP'):
            self.get_ICOHP()

        for key in self.keys:
            np.save(self.directory+"ICOHP"+str(key),self.ICOHP[key])
        '''
        for key in self.keys: # plot bands for each spin

            # write the TB parameters
            wannier_hr = open(self.directory + "ICOHP"+self.file_suff+str(key)+".txt", "w")
            wannier_hr.write("File generated by Emily Oliphant\n")

            num_trans = self.ICO_trans # self.num_trans
            each_dir = self.ICO_eachdir # self.num_each_dir
            wannier_hr.write(str(each_dir[0])+" "+str(each_dir[1])+" "+str(each_dir[2])+" "+str(self.num_orbs) + "\n")

            wannier_hr.write("start params\n")
            for a1 in range(num_trans[0]):
                trans1 = int(a1 - np.floor(num_trans[0] / 2))
                for a2 in range(num_trans[1]):
                    trans2 = int(a2 - np.floor(num_trans[1] / 2))
                    for a3 in range(num_trans[2]):
                        trans3 = int(a3 - np.floor(num_trans[2] / 2))
                        for orb1 in range(self.num_orbs):
                            for orb2 in range(self.num_orbs):
                                realpart = f"{self.ICOHP[key][orb1, orb2, a1, a2, a3].real:.6f}"
                                imagpart = f"{self.ICOHP[key][orb1, orb2, a1, a2, a3].imag:.6f}"
                                wannier_hr.write(
                                    '{:>5} {:>5} {:>5} {:>5} {:>5} {:>14} {:>14}'.format(str(trans1), str(trans2),
                                                                                         str(trans3),
                                                                                         str(orb1 + 1),
                                                                                         str(orb2 + 1), realpart,
                                                                                         imagpart) + "\n")
        '''

    def get_COOP(self, orbs: dict, NN: int = None, ylim: list = (-10, 10),sigma=0.1, include_onsite: bool = False, orbProj: bool = False,label:str=""):
        """
        Calculates and plots the projected COOP values on the density of states.

        Args:
            orbs (dict): Either a list of two dictionaries giving elements as keys and orbital types as items (eg [{"Pb":["s","d"],"O":["s","p"]},{"Pb":["s"]"O":["p"]}]) or give list of orb numbers [[1,2,3,5,6,7],[1,2,3,4,5,6,7,8]].
            NN (int): An integer for which nearest neighbor number to include (eg 1 for 1NN) or None or "All" for all nearest neighbors.
            ylim (list): The limits of the y-axis (energy) of the DOS plot.
            sigma (float): Sets the spread (in energy) of the gaussian for the each wavefunction COOP. The full pCOOP is created by summing the gaussians.
                        This should trade off with k-point resolution. Low resolution requires high sigma (0.2) to smear between points. High resoltion can have low sigma (0.02).
            include_onsite (bool): Includes atomic orbital energy terms (H_ab(R) where R=0 and a=b) instead of just bonding terms.

        Returns:
            png: Saves projected COOP DOS to self.directory+'COOP_DOS.png'.
        """
        self.NN_index, self.NN_dists = self.TB_model.get_neighbors(self)
        # make energies
        shifted_energies = {}
        all_shifted_ener = self.eigvals[:,:,:].flatten() - (self.efermi + self.energy_shift)
        for key in self.keys:
            shifted_energies[key] = self.eigvals[key,:,:].flatten() - (self.efermi + self.energy_shift)
        all_shifted_ener = all_shifted_ener[all_shifted_ener < ylim[1]]
        minx = np.min(all_shifted_ener) - sigma * 2
        maxx = np.max(all_shifted_ener)  # +sigma*2
        points = 200
        energies = np.linspace(minx, maxx, points)
        coop = {}
        flat_coop = {}
        integrated_coop = {}
        total_coop = {}
        for key in self.keys: # plot bands for each spin
            coop[key] = self.TB_model.get_COOP(self, orbs, NN=NN, include_onsite=include_onsite, spin = key)

            # now plot
            integrated_coop[key] = np.sum(coop[key].flatten()[self.eigvals[key][:, :].flatten() <= self.efermi + self.energy_shift])
            print("integrated occupied coop:",[key], integrated_coop[key])
            # print("max eigval:",self.eigval[:, self.num_orbs-1, 0])
            flat_coop[key] = coop[key].flatten()

            # create the DOS by histograms
            # myproj, x, _ = ax.hist(shifted_energies[shifted_energies < 1], bins=numbins,weights=flat_coop[shifted_energies < 1],color="white", orientation='horizontal')#- self.efermi
            # smooth_coop = smooth(myproj, 2)
            # smooth_coop = myproj

            # create the DOS by sum of gaussians
            flat_coop[key] = flat_coop[key][shifted_energies[key] < ylim[1]]
            shifted_energies[key] = shifted_energies[key][shifted_energies[key] < ylim[1]]
            sig = sigma  # 0.15#5/self.num_kpts**(1/2)
            #print(key,flat_coop[key].shape,shifted_energies[key].shape,energies.shape)
            all_coop = 1 / (sig * (2 * np.pi) ** (1 / 2)) * np.exp(
                -1 / 2 * ((energies[:, None] - shifted_energies[key][None, :]) / sig) ** 2) * flat_coop[key][None, :]
            total_coop[key] = np.sum(all_coop, axis=1)

        fig = plt.figure()
        fig.set_size_inches(3, 5)
        ax = fig.add_subplot(111)
        # plot vertically
        ax.plot(np.zeros(points), energies,'--', color="grey",linewidth=0.8)

        linestyle = ["","--"]
        color = ["dimgray","lightseagreen"]
        mini = []
        maxi = []
        data = {}
        for key in self.keys:
            data[key] = [total_coop[key],energies,linestyle[key],color[key]]
            ax.plot(total_coop[key], energies,linestyle[key], color=color[key])
            mini.append(np.min(total_coop[key]) - 0.05)
            maxi.append(np.max(total_coop[key]) + 0.05)
        mini = min(mini)
        maxi = max(maxi)
        # mini = -1.2
        # maxi = 0.5
        ax.plot([mini, maxi], [0, 0],"--", color="grey",linewidth=0.8)
        ax.set_ylabel("Energy (eV)")
        if orbProj:
            ax.set_xlabel(label)
        else:
            ax.set_xlabel("COOP")
        plt.ylim(ylim)
        plt.xlim((mini, maxi))
        print("showing plot")
        # plt.xlim((mini,maxi))
        #plt.gca().invert_xaxis()
        fig.tight_layout()
        if orbProj:
            #plt.savefig(self.directory+'projectedDOS.png',transparent=True,dpi=150,format='png')
            #plt.show()
            #plt.close()
            return ax, data
        else:
            plt.savefig(self.directory+'COOP_DOS.png',transparent=True,dpi=150,format='png')
            if self.show_figs:
                plt.show()
            plt.close()

    def get_projectedDOS(self, elem : str, ylim: list = (-10, 10), sigma: float=0.1) -> None:
        """
        Returns the COGITO-orbital projected density of states for a specific element. By construction, this performs the projection by Mulliken-style analysis.
        Projection without Mulliken is obfuscated by the nonorthogonal nature of the orbitals, which causes much larger projection at high energy states due to antibonding.

        Args:
            elem (str): String of the element label. For example, "Pb".
            ylim (tuple): The y-axis limits.
            sigma (float): Sets the spread (in energy) of the gaussian for the each wavefunction COOP. The full pCOOP is created by summing the gaussians.
                        This should trade off with k-point resolution. Low resolution requires high sigma (0.2) to smear between points. High resoltion can have low sigma (0.02).

        Returns:
            png: Saves the projection plot at self.directory+elem+'projectedDOS.png'.
        """
        self.NN_index, self.NN_dists = self.TB_model.get_neighbors(self)

        allorbs = {}
        for el in self.elements:
            allorbs[el] = ["s","p","d","f"]
        #elem = elem # list(orbdict.keys())[0]
        #orbs = list(orbdict.values())[0]
        #all_orb = ''
        #for orb in orbs:
        #    all_orb = all_orb+orb
        #full_label = "% "+elem+" "+all_orb

        iselemorb = np.array(self.elements)[np.array(self.orbatomnum)] == elem
        elemorbs = np.array(self.exactorbtype)[iselemorb]
        elemorbs = np.array([s[0] for s in elemorbs])
        uniqelemorbs = np.unique(elemorbs)

        # make energies
        shifted_energies = {}
        all_shifted_ener = self.eigvals[:, :, :].flatten() - (self.efermi + self.energy_shift)
        for key in self.keys:
            shifted_energies[key] = self.eigvals[key, :, :].flatten() - (self.efermi + self.energy_shift)
        all_shifted_ener = all_shifted_ener[all_shifted_ener < ylim[1]]
        minx = np.min(all_shifted_ener) - sigma * 2
        maxx = np.max(all_shifted_ener)  # +sigma*2
        points = 200
        energies = np.linspace(minx, maxx, points)

        allprojs = []
        for orb in uniqelemorbs:
            curorbdict = {elem:[orb]}

            total_coop = {}
            for key in self.keys:  # plot bands for each spin
                coop = self.TB_model.get_COOP(self, [curorbdict,allorbs], include_onsite=True, spin=key)

                # now plot
                integrated_coop = np.sum(
                    coop.flatten()[self.eigvals[key][:, :].flatten() <= self.efermi + self.energy_shift])
                print("integrated occupied coop:", [key], integrated_coop)
                # print("max eigval:",self.eigval[:, self.num_orbs-1, 0])
                flat_coop = coop.flatten()

                # create the DOS by sum of gaussians
                flat_coop = flat_coop[shifted_energies[key] < ylim[1]]
                use_energy = shifted_energies[key][shifted_energies[key] < ylim[1]]
                sig = sigma  # 0.15#5/self.num_kpts**(1/2)
                all_coop = 1 / (sig * (2 * np.pi) ** (1 / 2)) * np.exp(
                    -1 / 2 * ((energies[:, None] - use_energy[None, :]) / sig) ** 2) * flat_coop[None,:]
                total_coop[key] = np.sum(all_coop, axis=1)

            linestyle = ["", "--"]
            color = ["dimgray", "lightseagreen"]
            data = {}
            for key in self.keys:
                data[key] = [total_coop[key], energies, linestyle[key], color[key]]

            allprojs.append(data)

        allatmDOS = {}
        for key in self.keys:  # plot bands for each spin
            # now plot
            integrated_coop = np.sum(
                np.ones(len(self.eigvals[key][:, :].flatten()))[self.eigvals[key][:, :].flatten() <= self.efermi + self.energy_shift])
            print("integrated occupied coop:", [key], integrated_coop)
            # print("max eigval:",self.eigval[:, self.num_orbs-1, 0])
            flat_coop = np.ones(len(self.eigvals[key][:, :].flatten()))/self.num_kpts

            # create the DOS by sum of gaussians
            flat_coop = flat_coop[shifted_energies[key] < ylim[1]]
            use_energy = shifted_energies[key][shifted_energies[key] < ylim[1]]
            sig = sigma  # 0.15#5/self.num_kpts**(1/2)
            all_coop = 1 / (sig * (2 * np.pi) ** (1 / 2)) * np.exp(
                -1 / 2 * ((energies[:, None] - use_energy[None, :]) / sig) ** 2) * flat_coop[None,:]
            allatmDOS[key] = np.sum(all_coop, axis=1)


        fig = plt.figure()
        fig.set_size_inches(3, 5)
        ax = fig.add_subplot(111)
        # plot vertically
        #ax.plot(np.zeros(points), energies, '--', color="grey", linewidth=0.8)
        total_dos = {}
        for key in self.keys:
            total_dos[key] = np.zeros(len(allprojs[0][key][0]))
        for i,data in enumerate(allprojs):
            for key in self.keys:
                #data[key] = [total_coop[key],energies,linestyle[key],color[key]]
                ax.plot(data[key][0], data[key][1],data[key][2],label=uniqelemorbs[i])
                total_dos[key] += data[key][0]

        for key in self.keys:
            ax.plot(total_dos[key], data[key][1],'--',color="gray",label=elem,linewidth=1)
            ax.plot(allatmDOS[key], data[key][1],data[key][2],color="black",label="total",linewidth=1)
        max_proj = np.amax(allatmDOS[key][(data[key][1]>ylim[0]) & (data[key][1] < 0)])

        ax.plot([0,0], [ylim[0],ylim[1]],'--', color="grey",linewidth=0.8)

        ax.plot([-0.05, max_proj+0.1], [0, 0],"--", color="grey",linewidth=0.8)
        plt.xlim((-0.05, max_proj+0.1))
        plt.ylim(ylim)
        ax.set_xlabel("orbital pDOS")
        ax.set_ylabel("Energy (eV)")
        plt.legend()
        plt.tight_layout()
        fig.savefig(self.directory+elem+'projectedDOS.png',transparent=True,dpi=150,format='png')
        if self.show_figs:
            plt.show()
        plt.close()

    def get_ICOOP(self):
        """
        Generates an ICOOP matrix with from the TB_Model static function. Then reduces the matrix size. Used for fast calculation of energies in crystal bond plot or other integrated energy quantities.
        """
        icoop = {}
        for key in self.keys: # plot bands for each spin
            icoop[key] = self.TB_model.get_ICOOP(self, spin = key)

        self.ICOOP = icoop

        if not hasattr(self, 'ICO_trans'):
            # make the size of icoop smaller because most is zeros
            trans = self.num_trans
            old_each_dir = self.num_each_dir
            norbs = self.num_orbs
            round_icoop = np.around(self.ICOOP[0],decimals=5).flatten()
            non_zero_icoop = np.arange(norbs*norbs*trans[0]*trans[1]*trans[2])[round_icoop!=0]
            icoop_indices = np.unravel_index(non_zero_icoop, (norbs, norbs, trans[0], trans[1], trans[2]))
            eachdir1 = np.max(icoop_indices[2]-old_each_dir[0])
            eachdir2 = np.max(icoop_indices[3]-old_each_dir[1])
            eachdir3 = np.max(icoop_indices[4]-old_each_dir[2])
            new_each_dir = np.array([eachdir1,eachdir2,eachdir3],dtype=np.int_)
            print(new_each_dir)
            self.ICO_eachdir = new_each_dir
            self.ICO_trans = new_each_dir*2 + 1
            print("trans for ICOOP:",self.ICO_trans)

        small_overlap = {}
        small_hams = {}
        for key in self.keys: # reduce the size of ICOOP and overlap
            old_each_dir = self.num_each_dir
            new_each_dir = self.ICO_eachdir
            icoop[key] = icoop[key][:,:, old_each_dir[0] - new_each_dir[0]: old_each_dir[0] + new_each_dir[0] + 1,
                                    old_each_dir[1] - new_each_dir[1]:old_each_dir[1] + new_each_dir[1] + 1,
                                    old_each_dir[2] - new_each_dir[2]:old_each_dir[2] + new_each_dir[2] + 1]
            small_overlap[key] = self.overlaps_params[key][:,:, old_each_dir[0] - new_each_dir[0]: old_each_dir[0] + new_each_dir[0] + 1,
                                    old_each_dir[1] - new_each_dir[1]:old_each_dir[1] + new_each_dir[1] + 1,
                                    old_each_dir[2] - new_each_dir[2]:old_each_dir[2] + new_each_dir[2] + 1]
            small_hams[key] = self.TB_params[key][:,:, old_each_dir[0] - new_each_dir[0]: old_each_dir[0] + new_each_dir[0] + 1,
                                    old_each_dir[1] - new_each_dir[1]:old_each_dir[1] + new_each_dir[1] + 1,
                                    old_each_dir[2] - new_each_dir[2]:old_each_dir[2] + new_each_dir[2] + 1]

        self.ICOOP = icoop
        self.small_overlap = small_overlap
        self.small_hams = small_hams

        for key in self.keys:
            np.save(self.directory+"small_ham"+str(key),self.small_hams[key])
            np.save(self.directory+"small_over"+str(key),self.small_overlap[key])

        return icoop

    def get_bond_occup(self):
        """
        Generates an bond occupation matrix with from the TB_Model static function. Then reduces the matrix size. Used for fast calculation of energies in crystal bond plot or other integrated energy quantities.
        """
        bond_occup = {}
        for key in self.keys: # plot bands for each spin
            bond_occup[key] = self.TB_model.get_bond_occup(self, spin = key)

        if not hasattr(self, 'ICO_trans'):
            # make the size of bond_occup smaller because most is zeros
            trans = self.num_trans
            old_each_dir = self.num_each_dir
            norbs = self.num_orbs
            round_bond_occup = np.around(bond_occup[0],decimals=5).flatten()
            non_zero_bond_occup = np.arange(norbs*norbs*trans[0]*trans[1]*trans[2])[round_bond_occup!=0]
            bond_occup_indices = np.unravel_index(non_zero_bond_occup, (norbs, norbs, trans[0], trans[1], trans[2]))
            eachdir1 = np.max(bond_occup_indices[2]-old_each_dir[0])
            eachdir2 = np.max(bond_occup_indices[3]-old_each_dir[1])
            eachdir3 = np.max(bond_occup_indices[4]-old_each_dir[2])
            new_each_dir = np.array([eachdir1,eachdir2,eachdir3],dtype=np.int_)
            print(new_each_dir)
            self.ICO_eachdir = new_each_dir
            self.ICO_trans = new_each_dir*2 + 1
            print("trans for bond_occup:",self.ICO_trans)

        for key in self.keys: # reduce the size of bond_occup and overlap
            old_each_dir = self.num_each_dir
            new_each_dir = self.ICO_eachdir
            bond_occup[key] = bond_occup[key][:,:, old_each_dir[0] - new_each_dir[0]: old_each_dir[0] + new_each_dir[0] + 1,
                                    old_each_dir[1] - new_each_dir[1]:old_each_dir[1] + new_each_dir[1] + 1,
                                    old_each_dir[2] - new_each_dir[2]:old_each_dir[2] + new_each_dir[2] + 1]

        self.bond_occup = bond_occup

        return bond_occup

    def save_bond_occup(self):
        """
        Saves the smaller bond occupation (density) matrix to (self.directory+"bond_occup"+str(key)+".npy") for reading in later with COGITOico.
        """
        if not hasattr(self, 'bond_occup'):
            self.get_bond_occup()

        for key in self.keys:
            np.save(self.directory+"bond_occup"+str(key),self.bond_occup[key])

    def save_ICOOP(self):
        """
        Saves the smaller ICOOP matrix to (self.directory+"ICOOP"+str(key)+".npy") for reading in later with COGITOico.
        """
        if not hasattr(self, 'ICOOP'):
            self.get_ICOOP()

        for key in self.keys:
            np.save(self.directory+"ICOOP"+str(key),self.ICOOP[key])
        '''
        for key in self.keys: # plot bands for each spin

            # write the TB parameters
            wannier_hr = open(self.directory + "ICOOP"+self.file_suff+str(key)+".txt", "w")
            wannier_hr.write("File generated by Emily Oliphant\n")

            num_trans = self.ICO_trans # self.num_trans
            each_dir = self.ICO_eachdir # self.num_each_dir
            wannier_hr.write(str(each_dir[0])+" "+str(each_dir[1])+" "+str(each_dir[2])+" "+str(self.num_orbs) + "\n")

            wannier_hr.write("start params\n")
            for a1 in range(num_trans[0]):
                trans1 = int(a1 - np.floor(num_trans[0] / 2))
                for a2 in range(num_trans[1]):
                    trans2 = int(a2 - np.floor(num_trans[1] / 2))
                    for a3 in range(num_trans[2]):
                        trans3 = int(a3 - np.floor(num_trans[2] / 2))
                        for orb1 in range(self.num_orbs):
                            for orb2 in range(self.num_orbs):
                                realpart = f"{self.ICOOP[key][orb1, orb2, a1, a2, a3].real:.6f}"
                                imagpart = f"{self.ICOOP[key][orb1, orb2, a1, a2, a3].imag:.6f}"
                                wannier_hr.write(
                                    '{:>5} {:>5} {:>5} {:>5} {:>5} {:>14} {:>14}'.format(str(trans1), str(trans2),
                                                                                         str(trans3),
                                                                                         str(orb1 + 1),
                                                                                         str(orb2 + 1), realpart,
                                                                                         imagpart) + "\n")
        '''

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
            self.TB_model.read_orbitals()
            self.real_orbcoeffs = self.TB_model.real_orbcoeffs
            self.orb_radii = self.TB_model.orb_radii
            self.recip_orbcoeffs = self.TB_model.recip_orbcoeffs
            self.sph_harm_key = self.TB_model.sph_harm_key

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

    def get_bonds_figure(self, energy_cutoff: float = 0.1, bond_max: float = 3.0, elem_colors: list = [], atom_colors: list = [], atom_labels: list = [], plot_atom: int = None,
                         one_atom: bool = False, fovy: float = 10,return_fig: bool = False) -> None:
        """
        This is deprecated.
        This will plot the crystal structure atoms with line weighted by iCOHP.
        Each line should also be hoverable to reveal the number and amounts that are s-s,s-p, and p-p.

        Args:    
            energy_cutoff (float): This is the minimum bond magnitude that will be plotted.
            bond_max (float): The maximum bond distance that will be plotted outside the primitive cell.
            elem_colors (list): Colors for the elements based on order in tb_input. Length of list should be the number of
                        unique elements. Can either be integer list to reference the default colors or list of
                        plotly compatable colors.
            atom_colors (list): Colors for the atoms based on order in tb_input. Length of list should be the number of
                        atoms in the primitive cell. Can either be integer list to reference the default colors or
                        list of plotly compatable colors. If not set defaults to elem_colors.
            atom_labels (list): List of atom labels as a string.
            plot_atom (int): Set with one_atom=True, plots only one atom and it's bonds, this passes the atom number to plot.
            one_atom (bool): Whether only the atom defined in plot_atom should be plotted; default is False.
            fovy (float): field of view in the vertical direction. Use this tag to adjust depth perception in crystal.
                        Set between 3 (for close to orthographic) and 30 (for good perspective depth).
            return_fig (bool): If False, this function saves figure to crystal_bonds.html. If True, this function will return the plotly figure object.

        Returns:    
            html: Saves the crystal bond plot to (self.directory + "crystal_bonds.html"). Also plots figure if self.show_figs = True.
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

        # get the cohp
        if not hasattr(self, 'ICOHP'):
            self.get_ICOHP()
        icohp_orb = self.ICOHP
        # each_dir = np.array([1, 1, 1]) #self.num_each_dir
        cell = self.ICO_trans  # np.array(self.num_trans) # each_dir * 2 + 1
        vec_to_trans = np.zeros((cell[0], cell[1], cell[2], 3))  # self.vec_to_trans
        each_dir = self.ICO_eachdir  # self.num_each_dir

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
        fig.write_html(self.directory + "crystal_bonds.html",post_script=post_script)

        if self.show_figs:
            fig.show(post_script=post_script)

    def get_bonds_charge_figure(self, energy_cutoff: float = 0.1, bond_max: float = 3.0, elem_colors: list = [], atom_colors: list = [], atom_labels: list = [],
                                auto_label: str = "", plot_atom: int = None,
                         one_atom: bool = False, fovy: float = 10,return_fig: bool = False) -> None:
        """
        Plots the crystal structure atoms with bonds plotted based on iCOHP.
        Each line should also be hoverable to reveal the number and amounts that are s-s,s-p, and p-p.
        The charge and magnetic moment will also be plotted acording to auto_label.

        Args:    
            energy_cutoff (float): This is the minimum bond magnitude that will be plotted.
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
                        "mulliken" -Plots the onsite charge and mag (if spin_polar) on atoms by mulliken population (overrides atom_labels).
                        "full" - Plots the charge and magnetics moments (if spin_polar) on atoms and bonds (overrides atom_labels).
                        "color" - Colors the atoms and bonds based on their charge (overrides atom_colors or elem_colors).
                        "color mag" - Colors the atoms and bonds based on their magnetic moments (overrides atom_colors or elem_colors).
                        NOTE: Only use "mulliken" OR "full", NOT both
            plot_atom (int): Set with one_atom=True, plots only one atom and it's bonds, this passes the atom number to plot.
            one_atom (bool): Whether only the atom defined in plot_atom should be plotted; default is False.
            fovy (float): field of view in the vertical direction. Use this tag to adjust depth perception in crystal.
                        Set between 3 (for close to orthographic) and 30 (for good perspective depth).
            return_fig (bool): If False, this function saves figure to crystal_bonds.html. If True, this function will return the plotly figure object.

        Returns:    
            None: Depends on return_fig parameter.
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
        do_coop = False
        if "full" in auto_label:
            do_coop = True
        # get the cohp
        if not hasattr(self, 'ICOHP'):
            self.get_ICOHP()
        if do_coop:
            if not hasattr(self, 'ICOOP'):
                self.get_ICOOP()
            icoop_orb = self.ICOOP
        icohp_orb = self.ICOHP

        # each_dir = np.array([1, 1, 1]) #self.num_each_dir
        cell = self.ICO_trans  # np.array(self.num_trans) # each_dir * 2 + 1
        vec_to_trans = np.zeros((cell[0], cell[1], cell[2], 3))  # self.vec_to_trans
        each_dir = self.ICO_eachdir  # self.num_each_dir

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

    def get_bond_density_figure(self, energy_cutoff: float = 0.1, iso_max: float = 0.03, iso_min: float = -0.003, elem_colors: list = [], atom_colors: list = [], atom_labels: list = [],
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
            bond_max (float): Atoms are plotted outside the primitive cell only if they have a bond with
                        a primitive cell atom that is shorter than this amount. This does not filter bonds between atoms that are already plotted.
            fovy (float): field of view in the vertical direction. Use this tag to adjust depth perception in crystal.
                        Set between 3 (for close to orthographic) and 30 (for good perspective depth).
            return_fig (bool): If False, this function saves figure to crystal_bonds.html. If True, this function will return the plotly figure object.


        Returns:    
            html, or object: Depends on return_fig parameter. Default is to return nothing but save "crystal_bonds.html".
        """

        import plotly.graph_objs as go
        from skimage.measure import marching_cubes
        if not hasattr(self, 'NN_index'):
            self.get_neighbors()
        if not hasattr(self, 'real_orbcoeffs'):
            self.TB_model.read_orbitals()
            self.real_orbcoeffs = self.TB_model.real_orbcoeffs
            self.orb_radii = self.TB_model.orb_radii
            self.recip_orbcoeffs = self.TB_model.recip_orbcoeffs
            self.sph_harm_key = self.TB_model.sph_harm_key
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
        if not hasattr(self, 'ICOHP'):
            self.get_ICOHP()
        if do_coop:
            if not hasattr(self, 'ICOOP'):
                self.get_ICOOP()
            icoop_orb = self.ICOOP
        icohp_orb = self.ICOHP
        # each_dir = np.array([1, 1, 1]) #self.num_each_dir
        cell = self.ICO_trans  # np.array(self.num_trans) # each_dir * 2 + 1
        vec_to_trans = np.zeros((cell[0], cell[1], cell[2], 3))  # self.vec_to_trans
        each_dir = self.ICO_eachdir  # self.num_each_dir

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
            if self.verbose > 1:
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

                        if is_new_bond == True:
                            #all_start_atms = np.append(all_start_atms,np.array([start_atm]),axis=0)
                            #all_end_atms = np.append(all_end_atms,np.array([end_atm]),axis=0)
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
                                            end] + " " + orb2 + ": " + str(np.around(bond_eng, decimals=3)) + " eV" + "<br>"
                                        bond_label += bond_str
                            spin_label = ""
                            if self.spin_polar:
                                spin_label += "<br>Energy from spin up: " + str(
                                    np.around(bonds_spin[0][b_ind], decimals=3)) + " eV"
                                spin_label += "<br>Energy from spin down: " + str(
                                    np.around(bonds_spin[1][b_ind], decimals=3)) + " eV"
                            full_text = ("Bond energy: " + str(np.around(bond_energy, decimals=3)) + " eV" + "<br>Bond length: "
                                            + str(np.around(bond_dist, decimals=3)) + " Å" + " (" + str(NN) + "NN)" +
                                             spin_label + "<br>Breakdown into orbitals: <br>" + bond_label)
                            #all_atm_elems.append([atom_elems[start],atom_elems[end]])
                            #all_atm_labels.append([atom_labels[start],atom_labels[end]])
                            #atm1, atm2, cell1, cell2, cell3 = np.unravel_index(b_ind, (
                            #    self.numAtoms, self.numAtoms, cell[0], cell[1], cell[2]))
                            #all_atm_inds.append([atm1,atm2])
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
                            #all_colors.append([atm_colors[start], atm_colors[end]])
                            #all_energies.append(bond_energy)
                            #all_styles.append(style)
                            #all_atmrads.append([start_rad, end_rad])

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
                            if False:#style == 'dash':
                                closer_atom1 = (start_atm*5 + end_atm) / 6
                                closer_atom2 = (start_atm + end_atm*5) / 6
                                all_centers = np.append(all_centers,np.array([[centerx,centery,centerz],closer_atom1,closer_atom2]),axis=0)
                                all_text.append(full_text)
                                all_text.append(full_text)
                                all_text.append(full_text)
                                all_customdata.append(bond_label) #np.around(bond_energy,decimals=1))
                                all_customdata.append(bond_label)
                                all_customdata.append(bond_label)
                                data.append(go.Scatter3d(
                                    x=[start_atm[0], end_atm[0]],
                                    y=[start_atm[1], end_atm[1]],
                                    z=[start_atm[2], end_atm[2]],
                                    mode='lines',#'markers+lines',
                                    #marker=dict(size=[start_rad * 40, end_rad * 40],
                                    #            color=[atm_colors[start], atm_colors[end]], opacity=1.0,
                                    #            line=dict(color='black', width=20)),
                                    line=dict(
                                        color=color,
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
                                # instead generate the bond mesh with actually bond density
                                atm1, atm2, cell1, cell2, cell3 = np.unravel_index(b_ind, (
                                self.numAtoms, self.numAtoms, cell[0], cell[1], cell[2]))
                                orbs1 = np.arange(self.num_orbs)[self.orbatomnum == atm1]
                                orbs2 = np.arange(self.num_orbs)[self.orbatomnum == atm2]
                                bond_coop = icoop_orb[0][orbs1, :][:, orbs2][:, :, cell1, cell2, cell3]
                                #print(bond_coop)
                                orb_lap = self.small_overlap[0][orbs1, :][:, orbs2][:, :, cell1, cell2, cell3].flatten()
                                bond_coop = bond_coop.flatten()
                                bond_coop[orb_lap!=0] = bond_coop[orb_lap!=0] / orb_lap[orb_lap!=0]
                                bond_coop = np.reshape(bond_coop,(len(orbs1),len(orbs2)))
                                lowxyz = [np.amin([start_atm[0],end_atm[0]]),np.amin([start_atm[1],end_atm[1]]),np.amin([start_atm[2],end_atm[2]])]
                                highxyz = [np.amax([start_atm[0],end_atm[0]]),np.amax([start_atm[1],end_atm[1]]),np.amax([start_atm[2],end_atm[2]])]
                                res = 20
                                X, Y, Z = np.mgrid[lowxyz[0] - 1:highxyz[0] + 1:res*1j,
                                          lowxyz[1] - 1:highxyz[1] + 1:res*1j, lowxyz[2] - 1:highxyz[2] + 1:res*1j]
                                cartXYZ = np.array([X.flatten(), Y.flatten(), Z.flatten()])
                                bond_density = self.make_bond(atm1, atm2, start_atm, end_atm, bond_coop, cartXYZ)
                                print("bond density min and max:",np.amin(bond_density),np.amax(bond_density))
                                #print(bond_coop.shape,atm1,atm2,cell1, cell2, cell3)
                                #print(start_atm,end_atm)
                                #print(self.sph_harm_key[orbs1],self.sph_harm_key[orbs2])
                                #print(bond_coop)
                                dense_min = np.amin(bond_density)
                                dense_max = np.amax(bond_density)
                                isoval = np.array([dense_min,dense_max])[np.argmax([-dense_min,dense_max])]/2
                                if dense_max > iso_max:
                                    all_start_atms = np.append(all_start_atms, np.array([start_atm]), axis=0)
                                    all_end_atms = np.append(all_end_atms, np.array([end_atm]), axis=0)
                                    all_atm_elems.append([atom_elems[start],atom_elems[end]])
                                    all_atm_labels.append([atom_labels[start],atom_labels[end]])
                                    all_colors.append([atm_colors[start], atm_colors[end]])
                                    all_energies.append(bond_energy)
                                    all_styles.append(style)
                                    all_atmrads.append([start_rad, end_rad])
                                    atm1, atm2, cell1, cell2, cell3 = np.unravel_index(b_ind, (
                                        self.numAtoms, self.numAtoms, cell[0], cell[1], cell[2]))
                                    all_atm_inds.append([atm1,atm2])
                                    isoval = iso_max
                                    spacing = (np.array(highxyz) - np.array(lowxyz) + 2)/(res-1)
                                    verts, faces, _, _ = marching_cubes(np.reshape(bond_density, (res, res, res)), isoval,spacing=(spacing[0],spacing[1],spacing[2]))
                                    verts = verts + (np.array(lowxyz)-1)[None,:]
                                    data.append(go.Mesh3d(
                                        x = verts[:, 0],
                                        y = verts[:, 1],
                                        z = verts[:, 2],
                                        i = faces[:, 0],
                                        j = faces[:, 1],
                                        k = faces[:, 2],
                                        opacity=1.0,
                                        # color=color,
                                        color= color,
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
                                    all_atm_elems.append([atom_elems[start],atom_elems[end]])
                                    all_atm_labels.append([atom_labels[start],atom_labels[end]])
                                    atm1, atm2, cell1, cell2, cell3 = np.unravel_index(b_ind, (
                                        self.numAtoms, self.numAtoms, cell[0], cell[1], cell[2]))
                                    all_atm_inds.append([atm1,atm2])
                                    all_colors.append([atm_colors[start], atm_colors[end]])
                                    all_energies.append(bond_energy)
                                    all_styles.append(style)
                                    all_atmrads.append([start_rad, end_rad])
                                    isoval=iso_min
                                    spacing = (np.array(highxyz) - np.array(lowxyz) + 2)/(res-1)
                                    verts, faces, _, _ = marching_cubes(np.reshape(bond_density, (res, res, res)), isoval,spacing=(spacing[0],spacing[1],spacing[2]))
                                    verts = verts + (np.array(lowxyz)-1)[None,:]
                                    data.append(go.Mesh3d(
                                        x = verts[:, 0],
                                        y = verts[:, 1],
                                        z = verts[:, 2],
                                        i = faces[:, 0],
                                        j = faces[:, 1],
                                        k = faces[:, 2],
                                        opacity=1.0,
                                        # color=color,
                                        color= "black",
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

        #print(all_end_atms.shape,all_start_atms.shape)
        # find all unique atoms
        combnd_atms = np.append(all_start_atms[1:],all_end_atms[1:],axis=0)
        all_colors = np.array(all_colors)
        all_atmrads = np.array(all_atmrads)
        all_atm_elems = np.array(all_atm_elems)
        all_atm_labels = np.array(all_atm_labels)
        all_atm_inds = np.array(all_atm_inds)
        combnd_colors = np.append(all_colors[:,0],all_colors[:,1])
        combnd_atmrad = np.append(all_atmrads[:,0],all_atmrads[:,1])
        combnd_elems = np.append(all_atm_elems[:,0],all_atm_elems[:,1])
        combnd_labels = np.append(all_atm_labels[:,0],all_atm_labels[:,1])
        combnd_atminds = np.append(all_atm_inds[:,0],all_atm_inds[:,1])
        uniq_atms,indices = np.unique(combnd_atms,axis=0,return_index=True)
        uniq_colors = combnd_colors[indices]
        uniq_rad = combnd_atmrad[indices]
        uniq_elems = combnd_elems[indices]
        uniq_labels = combnd_labels[indices]
        uniq_atminds = combnd_atminds[indices]

        if self.verbose > 1:
            print("all atms:",uniq_atms)
            print(uniq_rad)
            print(uniq_colors)

        # append the bond labels
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

        # instead generate the bond mesh with actually bond density
        # Loop over atoms to create a sphere for each atom
        for ai,atm in enumerate(uniq_atms):
            atm1 = uniq_atminds[ai]
            orbs1 = np.arange(self.num_orbs)[self.orbatomnum == atm1]
            center = atm
            bond_coop = icoop_orb[0][orbs1, :][:, orbs1][:, :, each_dir[0],each_dir[1],each_dir[2]]
            #print(bond_coop)
            orb_lap = self.small_overlap[0][orbs1, :][:, orbs1][:, :, each_dir[0],each_dir[1],each_dir[2]].flatten()
            bond_coop = bond_coop.flatten()
            bond_coop[orb_lap != 0] = bond_coop[orb_lap != 0] / orb_lap[orb_lap != 0]
            bond_coop = np.reshape(bond_coop, (len(orbs1), len(orbs1)))
            bond_diag = np.diag(bond_coop)
            is_p = []
            for oind, orb in enumerate(orbs1):
                #print(self.exactorbtype[orb])
                if "p" in self.exactorbtype[orb]:
                    is_p.append(oind)
                if "s" in self.exactorbtype[orb]:
                    s_ind = oind
            #is_p = np.array(is_p)
            #print(is_p)
            total_p = np.sum(bond_diag[is_p])
            s_val = bond_diag[s_ind]
            hold_zeros = np.zeros(len(bond_diag))
            bond_diag = hold_zeros
            bond_diag[s_ind] = np.sum(np.abs(bond_coop))/20 # min(s_val,total_p)
            #print("orb vals:",total_p,min(s_val,total_p))
            np.fill_diagonal(bond_coop,0)
            combo_electrons = np.sum(np.abs(bond_coop))
            if np.sum(np.abs(bond_coop)) > 0.1: # only continue with decent amount of lone pair character
                bond_coop = bond_coop + np.diag(bond_diag)
                res = 30
                X, Y, Z = np.mgrid[center[0] - 1.5:center[0] + 1.5:res*1j,
                          center[1] - 1.5:center[1] + 1.5:res*1j, center[2] - 1.5:center[2] + 1.5:res*1j]

                cartXYZ = np.array([X.flatten(), Y.flatten(), Z.flatten()])
                atom_density = self.make_bond(atm1, atm1, center, center, bond_coop, cartXYZ)
                label_text = uniq_labels[ai] + "<br>Lone pair from " + str(np.around(combo_electrons,decimals=3)) + " sp overlap"
                #print("atom density min and max:", np.amin(atom_density), np.amax(atom_density))
                #print(bond_coop.shape, atm1, atm2)
                #print(center)
                #print(self.sph_harm_key[orbs1])
                #print(bond_coop)
                isoval = np.amax(np.abs(atom_density)) / 2
                spacing = 3 / (res - 1)
                verts, faces, _, _ = marching_cubes(np.reshape(atom_density, (res, res, res)), isoval, spacing=(spacing, spacing, spacing))
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
            fig.write_html(self.directory + "crystal_bond_density.html",post_script=post_script,div_id = "plotDiv")

            if self.show_figs:
                fig.show(post_script=post_script,div_id = "plotDiv")

    def jsonify_bonddata(self, minimum_cohp: float = 0.00001,make_bond_info_txt=True):
        """
        This function covers the raw ICOHP and ICOOP and bond_occup matrices into something more interpretable while still preserving norms.
        This constructs three json files:
        all_bonds.json : includes bonds between all atoms that have a elements above the minimum_cohp.
        all_unique_bonds : groups the all_bonds into unique ones for more interpretable viewing.
        all_atoms.json : all onsite atom information, including the bond orb_orb_dict + added onsite and mulliken
                        orbital occupations and partial charges + lone-pair-like energy (cohp) contributions.
        The structure of the all_bonds json:
        {"uniq_bonds by "el1_el2_dist_energy": {
                        "bondlength": float, "degeneracy": int, "cohp": float , "coop": float, "bondmagmom": float,
                "all atom-atom bonds in this set by 'atm1_atm2_T1_T2_T3':{  # just 'atm1_atm2' is not enough since atoms can bond outside unit cell
                                "atmorbs1, atmorbs2": [[int...],[int...]] # used to reference orbital energy, radius, and shape
                                "cohp": [float...] , "coop": [float...], "bondlength": float, "bondmagmom": float, "angle"?:float,  # same data just without averaging
                        "orb-orb e.g. 's-d'": {
                                        "cohp": [float...], "coop": [float...], # is referenced to spin state. For non spin-polarized: [float], for spin-polar: [float,float]
                                        "orbnums1, orbnums2": [[int...],[int...]] for orbital numbers e.g. for s-p: [[0],[1,2,3]],  # orbital number referenced to atom already selected
                                        "bond_occup_matrix": [numorbs1×numorbs2...], "orb_ediff_matrix": [numorbs1×numorbs2...],
                                        "cohp_matrix": [numorbs1×numorbs2...], "coop_matrix": [numorbs1×numorbs2...],
                                        "S_matrix": [numorbs1×numorbs2...], "H_matrix": [numorbs1×numorbs2...],
                        }
                }
        }
        @return:
        """

        do_coop = True
        # get the cohp
        if not hasattr(self, 'ICOHP'):
            self.get_ICOHP()
        if not hasattr(self, 'ICOOP'):
            self.get_ICOOP()
        if not hasattr(self, 'bond_occup'):
            self.get_bond_occup()
        icoop_orb = self.ICOOP
        icohp_orb = self.ICOHP
        overlaps = self.small_overlap
        hams = self.small_hams
        bond_occup = self.bond_occup
        # each_dir = np.array([1, 1, 1]) #self.num_each_dir
        cell = self.ICO_trans  # np.array(self.num_trans) # each_dir * 2 + 1
        vec_to_trans = np.zeros((cell[0], cell[1], cell[2], 3))  # self.vec_to_trans
        each_dir = self.ICO_eachdir  # self.num_each_dir

        for x in range(cell[0]):
            for y in range(cell[1]):
                for z in range(cell[2]):
                    vec_to_trans[x, y, z] = [x - each_dir[0], y - each_dir[1], z - each_dir[2]]

        num_total_atoms = self.numAtoms * cell[0] * cell[1] * cell[2]
        vec_to_atoms = vec_to_trans[None, :, :, :] + np.array(self.primAtoms)[:, None, None, None]
        vec_to_atoms = np.reshape(vec_to_atoms, (num_total_atoms, 3)).T
        # print(vec_to_atoms)
        cart_to_atoms = _red_to_cart((self._a[0], self._a[1], self._a[2]), vec_to_atoms.transpose())
        cart_to_atoms = np.reshape(cart_to_atoms,(self.numAtoms, cell[0], cell[1], cell[2],3))

        if self.spin_polar:
            keys = [0, 1]
            spin_degen = 1
        else:
            keys = [0]
            spin_degen = 2

        all_bonds_spin = {}
        bonds_spin = {}
        charge_spin = {}
        for key in keys:
            icohp_orb[key] = icohp_orb[key].real  # self.TB_model.get_ICOHP(self,spin=key).real
            icoop_orb[key] = icoop_orb[key].real
            overlaps[key] = overlaps[key].real
            hams[key] = hams[key].real

        added_atoms = 0
        rejected_atoms = 0
        bond_uniq_keys = []
        bond_group_keys = []
        all_atoms_bonds = {}
        all_atoms_onsite = {}
        for atm1 in range(self.numAtoms):
            for atm2 in range(self.numAtoms):
                # get atom orbital information
                orbs1 = np.arange(self.num_orbs)[self.orbatomnum == atm1]
                orbs2 = np.arange(self.num_orbs)[self.orbatomnum == atm2]
                orb_type1 = self.exactorbtype[orbs1]
                orb_type2 = self.exactorbtype[orbs2]
                # section orbitals by type (s, p, d, f)
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

                for T1 in range(cell[0]):
                    for T2 in range(cell[1]):
                        for T3 in range(cell[2]):
                            bond_key = str(atm1)+", "+str(atm2)+", "+str(T1-each_dir[0])+", "+str(T2-each_dir[1])+", "+str(T3-each_dir[2])

                            sum_cohp = {}
                            sum_coop = {}
                            sum_overlaps = {}
                            sum_hams = {}
                            sum_bond_occup = {}
                            for key in keys:
                                sum_cohp[key] = icohp_orb[key][orbs1, :][:, orbs2][:,:,T1,T2,T3] * spin_degen
                                sum_coop[key] = icoop_orb[key][orbs1, :][:, orbs2][:,:,T1,T2,T3] * spin_degen
                                sum_overlaps[key] = overlaps[key][orbs1, :][:, orbs2][:,:,T1,T2,T3]
                                sum_hams[key] = hams[key][orbs1, :][:, orbs2][:,:,T1,T2,T3]
                                sum_bond_occup[key] = bond_occup[key][orbs1, :][:, orbs2][:,:,T1,T2,T3] * spin_degen

                            orbkeytypes = ["s","p","d","f"]
                            orb_orb_dict = {}
                            total_bond_cohp = np.zeros(len(keys))
                            total_bond_coop = np.zeros(len(keys))
                            max_cohp = 0
                            for soi1, spec_orbs1 in enumerate([sorbs1,porbs1,dorbs1,forbs1]):
                                if len(spec_orbs1) != 0:
                                    for soi2, spec_orbs2 in enumerate([sorbs2, porbs2, dorbs2, forbs2]):
                                        if len(spec_orbs2) != 0:
                                            cohp = np.zeros((len(keys),len(spec_orbs1),len(spec_orbs2)))
                                            coop = np.zeros((len(keys),len(spec_orbs1),len(spec_orbs2)))
                                            orborb_overlaps = np.zeros((len(keys),len(spec_orbs1),len(spec_orbs2)))
                                            orborb_hams = np.zeros((len(keys),len(spec_orbs1),len(spec_orbs2)))
                                            energy_diff = np.zeros((len(keys),len(spec_orbs1),len(spec_orbs2)))
                                            orborb_bondoc = np.zeros((len(keys),len(spec_orbs1),len(spec_orbs2)))
                                            orb_key = orbkeytypes[soi1] + "-" + orbkeytypes[soi2]
                                            orb_orb_dict[orb_key] = {}
                                            orb_orb_dict[orb_key]["orbnums"] = [spec_orbs1.tolist(),spec_orbs2.tolist()]
                                            for key in keys:
                                                cohp[key] = sum_cohp[key][spec_orbs1, :][:, spec_orbs2]
                                                coop[key] = sum_coop[key][spec_orbs1, :][:, spec_orbs2]
                                                orborb_overlaps[key] = sum_overlaps[key][spec_orbs1, :][:, spec_orbs2]
                                                orborb_hams[key] = sum_hams[key][spec_orbs1, :][:, spec_orbs2]
                                                energy_diff[key] = self.atomic_energies[key][orbs1[spec_orbs1]][:,None] - self.atomic_energies[key][orbs2[spec_orbs2]][None,:]
                                                orborb_bondoc[key] = sum_bond_occup[key][spec_orbs1, :][:, spec_orbs2]
                                            mcohp = np.amax(np.abs(cohp))
                                            if mcohp > max_cohp:
                                                max_cohp = mcohp
                                            orb_orb_dict[orb_key]["cohp_matrix"] = np.around(cohp,decimals=12).tolist()
                                            orb_orb_dict[orb_key]["coop_matrix"] = np.around(coop,decimals=12).tolist()
                                            orb_orb_dict[orb_key]["overlaps"] = np.around(orborb_overlaps,decimals=12).tolist()
                                            orb_orb_dict[orb_key]["hams"] = np.around(orborb_hams,decimals=12).tolist()
                                            orb_orb_dict[orb_key]["bond_occup"] = np.around(orborb_bondoc,decimals=12).tolist()
                                            orb_orb_dict[orb_key]["orb_ediff"] = np.around(energy_diff,decimals=12).tolist()
                                            tot_cohp = np.sum(cohp,axis=(1,2))
                                            tot_coop = np.sum(coop,axis=(1,2))
                                            total_bond_cohp += tot_cohp
                                            total_bond_coop += tot_coop
                                            orb_orb_dict[orb_key]["cohp"] = np.around(tot_cohp,decimals=12).tolist()
                                            orb_orb_dict[orb_key]["coop"] = np.around(tot_coop,decimals=12).tolist()

                            if atm1==atm2 and T1==each_dir[0] and T2==each_dir[1] and T3==each_dir[2]: # add onsite terms to atom dictionary
                                all_atoms_onsite[str(atm1)] = {}
                                atmelem = self.elements[atm1]

                                # get the onsite orbital occupation and mulliken orbital occupation
                                # just do this per orbitals (any lone-pair like terms can be seen in the coop matrix and
                                # should mostly just be in the cohp/energy term anyways)
                                onsite_occup = np.zeros((len(keys), len(orbs1)))
                                mulli_occup = np.zeros((len(keys), len(orbs1)))
                                orb_energies = np.zeros((len(keys), len(orbs1)))
                                for key in keys:
                                    onsite_occup[key] = np.sum(sum_coop[key],axis=1)
                                    mulli_occup[key] = np.sum(icoop_orb[key][orbs1] * spin_degen,axis=(1,2,3,4))
                                    orb_energies[key] = np.diag(sum_hams[key])

                                onsite_occ_dict = {}
                                mulli_occ_dict = {}
                                onsite_occ_dict_ind = {}
                                mulli_occ_dict_ind = {}
                                grouporb_energies = {}
                                onsite_orbmag = {}
                                mulli_orbmag = {}
                                orbkeytypes = ["s", "p", "d", "f"]
                                for soi, spec_orbs in enumerate([sorbs1, porbs1, dorbs1, forbs1]):
                                    if len(spec_orbs) != 0:
                                        onsite_occ_dict[orbkeytypes[soi]] = np.around(np.sum(onsite_occup[:,spec_orbs],axis=1),decimals=6).tolist()
                                        onsite_occ_dict_ind[orbkeytypes[soi]] = np.around(onsite_occup[:,spec_orbs],decimals=12).tolist()
                                        mulli_occ_dict[orbkeytypes[soi]] = np.around(np.sum(mulli_occup[:,spec_orbs],axis=1),decimals=6).tolist()
                                        mulli_occ_dict_ind[orbkeytypes[soi]] = np.around(mulli_occup[:,spec_orbs],decimals=12).tolist()
                                        if self.spin_polar:
                                            onsite_orbmag[orbkeytypes[soi]] = np.around(np.sum(onsite_occup[0,spec_orbs])-np.sum(onsite_occup[1,spec_orbs]),decimals=6).tolist()
                                            mulli_orbmag[orbkeytypes[soi]] = np.around(np.sum(mulli_occup[0,spec_orbs])-np.sum(mulli_occup[1,spec_orbs]),decimals=6).tolist()
                                        grouporb_energies[orbkeytypes[soi]] = np.around(orb_energies[:,spec_orbs],decimals=6).tolist()

                                # now get lone-pair type terms
                                intra_atm_combos = {}
                                intra_atm_combos['cohp'] = {}
                                intra_atm_combos['coop'] = {}
                                intra_atm_combos['bond_occup'] = {}
                                # kinda dangerous to do but they are not used anywhere else
                                intra_cohp = np.zeros((len(keys),len(orbs1),len(orbs1)))
                                intra_coop = np.zeros((len(keys),len(orbs1),len(orbs1)))
                                intra_bond_occup = np.zeros((len(keys),len(orbs1),len(orbs1)))
                                for key in keys:
                                    intra_cohp[key] = sum_cohp[key]
                                    np.fill_diagonal(intra_cohp[key],0)
                                    intra_coop[key] = sum_coop[key]
                                    np.fill_diagonal(intra_coop[key],0)
                                    intra_bond_occup[key] = sum_bond_occup[key]
                                    np.fill_diagonal(intra_bond_occup[key],0)
                                for soi1, spec_orbs1 in enumerate([sorbs1, porbs1, dorbs1, forbs1]):
                                    if len(spec_orbs1) != 0:
                                        for soi2, spec_orbs2 in enumerate([sorbs2, porbs2, dorbs2, forbs2]):
                                            if len(spec_orbs2) != 0:
                                                tot_cohp = np.sum(intra_cohp[:,spec_orbs1,:][:,:, spec_orbs2])
                                                tot_coop = np.sum(intra_coop[:,spec_orbs1,:][:,:, spec_orbs2])
                                                tot_bond_occup = np.around(intra_bond_occup[:,spec_orbs1,:][:,:, spec_orbs2],decimals=6).tolist()
                                                orb_key = orbkeytypes[soi1] + "-" + orbkeytypes[soi2]
                                                if tot_cohp > 0.001:
                                                    intra_atm_combos['cohp'][orb_key] = float(np.around(tot_cohp,decimals=6))
                                                    intra_atm_combos['coop'][orb_key] = float(np.around(tot_coop,decimals=6))
                                                    intra_atm_combos['bond_occup'][orb_key] = tot_bond_occup

                                all_atoms_onsite[str(atm1)]['elem'] = atmelem
                                all_atoms_onsite[str(atm1)]['onsite_partial_chg'] = float(np.around(self.elec_count[atm1] - np.sum(onsite_occup),decimals=6))
                                all_atoms_onsite[str(atm1)]['mulli_partial_chg'] = float(np.around(self.elec_count[atm1] - np.sum(mulli_occup),decimals=6))

                                all_atoms_onsite[str(atm1)]['onsite_occup'] = onsite_occ_dict
                                all_atoms_onsite[str(atm1)]['onsite_occup_perorb'] = onsite_occ_dict_ind
                                all_atoms_onsite[str(atm1)]['mulli_occup'] = mulli_occ_dict
                                all_atoms_onsite[str(atm1)]['mulli_occup_perorb'] = mulli_occ_dict_ind
                                all_atoms_onsite[str(atm1)]['orb_energies'] = grouporb_energies
                                if self.spin_polar:
                                    all_atoms_onsite[str(atm1)]["onsite_magmom"] = float(np.around(total_bond_coop[0]-total_bond_coop[1],decimals=6))
                                    all_atoms_onsite[str(atm1)]["mulli_magmom"] = float(np.around(np.sum(mulli_occup[0])-np.sum(mulli_occup[1]),decimals=6))
                                    all_atoms_onsite[str(atm1)]["onsite_orbmagmom"] = onsite_orbmag
                                    all_atoms_onsite[str(atm1)]["mulli_orbmagmom"] = mulli_orbmag
                                all_atoms_onsite[str(atm1)]['inter_orb_combos'] = intra_atm_combos
                                all_atoms_onsite[str(atm1)]["orb-orb dict"] = orb_orb_dict
                                all_atoms_onsite[str(atm1)]["cohp"] = np.around(total_bond_cohp,decimals=12).tolist()
                                all_atoms_onsite[str(atm1)]["coop"] = np.around(total_bond_coop,decimals=12).tolist()


                            # add all the values to the atom-atom bond dictionary
                            if max_cohp > minimum_cohp:
                                added_atoms += 1
                                all_atoms_bonds[bond_key] = {}
                                atm_pos1 = cart_to_atoms[atm1,each_dir[0],each_dir[1],each_dir[2]]
                                atm_pos2 = cart_to_atoms[atm2,T1,T2,T3]
                                bond_vec = atm_pos1-atm_pos2
                                bond_length = np.around(np.linalg.norm(bond_vec),decimals=6)
                                bond_elems = np.sort([self.elements[atm1],self.elements[atm2]])
                                group_key = (bond_elems[0]+" "+bond_elems[1]+" "+str(np.around(bond_length,decimals=1))+
                                             " "+str(np.around(np.sum(total_bond_cohp),decimals=1)))
                                all_atoms_bonds[bond_key]["group_key"] = group_key
                                all_atoms_bonds[bond_key]["elements"] = [self.elements[atm1],self.elements[atm2]]
                                all_atoms_bonds[bond_key]["atomorbs1, atomorbs2"] = [orbs1.tolist(),orbs2.tolist()]
                                all_atoms_bonds[bond_key]["bond_length"] = float(bond_length)
                                all_atoms_bonds[bond_key]["orb-orb dict"] = orb_orb_dict
                                all_atoms_bonds[bond_key]["cohp"] = np.around(total_bond_cohp,decimals=12).tolist()
                                all_atoms_bonds[bond_key]["coop"] = np.around(total_bond_coop,decimals=12).tolist()
                                if self.spin_polar:
                                    all_atoms_bonds[bond_key]["bond_magmom"] = float(np.around(total_bond_coop[0]-total_bond_coop[1],decimals=6))

                                # for grouping into bond types
                                bond_uniq_keys.append(bond_key)
                                bond_group_keys.append(group_key)
                            else:
                                rejected_atoms += 1

        print("num atom-atom bonds added vs rejected:", added_atoms, rejected_atoms)
        #print(all_atoms_bonds)
        #print("num atom added vs rejected:", added_atoms, rejected_atoms)
        #print(all_atoms_bonds['0, 2, 0, 0, 0'])
        #print(all_atoms_bonds['2, 0, 0, 0, 0'])

        bond_uniq_keys = np.array(bond_uniq_keys)
        bond_group_keys = np.array(bond_group_keys)
        uniq_bonds, uniq_bond_inverse = np.unique(bond_group_keys, return_inverse=True)
        #print(uniq_bonds)
        #print(len(uniq_bonds),len(bond_uniq_keys),len(uniq_bond_inverse))
        grouped_uniq_keys = []
        for i in range(len(uniq_bonds)):
            grouped_uniq_keys.append(bond_uniq_keys[uniq_bond_inverse==i].tolist())
        #print(grouped_uniq_keys)
        #print(uniq_bond_inverse)

        all_group_bonds = {}
        for i in range(len(uniq_bonds)):
            avg_dist = 0
            avg_cohp = 0
            avg_coop = 0
            if self.spin_polar:
                avg_magmom = 0
            degen = len(grouped_uniq_keys[i])
            for j in grouped_uniq_keys[i]:
                avg_dist += all_atoms_bonds[j]["bond_length"]
                avg_cohp += sum(all_atoms_bonds[j]["cohp"])
                avg_coop += sum(all_atoms_bonds[j]["coop"])
                if self.spin_polar:
                    avg_magmom += all_atoms_bonds[j]["bond_magmom"]
            if avg_dist > 0:
                all_group_bonds[uniq_bonds[i]] = {}
                # Important note: Half of the degeneracies are trivial; from  <i|j+R> and <j|i-R>
                # One could either count that trivial degeneracy in the degen count or double the bond cohp and coop valeus
                # Convention often just doubles the cohp and coop values and considers only true degeneracy count
                all_group_bonds[uniq_bonds[i]]["degeneracy"] = int(degen/2)
                all_group_bonds[uniq_bonds[i]]["bond length"] = round(avg_dist/degen,12)
                all_group_bonds[uniq_bonds[i]]["cohp"] = round(avg_cohp/degen*2,12)
                all_group_bonds[uniq_bonds[i]]["coop"] = round(avg_coop/degen*2,12)
                if self.spin_polar:
                    all_group_bonds[uniq_bonds[i]]['bond_magmom'] = round(avg_magmom/degen*2,12)
                all_group_bonds[uniq_bonds[i]]["all bonds"] = grouped_uniq_keys[i] # will be twice as many keys as degen

        #print(all_group_bonds.keys())
        import json
        with open(self.directory + "all_unique_bonds"+self.file_suff+".json","w") as json_file:
            json.dump(all_group_bonds,json_file)
        with open(self.directory + "all_bonds"+self.file_suff+".json","w") as json_file:
            json.dump(all_atoms_bonds,json_file)
        with open(self.directory + "all_atoms"+self.file_suff+".json","w") as json_file:
            json.dump(all_atoms_onsite,json_file)

        if make_bond_info_txt:
            # print out to a file
            # Define the file name
            file_name = self.directory + "bond_info" + self.file_suff + ".txt"

            total_energy = 0
            total_charge = 0
            if self.spin_polar:
                total_magmom = 0
            # Open the file in write mode and write some data
            with open(file_name, "w") as file:
                file.write("Generated by COGITO\n")
                # Print header
                if self.spin_polar:
                    widths = [8, 8, 10, 12, 14, 12, 8, 10]
                    header = ["Atom1", "Atom2", "dist(A)", "eV/bond", "elec/bond", "mag/bond", "#/unit",
                              "ev/unit"]
                else:
                    widths = [8, 8, 10, 12, 14, 8, 10]
                    header = ["Atom1", "Atom2", "dist(A)", "eV/bond", "elec/bond", "#/unit", "ev/unit"]
                print("".join(f"{header[i]:<{10}}" for i in range(len(header))))
                file.write("".join(f"{header[i]:<{10}}" for i in range(len(header))))
                file.write("\n")

                # go through keys in order of bond length
                all_uniq_keys = list(all_group_bonds.keys())
                bond_dists = np.zeros(len(all_uniq_keys))
                for i, key in enumerate(all_uniq_keys):
                    bond_dists[i] = all_group_bonds[key]["bond length"] # ["degeneracy"] ["cohp"] ["coop"]
                args = np.argsort(bond_dists)
                sorted_keys = np.array(all_uniq_keys)[args]

                for key in sorted_keys:
                    [atom2, atom1] = key.split(" ")[:2]
                    degen = all_group_bonds[key]['degeneracy']
                    b_energy = float(np.around(all_group_bonds[key]['cohp'], decimals=4))
                    b_charge = float(np.around(all_group_bonds[key]['coop'], decimals=4))
                    tot_energy = float(np.around(all_group_bonds[key]['cohp'] * degen, decimals=4))
                    total_energy += all_group_bonds[key]['cohp'] * degen
                    total_charge += all_group_bonds[key]['coop'] * degen
                    dist = np.around(all_group_bonds[key]['bond length'], decimals=2)
                    all_vals = [atom1, atom2, dist, b_energy, b_charge, degen, tot_energy]
                    if self.spin_polar:
                        total_magmom += all_group_bonds[key]['bond_magmom'] * degen
                        b_magmom = float(np.around(all_group_bonds[key]['bond_magmom'], decimals=4))
                        all_vals = [atom1, atom2, dist, b_energy, b_charge, b_magmom, degen, tot_energy]
                    print("".join(f"{str(all_vals[i]):<{10}}" for i in range(len(all_vals))))
                    file.write("".join(f"{str(all_vals[i]):<{10}}" for i in range(len(all_vals))))
                    file.write("\n")
                file.write("\n")
                file.write("total eV (iCOHP) / unit:  " + str(np.around(total_energy, decimals=4)))
                file.write("\n")
                file.write("total bond elec (iCOOP) / unit:  " + str(np.around(total_charge, decimals=4)))
                if self.spin_polar:
                    file.write("\n")
                    file.write("total bond mag charge / unit:  " + str(np.around(total_magmom, decimals=4)))

    def get_bond_info(self):
        """
        Creates the bond_info.txt file which lists all the relavent bonds by there involved atoms, length, energy, charge, and multiplicity.
        """

        print("This function has deprecated, but the same file is generated by the function: jsonify_bonddata(make_bond_info_txt=True)")

    def get_COHP_DOS_bybond(self, sigma = 0.1, return_fig: bool=False):
        """
        Creates an interactive plot pCOHP DOS plot for all relevant bonds and their orbital decomposition.

        Returns:
            object: Plotly figure object.
        """
        if not hasattr(self, 'ICOHP'):
            self.get_ICOHP()
        if not hasattr(self, 'ICOOP'):
            self.get_ICOOP()

        icohp_orb = self.ICOHP
        icoop_orb = self.ICOOP
        # each_dir = np.array([1, 1, 1]) #self.num_each_dir
        cell = self.ICO_trans  # np.array(self.num_trans) # each_dir * 2 + 1
        vec_to_trans = np.zeros((cell[0], cell[1], cell[2], 3))  # self.vec_to_trans
        each_dir = self.ICO_eachdir  # self.num_each_dir

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
        if self.spin_polar:
            keys = [0, 1]
        else:
            keys = [0]

        bonds_spin = {}
        charge_spin = {}
        for key in keys:
            icohp_orb[key] = icohp_orb[key].real  # self.TB_model.get_ICOHP(self,spin=key).real
            # [s,p]*[s,p] = [s-s,s-p,p-s,p-p]
            # [s,p] * [s,p,d] = [s-s,s-p,p-s,p-p,s-d,p-d]
            icohp_atom = np.zeros((self.numAtoms, self.numAtoms, cell[0], cell[1], cell[2]))

            icoop_orb[key] = icoop_orb[key].real
            icoop_atom = np.zeros((self.numAtoms, self.numAtoms, cell[0], cell[1], cell[2]))
            for atm1 in range(self.numAtoms):
                for atm2 in range(self.numAtoms):
                    orbs1 = np.arange(self.num_orbs)[self.orbatomnum == atm1]
                    orbs2 = np.arange(self.num_orbs)[self.orbatomnum == atm2]
                    sum_cohp = icohp_orb[key][orbs1, :][:, orbs2]
                    icohp_atom[atm1, atm2] = np.sum(sum_cohp, axis=(0, 1))

                    sum_coop = icoop_orb[key][orbs1, :][:, orbs2]
                    icoop_atom[atm1, atm2] = np.sum(sum_coop, axis=(0, 1))
            #print(icohp_atom.shape)
            # print("cohp!",icohp_atom)
            bonds_spin[key] = icohp_atom.flatten()
            charge_spin[key] = icoop_atom.flatten()

        if self.spin_polar:
            bonds = bonds_spin[0] + bonds_spin[1]
            bond_charges = (- charge_spin[0] - charge_spin[1]) #* 2 # x2 for bond degeneracy (<i|j+R> and <j|i-R>)
            bond_magmoms = (charge_spin[0] - charge_spin[1]) #* 2 # x2 for bond degeneracy (<i|j+R> and <j|i-R>)
        else:
            bonds = bonds_spin[0] * 2
            bond_charges = - charge_spin[0] * 2 #* 2 # consider fact that electrons are negative charge # extra x2 for fact that bond charge comes from <i|j+R> and <j|i-R>

        # get the data in order
        num_bonds = len(bonds)
        bond_indices = np.unravel_index(np.arange(num_bonds), (self.numAtoms, self.numAtoms, cell[0], cell[1], cell[2]))
        start_atom = [bond_indices[0], np.ones(num_bonds, dtype=np.int_) * each_dir[0],
                      np.ones(num_bonds, dtype=np.int_) * each_dir[1], np.ones(num_bonds, dtype=np.int_) * each_dir[2]]
        start_ind = np.ravel_multi_index(start_atom, (self.numAtoms, cell[0], cell[1], cell[2]))
        end_atom = [bond_indices[1], bond_indices[2], bond_indices[3], bond_indices[4]]
        end_ind = np.ravel_multi_index(end_atom, (self.numAtoms, cell[0], cell[1], cell[2]))

        # find unique bond set by bond energy and bond distance
        all_bond_dists = []
        for b_ind in range(len(bonds)):
            #bond_energy = bonds[b_ind]
            start = start_ind[b_ind]
            end = end_ind[b_ind]
            start_atm = np.array([x_data[start], y_data[start], z_data[start]])
            end_atm = np.array([x_data[end], y_data[end], z_data[end]])
            bond_dist = np.linalg.norm(end_atm - start_atm)
            all_bond_dists.append(bond_dist)
        all_bond_dists = np.array(all_bond_dists)
        bond_data = np.array([np.around(all_bond_dists,decimals=1),np.around(bonds,decimals=3)]).T # ought to be rounded the same as get_neighbors()
        uniq_bonds, uniq_ind, uniq_inv, uniq_counts = np.unique(bond_data,axis=0,return_index=True,return_inverse=True,return_counts=True)
        uniq_bond_energies = uniq_bonds[:,1]
        if self.verbose > 1:
            print("Got bond groups!",uniq_bonds,uniq_counts)
            print("tot bond energies:",bonds[uniq_ind]*uniq_counts)
            print(self.NN_dists)
        all_uniq_bind = {}
        all_bond_energies = []
        all_bond_charges = []
        if self.spin_polar:
            all_bond_magmoms = []
        all_bond_counts = []
        bond_labels = []
        bnum = 0
        for bi in range(len(uniq_bonds[:80])):
            bond_dist, bond_energy, b_charge  = (all_bond_dists[uniq_ind[bi]], bonds[uniq_ind[bi]],bond_charges[uniq_ind[bi]])
            if self.spin_polar:
                b_magmom = bond_magmoms[uniq_ind[bi]]
            if bond_dist < 10 and np.abs(bond_energy)*uniq_counts[bi] > 0.01 and bond_dist != 0:
                start_atmnum = bond_indices[0][uniq_ind[bi]]
                end_atmnum = bond_indices[1][uniq_ind[bi]]
                NN_dists = self.NN_dists
                NN_num = np.arange(len(NN_dists))[NN_dists==uniq_bonds[bi][0]][0]
                all_uniq_bind[bnum] = [self.elements[start_atmnum],self.elements[end_atmnum],NN_num]
                all_bond_energies.append(bond_energy)#uniq_bond_energies[bi])
                all_bond_charges.append(b_charge)
                if self.spin_polar:
                    all_bond_magmoms.append(b_magmom)
                all_bond_counts.append(uniq_counts[bi])
                sorted_elems = np.sort([self.elements[start_atmnum], self.elements[end_atmnum]])
                bond_label = np.char.add(np.char.add(sorted_elems[0], sorted_elems[1]), str(NN_num))
                bond_labels.append(bond_label)
                bnum = bnum + 1

        actually_uniq, act_ind, inverse = np.unique(bond_labels,return_index=True,return_inverse=True)
        num_uniq = len(act_ind)
        newall_uniq_bind = {}
        newall_bond_energies = []
        newall_bond_charges = []
        if self.spin_polar:
            newall_bond_magmoms = []
        newbond_labels = []
        newbond_count = []
        newall_tot_energies = []
        for ind in range(num_uniq):
            newall_uniq_bind[ind] = all_uniq_bind[act_ind[ind]]
            if self.verbose > 2:
                print(inverse)
                print(len(all_bond_counts),len(inverse))
                print(inverse==ind)
            all_counts = np.sum(np.array(all_bond_counts)[inverse==ind])
            tot_energy = np.sum(np.array(all_bond_counts)[inverse==ind]*np.array(all_bond_energies)[inverse==ind])
            tot_charge = np.sum(np.array(all_bond_counts)[inverse==ind]*np.array(all_bond_charges)[inverse==ind])
            if self.spin_polar:
                tot_magmom = np.sum(np.array(all_bond_counts)[inverse==ind]*np.array(all_bond_magmoms)[inverse==ind])
            if self.verbose > 1:
                print(np.array(all_bond_counts)[inverse==ind],np.array(all_bond_energies)[inverse==ind])
                print("mean bond energy:",tot_energy/all_counts)
                print("first bond energy:",all_bond_energies[act_ind[ind]])
            newall_bond_energies.append(np.around(tot_energy/all_counts,decimals=3))#all_bond_energies[act_ind[ind]])
            newall_bond_charges.append(np.around(tot_charge/all_counts,decimals=3))
            if self.spin_polar:
                newall_bond_magmoms.append(np.around(tot_magmom/all_counts,decimals=3))
            newbond_count.append(all_counts)
            newbond_labels.append(bond_labels[act_ind[ind]])
        if self.verbose > 1:
            print("actually uniq:",newall_uniq_bind,newall_bond_energies,newbond_labels)

        all_uniq_bind = newall_uniq_bind
        all_bond_energies = newall_bond_energies
        all_bond_labels = newbond_labels

        ### - now get DOS COHP for each unique bond - ###
        # make energies
        shifted_energies = {}
        all_shifted_ener = self.eigvals[:, :, :].flatten() - (self.efermi + self.energy_shift)
        for key in self.keys:
            shifted_energies[key] = self.eigvals[key, :, :].flatten() - (self.efermi + self.energy_shift)
        #all_shifted_ener = all_shifted_ener[all_shifted_ener < ylim[1]]
        minx = np.min(all_shifted_ener) - sigma * 2
        #minx = np.max([minx,-50])
        maxx = np.max(all_shifted_ener)  # +sigma*2
        points = int(15*np.around(maxx-minx,decimals=0))
        energies = np.linspace(minx, maxx, points)

        all_bond_cohp = []
        all_orb_cohp = []
        all_bond_info = []
        for bond_num in range(num_uniq):
            elem1, elem2, NN = all_uniq_bind[bond_num]
            bond_energy = all_bond_energies[bond_num]
            iselemorb = np.array(self.elements)[np.array(self.orbatomnum)] == elem1
            elemorbs = np.array(self.exactorbtype)[iselemorb]
            elemorbs = np.array([s[0] for s in elemorbs])
            uniqelemorbs1 = np.unique(elemorbs)
            iselemorb = np.array(self.elements)[np.array(self.orbatomnum)] == elem2
            elemorbs = np.array(self.exactorbtype)[iselemorb]
            elemorbs = np.array([s[0] for s in elemorbs])
            uniqelemorbs2 = np.unique(elemorbs)
            #print("check elems:",elem1,elem1)
            orbs_dicts = [{str(elem1): uniqelemorbs1},{str(elem2): uniqelemorbs2}]
            #print("dict to pass get_COHP:",orbs_dicts,NN)

            #for orb in uniqelemorbs:
            #    curorbdict = {elem: [orb]}

            total_cohp = {}
            orb_cohp = {}
            for key in self.keys:  # plot bands for each spin
                #orbs_dict = [{"Si": ["s", "p", "d"]}, {"Si": ["s", "p", "d"]}]
                cohp = self.TB_model.get_COHP(self,orbs_dicts, NN=NN, spin=key)

                # now plot
                integrated_cohp = np.sum(
                    cohp.flatten()[self.eigvals[key][:, :].flatten() <= self.efermi + self.energy_shift])
                if self.verbose > 1:
                    print("integrated occupied cohp:", [key], integrated_cohp)
                # print("max eigval:",self.eigval[:, self.num_orbs-1, 0])
                flat_cohp = cohp.flatten()

                # create the DOS by sum of gaussians
                flat_cohp = flat_cohp #[shifted_energies[key] < ylim[1]]
                use_energy = shifted_energies[key] #[shifted_energies[key] < ylim[1]]
                sig = sigma  # 0.15#5/self.num_kpts**(1/2)
                all_cohp = 1 / (sig * (2 * np.pi) ** (1 / 2)) * np.exp(
                    -1 / 2 * ((energies[:, None] - use_energy[None, :]) / sig) ** 2) * flat_cohp[None, :]
                total_cohp[key] = np.sum(all_cohp, axis=1)

                # also get orbital breakdown COHP
                all_combos = np.array([["",""]])
                orb_cohp[key] = {}
                orb_keys = []
                for orb1 in uniqelemorbs1[::-1]:
                    for orb2 in uniqelemorbs2[::-1]:
                        cur_orb_dict =  [{str(elem1): [orb1]},{str(elem2): [orb2]}]
                        comp_list = [str(elem1)+" "+orb1,str(elem2)+" "+orb2]
                        same_prev = np.arange(len(orb_keys)+1)[(comp_list[0] == all_combos[:,1]) & (comp_list[1] == all_combos[:,0])]
                        #print(cur_orb_dict,same_prev)
                        if len(same_prev) == 0: # doesn't already exists
                            orb_name = comp_list[0] + " - " + comp_list[1]

                            cohp = self.TB_model.get_COHP(self, cur_orb_dict, NN=NN, spin=key)
                            # now plot
                            integrated_cohp = np.sum(
                                cohp.flatten()[self.eigvals[key][:, :].flatten() <= self.efermi + self.energy_shift])
                            #print("integrated occupied cohp:", [key], integrated_cohp)
                            # print("max eigval:",self.eigval[:, self.num_orbs-1, 0])
                            flat_cohp = cohp.flatten()

                            # create the DOS by sum of gaussians
                            flat_cohp = flat_cohp  # [shifted_energies[key] < ylim[1]]
                            use_energy = shifted_energies[key]  # [shifted_energies[key] < ylim[1]]
                            sig = sigma  # 0.15#5/self.num_kpts**(1/2)
                            all_cohp = 1 / (sig * (2 * np.pi) ** (1 / 2)) * np.exp(
                                -1 / 2 * ((energies[:, None] - use_energy[None, :]) / sig) ** 2) * flat_cohp[None, :]
                            orb_cohp[key][orb_name] = [np.sum(all_cohp, axis=1),integrated_cohp]
                            all_combos = np.append(all_combos,[comp_list],axis=0)
                            orb_keys.append(orb_name)
                        else: # add it to the current version
                            #print(orb_keys)
                            orb_name = orb_keys[same_prev[0]-1]

                            cohp = self.TB_model.get_COHP(self, cur_orb_dict, NN=NN, spin=key)
                            # now plot
                            integrated_cohp = np.sum(
                                cohp.flatten()[self.eigvals[key][:, :].flatten() <= self.efermi + self.energy_shift])
                            #print("integrated occupied cohp:", [key], integrated_cohp)
                            # print("max eigval:",self.eigval[:, self.num_orbs-1, 0])
                            flat_cohp = cohp.flatten()

                            # create the DOS by sum of gaussians
                            flat_cohp = flat_cohp  # [shifted_energies[key] < ylim[1]]
                            use_energy = shifted_energies[key]  # [shifted_energies[key] < ylim[1]]
                            sig = sigma  # 0.15#5/self.num_kpts**(1/2)
                            all_cohp = 1 / (sig * (2 * np.pi) ** (1 / 2)) * np.exp(
                                -1 / 2 * ((energies[:, None] - use_energy[None, :]) / sig) ** 2) * flat_cohp[None, :]
                            orb_cohp[key][orb_name][0] += np.sum(all_cohp, axis=1)
                            orb_cohp[key][orb_name][1] += integrated_cohp


            linestyle = ["", "--"]
            color = ["dimgray", "lightseagreen"]
            data = {}
            info = {}
            for key in self.keys:
                data[key] = [total_cohp[key], energies, linestyle[key], color[key]]
                info[key] = all_bond_labels[bond_num]

            all_bond_cohp.append(data)
            all_orb_cohp.append(orb_cohp)
            all_bond_info.append(info)
        if self.verbose > 0:
            print("bonds:",all_bond_info)
        ### - plot the bonds and total COHP - ###
        import plotly.graph_objects as go
        fig = go.Figure()
        # plot vertically
        # ax.plot(np.zeros(points), energies, '--', color="grey", linewidth=0.8)
        total_dos = {}
        colors = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf']
        colors = ['rgb(31, 119, 180)', 'rgb(255, 127, 14)','rgb(44, 160, 44)', 'rgb(214, 39, 40)','rgb(148, 103, 189)',
                  'rgb(140, 86, 75)','rgb(227, 119, 194)', 'rgb(127, 127, 127)', 'rgb(188, 189, 34)', 'rgb(23, 190, 207)',
                  'rgb(31, 119, 180)', 'rgb(255, 127, 14)','rgb(44, 160, 44)', 'rgb(214, 39, 40)','rgb(148, 103, 189)',
                  'rgb(140, 86, 75)','rgb(227, 119, 194)', 'rgb(127, 127, 127)', 'rgb(188, 189, 34)', 'rgb(23, 190, 207)']
        prev_y = {}
        for key in self.keys:
            total_dos[key] = np.zeros(len(all_bond_cohp[0][key][0]))
            prev_y[key] = np.zeros_like(all_bond_cohp[0][self.keys[0]][1])
        for i, data in enumerate(all_bond_cohp):
            for key in self.keys:
                # data[key] = [total_cohp[key],energies,linestyle[key],color[key]]
                #ax.plot(data[key][0], energies, data[key][2], label=uniqelemorbs[i])
                total_dos[key] += data[key][0]
                y_combined = np.concatenate([energies, energies[::-1]])
                x_combined = np.concatenate([total_dos[key], prev_y[key][::-1]])
                # start by adding filled in shape
                #fig.add_trace(go.Scatter(x=data[key][0],y=energies,mode='lines', line=dict(color="rgba"+colors[i][3:][:-1]+",1.00)"),
                #                         fill='tozerox',fillcolor="rgba"+colors[i][3:][:-1]+",0.65)",customdata=[all_bond_info[i][key]]))
                # now just add line
                #fig.add_trace(go.Scatter(x=data[key][0],y=energies,mode='lines', line=dict(color='black',width=1),hoverinfo='none',customdata=[all_bond_info[i][key]]))
                prev_y[key] = total_dos[key].copy()

                # add all the orbital stuff
                orbs_combos = all_orb_cohp[i][key] # dictionary of COHP values
                all_energies = []
                for orb_key in orbs_combos.keys():
                    all_energies.append(orbs_combos[orb_key][1])
                sort_ind = np.argsort(np.abs(all_energies))[::-1]
                tot_bond_dos = np.zeros(len(all_bond_cohp[0][key][0]))
                prev_orb = np.zeros_like(all_bond_cohp[0][self.keys[0]][1])
                for oi in sort_ind:# enumerate(orbs_combos.keys()):
                    orb_key = list(orbs_combos.keys())[oi]
                    orb_dos = orbs_combos[orb_key][0]
                    tot_bond_dos += orb_dos
                    y_combined = np.concatenate([energies, energies[::-1]])
                    x_combined = np.concatenate([tot_bond_dos, prev_orb[::-1]])
                    # start by adding filled in shape
                    fig.add_trace(go.Scatter(name = orb_key, x=x_combined, y=y_combined, mode='none',
                                             fill='toself', fillcolor="rgba" + colors[oi][3:][:-1] + ",0.00)",
                                             customdata=[all_bond_info[i][key]],showlegend=False,hoverinfo='none'))
                    fig.add_trace(go.Scatter(name = orb_key, x=tot_bond_dos, y=energies, mode='lines',
                                             line=dict(color="rgba" + colors[oi][3:][:-1] + ",0.00)",width=2),
                                             customdata=[all_bond_info[i][key]],showlegend=False,hoverinfo='none'))
                    # now just add line
                    # fig.add_trace(go.Scatter(x=data[key][0],y=energies,mode='lines', line=dict(color=colors[i]),hoverinfo='none'))
                    prev_orb = tot_bond_dos.copy()

                fig.add_trace(go.Scatter(name ="bond COHP",x=data[key][0],y=energies,mode='lines', line=dict(color="rgba(1, 1, 1, 0.00)",width=1.5),
                                         customdata=[all_bond_info[i][key]],showlegend=False,hoverinfo='none'))


        for key in self.keys:
            #ax.plot(total_dos[key], energies, '--', color="gray", label=elem, linewidth=1)
            #ax.plot(allatmDOS[key],energies, data[key][2], color="black", label="total", linewidth=1)
            fig.add_trace(go.Scatter(x=total_dos[key],y= energies,mode="lines",line=dict(color='black',width=3),name="total COHP   "))

        mini = np.min(total_dos[key]) - 0.05
        maxi = np.max(total_dos[key]) + 0.05
        fig.add_trace(go.Scatter(x=[0,0], y=[energies[0],energies[-1]],mode='lines', line=dict(color="grey",dash="dash",width=1.5),showlegend=False))

        fig.add_trace(go.Scatter(x=[mini, maxi], y=[0, 0],mode='lines', line=dict(color="grey",dash="dash",width=1.5),showlegend=False))
        max_occ = np.amax(total_dos[key][(energies > -15) & (energies < 0)])
        min_occ = np.amin(total_dos[key][(energies > -15) & (energies < 0)])

        # Customize layout
        fig.update_layout(
            title='COHP Density of States Breakdown with Negative Contributions',
            xaxis_title='COHP DOS',
            yaxis_title='Energy',
            legend_title='Contributions',
            hovermode='x unified'
        )

        fig.update_xaxes(range=[max_occ+1,min_occ-1])
        miny = np.max([minx,-20])
        fig.update_yaxes(range=[miny,5])
        #ax.plot([0, 0], [ylim[0], ylim[1]], '--', color="grey", linewidth=0.8)

        #ax.plot([-0.05, max_proj + 0.1], [0, 0], "--", color="grey", linewidth=0.8)
        #plt.xlim((-0.05, max_proj + 0.1))
        #plt.ylim(ylim)
        #ax.set_xlabel("orbital pDOS")
        #ax.set_ylabel("Energy (eV)")
        #plt.legend()
        #plt.tight_layout()
        #fig.savefig(self.directory + elem + 'projectedDOS.png', transparent=True, dpi=150, format='png')
        #plt.show()
        #plt.close()

        if return_fig:
            return fig
        else:
            # Display the plot
            if self.show_figs:
                fig.show()

    def get_crystal_plus_COHP(self, energy_cutoff: float=0.05, bond_max:float=3, auto_label: str='mulliken',fovy:float=10) -> None:
        """
        The will plot the crystal bond plot on the left with interactivity to a COHP DOS plot on the right.

        Args:    
            energy_cutoff (float): This is the minimum bond magnitude that will be plotted.
            bond_max (float): Atoms are plotted outside the primitive cell only if they have a bond with
                        a primitive cell atom that is shorter than this amount. This does not filter bonds between atoms that are already plotted.
            auto_label (str): Different options for plotting includes: (can include multiple in the string)
                        "mulliken" - Plots the onsite charge and mag (if spin_polar) on atoms by mulliken population (overrides atom_labels).
                        "full" - Plots the charge and magnetics moments (if spin_polar) on atoms and bonds (overrides atom_labels).
                        "color" - Colors the atoms and bonds based on their charge (overrides atom_colors or elem_colors).
                        "color mag" - Colors the atoms and bonds based on their magnetic moments (overrides atom_colors or elem_colors).
                        NOTE: Only use "mulliken" OR "full", NOT both.
            fovy (float): field of view in the vertical direction. Use this tag to adjust depth perception in crystal.

        Returns:    
            None: Nothing, but saved to 'bond_cohp_plot.html'.
        """

        fig1, post_script = self.get_bonds_charge_figure(energy_cutoff=energy_cutoff, bond_max=bond_max, auto_label=auto_label,fovy=fovy,return_fig=True)
        if not hasattr(self, 'ICOHP'):
            self.get_ICOHP()
        if not hasattr(self, 'ICOOP'):
            self.get_ICOOP()

        if self.spin_polar:
            self.keys = [0]
            fig2 = self.get_COHP_DOS_bybond(return_fig=True)
            self.keys = [1]
            fig3 = self.get_COHP_DOS_bybond(return_fig=True)
            self.keys = [0,1]
        else:
            fig2 = self.get_COHP_DOS_bybond(return_fig=True)

        from plotly.subplots import make_subplots

        # Assume you have two existing figures: fig1 and fig2
        # Each created with fig = go.Figure(data=data)

        # Create a new figure with subplots
        if self.spin_polar:
            combined_fig = make_subplots(
                rows=1, cols=3,  # Adjust the number of rows and columns as needed
                specs=[[{'type': 'scene'}, {'type': 'xy'}, {'type': 'xy'}]],
                column_widths=[0.6, 0.2, 0.2],
                subplot_titles=(fig1.layout.title.text, "spin up COHP for bond","spin down COHP for bond")
            )
        else:
            combined_fig = make_subplots(
                rows=1, cols=2,  # Adjust the number of rows and columns as needed
                specs=[[{'type': 'scene'}, {'type': 'xy'}]],
                column_widths=[0.7, 0.3],
                subplot_titles=(fig1.layout.title.text, "COHP for selected bond")
            )

        # Add traces from fig1 to the first subplot
        for trace in fig1.data:
            combined_fig.add_trace(trace, row=1, col=1)

        # Add traces from fig2 to the second subplot
        for trace in fig2.data:
            combined_fig.add_trace(trace, row=1, col=2)

        if self.spin_polar:
            # Add traces from fig2 to the second subplot
            for trace in fig3.data:
                combined_fig.add_trace(trace, row=1, col=3)

        # Update x-axis and y-axis titles for the first subplot
        combined_fig.update_xaxes(fig1.layout.xaxis.to_plotly_json(), row=1, col=1)
        combined_fig.update_yaxes(fig1.layout.yaxis.to_plotly_json(), row=1, col=1)
        #combined_fig.update_xaxes(title_text=fig1.layout.xaxis.title.text, row=1, col=1)
        #combined_fig.update_yaxes(title_text=fig1.layout.yaxis.title.text, row=1, col=1)

        # Update x-axis and y-axis titles for the second subplot
        combined_fig.update_xaxes(fig2.layout.xaxis.to_plotly_json(), row=1, col=2)
        combined_fig.update_yaxes(fig2.layout.yaxis.to_plotly_json(), row=1, col=2)
        #combined_fig.update_xaxes(title_text=fig2.layout.xaxis.title.text, row=1, col=2)
        #combined_fig.update_yaxes(title_text=fig2.layout.yaxis.title.text, row=1, col=2)
        if self.spin_polar:
            # Update x-axis and y-axis titles for the third subplot
            combined_fig.update_xaxes(fig3.layout.xaxis.to_plotly_json(), row=1, col=3)
            combined_fig.update_yaxes(fig3.layout.yaxis.to_plotly_json(), row=1, col=3)

        # If there are other layout properties you want to transfer, such as legends or annotations, you can update them accordingly.

        # Update the overall layout
        combined_fig.update_layout(
            title_text='Crystal bonds from COGITO',
            showlegend=True  # or False, depending on your preference
        )
        combined_fig.update_layout(hoverdistance=1000, margin=dict(l=0, r=0),
                          paper_bgcolor='rgba(0,0,0,0)',  # Transparent background for the whole figure
                          plot_bgcolor='rgba(0,0,0,0)')  # ,hovermode="x unified")
        combined_fig.update_layout(
            scene=dict(xaxis=dict(showbackground=False, showspikes=False, showticklabels=False, visible=False),
                       yaxis=dict(showbackground=False, showspikes=False, showticklabels=False, visible=False),
                       zaxis=dict(showbackground=False, showspikes=False, showticklabels=False, visible=False),
                       aspectmode='data'))

        # JavaScript code to handle hover events
        hover_js = """
        <script>
        var myPlot = document.getElementById('plotDiv');

        var hoveredCohpTraceIndex = null;
        var current_energy = null;
        var old_energy = null
        var hoverTimer = null;

        myPlot.on('plotly_hover', function(data){
        
    
            // Clear any existing hover timer
            if (hoverTimer !== null) {
                clearTimeout(hoverTimer);
                hoverTimer = null;
            }

            // Start a new hover timer
            hoverTimer = setTimeout(function() {
                var point = data.points[0];
                var curveNumber = point.curveNumber;
    
                // Check if the hovered trace is a bond in the 3D plot
                if (point.fullData.type === 'scatter3d' && point.fullData.mode === 'markers') {
                    // Access bond energy from point.data.customdata using point.pointNumber
                    var bondEnergy = point.data.customdata[point.pointNumber];
                    console.log("new bond energy", bondEnergy, current_energy, old_energy);
                

                    // only add trace if it's not already activated
                    if (current_energy !== bondEnergy) {

                        // Remove any previously added hovered COHP trace
                        if (hoveredCohpTraceIndex !== null) {
                            console.log("delete", hoveredCohpTraceIndex, myPlot.data.length);

                            // Store the index to delete and reset the hovered index before deletion
                            var traceIndexToDelete = hoveredCohpTraceIndex;
                            hoveredCohpTraceIndex = null;
                            current_energy = null;
                            Plotly.deleteTraces(myPlot, traceIndexToDelete);
                            console.log("delete", hoveredCohpTraceIndex, myPlot.data.length);
                        }

                        // Store the original number of traces before adding new ones
                        var originalDataLength = myPlot.data.length;
                        console.log(originalDataLength);

                        // Find the COHP trace with matching bond energy
                        var cohpTraceIndex = null;
                        for (var i = 0; i < originalDataLength; i++) {
                            var trace = myPlot.data[i];
                            console.log(trace.xaxis);
                            // Check if the trace is in the second subplot (col=2)
                            if (trace.xaxis === 'x' || trace.xaxis === 'x2') {
                                if ('customdata' in trace) {
                                    console.log(trace.customdata[0], bondEnergy);
                                    console.log((trace.customdata[0] === bondEnergy));
                                    if (trace.customdata[0] === bondEnergy) {
                                        console.log("passed");
                                        cohpTraceIndex = i;
                                        break;
                                    }
                                }
                            }
                        }

                        if (cohpTraceIndex !== null) {
                            // Clone the COHP trace
                            var cohpTrace = myPlot.data[cohpTraceIndex];
                            var newTrace = JSON.parse(JSON.stringify(cohpTrace));

                            // Modify the new trace to be black
                            prev_color = cohpTrace.fillcolor
                            new_color = prev_color.substring(0, prev_color.length - 5) + "1.0)";
                            console.log(new_color)
                            newTrace.fillcolor = new_color;
                            newTrace.opacity = 1.0;
                            newTrace.hoverinfo = 'skip'; // Avoid recursive hover events
                            newTrace.showlegend = false; // Do not show in legend
                            newTrace.name = ''; // Remove name to avoid tooltip

                            // Add the new trace to the plot
                            Plotly.addTraces(myPlot, newTrace);

                            var originalDataLength = myPlot.data.length;
                            console.log(originalDataLength);

                            // Store the index of the added trace
                            hoveredCohpTraceIndex = myPlot.data.length - 1;
                            current_energy = bondEnergy;
                        }
                    }
                }
            }, 50); // Delay of 0.15 second
        }); 

        myPlot.on('plotly_unhover', function(data){
            // Clear the hover timer
            if (hoverTimer !== null) {
                clearTimeout(hoverTimer);
                hoverTimer = null;
            }

            // Remove the added hovered COHP trace
            if (hoveredCohpTraceIndex !== null) {
                console.log(hoveredCohpTraceIndex, myPlot.data.length);

                // Store the index to delete and reset the hovered index before deletion
                var traceIndexToDelete = hoveredCohpTraceIndex;
                hoveredCohpTraceIndex = null;
                current_energy = null;
                Plotly.deleteTraces(myPlot, traceIndexToDelete);
                console.log("passed once");
            }
        });

        </script>
        """

        hover_js = """
        <script>
        var myPlot = document.getElementById('plotDiv');

        var hoverTimer = null;
        var currentCohpTraceIndex = [];
        var current_energy = null;
        var clicked_energy = null;

        // create click feature which prevents unhover from removing plot
        
        myPlot.on('plotly_click', function(data){
            var point = data.points[0];
            // Check if the hovered trace is a bond in the 3D plot
            if (point.fullData.type === 'scatter3d' && point.fullData.mode === 'markers') {
                // Access bond energy from point.data.customdata using point.pointNumber
                var bondEnergy = point.data.customdata[point.pointNumber];
                clicked_energy = bondEnergy;
            }
        });
                    

        myPlot.on('plotly_hover', function(data){
            // Clear any existing hover timer
            if (hoverTimer !== null) {
                clearTimeout(hoverTimer);
                hoverTimer = null;
            }

            // Start a new hover timer
            hoverTimer = setTimeout(function() {
                var point = data.points[0];

                // Check if the hovered trace is a bond in the 3D plot
                if (point.fullData.type === 'scatter3d' && point.fullData.mode === 'markers') {
                    // Access bond energy from point.data.customdata using point.pointNumber
                    var bondEnergy = point.data.customdata[point.pointNumber];
                    // console.log("new bond energy", bondEnergy, current_energy);

                    // Only change opacity if it's not already activated
                    if (current_energy !== bondEnergy) {
                        // Reset previous highlighted trace
                        if (currentCohpTraceIndex !== []) {
                            var hold_index = currentCohpTraceIndex
                            currentCohpTraceIndex = [];
                            current_energy = null;
                            var fill_colors = []
                            var fill_legends = []
                            var fill_indices = []
                            var line_colors = []
                            var line_legends = []
                            var line_indices = []
                            for (var i = 0; i < hold_index.length; i++) {
                                var cohpTrace = myPlot.data[hold_index[i]];
                                
                                // Modify the new trace to be black
                                if ('fillcolor' in cohpTrace) {
                                    prev_color = cohpTrace.fillcolor;
                                    new_color = prev_color.substring(0, prev_color.length - 5) + "0.00)";
                                    fill_colors.push(new_color)
                                    fill_legends.push(false)
                                    fill_indices.push(hold_index[i])
                                    // Plotly.restyle(myPlot, {'fillcolor': new_color, 'showlegend':false}, [hold_index[i]]);
                                }
                                
                                if ('line' in cohpTrace) {
                                    prev_color = cohpTrace.line.color;
                                    line_color = prev_color.substring(0, prev_color.length - 5) + "0.00)";
                                    line_colors.push(line_color)
                                    line_legends.push(false)
                                    line_indices.push(hold_index[i])
                                    // Plotly.restyle(myPlot, { 'line.color': line_color,'hoverinfo':'none'}, [hold_index[i]]);
                                }
                                // prev_color = cohpTrace.fillcolor;
                                // new_color = prev_color.substring(0, prev_color.length - 5) + "0.00)";
                                // Plotly.restyle(myPlot, {'fillcolor': new_color, 'line.color': new_color}, [hold_index[i]]);
                            }
                            
                            Plotly.restyle(myPlot, {'fillcolor': fill_colors,'showlegend':fill_legends}, fill_indices);
                            var all_hovers = new Array(line_indices.length).fill('none')
                            Plotly.restyle(myPlot, {'line.color': line_colors,'showlegend':line_legends,'hoverinfo':all_hovers}, line_indices);
                            
                        }

                        // Find the COHP trace with matching bond energy
                        var cohpTraceIndex = [];
                        for (var i = 0; i < myPlot.data.length; i++) {
                            var trace = myPlot.data[i];
                            // console.log(trace.xaxis);
                            // Check if the trace is in the second subplot (col=2)
                            if (trace.xaxis === 'x' || trace.xaxis === 'x2') {
                                // console.log(trace.customdata)
                                if ('customdata' in trace) {
                                    // console.log(trace.customdata[0], bondEnergy);
                                    // console.log((trace.customdata[0] === bondEnergy));
                                    if (trace.customdata[0] === bondEnergy) {
                                        // console.log("passed:");
                                        cohpTraceIndex.push(i);
                                    }
                                }
                            }
                        }

                        console.log(cohpTraceIndex);
                        if (cohpTraceIndex !== []) {
                            // Set the opacity of the trace to 0.65 to highlight it
                            // console.log("all lines:",cohpTraceIndex)
                            
                            // Plotly.restyle(myPlot, {'opacity': 0.95}, [cohpTraceIndex]); 
                            var fill_colors = []
                            var fill_legends = []
                            var fill_indices = []
                            var line_colors = []
                            var line_legends = []
                            var line_indices = []
                            for (var i = 0; i < cohpTraceIndex.length; i++) {
                                var cohpTrace = myPlot.data[cohpTraceIndex[i]];
                                // Modify the new trace to be black
                                if ('fillcolor' in cohpTrace) {
                                    prev_color = cohpTrace.fillcolor;
                                    new_color = prev_color.substring(0, prev_color.length - 5) + "0.55)";
                                    // console.log("fill",i,prev_color,new_color)
                                    fill_colors.push(new_color)
                                    fill_legends.push(false)
                                    fill_indices.push(cohpTraceIndex[i])
                                    // Plotly.restyle(myPlot, {'fillcolor': new_color}, [cohpTraceIndex[i]]);
                                    if (cohpTrace.xaxis === 'x'){
                                        fill_legends[fill_legends.length-1] = true
                                        // Plotly.restyle(myPlot, {'showlegend':true}, [cohpTraceIndex[i]]);
                                    }
                                }
                                
                                if ('line' in cohpTrace) {
                                    prev_color = cohpTrace.line.color;
                                    line_color = prev_color.substring(0, prev_color.length - 5) + "1.00)";
                                    // console.log("line",i,prev_color,line_color)
                                    // Plotly.restyle(myPlot, { 'line.color': line_color,'hoverinfo':'x+y+name'}, [cohpTraceIndex[i]]);
                                    line_colors.push(line_color)
                                    line_legends.push(false)
                                    line_indices.push(cohpTraceIndex[i])
                                    if ("bond COHP" === cohpTrace.name) {
                                        line_legends[line_legends.length-1] = true
                                        // Plotly.restyle(myPlot, {'showlegend':true,'hoverinfo':'x+y+name'}, [cohpTraceIndex[i]]);
                                    }
                                }
                                // Plotly.restyle(myPlot, {'line.color': all_colors,'showlegend':all_legends,'hoverinfo':all_hovers}, all_indices);
                                // console.log(new_color);
                                // Plotly.restyle(myPlot, {'line': new_line}, [cohpTraceIndex[i]]);
    
                            }
                            
                            Plotly.restyle(myPlot, {'fillcolor': fill_colors,'showlegend':fill_legends}, fill_indices);
                            var all_hovers = new Array(line_indices.length).fill('x+y+name')
                            Plotly.restyle(myPlot, {'line.color': line_colors,'showlegend':line_legends,'hoverinfo':all_hovers}, line_indices);
                            
                            // Store the index of the highlighted trace
                            currentCohpTraceIndex = cohpTraceIndex;
                            current_energy = bondEnergy;
                        }
                    }
                }
            }, 100); // Delay of .1 second
        }); 

        myPlot.on('plotly_unhover', function(data){
            // Clear the hover timer
            if (hoverTimer !== null) {
                clearTimeout(hoverTimer);
                hoverTimer = null;
            }

            // Reset the opacity of the highlighted trace
            if (currentCohpTraceIndex !== [] && clicked_energy !== current_energy) {
                var hold_index = currentCohpTraceIndex
                currentCohpTraceIndex = [];
                current_energy = null;
                var fill_colors = []
                var fill_legends = []
                var fill_indices = []
                var line_colors = []
                var line_legends = []
                var line_indices = []
                for (var i = 0; i < hold_index.length; i++) {
                    var cohpTrace = myPlot.data[hold_index[i]];
                    // Modify the new trace to be black
                    if ('fillcolor' in cohpTrace) {
                        prev_color = cohpTrace.fillcolor;
                        new_color = prev_color.substring(0, prev_color.length - 5) + "0.00)";
                        fill_colors.push(new_color)
                        fill_legends.push(false)
                        fill_indices.push(hold_index[i])
                        // console.log("fill",i,prev_color,new_color)
                        // Plotly.restyle(myPlot, {'fillcolor': new_color, 'showlegend':false}, [hold_index[i]]);
                    }
                    
                    if ('line' in cohpTrace) {
                        prev_color = cohpTrace.line.color;
                        line_color = prev_color.substring(0, prev_color.length - 5) + "0.00)";
                        line_colors.push(line_color)
                        line_legends.push(false)
                        line_indices.push(hold_index[i])
                        // console.log("line",i,prev_color,line_color)
                        // Plotly.restyle(myPlot, { 'line.color': line_color,'hoverinfo':'none'}, [hold_index[i]]);
                    }
                    
                    // Modify the new trace to be black
                    // prev_color = cohpTrace.fillcolor;
                    // new_color = prev_color.substring(0, prev_color.length - 5) + "0.00)";
                    // console.log(new_color);
                    // Plotly.restyle(myPlot, {'line': new_line}, [hold_index[i]]);
                    // Plotly.restyle(myPlot, {'fillcolor': new_color, 'line.color': new_color}, [hold_index[i]]);
                    // Plotly.restyle(myPlot, {'opacity': 0.0}, [hold_index[i]]);
                }
                
                Plotly.restyle(myPlot, {'fillcolor': fill_colors,'showlegend':fill_legends}, fill_indices);
                var all_hovers = new Array(line_indices.length).fill('none')
                Plotly.restyle(myPlot, {'line.color': line_colors,'showlegend':line_legends,'hoverinfo':all_hovers}, line_indices);
                            
            }
        });
        </script>
        """

        # Display the combined figure
        #combined_fig.show(post_script=post_script)

        import plotly.io as pio
        #"console.log(`Updated FOV to ${camera.fovy} radians.`);")
        html_str = pio.to_html(combined_fig, include_plotlyjs='cdn', full_html=True, div_id='plotDiv', post_script=post_script)

        # Insert the JavaScript code before the closing </body> tag
        insertion_point = html_str.rfind('</body>')
        modified_html_str = html_str[:insertion_point] + hover_js + html_str[insertion_point:]

        # Save the modified HTML string to a file
        with open(self.directory+'bond_cohp_plot.html', 'w') as f:
            f.write(modified_html_str)

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

class COGITO_BS_widget(object): # ought to pass the class COGITO_TB_Model
    def __init__(self, TB_model: object, num_kpts: int = 100):
        """
        This class deals with all post-processing band structure analysis.

        Args:    
            TB_model (object): Requires an object of the class COGITO_TB_Model.
        """
        # copy all attributes of the TB_model
        attributes = vars(TB_model)
        for attr, value in attributes.items():
            setattr(self, attr, value)
        # save TB_model to self to use TB_model functions
        self.TB_model = TB_model
        # print("is polarized?",self.spin_polar)
        # calculate the bandstructure
        self.get_bandstructure(num_kpts)
        self.NN_index, self.NN_dists = self.TB_model.get_neighbors(self)

    def get_bandstructure(self, num_kpts: int = 100) -> None:
        """
        The function automatically generates a kpath using pymatgen and kpathseek.
        Then calculates the band energies and vectors for the kpath.

        Args:    
            num_kpts (int): The number of kpoints between EACH kpath.

        Returns:    
            None: Nothing
        """

        pymat_struc = struc.Structure(self._a, self.elements, self.primAtoms)
        print(pymat_struc)
        kpts_obj = kpath.KPathSeek(pymat_struc)
        high_sym = kpts_obj.get_kpoints(1, coords_are_cartesian=False)

        num_each = int(num_kpts / (len(high_sym) / 2))
        print("num kpts per path:", num_each)
        kpoints = kpts_obj.get_kpoints(num_each, coords_are_cartesian=False)
        self.num_kpts = len(kpoints[0])
        print(self.num_kpts)
        self.kpoints = np.array(kpoints[0])
        self.kpoint_labels = np.array(kpoints[1])

        if self.spin_polar:
            keys = [0,1]
        else:
            keys = [0]
        self.keys = keys

        eigvals = np.zeros((len(keys), self.num_kpts, self.num_orbs))
        eigvecs = np.zeros((len(keys), self.num_kpts, self.num_orbs, self.num_orbs), dtype=np.complex128)
        kpt_olaps = np.zeros((len(keys), self.num_kpts, self.num_orbs, self.num_orbs), dtype=np.complex128)

        for key in keys:
            for kpt in range(self.num_kpts):
                print("-", end="")
            print("")
            for kpt in range(self.num_kpts):
                print("-", end="")
                eigvals[key][kpt], eigvecs[key][kpt], kpt_olaps[key][kpt] = self.TB_model.get_ham(self, self.kpoints[kpt], return_overlap=True,return_truevec=True,spin=key)
            print("")
        self.eigvals = eigvals
        self.eigvecs = eigvecs
        self.kpt_olaps = kpt_olaps
        self.kpt_weights = np.ones(self.num_kpts)

    def plotlyBS(self,ylim=(-10,10),selectedDot=None,plotnew=False) -> object:
        """
        Plots bandstructure (or projected bandstructure) using plotly graph_objects.

        Args:    
            ylim (unknown): Limits on the y-axis of plot.

        Returns:    
            object: Plotly figure with bandstructure plotted.
        """

        #reformat data to find nodes and stop repeating them
        num_nodes = 1
        nodes = [0]
        labels = [self.kpoint_labels[0]]
        kpt_line = np.arange(self.num_kpts)
        next_label = ''
        for kpt_ind,name in enumerate(self.kpoint_labels[1:-1]):
            if name != '' and name != next_label:
                nodes.append(kpt_ind-num_nodes+2)
                next_label = self.kpoint_labels[kpt_ind+2]
                if name == next_label:
                    labels.append(name)
                else:
                    labels.append(name+"|"+next_label)
                kpt_line[kpt_ind+2:] = kpt_line[kpt_ind+2:]-1
                num_nodes += 1

        import plotly.graph_objects as go
        bs_figure = go.Figure()
        self.k_node = nodes
        self.k_label = labels


        #plot the bands, i loops over each band
        for key in self.keys: # plot bands for each spin
            for i in range(self.num_orbs):
                bs_figure.add_trace(go.Scatter(x=kpt_line,
                                   y= self.eigvals[key][:,i] - (self.efermi+self.energy_shift),
                                   marker=dict(color="black", size=2),showlegend=False))
        if plotnew==True:
            #for key in self.keys: # plot bands for each spin
            for i in range(self.num_orbs):
                bs_figure.add_trace(go.Scatter(x=kpt_line,
                                   y= self.new_evals[:,i] - (self.efermi+self.energy_shift),
                                   marker=dict(color="red", size=2),showlegend=False))

        if ylim == None:
            minen = np.amin(self.eigvals.flatten() - (self.efermi+self.energy_shift))
            maxen = np.amax(self.eigvals.flatten() - (self.efermi+self.energy_shift))
            ylim = (minen-0.5,maxen+0.5)

        if ylim == (-10,10):
            amin = np.amin(self.eigvals[0]) - (self.efermi+self.energy_shift)
            amax = np.amax(self.eigvals[0]) - (self.efermi+self.energy_shift)
            ylim = (amin-1,amax+1)

        # Add custom grid lines
        for x_val in nodes:
            bs_figure.add_shape(type='line',
                                   x0=x_val, x1=x_val, y0=ylim[0], y1=ylim[1],
                                   line=dict(color='black', width=1))

        bs_figure.update_layout(xaxis={'title': 'Path in k-space'},
                           yaxis={'title': 'Energy (eV)'})

        # Update layout to set axis limits and ticks
        bs_figure.update_layout(margin=dict(l=20, r=20, t=50, b=0),
            xaxis=dict(showgrid=False,range=[kpt_line[0], kpt_line[-1]], tickvals=nodes,ticktext=labels,
                       showline=True, linewidth=1.5, linecolor='black', mirror=True,tickfont=dict(size=16), titlefont=dict(size=18)),
            yaxis=dict(showgrid=False,range=[ylim[0], ylim[1]],showline=True, linewidth=2, linecolor='black', mirror=True,
                       tickfont=dict(size=16), titlefont=dict(size=18)),
            title={'text':'<b>Bandstructure</b>', 'font':dict(size=26, color='black',family='Times')},
            #plot_bgcolor='rgba(0, 0, 0, 0)',
            title_x = 0.5,  # Center the title
            paper_bgcolor='rgba(0,0,0,0)',  # Transparent background for the whole figure
            plot_bgcolor='rgba(0,0,0,0)'
        )

        if selectedDot != None:
            xcenter = kpt_line[selectedDot[0]] # should just be the number of kpoint
            ycenter = self.eigvals[selectedDot[2]][selectedDot[0],selectedDot[1]] - (self.efermi+self.energy_shift)  #self.evals[selectedDot[1]][selectedDot[0]] - fermi_energy
            print("point center", xcenter, ycenter)
            bs_figure.add_trace(go.Scatter(x=[xcenter], y=[ycenter],
                                           marker=dict(color="limegreen", size=12), showlegend=False))

        bs_figure.update_layout(autosize=False, width=800, height=900)

        return bs_figure

    def plotBS(self, ax=None, selectedDot=None, plotnew=False,ylim=None):
        """

        Args:    
            ax (unknown): (matplotlib.pyplot axis): Axis to save the bandstructure to, otherwise generate new axis.
            selectedDot (unknown): (1D integer array): Gives kpoint and band index of the dot selected to make green circle.
                        eg: [3,4]

        Returns:    
            unknown: matplotlib.pyplot axis with bandstructure plotted.
        """
        fermi_energy = 4.619735#2.5529#4.619735  # best si: 7.089 #used to set the valence band maximum to zero
        #print("tick info:", self.k_node, self.k_label)

        bs_figure = go.Figure()

        if plotnew == True:
            for i in range(self.new_evals.shape[0]):
                bs_figure.add_trace(go.Scatter(x=self.k_dist,
                                               y=self.new_evals[i] - fermi_energy,
                                               marker=dict(color="red", size=2), showlegend=False))
        for i in range(self.evals.shape[0]):
            bs_figure.add_trace(go.Scatter(x=self.k_dist,
                               y=self.evals[i] - fermi_energy,
                               marker=dict(color="black", size=2),showlegend=False))
        if ylim == None:
            minen = np.amin(self.evals.flatten() - fermi_energy)
            maxen = np.amax(self.evals.flatten() - fermi_energy)
            ylim = (minen-0.5,maxen+0.5)

        # Add custom grid lines
        for x_val in self.k_node:
            bs_figure.add_shape(type='line',
                                   x0=x_val, x1=x_val, y0=ylim[0], y1=ylim[1],
                                   line=dict(color='black', width=1))

        bs_figure.update_layout(xaxis={'title': 'Path in k-space'},
                           yaxis={'title': 'Energy (eV)'})

        # Update layout to set axis limits and ticks
        bs_figure.update_layout(margin=dict(l=20, r=20, t=50, b=0),
            xaxis=dict(showgrid=False,range=[self.k_dist[0], self.k_dist[-1]], tickvals=self.k_node,ticktext=self.k_label,
                       showline=True, linewidth=1.5, linecolor='black', mirror=True,tickfont=dict(size=16), titlefont=dict(size=18)),
            yaxis=dict(showgrid=False,range=[ylim[0], ylim[1]],showline=True, linewidth=2, linecolor='black', mirror=True,
                       tickfont=dict(size=16), titlefont=dict(size=18)),
            title={'text':'<b>Bandstructure</b>', 'font':dict(size=26, color='black',family='Times')},
            plot_bgcolor='rgba(0, 0, 0, 0)',
            title_x = 0.5  # Center the title
        )

        #make the selected dot green
        #selectedDot = [30,4]
        if selectedDot != None:
            xcenter = self.k_dist[selectedDot[0]]
            ycenter = self.evals[selectedDot[1]][selectedDot[0]] - fermi_energy
            print("point center",xcenter,ycenter)
            bs_figure.add_trace(go.Scatter(x=[xcenter],y=[ycenter],
                               marker=dict(color="limegreen", size=12),showlegend=False))


        # plot new bandstructures
        #if plotnew == True:
        #    for i in range(self.evals.shape[0]):
        #        ax.plot(self.k_dist, self.new_evals[i] - fermi_energy, c='red', linewidth=2, zorder=1)

        #make the selected dot green
        #if selectedDot != None:
        #    ax.plot(self.k_dist[selectedDot[0]], self.evals[selectedDot[1]][selectedDot[0]] - fermi_energy, 'o',
        #            c='limegreen', picker=True, label=str(i), markersize=10)
        bs_figure.update_layout(autosize=False, width=800, height=900)
        return bs_figure

    def get_significant_bonds(self, band, kpoint, spin):
        value = 0
        # define diff_evals

        #print(band, kpoint)
        diff_evals_vals = []
        diff_evals_keys = []
        orig_tb_params = []
        orbitals = np.arange(0,self.num_orbs)
        self.selected_band = band
        self.selected_spin = spin
        eigen_var = copy.deepcopy(self.eigvecs[spin][kpoint,:,band])
        sig_orbitals = orbitals[abs(eigen_var)>0.1]
        #print("Sig_orbitals", sig_orbitals)
        num_trans = self.num_each_dir
        numT = np.array([int(num_trans[0]*2+1),int(num_trans[1]*2+1),int(num_trans[2]*2+1)])

        #do above the smart way that is much better
        scaledTBparams = self.TB_model.get_fullHam(self,self.kpoints[kpoint],spin=spin)
        TBparams = copy.deepcopy(self.TB_params[spin]).flatten()

        energyCont = np.zeros((self.num_orbs, self.num_orbs,numT[0], numT[1], numT[2]), dtype=np.complex_)
        for orb1 in range(self.num_orbs):
            for orb2 in range(self.num_orbs):
                energyCont[orb1,orb2] = scaledTBparams[orb1,orb2]*np.conj(eigen_var[orb1])*eigen_var[orb2]
        energyCont = energyCont.flatten()
        orb1,orb2,trans1,trans2,trans3 = np.mgrid[0:self.num_orbs,0:self.num_orbs,-num_trans[0]:num_trans[0]+1,-num_trans[1]:num_trans[1]+1,-num_trans[2]:num_trans[2]+1]
        (orb1, orb2, trans1, trans2, trans3) = (orb1.flatten(), orb2.flatten(), trans1.flatten(), trans2.flatten(), trans3.flatten())
        large_TB = np.abs(energyCont.real)>0.005
        energyCont = np.around(energyCont[large_TB],decimals=6)
        TBparams = TBparams[large_TB]
        (orb1, orb2, trans1, trans2, trans3) = (orb1[large_TB], orb2[large_TB], trans1[large_TB], trans2[large_TB], trans3[large_TB])
        onsite_term = [(orb1==orb2) & (trans1==0) & (trans2==0) & (trans3==0)]
        notonsite_term = np.invert(onsite_term)[0]
        energyCont = energyCont[notonsite_term]
        TBparams = TBparams[notonsite_term]
        (orb1,orb2) = (orb1[notonsite_term],orb2[notonsite_term])
        (trans1,trans2,trans3) = (trans1[notonsite_term],trans2[notonsite_term],trans3[notonsite_term])
        sorting_ind = np.flip(np.argsort(np.abs(energyCont)))
        sortedEnergyCont = energyCont[sorting_ind]
        sort_TBparams = TBparams[sorting_ind]
        (Sorb1, Sorb2, Strans1, Strans2, Strans3) = (orb1[sorting_ind], orb2[sorting_ind], trans1[sorting_ind], trans2[sorting_ind], trans3[sorting_ind])
        stSorb1 = np.char.mod('%d', Sorb1+1)
        stSorb2 = np.char.mod('%d', Sorb2+1)
        #print("to string?", Sorb1, stSorb1)
        sorted_keys = np.char.add(stSorb1,stSorb2)

        sample_keys = sorted_keys# diff_evals_keys[sorted_sample_indices]
        sorted_sample_val = sortedEnergyCont #sample_val[sorted_sample_indices] # sorted_sample_val but with +/-
        sorted_sample_mag_val = np.abs(sortedEnergyCont.real)# np.flip(np.sort(abs(sample_val))) # This used to be sorted_sample_val
        #print(sorted_sample_val)
        #print(sample_keys[:20])
        #sample_params = diff_tb_params[sorted_sample_indices]
        #print(sample_params)
        (group_mag_vals, group_vals, bool_diff, count_list, indices) = ([sorted_sample_mag_val[0]], [sorted_sample_val[0]],
                                                                        [True], [], [])
        count = 1
        groups = []
        for i in range(1, np.size(sorted_sample_mag_val)):
            #check that the parameters have the same magnitude of real energy and TB params are the same
            energy_diff = abs(abs(sorted_sample_val[i].real) - abs(sorted_sample_val[i - 1].real))
            tbval_diff = abs(abs(sort_TBparams[i])-abs(sort_TBparams[i-1]))
            same_bond_set = energy_diff < 0.004 and tbval_diff < 0.001
            #print("check params:",i,sorted_sample_val[i].real,sorted_sample_val[i - 1].real,sorted_sample_val[i].imag,sorted_sample_val[i - 1].imag,tbval_diff,same_bond_set)
            if not same_bond_set:
                group_vals.append(sorted_sample_val[i])
                group_mag_vals.append(sorted_sample_mag_val[i])
            bool_diff.append(not same_bond_set)
        #print("bool_diff", bool_diff)
        for i in range(0, len(bool_diff) - 1):
            if not bool_diff[i + 1]:
                count += 1
            else:
                count_list.append(count)
                count = 1
        count_list.append(count)

        #will recalculate groups later cause this is no longer correct
        for n in range(len(group_vals)):
            groups.append((group_vals[n]*count_list[n], count_list[n]))


        for index in range(0, np.size(sorted_sample_mag_val)):
            if sorted_sample_mag_val[index] in group_mag_vals:
                indices.append(index)
        next_ind = []
        for i in range(len(bool_diff)):
            if bool_diff[i]:
                next_ind.append(i)
        next_ind = np.array(next_ind)
        first_vec_of_keys = []
        first_vec_of_orbs = []
        first_vec_of_trans = []
        first_vec_of_tbparams_uq = []
        first_vec_of_tbparams = []
        new_groups = []
        for i in range(len(next_ind)-1):
            first_vec_of_keys.append(np.unique(sample_keys[next_ind[i]:next_ind[i+1]]))
            first_vec_of_tbparams_uq.append(np.unique(np.around(sort_TBparams[next_ind[i]:next_ind[i + 1]].real,decimals=3)).tolist())
            first_vec_of_tbparams.append(sort_TBparams[next_ind[i]:next_ind[i + 1]].tolist())
            first_vec_of_orbs.append([Sorb1[next_ind[i]:next_ind[i + 1]].tolist(),Sorb2[next_ind[i]:next_ind[i + 1]].tolist()])
            first_vec_of_trans.append([Strans1[next_ind[i]:next_ind[i + 1]].tolist(),Strans2[next_ind[i]:next_ind[i + 1]].tolist(),Strans3[next_ind[i]:next_ind[i + 1]].tolist()])
            new_groups.append((np.sum(sorted_sample_val[next_ind[i]:next_ind[i+1]]),groups[i][1]))
        # don't forget to append the last element
        first_vec_of_keys.append(np.unique(sample_keys[next_ind[-1]:]))
        first_vec_of_tbparams_uq.append(np.unique(np.around(sort_TBparams[next_ind[-1]:].real,decimals=3)).tolist())
        first_vec_of_tbparams.append(sort_TBparams[next_ind[-1]:].tolist())
        first_vec_of_orbs.append([Sorb1[next_ind[-1]:].tolist(),Sorb2[next_ind[-1]:].tolist()])
        first_vec_of_trans.append([Strans1[next_ind[-1]:].tolist(),Strans2[next_ind[-1]:].tolist(),Strans3[next_ind[-1]:].tolist()])
        new_groups.append((np.sum(sorted_sample_val[next_ind[-1]:]), groups[-1][1]))
        groups = np.array(new_groups).real
        #print("First_vec_of_keys, ",first_vec_of_keys)

        one_d_vec_of_keys = []
        for arr in first_vec_of_keys:
            con = ''
            for ind, elem in enumerate(arr):
                if ind < 6:
                    con += elem + ", "
            one_d_vec_of_keys.append(con[0:-2])
        first_vec_keys_1D = np.array(one_d_vec_of_keys)
        first_vec_tbparams_1D = np.array(first_vec_of_tbparams_uq,dtype=object)
        #print(first_vec_tbparams_1D)

        #sort based on (energy change * num params) instead of just energy
        #print(np.abs(groups[:,0]))
        #print(first_vec_keys_1D)
        new_sort = np.flip(np.argsort(np.abs(groups[:,0])))

        #print("new sorting:",new_sort)

        self.groups = np.around(groups[new_sort],decimals=3)
        self.trans = np.array(first_vec_of_trans,dtype=object)[new_sort]
        self.tbparams = np.array(first_vec_of_tbparams,dtype=object)[new_sort]
        self.orbs = np.array(first_vec_of_orbs,dtype=object)[new_sort]
        self.coeffs = eigen_var #will use this to determine phase of orbitals
        #    print("orbs",self.orbs,"  trans",self.trans)
        self.orbkeys = first_vec_keys_1D[new_sort]
        self.tbparams_uq= first_vec_tbparams_1D[new_sort]
        self.orig_oneTB = np.zeros(6)
        #print(first_vec_tbparams_1D[new_sort].T)
        if len(new_sort) < 6:
            loop_over = range(len(new_sort))
        else:
            loop_over = range(6)
        for param in loop_over:
            uq_params =  first_vec_tbparams_1D[new_sort][param]
            one_value = np.abs(uq_params[0])
            self.orig_oneTB[param] = one_value
        #self.orig_oneTB[:len(new_sort)] = first_vec_tbparams_1D[new_sort].T[0].T
        print("og TB", self.orig_oneTB)
        #self.new_evals = np.around(new_evals, decimals=5)
        return ([self.orbkeys, self.groups, self.tbparams_uq])

    def plot_bond_run(self,num_bond = 0):
        # make it work for plotly.graph_objects
        #print(self.orbs)
        orbs = self.orbs[num_bond]
        trans = self.trans[num_bond]
        tbparams = self.tbparams[num_bond]
        print("bond run info:",orbs,trans,tbparams)
        #coeffs = self.coeffs
        #coeffs[np.abs(coeffs)>0.001] = coeffs[np.abs(coeffs)>0.001]/np.abs(coeffs)[np.abs(coeffs)>0.001]
        #phase = np.conj(coeffs[orbs[0][0]])*coeffs[orbs[1][0]]
        #print("phase from coeffs:",phase)
        ind_l = self.close_kinds[0]
        ind_r = self.close_kinds[1]
        print(ind_l,ind_r)
        kpnts = self.kpoints[ind_l:ind_r]
        eig_vecs = self.eigvecs[self.selected_spin][ind_l:ind_r,:,self.selected_band]  # [band,kpt,orbs] --> [kpnts,orbs]

        # actually only use single point values if band is degenerate
        cur_band = self.selected_band
        ind_betwn = int((ind_l + ind_r)/2)
        if cur_band == 0 :
            same_below = False
        else:
            same_below = abs(self.eigvals[self.selected_spin][ind_betwn, cur_band] - self.eigvals[self.selected_spin][ind_betwn, cur_band - 1]) < 0.001
        if cur_band == self.num_orbs-1:
            same_above = False
        else:
            same_above = abs(self.eigvals[self.selected_spin][ind_betwn, cur_band]-self.eigvals[self.selected_spin][ind_betwn, cur_band+1])<0.001
        if same_above or same_below:
            print("warning: bands are degenerate so only using the coefficients of selecting point for bond run")
            eig_vecs[:,:] = self.coeffs
        num_kpts = len(kpnts)

        bond_run = np.zeros((num_kpts),dtype=np.complex_)
        bond_energy = np.zeros((num_kpts),dtype=np.complex_)
        coeffs_mag = []
        for ind,k in enumerate(kpnts):
            #get and normalize eigvecs but keep phase
            coeffs = eig_vecs[ind]
            #coeffs[np.abs(coeffs) > 0.00001] = coeffs[np.abs(coeffs) > 0.00001] / np.abs(coeffs)[np.abs(coeffs) > 0.00001]
            for bond in range(len(tbparams)):
                vec = np.array([1,0,0])*trans[0][bond]+np.array([0,1,0])*trans[1][bond]+np.array([0,0,1])*trans[2][bond]
                vec = vec + self.orb_redcoords[orbs[1][bond]] - self.orb_redcoords[orbs[0][bond]]
                #print("check vec",vec)
                coeff_mag = np.abs(coeffs[orbs[0][bond]]+0.000000001)*np.abs(coeffs[orbs[1][bond]]+0.000000001)
                #print(ind,coeff_mag)
                phase_scale = np.conj(coeffs[orbs[0][bond]])*coeffs[orbs[1][bond]]/coeff_mag
                exp_fac = np.exp(2j * np.pi * np.dot(k, vec))*phase_scale#*np.conj(coeffs[orbs[0][bond]])*coeffs[orbs[1][bond]]
                #print(np.exp(-2j * np.pi * np.dot(k, vec)),exp_fac)
                bond_run[ind] = bond_run[ind] + tbparams[bond]*exp_fac
                bond_energy[ind] = bond_energy[ind] + tbparams[bond]*exp_fac*coeff_mag
            coeffs_mag.append(coeff_mag)
        coeffs_mag = np.array(coeffs_mag)
        bond_run = bond_run #/2# seems like it's twice as large as is should be
        #print("should be zero:",bond_run.imag)
        bond_run = bond_run.real
        xaxis = np.linspace(0,1,num=num_kpts)
        #bond run should have one consistant phase

        # build go.Figure
        import plotly.graph_objects as go
        bondrun_fig = go.Figure()

        if not (same_above or same_below):
            bondrun_fig.add_trace(go.Scatter(x=xaxis,y=bond_run.real / 2,
                               marker=dict(color="blue", size=2),name="band-dep bond run")) # divide by 2 because maximum presence would be if orbitals were perfectly split between orb 1 and orb 2 (or 1/√2 * 1/√2)
            bondrun_fig.add_trace(go.Scatter(x=xaxis,y=bond_energy.real,
                               marker=dict(color="pink", size=2),name="bond energy"))

        else:
            bondrun_fig.add_trace(go.Scatter(x=xaxis, y=bond_run.real / 2,
                                           marker=dict(color="blue", size=2), name="band-dep bond run"))

        max = 2
        min = -2
        if (bond_run > 2).any() or (bond_energy > 2).any():
            max = np.amax([np.amax(bond_run / 2) + 1,np.amax(bond_energy.real) + 1])
        if (bond_run < -2).any() or (bond_energy < 2).any():
            min = np.amin([np.amin(bond_run / 2) - 1,np.amin(bond_energy.real) - 1])
        ylim = (min,max)

        bondrun_fig.update_layout(xaxis={'title': 'Path in k-space'},
                           yaxis={'title': 'Energy (eV)'})

        # Update layout to set axis limits and ticks
        bondrun_fig.update_layout(margin=dict(l=20, r=20, t=50, b=0),
            xaxis=dict(showgrid=False, tickvals=[0,1],ticktext=self.close_labels,
                       showline=True, linewidth=1.5, linecolor='black', mirror=True,tickfont=dict(size=16), titlefont=dict(size=18)),
            yaxis=dict(showgrid=False,range=[ylim[0], ylim[1]],showline=True, linewidth=2, linecolor='black', mirror=True,
                       tickfont=dict(size=16), titlefont=dict(size=18)),
            title={'text':'<b>Bond run of selected bond</b>', 'font':dict(size=20, color='black',family='Times')},
            plot_bgcolor='rgba(0, 0, 0, 0)',
            title_x = 0.3  # Center the title
        )
        #bondrun_fig.update_layout(autosize=False, width=600, height=350)
        return bondrun_fig

    def change_sig_bonds(self,old_vals,new_vals,tbvals=None,num_bond=None):
        """
        To change tight-binding parameter, need to know two orbitals and three translations.
        """
        # if num_bond is set than tbvals need to be a float otherwise it is a list of floats

        #reassign all TB params with the same value as the one that has been changed
        chang_model = copy.deepcopy(self.TB_model)
        spin = self.selected_spin
        chang_hops = copy.deepcopy(self.TB_params[spin])
        for i in range(len(old_vals)):
            old_val = old_vals[i]
            new_val = new_vals[i]
            #print(old_val)
            same_hop = np.around(np.abs(chang_hops),decimals=3)==np.around(np.abs(old_val),decimals=3)
            print("found hops:",chang_hops[same_hop])
            print(new_val*np.sign(chang_hops[same_hop]))
            chang_hops[same_hop] = new_val*np.sign(chang_hops[same_hop])
            #print(chang_hops[same_hop])
            chang_model.TB_params[spin] = chang_hops

        chang_model.restrict_params(maximum_dist=15, minimum_value=0.00001)
        int_evals = []
        int_evecs = []
        for kpt in self.kpoints:
            (evals, evecs) = chang_model.get_ham(chang_model,kpt,spin=spin)
            int_evals.append(evals)
            int_evecs.append(evecs)

        self.new_evals = np.array(int_evals)

    def make_BS_widget(self,app=None): # returns the dash app
        # new dash version
        import dash
        from dash import dcc, html
        from dash.dependencies import Input, Output, State
        import dash_ag_grid as ag

        # Dash app initialization
        if app == None:
            app = dash.Dash(__name__)

        # orbital character table info
        titles = ["orbital type", "percent character"]
        values = [[""], [0]]

        # bond run table info
        bondinfo = ['Orbitals', 'Total Energy', 'Degeneracy', 'TB parameters', 'Modified TB']
        bonds = [["", 0, 0, '', 0]]

        # Layout of the app with subplot
        app.layout = html.Div([html.H1("Bonds behind Bandstructure",
                                       style={'margin': '15px', 'text-align': 'center', 'font-size': '48px',
                                              'height': 80}),
                               html.Div([
                                   html.Div(
                                       dcc.Graph(
                                           id='bandstructure',
                                           # figure=px.scatter(data, x='sepal_width', y='sepal_length', color='species', title='Click a point')
                                       ), style={'height': 800, 'width': 850, 'display': 'inline-block',
                                                 "marginLeft": 20}),
                                   html.Div([
                                       html.Div([html.H1("Orbital character of selected point",
                                                         style={'margin': '0px', 'text-align': 'center',
                                                                'font-size': '20px'}),
                                                 ag.AgGrid(
                                                     id='orbital_char',
                                                     columnDefs=[
                                                                    {'headerName': "titles", 'field': "titles",
                                                                     'width': 170}
                                                                ] + [
                                                                    {'headerName': f'Value_{i + 1}',
                                                                     'field': f'value_{i + 1}', 'width': 100}
                                                                    for i in range(len(values[0]))
                                                                ],
                                                     rowData=[
                                                         {"titles": title, **dict(
                                                             zip([f'value_{i + 1}' for i in range(len(values[0]))],
                                                                 values[index]))}
                                                         for index, title in enumerate(titles)
                                                     ],
                                                     dashGridOptions={"headerHeight": 0},
                                                     style={'height': 90, 'text-align': 'center'})],
                                                style={'height': 150, 'width': 700, 'display': 'inline-block'}),
                                       html.Div([html.H1("Important bonds of selected point",
                                                         style={'margin': '0px', 'text-align': 'center',
                                                                'font-size': '20px'}),
                                                 ag.AgGrid(
                                                     id='bond_table',
                                                     columnDefs=[
                                                         {'headerName': header, 'field': header, 'width': 140,
                                                          'editable': header == "Modified TB"} for header in bondinfo
                                                     ],
                                                     rowData=[{bondinfo[i]: bonds[row_index][i] for i in
                                                               range(len(bondinfo))} for row_index in
                                                              range(len(bonds))],
                                                     getRowStyle={"styleConditions": [{
                                                         "condition": "params.rowIndex === 100",
                                                         "style": {"backgroundColor": "grey"}}]},
                                                     style={'height': 225, 'align': 'right'}
                                                 )], style={'height': 250, 'width': 700, 'display': 'inline-block',
                                                            "marginBottom": 0}),
                                       html.Div([html.Button("Calculate new", id='recalc', n_clicks=0)],
                                                style={"marginLeft": 580, "marginTop": 0, 'display': 'inline-block',
                                                       'height': 25, "marginBottom": 0}),
                                       html.Div(
                                           dcc.Graph(
                                               id='bond_run',
                                               # figure=px.scatter(data, x='sepal_width', y='sepal_length', color='species', title='Click a point')
                                           ), style={'height': 400, 'width': 700, 'display': 'inline-block'})
                                   ], style={'height': 900, 'width': 700, 'display': 'inline-block'})
                               ], style={'height': 800, 'width': 1600, 'display': 'inline-block'})
                               ], style={'height': 950, 'width': 1600, 'display': 'inline-block'})

        #
        @app.callback(
            [Output('bandstructure', 'figure'),
             Output('orbital_char', 'columnDefs'),
             Output('orbital_char', 'rowData'),
             Output('bond_table', 'rowData')],
            [Input('bandstructure', 'clickData')]
        )
        def update_selected_data(clickData):
            titles = ["Orbital Type", "Percent Character"]
            values = [[""], [0]]
            bonds = [["", 0, 0, '', 0]]
            if clickData is None:
                # plot bandstructure
                bs_figure = self.plotlyBS()
                # initialize the character table
                orb_cols = [{'headerName': "titles", 'field': "titles", 'width': 170}
                            ] + [{'headerName': f'Value_{i + 1}', 'field': f'value_{i + 1}', 'width': 100}
                                 for i in range(len(values[0]))]
                orbchar_data = [
                    {"titles": title, **dict(zip([f'value_{i + 1}' for i in range(len(values[0]))], values[index]))}
                    for index, title in enumerate(titles)]

                # initialize the bond table
                bond_info = [{bondinfo[i]: bonds[row_index][i] for i in range(len(bondinfo))}
                             for row_index in range(len(bonds))]

            else:
                # Extracting information from clickData
                point_data = clickData['points'][0]
                print(point_data)
                kpoint = point_data['pointIndex']
                band = point_data['curveNumber'] % self.num_orbs
                spin = int(point_data['curveNumber'] / self.num_orbs) % 2
                self.selected_dot = [kpoint, band, spin]
                bs_figure = self.plotlyBS(selectedDot=[kpoint, band, spin])

                # update the table
                orb_type = []
                orb_char = []
                print(np.abs(self.eigvecs[spin][kpoint,:,band]))
                #percent_evec = np.around(np.abs(self.eigvecs[spin][kpoint,:,band]) * 100, decimals=1)
                orb_occup =  np.conj(self.eigvecs[spin][kpoint, :,band][:,None])*self.eigvecs[spin][kpoint,:,band][None,:] * self.kpt_olaps[spin][kpoint]
                print("orb occup:",orb_occup)
                print("orb percent:",np.sum(orb_occup,axis=0) * 100)
                percent_evec = np.around(np.sum(orb_occup,axis=0).real * 100, decimals=1)
                print(percent_evec)
                for i in range(0, self.num_orbs):  # i loops over orbitals
                    # old evecs arrangement: evecs[band][kpoint][0][i]

                    if percent_evec[i] > 5:  # only include orbitals present at point
                        # set table values
                        evec_val = percent_evec[i]
                        orb_type.append(str(i + 1))# + " - " + self.character[i])
                        orb_char.append(evec_val)

                values = [orb_type, orb_char]
                orb_cols = [{'headerName': "titles", 'field': "titles", 'width': 170}
                            ] + [{'headerName': f'Value_{i + 1}', 'field': f'value_{i + 1}', 'width': 100}
                                 for i in range(len(values[0]))]
                orbchar_data = [
                    {"titles": title, **dict(zip([f'value_{i + 1}' for i in range(len(values[0]))], values[index]))}
                    for index, title in enumerate(titles)]

                # get the bond info

                # update the second table
                keys_and_groups = self.get_significant_bonds(band=band, kpoint=kpoint,spin=spin)
                # self.change_sig_bonds(tbvals=0.3,num_bond=0)
                # self.ax_bond.clear()
                # self.ax_bond.set_title("Bond run")
                # self.ax_bond = self.plot_bond_run(ax=self.ax_bond,num_bond=bondind)
                num_bonds = len(keys_and_groups[2])
                # bond_vals = np.zeros((num_bonds,4))
                bond_vals = []
                org_tbparam = []
                for bond in range(num_bonds):
                    if abs(keys_and_groups[1][bond][0]) > 0.01:
                        tb_param = abs(keys_and_groups[2][bond][0])
                        [one, two, thr, fou] = [keys_and_groups[0][bond], keys_and_groups[1][bond][0],
                                                keys_and_groups[1][bond][1], str(keys_and_groups[2][bond])]
                        bond_vals.append([one, two, thr, fou, tb_param])
                        org_tbparam.append(tb_param)
                self.org_tbparam = org_tbparam
                bonds = bond_vals  # [["", 1, 2, 3]]
                bond_info = [{bondinfo[i]: bonds[row_index][i] for i in range(len(bondinfo))} for row_index in
                             range(len(bonds))]

                # get closest high sym points
                is_node = np.array(self.kpoint_labels) != ''
                node_inds = np.arange(self.num_kpts)[is_node] #self.k_node
                print(node_inds)
                print(kpoint)
                dis_ind = node_inds - kpoint
                neg_ind = copy.deepcopy(dis_ind)
                pos_ind = copy.deepcopy(dis_ind)
                neg_ind[neg_ind >= 0] = 1000
                pos_ind[pos_ind < 0] = 1000
                ind_right = np.argmin(pos_ind)
                ind_left = np.argmin(abs(neg_ind))
                closest_labels = [self.kpoint_labels[node_inds[ind_left]], self.kpoint_labels[node_inds[ind_right]]]
                print("get 1 kpath:",ind_left,ind_right,closest_labels)
                if closest_labels[0] == closest_labels[1]:
                    ind_right = ind_right + 1
                    closest_labels[1] = self.kpoint_labels[ind_right]
                self.close_labels = closest_labels
                [dist_left, dist_right] = [node_inds[ind_left], node_inds[ind_right]]
                kind_l = np.abs(np.arange(self.num_kpts) - dist_left).argmin()
                kind_r = np.abs(np.arange(self.num_kpts) - dist_right).argmin()
                self.close_kinds = [kind_l, kind_r]

            return bs_figure, orb_cols, orbchar_data, bond_info

        @app.callback(
            [Output('bond_table', 'getRowStyle'),
             Output('bond_run', 'figure')],
            [Input('bond_table', 'cellClicked')])
        def display_selected_row(selected_rows):
            if selected_rows:
                row = selected_rows['rowIndex']
                new_style = {"styleConditions": [{
                    "condition": "params.rowIndex === " + str(row),
                    "style": {"backgroundColor": "lightgrey"}}]}
                bond_figure = self.plot_bond_run(num_bond=row)
                return new_style, bond_figure
            else:
                import plotly.graph_objects as go
                empty_fig = go.Figure()
                new_style = {"styleConditions": [{
                    "condition": "params.rowIndex === 100",
                    "style": {"backgroundColor": "lightgrey"}}]}
                return new_style, empty_fig

        # Callback to capture edited values and print them when the button is clicked
        @app.callback(
            Output('bandstructure', 'figure', allow_duplicate=True),
            Input('recalc', 'n_clicks'),
            State('bond_table', 'rowData'),
            prevent_initial_call=True)
        def display_edited_values(n_clicks, row_data):
            if n_clicks > 0:
                edited_values = [row['Modified TB'] for row in row_data]
                print(edited_values)
                orig_values = self.org_tbparam
                change_old = []
                change_new = []
                for i in range(len(edited_values)):
                    if abs(orig_values[i] - edited_values[i]) > 0.001:
                        change_new.append(edited_values[i])
                        change_old.append(orig_values[i])
                print("vals to change:",change_new, change_old)
                # calculate new eigenvals
                self.change_sig_bonds(old_vals=change_old, new_vals=change_new)
                bs_figure = self.plotlyBS(selectedDot=self.selected_dot, plotnew=True)
                return bs_figure

        return app

    def get_coulomb(self):
        """
        This function is for me trying out building the coulomb operator 4D kernel from the COGITO gaussian functions.
        In this I will test first using outside method to calculate overlap / coulomb operator. (Compare directly calculated overlap to overlap params!)
        Then I will verify whether the sum of terms converges or diverges (should diverge).
        Then I will try out how to repartition the coulomb operator to a convergent sum.
        My main method for this is to take the next closest density in a equal amount of charge and have it cancel.
        This should converge with the decrease in distance between consequetive layers.
        Test on PbO.
        @return:
        """
        make_orbital

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
        prim_vec (unknown): Three float tuples representing the primitive vectors.
        prim_coord (unknown): List of float tuples for primitive coordinates.

    Returns:    
        unknown: List of float tuples for cartesian coordinates.
                        ex: cart_coord = _red_to_cart((a1,a2,a3),prim_coord).
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
    else: # f_z3, f_xz2, f_yz2, f_xyz, f_z(x2-y2), f_x(x2-3y2), f_y(3x2-y2) # no longer this order (is just order below)
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

def run_cogito_model(dir:str="./",tag:str="",eigfile:str="EIGENVAL",save_quality_info:bool=True,
                     save_crystal_bonds:bool=True,save_bondswCOHP:bool=True,save_ico:bool=True,max_dist:float=15,
                     min_val:float=0.00001,auto_label:str="full",densify:float=1,energy_cutoff:float=0.2,
                     bond_max:float=3,crystal_field=[]):
    """
    This function solves the COGITO TB model for specific k-points. Options for grids are uniform, band structure, or provided (via EIGENVAL).
    Primarily, this function saves all atom and bond data (from a uniform grid) to jsons for later analysis.
    By toggling the various 'save_*' options (all default true), this function can also plot parameter decay and band error,
    make the crystal bond diagrams (w or w/o the interactive COHP plots), and save ICOHP and ICOOP for more computationally effecient analysis.
    This can also plot the crystal field splitting of orbital energy levels on an atoms specified by a supplied 'crystal_field' list.
    Args:
        dir (str): The directory that contains the COGITO output files.
        tag (str): The tag appended to COGITO output files.
        eigfile (str): The EIGVAL file to use for compare_to_DFT(). Note: is appended to dir.
        save_quality_info (bool): Whether to make hopping/overlap decay plots and DFT band error analysis.
        save_crystal_bonds (bool): Whether to make and save the crystal bonds plot.
        save_bondswCOHP (bool): Whether to make and save the crystal bonds plot interacting with the COHP vs energy.
        save_ico (bool): Whether to save integrated COHP and COOP data for speedy use in COGITOico.py.
        max_dist (float): Maximum interatomic distance for including hopping or overlap parameters.
        min_val (float): The minimum value cutoff to include parameters.
        auto_label (str): Determines how to partition charge for crystal bond plots. 'mulliken' does Mulliken paritioning to atomic site, 'full' doesn't repartition but plots charges on atoms and bonds.
        densify (float): Increase k-point density by this factor.
        energy_cutoff (float): The minimum energy cutoff to plot a bond in crystal bond plots.
        bond_max (float): Atoms are plotted outside the primitive cell only if they have a bond with
                        a primitive cell atom that is shorter than this amount. This does not filter bonds between atoms that are already plotted.
        crystal_field (list): List of all the atoms that the crystal field diagram is plotted for (e.g. '0 1 3').

    Returns:

    """

    COGITOTB = COGITO_TB_Model(dir, file_suffix=tag)  # should be maximum distance
    COGITOTB.normalize_params()
    for ai in range(len(crystal_field)):
        COGITOTB.plot_crystal_field(COGITOTB, crystal_field[ai])
    COGITOTB.restrict_params(maximum_dist=max_dist, minimum_value=min_val)

    if save_quality_info:
        # plot log decay of hopping and overlaps
        COGITOTB.plot_hopping(COGITOTB)
        COGITOTB.plot_overlaps(COGITOTB)

        # compare COGITO bands with DFT bands
        from pathlib import Path
        if Path(dir + eigfile).is_file(): # check that file actually exists
            errors = COGITOTB.compare_to_DFT(COGITOTB, dir + eigfile)
        else:
            print("WARNING: EIGENVAL file at ",dir + eigfile, " not found!")
            print("Skipping compare_to_DFT for checking band error.")

    # now do the uniform stuff
    import numpy as np
    my_CoUN = COGITO_UNIFORM(COGITOTB, grid=np.array(np.around(np.array(COGITOTB.num_trans) * densify, decimals=0),dtype=int))
    my_CoUN.jsonify_bonddata()
    # my_CoUN.get_occupation() # prints electron occupation data for orbitals and atoms with full and mulliken schemes

    if save_crystal_bonds:
        my_CoUN.get_bonds_charge_figure(energy_cutoff=energy_cutoff, bond_max = bond_max,
                                        auto_label= auto_label)
    if save_bondswCOHP:
        if densify == 1:
            print("NOTE: If you want a smoother COHP vs energy plot, increase set '--densify 2' or something higher.")
        my_CoUN.get_crystal_plus_COHP(energy_cutoff= energy_cutoff, bond_max= bond_max,
                                      auto_label= auto_label)

    if save_ico:
        my_CoUN.save_ICOHP()
        my_CoUN.save_ICOOP()
        my_CoUN.save_bond_occup()
    return my_CoUN

def main(argv=None):
    import argparse
    analyzer_args = argparse.ArgumentParser(description="This function solves the COGITO TB model for specific k-points. Options for grids are uniform, band structure, or provided (via EIGENVAL). "
                                                        "Primarily, this function saves all atom and bond data (from a uniform grid) to jsons for later analysis. "
                                                        "By toggling the various 'save_*' options (all default true), this function can also plot parameter decay and band error, "
                                                        "make the crystal bond diagrams (w or w/o the interactive COHP plots), and save ICOHP and ICOOP for more computationally effecient analysis. "
                                                        "This can also plot the crystal field splitting of orbital energy levels on an atoms specified by a supplied 'crystal_field' list.")
    analyzer_args.add_argument("--dir",type=str,help="The directory that contains the COGITO output files.",default="./")
    analyzer_args.add_argument("--tag",type=str,help="The tag appended to COGITO output files.",default="")
    analyzer_args.add_argument("--eigfile",type=str,help="The EIGVAL file to use for compare_to_DFT(). Note is appended to dir",default="EIGENVAL")
    analyzer_args.add_argument("--save_quality_info",type=bool,help="Whether to make hopping/overlap decay plots and DFT band error analysis.",default=True)
    analyzer_args.add_argument("--save_crystal_bonds",type=bool,help="Whether to make and save the crystal bonds plot.",default=True)
    analyzer_args.add_argument("--save_bondswCOHP",type=bool,help="Whether to make and save the crystal bonds plot interacting with the COHP vs energy.",default=True)
    analyzer_args.add_argument("--save_ico",type=bool,help="Whether to save integrated COHP and COOP data for speedy use in COGITOico.py.",default=True)
    analyzer_args.add_argument("--max_dist",type=float,help="Maximum interatomic distance for including hopping or overlap parameters.",default=15)
    analyzer_args.add_argument("--min_val",type=float,help="The minimum value cutoff to include parameters.",default=0.00001)
    analyzer_args.add_argument("--auto_label",type=str,help="Determines how to partition charge for crystal bond plots. 'mulliken' does Mulliken paritioning to atomic site, 'full' doesn't repartition but plots charges on atoms and bonds.",default="full")
    analyzer_args.add_argument("--densify",type=float,help="Increase k-point density by this factor.",default=1)
    analyzer_args.add_argument("--energy_cutoff",type=float,help="The minimum energy cutoff to plot a bond in crystal bond plots.",default=0.2)
    analyzer_args.add_argument("--bond_max",type=float,help="Defines the maximum bond length plotted outside the primitive cell.",default=3)
    analyzer_args.add_argument("--crystal_field",type=int,help="List of all the atoms that the crystal field diagram is plotted for (e.g. '0 1 3').",nargs="*",default=[])

    args = analyzer_args.parse_args(argv)

    run_cogito_model(dir=args.dir,tag=args.tag,eigfile=args.eigfile,save_quality_info=args.save_quality_info,
                     save_crystal_bonds=args.save_crystal_bonds,save_bondswCOHP=args.save_bondswCOHP,
                     save_ico=args.save_ico,max_dist=args.max_dist,min_val=args.min_val,auto_label=args.auto_label,
                     densify=args.densify,energy_cutoff=args.energy_cutoff,bond_max=args.bond_max,
                     crystal_field=args.crystal_field,)


if __name__ == "__main__":
    raise SystemExit(main())


# working on / failed but don't want to delete yet...
'''
    def make_charge_density(self):
        # this must be done in real space
        # first this code generates a 3x3x3 cell of the crystal
        # then all possible atom combinations are run through the make_bond function
        # finally all the cells are summed onto the central cell

        ## this is not working.... I will have to just do it based on the wavefunctions later :((
        if not hasattr(self, 'ICOHP'):
            self.get_ICOHP()

        grid = [0, 0, 0]
        grid[0] = int(np.around(np.linalg.norm(self._a[0]) * 15, decimals=0))
        grid[1] = int(np.around(np.linalg.norm(self._a[1]) * 15, decimals=0))
        grid[2] = int(np.around(np.linalg.norm(self._a[2]) * 15, decimals=0))
        print("grid:", grid)
        grid = [60, 60, 60]

        # make in reduced coordinates first
        A, B, C = np.mgrid[-1:2 - 1 / grid[0]:(3 * grid[0]) * 1j, -1:2 - 1 / grid[1]:(3 * grid[1]) * 1j,
                  -1:2 - 1 / grid[2]:(3 * grid[2]) * 1j]
        cartXYZ = _red_to_cart((self._a[0], self._a[1], self._a[2]),
                               np.array([A.flatten(), B.flatten(), C.flatten()]).transpose()).T

        # now reconstruct charge density based on ICOHP/tb_params
        old_each_dir = self.num_each_dir
        new_each_dir = self.ICO_eachdir
        reduce_tb = self.TB_params[0][:, :, old_each_dir[0] - new_each_dir[0]: old_each_dir[0] + new_each_dir[0] + 1,
                    old_each_dir[1] - new_each_dir[1]:old_each_dir[1] + new_each_dir[1] + 1,
                    old_each_dir[2] - new_each_dir[2]:old_each_dir[2] + new_each_dir[2] + 1]

        coeff_multi = self.ICOHP[0] / reduce_tb
        print(coeff_multi[:2, :2, 1, 1, 1])
        print(coeff_multi[:2, :2, 2, 2, 2])
        print(coeff_multi[:2, :2, 1, 2, 1])

        cell = self.ICO_trans  # np.array(self.num_trans) # each_dir * 2 + 1
        vec_to_trans = np.zeros((cell[0], cell[1], cell[2], 3))  # self.vec_to_trans
        each_dir = self.ICO_eachdir  # self.num_each_dir

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
        num_bonds = self.numAtoms * self.numAtoms * cell[0] * cell[1] * cell[2]
        bond_indices = np.unravel_index(np.arange(num_bonds), (self.numAtoms, self.numAtoms, cell[0], cell[1], cell[2]))

        start_atom = [bond_indices[0], np.ones(num_bonds, dtype=np.int_) * each_dir[0],
                      np.ones(num_bonds, dtype=np.int_) * each_dir[1], np.ones(num_bonds, dtype=np.int_) * each_dir[2]]
        start_ind = np.ravel_multi_index(start_atom, (self.numAtoms, cell[0], cell[1], cell[2]))
        end_atom = [bond_indices[1], bond_indices[2], bond_indices[3], bond_indices[4]]
        end_ind = np.ravel_multi_index(end_atom, (self.numAtoms, cell[0], cell[1], cell[2]))

        # make orbitals in central cell first and just translate
        central_orbs = np.reshape(self.TB_model.make_orbitals(self.TB_model, cartXYZ),
                                  (self.num_orbs, 3 * grid[0], 3 * grid[1], 3 * grid[2]))

        charge_density = np.zeros((3 * grid[0], 3 * grid[1], 3 * grid[2]))
        for b_ind in range(num_bonds):
            [cell1, cell2, cell3] = [bond_indices[2][b_ind], bond_indices[3][b_ind], bond_indices[4][b_ind]]
            [atm1, atm2] = [bond_indices[0][b_ind], bond_indices[1][b_ind]]
            start = start_ind[b_ind]
            end = end_ind[b_ind]
            start_atm = np.array([x_data[start], y_data[start], z_data[start]])
            end_atm = np.array([x_data[end], y_data[end], z_data[end]])

            orbs1 = np.arange(self.num_orbs)[self.orbatomnum == atm1]
            orbs2 = np.arange(self.num_orbs)[self.orbatomnum == atm2]
            orbitals1 = central_orbs[orbs1]
            orbitals2 = central_orbs[orbs2]
            # shift orbitals2 to site out of central cell
            trans123 = np.array([cell1, cell2, cell3]) - np.array(self.ICO_eachdir)
            padx = [0, 0]
            pady = [0, 0]
            padz = [0, 0]
            if trans123[0] > 0:
                padx[0] = grid[0] * trans123[0]
            elif trans123[0] < 0:
                padx[1] = grid[0] * abs(trans123[0])
            if trans123[1] > 0:
                pady[0] = grid[1] * trans123[1]
            elif trans123[1] < 0:
                pady[1] = grid[1] * abs(trans123[1])
            if trans123[2] > 0:
                padz[0] = grid[2] * trans123[2]
            elif trans123[2] < 0:
                padz[1] = grid[2] * abs(trans123[2])

            print("orb shape", orbitals2.shape, orbitals1.shape)
            print("padding:", trans123, padx, pady, padz)
            # orbitals2 = np.pad(orbitals2, ([0,0],padx,pady,padz)) #[:,padx[1]:-padx[0],pady[1]:-pady[0],padz[1]:-padz[0]]
            if trans123[0] > 0:
                orbitals2[:, padx[0]:] = orbitals2[:, :-padx[0]]
                orbitals2[:, :padx[0]] = 0
            elif trans123[0] < 0:
                orbitals2[:, :-padx[1]] = orbitals2[:, padx[1]:]
                orbitals2[:, -padx[1]:] = 0
            if trans123[1] > 0:
                orbitals2[:, :, pady[0]:] = orbitals2[:, :, :-pady[0]]
                orbitals2[:, :, :pady[0]] = 0
            elif trans123[1] < 0:
                orbitals2[:, :, :-pady[1]] = orbitals2[:, :, pady[1]:]
                orbitals2[:, :, -pady[1]:] = 0
            if trans123[2] > 0:
                orbitals2[:, :, :, padz[0]:] = orbitals2[:, :, :, :-padz[0]]
                orbitals2[:, :, :, :padz[0]] = 0
            elif trans123[2] < 0:
                orbitals2[:, :, :, :-padz[1]] = orbitals2[:, :, :, padz[1]:]
                orbitals2[:, :, :, -padz[1]:] = 0
            print("post pad")
            bond_coeffs = coeff_multi[orbs1, :][:, orbs2][:, :, cell1, cell2, cell3]
            bond_density = np.zeros(charge_density.shape)
            for orb1 in range(len(orbs1)):
                for orb2 in range(len(orbs2)):
                    if abs(bond_coeffs[orb1, orb2]) > 0.001:
                        bond_density = bond_density + orbitals1[orb1] * orbitals2[orb2] * bond_coeffs[orb1, orb2]
            # bond_density = np.sum(orbitals1[:,None] * orbitals2[None,:] * bond_coeffs[:, :,None,None,None],axis = (0,1))
            # bond_density = self.make_bond(atm1, atm2, start_atm, end_atm, bond_coeffs, cartXYZ)
            print("bond min max:", np.amin(bond_density), np.amax(bond_density))
            charge_density = charge_density + bond_density

            print("charge min max:", np.amin(charge_density), np.amax(charge_density))
        # finally map back into the central cell

        # recip_charge_density = np.zeros(len(recip_rad),dtype=np.complex128)
        # for orb1 in range(len(recip_orbs)):
        #    for orb2 in range(len(recip_orbs)):
        #        if orbCOOP[orb1,orb2] != 0:
        #            self.make_bond()
        #            print("bond min max:",orb1, orb2, np.amin(orbitals1[orb1] * orbitals2[orb2]),np.amax(orbitals1[orb1] * orbitals2[orb2]))
        #            bond_density = bond_density + orbitals1[orb1] * orbitals2[orb2] * orbCOOP[orb1,orb2]
        #            print("bond min max:",np.amin(bond_density),np.amax(bond_density))

    def make_charge_density_recip(self):
        if not hasattr(self, 'ICOHP'):
            self.get_ICOHP()

        # generate the gpoint grid
        _nbmax = [0, 0, 0]
        _nbmax[0] = int(np.around(np.linalg.norm(self._a[0]) * 2, decimals=0))
        _nbmax[1] = int(np.around(np.linalg.norm(self._a[1]) * 2, decimals=0))
        _nbmax[2] = int(np.around(np.linalg.norm(self._a[2]) * 2, decimals=0))
        print("nbmax:", _nbmax)

        i3, j2, k1 = np.mgrid[-_nbmax[2]:_nbmax[2]:(2 * _nbmax[2] + 1) * 1j,
                     -_nbmax[1]:_nbmax[1]:(2 * _nbmax[1] + 1) * 1j,
                     -_nbmax[0]:_nbmax[0]:(2 * _nbmax[0] + 1) * 1j]
        i3 = i3.flatten()
        j2 = j2.flatten()
        k1 = k1.flatten()
        G = np.array([k1, j2, i3])
        # get cartisean and spherical coordinates
        # calculate reciprocal lattice
        b = np.array([
            np.cross(self._a[1, :], self._a[2, :]),
            np.cross(self._a[2, :], self._a[0, :]),
            np.cross(self._a[0, :], self._a[1, :]), ])
        b = 2 * np.pi * b / self.vol
        # print("b_vec calc",b,self.vol)
        self.b_vecs = b
        cart_gplusk = _red_to_cart((self.b_vecs[0], self.b_vecs[1], self.b_vecs[2]), G.transpose()).T

        # not relevant cause you need to generate too many version of the orbitals...  so just use make_bond()
        vec = cart_gplusk  # point - np.array([atm]).transpose()
        recip_rad = np.linalg.norm(vec, axis=0)
        recip_rad[recip_rad == 0] = 0.0000000001
        recip_theta = np.arccos(vec[2] / recip_rad)
        recip_theta[recip_rad == 0.0000000001] = 0
        recip_phi = np.arctan2(vec[1], vec[0])

        # try out the recip orbital coeffs
        recip_orbs = np.zeros((self.num_orbs, len(recip_rad)), dtype=np.complex128)  # {}
        for orb in range(self.num_orbs):
            recip_coeff = self.recip_orbcoeffs[orb]
            [ak, bk, ck, dk, ek, fk, gk, hk, l] = recip_coeff
            orb_peratom = self.sph_harm_key[orb]
            angular_part = complex128funs(recip_phi, recip_theta, orb_peratom)
            atomnum = self.orbatomnum[orb]
            center = self.primAtoms[atomnum]  # center the projector correctly
            exp_term = np.exp(-2j * np.pi * np.dot(G.T, center))
            radial_part = func_for_rad(recip_rad, ak, bk, ck, dk, ek, fk, gk, hk, l)
            flat_recip_orb = (-1j) ** (l) * radial_part * angular_part * exp_term

            # structure constant
            const = ((2 * np.pi) ** 3 / self.vol) ** (1 / 2)
            recip_orbs[orb] = flat_recip_orb * const

        # now reconstruct charge density based on ICOHP/tb_params
        old_each_dir = self.num_each_dir
        new_each_dir = self.ICO_eachdir
        reduce_tb = self.TB_params[0][:, :, old_each_dir[0] - new_each_dir[0]: old_each_dir[0] + new_each_dir[0] + 1,
                    old_each_dir[1] - new_each_dir[1]:old_each_dir[1] + new_each_dir[1] + 1,
                    old_each_dir[2] - new_each_dir[2]:old_each_dir[2] + new_each_dir[2] + 1]

        coeff_multi = self.ICOHP[0] / reduce_tb
        print(coeff_multi[:2, :2, 1, 1, 1])
        print(coeff_multi[:2, :2, 2, 2, 2])
        print(coeff_multi[:2, :2, 1, 2, 1])

        recip_charge_density = np.zeros(len(recip_rad), dtype=np.complex128)
        for orb1 in range(len(recip_orbs)):
            for orb2 in range(len(recip_orbs)):
                if orbCOOP[orb1, orb2] != 0:
                    print("bond min max:", orb1, orb2, np.amin(orbitals1[orb1] * orbitals2[orb2]),
                          np.amax(orbitals1[orb1] * orbitals2[orb2]))
                    bond_density = bond_density + orbitals1[orb1] * orbitals2[orb2] * orbCOOP[orb1, orb2]
                    print("bond min max:", np.amin(bond_density), np.amax(bond_density))

    def read_vasp_grid(filename):  # generated with Chatgpt after some guidance
        with open(filename, 'r') as f:
            lines = f.readlines()

        # Locate the grid size by finding the first non-empty line after an empty line
        for i in range(len(lines) - 1):
            if lines[i].strip() == "":
                grid_size = list(map(int, lines[i + 1].split()))
                grid_start = i + 2  # Data starts right after grid size
                break

        # Read the 3D grid data
        grid_data = []
        for line in lines[grid_start:]:
            grid_data.extend(map(float, line.split()))

        grid_data = np.array(grid_data).reshape(grid_size)  # Reshape into 3D array

        return grid_data
'''
