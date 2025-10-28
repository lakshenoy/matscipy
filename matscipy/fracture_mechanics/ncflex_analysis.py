################################################
#  Routines for analyzing ncflex simutations  #
################################################

import numpy as np
import os, h5py, sys
import matplotlib.pyplot as plt

from scipy.interpolate import make_splrep 
# from scipy.optimize import root, minimize

from matscipy.fracture_mechanics.crack import CubicCrystalCrack, SinclairCrack
from matscipy import parameter

from ase.units import GPa
# from ase.constraints import FixAtoms
# from ase.optimize import LBFGSLineSearch

class NcflexAnalysis:

    def __init__(self, fpath, crksys='crack', traj_file = 'x_traj.h5'):

        # Change to data directory
        os.chdir(fpath) 

        # Import the params file
        self.root = os.getcwd()
        sys.path.insert(0,self.root)
        if 'params' in sys.modules:
             sys.modules.pop('params')
        import params
        sys.path.remove(self.root)
        
        # Find Griffith's k1
        crk = CubicCrystalCrack(parameter('crack_surface'),
                                parameter('crack_front'),
                                C=parameter('C')/GPa,cauchy_born=None)
        k1g = crk.k1g(parameter('surface_energy'))

        # Extract other params
        cluster = params.cluster.copy()
        calc = parameter('calc')
        vacuum = parameter('vacuum', 10.0)
        flexible = parameter('flexible', True)
        extended_far_field = parameter('extended_far_field', False)
        k0 = parameter('k0', 1.0)
        rI = parameter('r_I')
        rIII = parameter('r_III')
        cutoff = parameter('cutoff')
        alpha0 = parameter('alpha0', 0.0) # initial guess for crack position
        
        # Initialize Sinclair crack class
        sc = SinclairCrack(crk, cluster, calc, k0 * k1g, alpha=alpha0,
                            vacuum=vacuum, variable_alpha=flexible,
                            extended_far_field=extended_far_field,
                            rI=rI, rIII=rIII, cutoff=cutoff,
                            incl_rI_f_alpha=True, variable_k=True)

        # Read trajectory and sort by alpha
        hf = h5py.File(traj_file, 'r')
        x_traj = hf['x']
        x = x_traj[:,:]
        hf.close()
        x_sort = x[x[:,-2].argsort()]
        
        # Store attributes
        self.crksys = crksys # crack system name
        self.k1g = k1g # Griffith's k1
        self.a0 = parameter('a0') # lattice constant
        self.sc = sc # instance of SinclairCrack
        self.x = x_sort # sorted ncflex trajectory

        # Store reference energies
        a0 = sc.atoms.copy()
        a0.calc = calc
        self.regFF = (sc.regionII | sc.regionIII) if extended_far_field else sc.regionII
        self.regSim = sc.regionI | self.regFF
        self.x0 = a0.get_positions()
        self.E0 = a0.get_potential_energies()[self.regSim].sum()
        self.fbar0 = a0.get_forces()[self.regFF]
    

    def find_turning_points(self, alphas=None, Ks=None, alpha_period=None, Kratio=0.05):
        
        tp_alphas, Kmins, Kmaxs = [], [], []

        if alphas is None:
            alphas = self.x_spline[:,-2]
        if Ks is None:
            Ks = self.x_spline[:,-1]
        if alpha_period is None:
            alpha_period = self.a0/2
        half_period = alpha_period / 2.0
        # print(f'Assuming spacing between consecutive Kmax > {half_period} Angstroms')

        # Tolerance for identifying min and max Ks
        min_K = np.min(Ks)
        max_K = np.max(Ks)
        Ktol = Kratio * (max_K - min_K)
        # print(f'Finding turning points with Ktol = {Ktol}')
        
        # Find indices of Ks where K is within Ktol of min_K
        i_minK = np.where( np.abs(Ks - min_K) < Ktol )[0]
        # Group corresponding alphas to identify turning points
        sub_alphas = alphas[i_minK]
        i_alpha_groups = np.split(i_minK, np.where(np.diff(sub_alphas)>half_period)[0]+1)
        for igroup in i_alpha_groups:
            imin = igroup[np.argmin(Ks[igroup])]
            if imin==0 or imin==len(alphas)-1:
                continue
            tp_alphas += [ alphas[imin] ]
            Kmins += [ Ks[imin] ]

        # Find indices of Ks where K is within Ktol of max_K
        i_maxK = np.where( np.abs(Ks - max_K) < Ktol )[0]
        # Group corresponding alphas to identify turning points 
        sub_alphas = alphas[i_maxK]
        i_alpha_groups = np.split(i_maxK, np.where(np.diff(sub_alphas)>half_period)[0]+1)
        for igroup in i_alpha_groups:
            imax = igroup[np.argmax(Ks[igroup])]
            if imax==0 or imax==len(alphas)-1:
                continue
            tp_alphas += [ alphas[imax] ]
            Kmaxs += [ Ks[imax] ]

        tp_alphas = np.sort(tp_alphas) 
        # print('Turning points found: ', tp_alphas)
        # print('Kmins: ', Kmins)
        # print('Kmaxs: ', Kmaxs)
        return tp_alphas, Kmins, Kmaxs
    

    def prep_x_spline(self,  alpha_tol=0.001, K_tol=0.001, thinning_step = 2, plot_Ks=False):
        
        x_spline = self.x.copy()

        # Filter out close alphas
        iclose = np.where(np.abs(np.diff(x_spline[:,-2])) < alpha_tol)[0]
        if len(iclose)>0:
            iter = 0
            print('Removing data points with close alphas:')
            print(f'iter {iter}: ', iclose)
            while len(iclose)>0 and iter<10:
                idel = iclose[::2] # every other index
                x_spline = np.delete(x_spline, idel, axis=0)
                iclose = np.where(np.abs(np.diff(x_spline[:,-2])) < alpha_tol)[0]
                print(f'iter {iter}: ', iclose)
                iter += 1
            if iter==10:
                print('Warning: spline fit failed due to too many repeated alphas')
        
        # Filter out close Ks, and adjacent Ks with the wrong sign
        tp_alphas, _, _ = self.find_turning_points(alphas=x_spline[:,-2], Ks=x_spline[:,-1])
        idel = []
        alphas = x_spline[:,-2]
        Ks = x_spline[:,-1]
        for i in range(len(tp_alphas)-1):
            alpha1 = tp_alphas[i]
            alpha2 = tp_alphas[i+1]
            idx_ran = np.intersect1d( np.where(alphas>=alpha1)[0], np.where(alphas<=alpha2)[0] )

            diff = np.diff(Ks[idx_ran])
            iclose = np.where( np.abs(diff) < K_tol )[0]

            diff_sign = np.sign( diff )
            direc = np.sign(sum(diff_sign))
            idirec = np.where( np.sign(diff) != direc )[0]
            
            idel_sub = np.unique( np.concatenate( (iclose, idirec) ) )
            idel += list( idx_ran[idel_sub] )
        print('Removing data points with close Ks, or wrong sign of consec Ks: ', idel)
        x_spline = np.delete(x_spline, idel, axis=0)

        # Thin the data to remove backtracking noise or oscillations
        # To disable, set thinning_step = 1 
        x_spline = x_spline[::thinning_step]

        # Store the processed trajectory data, to be used for spline fitting
        self.x_spline = x_spline

        # Plot alphas vs K
        if plot_Ks:
            alphas = x_spline[:,-2]
            Ks = x_spline[:,-1] #/ self.k1g
            plt.scatter(alphas, Ks, s=2) ; plt.title(self.crksys) 
            plt.xlabel(r'$\alpha$ crack tip position '+r'($\mathrm{\AA}$)')
            #plt.ylabel(r'$K/K_{\mathrm{g}}$ relative stress intensity factor')
            plt.ylabel(r'$K$ stress intensity factor')
            plt.tight_layout() ; plt.grid() 
            plt.savefig('spline_data.png') ; plt.clf()
      

    def fit_spline(self, degree=3, smoothness=0.0, only_Ks=False):

        self.splines = {}
        x_spline = self.x_spline

        # Fit spline for K vs alpha 
        alphas = x_spline[:,-2]
        Ks = x_spline[:,-1]
        cs = make_splrep(alphas, Ks, k=degree, s=smoothness) # smoothness parameter s=0 -> 1/std^2
        self.splines['K'] = cs

        if not only_Ks:
            nDOF = len(x_spline[0])
            for i in range(nDOF-2):
                Ui = x_spline[:,i]
                cs = make_splrep(alphas, Ui, k=degree, s=smoothness) 
                self.splines[f'U{i}'] = cs   

    def find_spline_turning_points(self,npts=2000, alpha_period=None, Kratio=0.05):

        x_spline = self.x_spline
        alphas = x_spline[:,-2]
        cs = self.splines['K']

        # Find K spline predictions
        alphas_cs = np.linspace(alphas[0],alphas[-1],npts)
        Ks_cs = cs(alphas_cs) 

        # Find lattice trapping range turning points
        tp_alphas, Kmins, Kmaxs = self.find_turning_points(alphas_cs, Ks_cs, alpha_period, Kratio)
        
        # Print results
        tp_alphas = np.sort(tp_alphas)
        print('Turning points: ', tp_alphas)
        latrap = [ np.mean(Kmins) , np.mean(Kmaxs) ] 
        print('Lattice trapping range: ', latrap)

        self.tp_alphas = tp_alphas
        self.latrap = latrap
    
    def label_branches(self):

        x_spline = self.x_spline
        alphas = x_spline[:,-2]
        tp_alphas = self.tp_alphas

        # Define alpha ranges, and whether they are stable or unstable
        alpha_ranges = np.array([[alphas[0],tp_alphas[0]]] + 
                                [ [tp1,tp2] for tp1,tp2 in zip(tp_alphas[:-1],tp_alphas[1:]) ] + 
                                [[tp_alphas[-1],alphas[-1]]])
        stable = np.zeros(len(alpha_ranges),dtype=bool) # initialize all as unstable (ie False)

        cs = self.splines['K']
        Kdiff_branch0 = cs(tp_alphas[0]) - cs(alphas[0])
        if Kdiff_branch0>0:
            stable[::2] = True # First alpha range is stable
        else:
            stable[1::2] = True # First alpha range is unstable
        print(alpha_ranges)
        print(stable)

        self.alpha_ranges = alpha_ranges
        self.stable = stable

    def identify_central_branches(self):

        alpha_ranges = self.alpha_ranges
        stable = self.stable

        # Find index of the branch containing alpha=0
        ranmin = alpha_ranges[:,0]
        ib0 = np.argmin(np.abs(ranmin))
        if ranmin[ib0] > 0.0 and ib0>0:
            ib0-=1  # index of branch containing alpha=0

        # Find indices of the three central branch numbers
        if not stable[ib0]:
            ib = [ib0-1,ib0,ib0+1] # ib0 is unstable branch crossing alpha=0
        elif ib0-2 > 0:
            ib = [ib0-2,ib0-1,ib0] # ib0-2 is not the first branch, so should be complete
        else:
            ib = [ib0,ib0+1,ib0+2] 
            if ib0+2 == len(alpha_ranges)-1 :
                # if ib0+2 is the last branch,it is likely incomplete
                print('WARNING: No set of three complete branches in this simulation!\n' +
                      '         Either repeat ncflex with a larger alpha range, or limit\n' +
                      '         your analysis to the sampled part of the lattice trapping range.')
        
        print('Indices of chosen central branches : ', ib)

        self.ib = ib
    
    def plot_spline(self, only_Ks=True, npts=2000, rel_kg=False, figname='spline_fit.png'):

        alpha_ranges = self.alpha_ranges
        stable = self.stable  
        cs = self.splines['K']
        alphas = self.x_spline[:,-2]
        alphas_cs = np.linspace(alphas[0],alphas[-1],npts)
        Ks_cs = cs(alphas_cs) 
        k1g = self.k1g
        if rel_kg:
            Ks_cs /= k1g
            Klabel = r'$K/K_{\mathrm{g}}$ relative stress intensity factor'
        else:
            Ks_cs /= 100
            Klabel = r'$K$ stress intensity factor '+r'(MPa$\sqrt{\mathrm{m}}$)'

        def get_subset(alpha_cs,K_cs,ran):
            idx_ran = np.intersect1d(np.where(alpha_cs>=ran[0])[0], np.where(alpha_cs<=ran[1])[0])
            return alpha_cs[idx_ran], K_cs[idx_ran]
        
        istable = np.where(stable==True)[0] 
        for i in istable:
            sub_alphas, sub_Ks = get_subset(alphas_cs,Ks_cs,alpha_ranges[i])
            plt.plot(sub_Ks,sub_alphas,color='C1',linestyle='solid')
            #plt.plot(sub_alphas,sub_Ks,color='black',linestyle='solid')

        iunstable = np.where(stable==False)[0]
        for i in iunstable:
            sub_alphas, sub_Ks = get_subset(alphas_cs,Ks_cs,alpha_ranges[i])
            plt.plot(sub_Ks,sub_alphas,color='black')#,linestyle='dashed')
            #plt.plot(sub_alphas,sub_Ks,color='black',linestyle='dashed')

        # Plot original data points
        # x_spline = self.x_spline
        # alphas = x_spline[:,-2]
        # Ks = x_spline[:,-1]
        # plt.scatter(Ks,alphas,s=1,color='C0')

        # Axis labels
        plt.xlabel(Klabel)
        plt.ylabel(r'$\alpha$ crack tip position '+r'($\mathrm{\AA}$)')
        plt.title(self.crksys)
        plt.grid()
        plt.savefig(figname, bbox_inches='tight')
        plt.clf()

        if not only_Ks:
            
            nDOF = len(self.x[0])
            for i in range(nDOF-2):
                cs = self.splines[f'U{i}'] 
                Ui_cs = cs(alphas_cs)
                plt.plot(alphas_cs,Ui_cs,color=f'C{i}',linestyle='dashed')

            # Axis labels
            plt.xlabel(r'crack tip position $\alpha$')
            plt.ylabel(r'region-I displacments $U_i$')
            plt.title(self.crksys)
            plt.tight_layout()
            plt.grid()
            plt.savefig(f'spline_Us.png')
            plt.clf() 