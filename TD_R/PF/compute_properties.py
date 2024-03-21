import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from numba import jit
#from pygifsicle import optimize as gifOPT # This needs to be installed somewhere
#from PIL import Image, ImageDraw, ImageFont
#import imageio
import subprocess as sp
from time import time, sleep
from scipy.integrate import simps
from scipy.interpolate import interp1d
from scipy.interpolate import interp2d
from scipy.ndimage import gaussian_filter
from scipy.special import hermite
from scipy.special import factorial
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

##### Obtained from Braden M. Weight #####
# Compute various properties, including difference density

"""
Install pygifcicle:
pip3 install pygifsicle


Install gifsicle: ( echo "$(pwd)" = /scratch/bweight/software/ )
curl -sL http://www.lcdf.org/gifsicle/gifsicle-1.91.tar.gz | tar -zx
cd gifsicle-1.91
./configure --disable-gifview
make install exec_prefix=$(pwd) prefix=$(pwd) datarootdir=$(pwd)
"""

def getGlobals():
    global A0_LIST, WC, NM, NF, d, NPolCompute, NA0, write_TD_Files, data_PF, print_level
    global CMomFile, DMomFile, QMomFile, QGrid, plotDIM, plotDIM2D
    global EVEC_INTS, EVEC_NORM, EVEC_OUT, RPA
    #A0_LIST = np.arange( 0.0, 0.2+0.001, 0.001 ) # For spectra
    #A0_LIST = np.arange( 0.0, 0.02+0.01, 0.01 ) # For orbitals
    A0_LIST = np.array( [0.0, 0.05, 0.1, 0.15, 0.2] ) # For orbitals
    #A0_LIST = np.array( [0.0, 0.05, 0.2, 0.3, 0.4, 0.5] ) # For orbitals
    #A0_LIST = np.arange( 0.0, 0.5+0.01, 0.01 )
    #A0_LIST = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]) # For orbitals
    NA0 = len(A0_LIST)
    WC = 10.0   # eV
    NM = 50
    NF = 10
    RPA = True # Look for TD-DFT/RPA data rather than TD-DFT/TDA data
    EVEC_INTS = np.array([ 1,0,0 ]) # Cavity Polarization Vector (input as integers without normalizing)
    #NPolCompute = NM # For spectra
    NPolCompute = 2 # For Orbitals
    QGrid = np.arange( -20, 20, 0.2 )
    write_TD_Files = True
    print_level = 0 # 0 = Minimal, 1 = Some, 2 = Debugging --> This was a goal that never happened.
    d = 'x'
    plotDIM   = 'x' # Dimension for plotting 2D transition density
    plotDIM2D = (0,1) # Plotting these two dimensions as heatmap/contour
    data_PF = "data_PF"
    EVEC_NORM = EVEC_INTS / np.linalg.norm(EVEC_INTS)
    EVEC_OUT = "_".join(map(str,EVEC_INTS))

#@jit(nopython=True)
def get_TD_fast_1r( Upol, TD_matter, NPolCompute ):
    # EXACT:  <P0| TD(R) |Pa> = SUM_{j k n} C_{jn}^(0) C_{kn}^(a) TD(R)_{jk}
    # Note: We only have T_{00} and T_{0k} matrix elements
    sp.call(f"rm data_TD/transition_density_contributions_{EVEC_OUT}_WC{WC}.dat",shell=True)
    TD_pol = np.zeros(( len(Upol), NPolCompute, Nxyz[0]*Nxyz[1]*Nxyz[2] )) # A0, NPol, NGrid, assuming calculation of P0 to Pa, a = 0,1,2
    for A0IND in range(NA0):
        print(f"Working on TD for A0: {A0IND+1} of {NA0}")
        for a in range( NPolCompute ):                       # Choose only up to what is needed..
            print(f"Working on TD for POL: {a+1} of {NPolCompute}")
            sp.call(f'''echo "Non-zero contributions to transition Density for state P{a} for A0 = {round(A0_LIST[A0IND],3)}:" >> data_TD/transition_density_contributions_{EVEC_OUT}_WC{WC}.dat''',shell=True)
            sp.call('''echo "n, j, k, C_{jn}^p, C_{kn}^p, C_{jn}^p * C_{kn}^p"''' + f">> data_TD/transition_density_contributions_{EVEC_OUT}_WC{WC}.dat",shell=True)
            for n in range( NF ):                         # Photons
                for state_j in range( NM ):               # Matter
                    polIND1 = state_j * NF  + n           # Only all | 0,n > basis here
                    for state_k in range( NM ):           # Matter
                        if ( state_j == state_k or state_j == 0 or state_k == 0 ): # We only have 0-->k and k-->k densities !!!!
                            polIND2 = state_k * NF  + n       # Including all | S,n > basis here
                            TD_pol[A0IND,a,:] += Upol[A0IND,polIND1,0] * Upol[A0IND,polIND2,a] * TD_matter[state_j,state_k,:,:,:].flatten()
                            if ( Upol[A0IND,polIND1,0] * Upol[A0IND,polIND2,a] > 1e-3 ):
                                out = "\t".join(map(str,[n,state_j,state_k,round(Upol[A0IND,polIND1,0],4), round(Upol[A0IND,polIND2,a],4), round(Upol[A0IND,polIND1,0] * Upol[A0IND,polIND2,a],4)]))
                                sp.call(f"echo {out} >> data_TD/transition_density_contributions_{EVEC_OUT}_WC{WC}.dat",shell=True)
                                print( n,state_j,state_k,round(Upol[A0IND,polIND1,0],4), round(Upol[A0IND,polIND2,a],4), round(Upol[A0IND,polIND1,0] * Upol[A0IND,polIND2,a],4) )
                        else:
                            if ( Upol[A0IND,polIND1,a] * Upol[A0IND,polIND2,a] > 0.01 ): # Tell if missing > 1%
                                out = "MISSING\t" + "\t".join(map(str,[n,state_j,state_k,round(Upol[A0IND,polIND1,a],4), round(Upol[A0IND,polIND2,a],4), round(Upol[A0IND,polIND1,a] * Upol[A0IND,polIND2,a],4)]))
                                sp.call(f"echo {out} >> data_TD/transition_density_contributions_{EVEC_OUT}_WC{WC}.dat",shell=True)
                                print( "MISSING!!!",n,state_j,state_k,round(Upol[A0IND,polIND1,a],4), round(Upol[A0IND,polIND2,a],4), round(Upol[A0IND,polIND1,a] * Upol[A0IND,polIND2,a],4) )
    return TD_pol


#@jit(nopython=True)
def get_TDM_fast_1r( Upol, TDM_matter, NPolCompute ):
    # EXACT:  <P0| TDM(R,R') |Pa> = SUM_{j k n} C_{jn}^(0) C_{kn}^(a) TDM(R,R')_{jk}
    # Note: We only have T_{0k} matrix elements
    TDM_pol = np.zeros(( len(Upol), NPolCompute, len(TDM_matter[0,0]) * len(TDM_matter[0,0]) )) # A0, NPol, NGrid, assuming calculation of P0 to Pa, a = 0,1,2
    for A0IND in range(NA0):
        #print (A0IND)
        for a in range( NPolCompute ):                       # Choose only up to what is needed..
            for n in range( NF ):                         # Photons
                for state_j in range( NM ):               # Matter
                    polIND1 = state_j * NF  + n           # Only all | 0,n > basis here
                    for state_k in range( NM ):           # Matter
                        polIND2 = state_k * NF  + n       # Including all | S,n > basis here
                        if ( state_j == 0 or state_k == 0 ): # We only have 0-->k NTOs !!!!
                            TDM_pol[A0IND,a,:] += Upol[A0IND,polIND1,0] * Upol[A0IND,polIND2,a] * TDM_matter[state_j,state_k,:,:].flatten()
    return TDM_pol


@jit(nopython=True)
def get_NTO_fast_1r( Upol, NTO_matter, NPolCompute ):
    # EXACT:  <P0| TD(R) |Pa> = SUM_{j k n} C_{jn}^(0) C_{kn}^(a) TD(R)_{jk}
    # Note: We only have T_{00} and T_{0k} matrix elements
    NTO_pol = np.zeros(( len(Upol), NPolCompute, 2, Nxyz[0]*Nxyz[1]*Nxyz[2] )) # A0, NPol, NGrid, assuming calculation of P0 to Pa, a = 0,1,2
    for A0IND in range(NA0):
        #print (A0IND)
        for ind in range(2): # HOTO/LUTO
            for a in range( NPolCompute ):                       # Choose only up to what is needed..
                for n in range( NF ):                         # Photons
                    for state_j in range( NM ):               # Matter
                        polIND1 = state_j * NF  + n           # Only all | 0,n > basis here
                        for state_k in range( NM ):           # Matter
                            polIND2 = state_k * NF  + n       # Including all | S,n > basis here
                            NTO_pol[A0IND,a,ind,:] += Upol[A0IND,polIND1,0] * Upol[A0IND,polIND2,a] * NTO_matter[state_j,state_k,ind,:,:,:].flatten()
    return NTO_pol


def get_diag_density_fast_1r( Upol, TD_matter, NPolCompute ):
    # EXACT:  <Pa| TD(R) |Pa> = SUM_{j k n} C_{jn}^(a) C_{kn}^(a) TD(R)_{jk}
    # Note: We only have T_{kk} and T_{0k} matrix elements
    sp.call(f"rm data_diagonal_density/diagonal_density_contributions_{EVEC_OUT}_WC{WC}.dat",shell=True)
    sp.call(f"touch data_diagonal_density/diagonal_density_contributions_{EVEC_OUT}_WC{WC}.dat",shell=True)
    
    DIAG_DENSITY = np.zeros(( len(Upol), NPolCompute, Nxyz[0]*Nxyz[1]*Nxyz[2] )) # A0, NPol, NGrid, assuming calculation of P0 to Pa, a = 0,1,2
    for A0IND in range(NA0):
        print(f"Diagonal Density (A0 = {round(A0_LIST[A0IND],3)})")
        for a in range( NPolCompute ):                       # Choose only up to what is needed..
            print(f"Non-zero contributions to diagonal Density for state P{a}:")
            print("n, j, k, C_{jn}, C_{kn}, P_{jn,kn}")
            sp.call(f'''echo "Non-zero contributions to diagonal Density for state P{a} for A0 = {round(A0_LIST[A0IND],3)}:" >> data_diagonal_density/diagonal_density_contributions_{EVEC_OUT}_WC{WC}.dat''',shell=True)
            sp.call('''echo "n, j, k, C_{jn}, C_{kn}, P_{jn,kn}"''' + f">> data_diagonal_density/diagonal_density_contributions_{EVEC_OUT}_WC{WC}.dat",shell=True)
            for n in range( NF ):                         # Photons
                for state_j in range( NM ):               # Matter
                    polIND1 = state_j * NF  + n           # Only all | j,n > basis here
                    for state_k in range( NM ):           # Matter
                        polIND2 = state_k * NF  + n       # Including all | k,n > basis here
                        if ( state_j == state_k or state_j == 0 or state_k == 0 ): # We only have 0-->k and k-->k densities !!!!
                            DIAG_DENSITY[A0IND,a,:] += Upol[A0IND,polIND1,a] * Upol[A0IND,polIND2,a] * TD_matter[state_j,state_k,:,:,:].flatten()
                            if ( Upol[A0IND,polIND1,a] * Upol[A0IND,polIND2,a] > 1e-3 ):
                                out = "\t".join(map(str,[n,state_j,state_k,round(Upol[A0IND,polIND1,a],4), round(Upol[A0IND,polIND2,a],4), round(Upol[A0IND,polIND1,a] * Upol[A0IND,polIND2,a],4)]))
                                sp.call(f"echo {out} >> data_diagonal_density/diagonal_density_contributions_{EVEC_OUT}_WC{WC}.dat",shell=True)
                                print( n,state_j,state_k,round(Upol[A0IND,polIND1,a],4), round(Upol[A0IND,polIND2,a],4), round(Upol[A0IND,polIND1,a] * Upol[A0IND,polIND2,a],4) )
                        else:
                            if ( Upol[A0IND,polIND1,a] * Upol[A0IND,polIND2,a] > 0.01 ): # Tell if missing > 1%
                                out = "\t".join(map(str,[n,state_j,state_k,round(Upol[A0IND,polIND1,a],4), round(Upol[A0IND,polIND2,a],4), round(Upol[A0IND,polIND1,a] * Upol[A0IND,polIND2,a],4)]))
                                sp.call(f"echo ' MISSING: {out}' >> data_diagonal_density/diagonal_density_contributions_{EVEC_OUT}_WC{WC}.dat",shell=True)
                                print( "MISSING!!!",n,state_j,state_k,round(Upol[A0IND,polIND1,a],4), round(Upol[A0IND,polIND2,a],4), round(Upol[A0IND,polIND1,a] * Upol[A0IND,polIND2,a],4) )


    return DIAG_DENSITY


def getHO_q( n ):
    print ( f"HO_q_{n}" ,n )
    WC_au = WC / 27.2114
    HO_q_n = 1/np.sqrt(2**n * factorial(n) ) * ( WC_au/np.pi )**(1/4) * np.exp( - WC_au * QGrid**2 / 2 ) * hermite(n)( np.sqrt(WC_au) * QGrid)
    HO_q_n /= np.max( HO_q_n )

    plt.plot( QGrid, HO_q_n )
    plt.xlabel( r'q$_c$ (a.u.)', fontsize=15 )
    plt.ylabel( r'$\phi (q_c)$', fontsize=15 )
    plt.tight_layout()
    plt.savefig(f"HO_q_{n}.jpg" )
    plt.clf()

    return HO_q_n 

def getTraceTD( TD_matter ):
    dict_xyz = { 'x':(1+2,2+2), 'y':(0+2,2+2), 'z':(0+2,1+2) } # Adding two for state_j and state_k labels
    TD_projected = np.sum( TD_matter, axis=dict_xyz[plotDIM] )
    return TD_projected

@jit(nopython=True)
def get_TD_fast_1r1q( Upol, TD_projected, NPolCompute, HO_WFNs  ):
    # EXACT:  <P0| TD(R,q) |Pa> = SUM_{j k n m} C_{jn}^(0) C_{km}^(a) TD(R)_{jk} \phi(q)_n^HO \phi(q)_m^HO
    # Note: We only have T_{00} and T_{0k} matrix elements

    TD_pol = np.zeros(( len(Upol), NPolCompute, np.shape(TD_projected)[-1],  len(QGrid) )) # A0, NPol, RGrid, QGrid ... assuming calculation of P0 to Pa, a = 0,1,2
    for A0IND in range(NA0):
        #for a in range( NPolCompute ):                       # Choose only up to what is needed..
        for a in [ 0, 1, 6, 7, 21, 22 ]:                                          # Choose only up to what is needed..
            print ( a )
            for n in range( NF ):                         # Photon
                for state_j in range( NM ):               # Matter
                    polIND1 = state_j * NF  + n               # Only all | S_j,n > basis here
                    for state_k in range( NM ):           # Matter
                        for m in range( NF ):             # Photon
                            polIND2 = state_k * NF  + m       # Including all | S_k,n > basis here
                            for qi in range( len(QGrid) ):
                                TD_pol[A0IND,a,:,qi] += Upol[A0IND,polIND1,0] * Upol[A0IND,polIND2,a] * TD_projected[state_j,state_k,:] * (HO_WFNs[n])[qi] * (HO_WFNs[m])[qi]
    return TD_pol #.reshape(( len(Upol), NPolCompute, np.shape(TD_projected)[-1]  , len(QGrid) ))


#@jit(nopython=True)
def compute_dipole_1_fast( NF, Upol, MU, NPolCompute ):
    # <P0| µ |Pa> = SUM_{j j' n} C_{in}^(0) C_{jn}^(a) µ_{ij}
    eff_dipole_1 = np.zeros(( len(Upol), NPolCompute ))
    for a in range( NPolCompute ): # This will give ground to excited state "a" dipole matrix elements. Choose only up to what is needed...
        for n in range( NF ):                         # Photon
            for i in range( len(Upol[0])//NF ):                   # Matter
                polIND1 = i *NF  + n
                for j in range( len(Upol[0])//NF ):               # Matter
                    polIND2 = j *NF  + n
                    eff_dipole_1[:,a] += Upol[:,polIND1,0] * Upol[:,polIND2,a] * MU[i,j] # Implicit loop over A0
    return eff_dipole_1

#@jit(nopython=True)
def compute_eff_osc_str_fast( Npolar, Epol, eff_dipole_1 ):
    eff_osc_str = np.zeros(( len(eff_dipole_1), Npolar ))
    for a in range( NPolCompute ):
        dE = (Epol[:,a] - Epol[:,0]) / 27.2114 # eV to a.u.
        eff_osc_str[:,a] = (2/3) * dE[:] * eff_dipole_1[:,a] ** 2 # Implicit loop over A0
    return eff_osc_str

#@jit(nopython=True)
def trace_fock( Upol ):
    Upol_ad = np.zeros(( len(A0_LIST), NPolar, NM + 1 ))      # Coupling Strength, # of Polaritons, # of Adiabatic states
    for j in range( NM + 1):                               # LOOP OVER MATTER BASIS
        for n in range( NF ):                            # LOOP OVER FOCK BASIS
            polIND1 = j * NF  + n                            # POL. SUPER-INDEX 1, DEFINES STATE OF INTEREST
            for jp in range( NM + 1 ):                     # LOOP OVER MATTER BASIS
                for npp in range( NF ):                  # LOOP OVER FOCK BASIS
                    polIND2 = jp * NF + npp                  # POL. SUPER-INDEX 2, DEFINES EXPANSION COMPONENT
                    Upol_ad[ :, polIND1, jp ] += Upol[ :, polIND2, polIND1 ] ** 2 # (g,0), (g,1), ..., (g,NF-1), (e,0), (e,1), ..., 
    return Upol_ad

#@jit(nopython=True)
def trace_ad( Upol ):
    Upol_fock = np.zeros(( len(A0_LIST), NPolar, NF ))
    for j in range( NM + 1 ):
        for n in range( NF ):
            polIND1 = j * NF  + n
            for jp in range( NM + 1 ):
                for npp in range( NF ):
                    polIND2 = jp * NF + npp
                    Upol_fock[ :, polIND1, npp ] += Upol[ :, polIND2, polIND1 ] ** 2 # (g,0), (g,1), ..., (e,0), (e,1), ..., 
    return Upol_fock


def plot_dipole( MU ):

    sp.call('mkdir -p data_dipole',shell=True)

    xyz_dict = { 0:"x", 1:"y", 2:"z" }

    for dim in range(4):

        fig = plt.figure( figsize=[5.,5.] )

        polarization      = np.zeros((3))

        if ( dim == 3 ):
            polarization[:] = np.array([1.0,1.0,1.0])
            MU_d = np.einsum("JKd,d->JK", MU[:,:,:], polarization[:] / np.sqrt(3))
            name = "Total"
        else:
            polarization[dim] = 1.0
            MU_d = np.einsum("JKd,d->JK", MU[:,:,:], polarization[:])
            name = xyz_dict[dim]

        # Get max and round to nearest 0.5
        MU1_MAX = np.max( np.abs(MU_d) )
        MU1_MAX = round( MU1_MAX*5 )/5

        # Get max and round to nearest 10
        MU2_MAX = np.max( np.abs(MU_d @ MU_d) )
        MU2_MAX = round( MU2_MAX*10 )/10

        # Plot dipole matrix
        np.savetxt(f"data_dipole/DIPOLE_MATRIX_{name}.dat", np.abs(MU_d))
        plt.imshow( np.abs(MU_d), origin='lower', cmap='afmhot_r', vmin=0.0, vmax=MU1_MAX )
        plt.colorbar(pad=0.01)
        plt.xlim(-0.5,len(MU_d)-1)
        plt.ylim(-0.5,len(MU_d)-1)
        plt.savefig(f"data_dipole/DIPOLE_MATRIX_{name}.jpg",dpi=600)
        plt.clf()

        # Plot dipole matrix squared (0-J elements)
        np.savetxt(f"data_dipole/DIPOLE_MATRIX_0J_{name}.dat", np.abs(MU_d)[0,:] )
        plt.stem( np.abs(MU_d)[0,:] )
        plt.xlim(-0.5,len(MU_d)-1)
        plt.ylim(0.0)
        plt.xlabel("Electronic State $\\alpha$",fontsize=15)
        plt.ylabel("Dipole Matrix, $|\mu_{0\\alpha}|$ (a.u.)",fontsize=15)
        plt.savefig(f"data_dipole/DIPOLE_MATRIX_0J_{name}.jpg",dpi=600)
        plt.clf()

        # Plot dipole matrix squared
        np.savetxt(f"data_dipole/DIPOLE_MATRIX_SQUARED_{name}.dat", np.abs(MU_d @ MU_d))
        plt.imshow( np.abs(MU_d @ MU_d), origin='lower', cmap='afmhot_r', vmin=0.0, vmax=MU2_MAX )
        plt.colorbar(pad=0.01)
        plt.xlim(-0.5,len(MU_d)-1)
        plt.ylim(-0.5,len(MU_d)-1)
        plt.savefig(f"data_dipole/DIPOLE_MATRIX_SQUARED_{name}.jpg",dpi=600)
        plt.clf()
        
        # Plot dipole matrix squared (0-J elements)
        np.savetxt(f"data_dipole/DIPOLE_MATRIX_SQUARED_0J_{name}.dat", np.abs(MU_d @ MU_d)[0,:] )
        plt.stem( np.abs(MU_d @ MU_d)[0,:] )
        plt.xlim(-0.5,len(MU_d)-1)
        plt.ylim(0.0)
        plt.xlabel("Electronic State $\\alpha$",fontsize=15)
        plt.ylabel("Square Dipole Matrix, $|(\hat{\mu}^2)_{0\\alpha}|$ (a.u.)",fontsize=15)
        plt.savefig(f"data_dipole/DIPOLE_MATRIX_SQUARED_0J_{name}.jpg",dpi=600)
        plt.clf()

def get_HadMU():
    global NPolar

    # Get Data
    NPolar = NM * NF
    Epol = np.zeros(( NA0, NPolar ))
    Upol = np.zeros(( NA0, NPolar, NPolar ))
    for j in range( NA0 ):
        A0 = round( A0_LIST[j], 8 )
        print ( f"Reading Upol file {j} of {NA0}" )
        Epol[j,:] = np.loadtxt(f"{data_PF}/E_{EVEC_OUT}_A0_{round(A0,6)}_WC_{round(WC,6)}_NF_{NF}_NM_{NM}.dat")
        #Upol[j,:,:] = np.loadtxt(f"{data_PF}/U_{EVEC_OUT}_A0_{round(A0,6)}_WC_{round(WC,6)}_NF_{NF}_NM_{NM}.dat")
        Upol[j,:,:] = np.load(f"{data_PF}/U_{EVEC_OUT}_A0_{round(A0,6)}_WC_{round(WC,6)}_NF_{NF}_NM_{NM}.dat.npy")

    # Get molecular dipoles and energies
    EAD  = np.zeros(( NM ))
    MU = np.zeros(( NM, NM, 3 ))
    if ( RPA == True ):
        EAD += np.loadtxt("../PLOTS_DATA/ADIABATIC_ENERGIES_RPA.dat")[:NM]
        MU  += np.load("../PLOTS_DATA/DIPOLE_RPA.dat.npy")[:NM,:NM]
    else:
        EAD += np.loadtxt("../PLOTS_DATA/ADIABATIC_ENERGIES_TDA.dat")[:NM]
        MU  += np.load("../PLOTS_DATA/DIPOLE_TDA.dat.npy")[:NM,:NM]



    assert( NPolCompute <= len(Upol[0]) ), "Number of requested polaritons to compute is larger than the number of polaritons."

    plot_dipole( MU ) # Only plot in direction needed

    MU = np.einsum("JKd,d->JK", MU, EVEC_NORM)

    return Epol, Upol, EAD, MU

def compute_Electrostatic_Moments( TD, state_j, state_k ):
    """
    Input:
        TD: nd.array(( NStates,NStates,Nx,Ny,Nz )) - transition density (position-basis)
        state_j: int - electronic state j
        state_k: int - electronic state k
    Returns:
        None
    """

    # Instantiate Files
    if ( state_j == state_k and state_k == 0 ):
        CMomFile = open("data_ElectricMoments/Exciton_Transition_Charge.dat","w")
        DMomFile = open("data_ElectricMoments/Exciton_Transition_Dipole.dat","w")
        QMomFile = open("data_ElectricMoments/Exciton_Transition_Quadrupole.dat","w")
        CMomFile.write(f"# j k Charge\n")
        DMomFile.write(f"# j k Dx Dy Dz\n")
        QMomFile.write(f"# j k Qxx Qyy Qzz Qxy Qxz Qyz\n")
    else:
        CMomFile = open("data_ElectricMoments/Exciton_Transition_Charge.dat","a")
        DMomFile = open("data_ElectricMoments/Exciton_Transition_Dipole.dat","a")
        QMomFile = open("data_ElectricMoments/Exciton_Transition_Quadrupole.dat","a")

    xyz_dic = { (0,1):2, (0,2):1, (1,2):0 }

    # Charge
    charge = np.sum(TD[state_j,state_k,:,:,:])*dLxyz[0]*dLxyz[1]*dLxyz[2]
    CMomFile.write( f"{state_j}\t{state_k}\t{charge}\n" )
    if ( print_level >= 1 ):
        print (f'\t\t,CHARGE of T_{state_j}-{state_k} = {charge} =? 0.000')


    # Diagonal dipoles (perm. dipoles) are not correct. Transition dipoles should be okay.

    # Dipole
    op_Rx = np.arange(0,Nxyz[0])*dLxyz[0]
    op_Ry = np.arange(0,Nxyz[1])*dLxyz[1]
    op_Rz = np.arange(0,Nxyz[2])*dLxyz[2]
    op_R = [ op_Rx, op_Ry, op_Rz ]

    Dx = np.sum( np.sum(TD[state_j,state_k,:,:,:],axis=(1,2)) * op_R[0] )*dLxyz[0]*dLxyz[1]*dLxyz[2] / ( 1 * (state_j != state_k) + charge * (state_j == state_k) ) 
    Dy = np.sum( np.sum(TD[state_j,state_k,:,:,:],axis=(0,2)) * op_R[1] )*dLxyz[0]*dLxyz[1]*dLxyz[2] / ( 1 * (state_j != state_k) + charge * (state_j == state_k) ) 
    Dz = np.sum( np.sum(TD[state_j,state_k,:,:,:],axis=(0,1)) * op_R[2] )*dLxyz[0]*dLxyz[1]*dLxyz[2] / ( 1 * (state_j != state_k) + charge * (state_j == state_k) ) 

    DMomFile.write( f"{state_j}\t{state_k}\t{Dx}\t{Dy}\t{Dz}\n" )

    if ( print_level >= 1 ):
        print (f'\t\t\tDipole Moment (X) of T_{state_j}-{state_k} = {Dx}')
        print (f'\t\t\tDipole Moment (Y) of T_{state_j}-{state_k} = {Dy}')
        print (f'\t\t\tDipole Moment (Z) of T_{state_j}-{state_k} = {Dz}')
    

    # Quadrupoles are not correct yet. Probably we don't need.

    # Quadrupole, depends on location of origin. Shift coordinates to Q_COM
    # V_QUAD = 1/(4 \pi \eps_o) 1/(2 r^3) SUM_[jk] \hat{r}\hat{j} Q_{jk}
    # Q_{jk} = INT ( 3 r_j r_k - r_j^2 (j==k) ) \rho(r)  dr d^3r
    Q_COM = [ np.average(op_R[0]), np.average(op_R[1]), np.average(op_R[2])  ] # Find center of box
    op_R[0] -= Q_COM[0] # Shift coordinates to center of box
    op_R[1] -= Q_COM[1] # Shift coordinates to center of box
    op_R[2] -= Q_COM[2] # Shift coordinates to center of box
    op_Q = np.zeros(( 3, 3 ))
    for d1 in range( 3 ): # x,y,z
        for d2 in range( d1, 3 ): # x,y,z
            
            if ( d1 == d2 ):
                ind_1 = (d1+1)%3
                ind_2 = (d1+2)%3
                T_proj = np.sum(TD[state_j,state_k,:,:,:],axis=(ind_1, ind_2)) * dLxyz[ind_1] * dLxyz[ind_2]
                #op_Q[d1,d2] = np.sum( np.sum( T_proj[:] * op_R[d1]**2,axis=0 ) * op_R[d2] )*dLxyz[d1]
                op_Q[d1,d2] = 2 * np.sum( T_proj[:] * op_R[d1]**2 ) * dLxyz[d1]

            else:
                dTr = xyz_dic[ (d1,d2) ]
                T_proj = np.sum( TD[state_j,state_k,:,:,:],axis=(dTr) ) * dLxyz[dTr]
                tmp = []
                for j in range( Nxyz[d2] ):
                    tmp.append( 3 * np.sum( T_proj[:,d2] * op_R[d1]**2, axis=0 ) * dLxyz[d1] )
                op_Q[d1,d2] = np.sum( np.array(tmp) * op_R[d2] )*dLxyz[d2]
                op_Q[d2,d1] = op_Q[d1,d2]

    # j k Qxx Qyy Qzz Qxy Qxz Qyz
    QMomFile.write( f"{state_j}\t{state_k}\t{op_Q[0,0]}\t{op_Q[1,1]}\t{op_Q[2,2]}\t{op_Q[0,1]}\t{op_Q[0,2]}\t{op_Q[1,2]}\n" )

    if ( print_level >= 1 ):
        print ( "Octopole Tensor\n", op_Q )
        print ( 'Tr[Q] =', np.round( np.sum(op_Q[np.diag_indices(3)]) ,4) )



def get_NTO_Data():
    """
    NOTE: WE ARE ONLY CALCULATING S0 to Sk NTOs. IN PRINCIPLE, WE NEED ALL TERMS.
    """

    print ("\tStarting to Read NTO Files.")

    # Get size from first TD file
    global NAtoms, NGrid, Nxyz, dLxyz, Lxyz, coords
    header = np.array([ np.array(j.split()) for j in open(f"TDMat/S_1.HOTO.cube","r").readlines()[1:6] ])
    NAtoms = abs(int(header[1][0]))
    Nxyz   = np.array([ header[2+j][0] for j in range(3) ]).astype(int)
    NGrid  = Nxyz[0]*Nxyz[1]*Nxyz[2]
    dLxyz  = np.array([header[2][1],header[3][2],header[4][3] ]).astype(float)
    Lxyz   = np.array([ header[1][1], header[1][2], header[1][3] ]).astype(float)
    if ( Lxyz[0] < 0 ): 
        Lxyz *= -1.000 # Switch Sign, Already Angstroms
        #dLxyz *= -1.000 # Switch Sign, Already Angstroms
    elif ( Lxyz[0] > 0 ): 
        Lxyz *= 0.529 # Convert from Bohr to Angstroms
        dLxyz *= 0.529 # Convert from Bohr to Angstroms
    Vol    = Lxyz[0] * Lxyz[1] * Lxyz[2]
    

    print (f'\tNAtoms   = {NAtoms}')
    print (f'\tTD Grid  = {NGrid}')
    print (f'\tNx Ny Nz = {Nxyz[0]} {Nxyz[1]} {Nxyz[2]}')
    print (f'\tLx Ly Lz = {Lxyz[0]} {Lxyz[1]} {Lxyz[2]} A')
    print (f'\tdLx dLy dLz = {dLxyz[0]} {dLxyz[1]} {dLxyz[2]} A')
    print (f'\tVolume   = {Vol} A^3')

    NStart = NAtoms + 6 + 1 # Extra 1 for NTOS

    coords = np.array([ j for j in open(f"TDMat/S_1.HOTO.cube","r").readlines()[6:NStart] ])


    NTO = np.zeros(( NM, NM, 2, Nxyz[0], Nxyz[1], Nxyz[2]  )) # 2 for HOTO/LUTO
    print(f"\tMemory size of transition density array in (MB, GB): ({round(NTO.size * NTO.itemsize * 10 ** -6,2)},{round(NTO.size * NTO.itemsize * 10 ** -9,2)})" )
    for state_j in range(1,NM):
        print(f"\tReading NTO {state_j}")
        for ind,eh in enumerate(['HOTO','LUTO']):
            temp = []
            try:
                lines = open(f"TDMat/S_{state_j}.{eh}.cube",'r').readlines()[NStart:]
            except FileNotFoundError:
                print (f'\t****** File "S_{state_j}.{eh}.cube" not found. Skipping this matrix element. ******')
                continue
            for count, line in enumerate(lines):
                t = line.split('\n')[0].split()
                for j in range(len(t)):
                    temp.append( float(t[j]) )
            NTO[0,state_j,ind,:,:,:] = np.array( temp ).reshape(( Nxyz[0],Nxyz[1],Nxyz[2] ))
            NTO[state_j,0,ind,:,:,:] = 1.00 * NTO[0,state_j,ind,:,:,:] # SHOULD THIS SYMMETRIC, RIGHT ???               

            #print( state_j, eh, f"../TD/NTOs/S_{state_j}.{eh}.cube" , np.max( NTO[0,j,ind] ), NTO[0,j,ind,Nxyz[0]//2,Nxyz[1]//2,Nxyz[2]//2] )
    print("I FINISHED READING NTOs.")
    #plt.plot( np.arange(Nxyz[0]), np.sum( NTO[0,1,0,:,:,:], axis=(1,2) ) )
    #plt.savefig('NTO_HOTO_0_1.jpg',dpi=300)
    #plt.clf()

    return NTO

def get_TD_Data():
    """
    NOTE: WE ARE ONLY CALCULATING S0 to Sk TRANSITION DENSITY. IN PRINCIPLE, WE NEED ALL TERMS.
    """

    sp.call('mkdir -p data_ElectricMoments',shell=True)

    #print ("\tStarting to Read TD Files.")

    # Get size from first TD cube file from QCHEM (not Gaussian)
    global NAtoms, NGrid, Nxyz, dLxyz, Lxyz, coords
    header = np.array([ np.array(j.split(),dtype=float) for j in open(f"../QCHEM.plots/dens.0.cube","r").readlines()[2:6] ])
    NAtoms = int(header[0,0])
    NGrid  = int( header[0,1] )
    Nxyz   = (header[1:,0]).astype(int)
    dLxyz  = np.array([header[1,1],header[2,2],header[3,3] ]).astype(float)
    Lxyz   = np.array([ header[0,1], header[0,2], header[0,3] ]).astype(float)
    if ( Lxyz[0] < 0 ): Lxyz *= -1.000 # Switch Sign, Already Angstroms
    if ( Lxyz[0] > 0 ): Lxyz *= 0.529 # Convert from Bohr to Angstroms
    Vol    = Lxyz[0] * Lxyz[1] * Lxyz[2]
    
    # QCHEM EXCITED DENSITIES MAY BE WRONG... THEY FORGET to MULTIPLY BY 2 FOR RESTRICTED DFT...
    is_QCHEM = False # Let's check this and fix if necessary for correct diagonal density normalization
    GS_NORM = 0.0 # Compare excited state norms to ground state for the check. If within 1 e-, is probably fine.

    print (f'\tNAtoms   = {NAtoms}')
    print (f'\tTD Grid  = {NGrid}')
    print (f'\tNx Ny Nz = {Nxyz[0]} {Nxyz[1]} {Nxyz[2]}')
    print (f'\tLx Ly Lz = {Lxyz[0]} {Lxyz[1]} {Lxyz[2]} A')
    print (f'\tVolume   = {Vol} A^3')

    NStart = NAtoms + 6

    coords = np.array([ j for j in open(f"../QCHEM.plots/dens.0.cube","r").readlines()[6:NStart] ])


    TD = np.zeros(( NM, NM, Nxyz[0], Nxyz[1], Nxyz[2]  ))
    print(f"\tMemory size of transition density array in (MB, GB): ({round(TD.size * TD.itemsize * 10 ** -6,2)},{round(TD.size * TD.itemsize * 10 ** -9,2)})" )
    for state_j in range(NM):
        for state_k in range(state_j,NM):
            print (f'\tReading Transition Density: {state_j}-{state_k}.')
            temp = []
            try:
                if ( state_j == state_k ):
                    lines = open(f"../QCHEM.plots/dens.{state_j}.cube",'r').readlines()[NStart:]
                elif ( state_j == 0 and state_k != 0 ):
                    lines = open(f"../QCHEM.plots/trans.{state_k}.cube",'r').readlines()[NStart:]
                elif ( state_j != 0 and state_k == 0 ):
                    lines = open(f"../QCHEM.plots/trans.{state_j}.cube",'r').readlines()[NStart:]
                else:
                    raise FileNotFoundError
            except FileNotFoundError:
                ####print (f'\t****** File "trans-{state_j}_{state_k}.cube" not found. Skipping this matrix element. ******')
                continue
            for count, line in enumerate(lines):
                t = line.split('\n')[0].split()
                for j in range(len(t)):
                    temp.append( float(t[j]) )
            TD[state_j,state_k,:,:,:] = np.array( temp ).reshape(( Nxyz[0],Nxyz[1],Nxyz[2] ))
            if ( state_j != state_k ):
                TD[state_k,state_j,:,:,:] = 1.00 * TD[state_j,state_k,:,:,:] # SHOULD THIS SYMMETRIC, RIGHT ??? All are real-valued.
            elif( state_j == state_k ):
                norm = np.sum( TD[state_j,state_k,:,:,:],axis=(0,1,2) ) * dLxyz[0]*dLxyz[1]*dLxyz[2]
                if ( state_j == state_k and state_j == 0 ):
                    GS_NORM = norm * 1.0
                    GS_NORM = round(GS_NORM/2)*2 # Round to nearest even electron number
                    TD[state_j,:,:,:] *= GS_NORM/norm
                    print("Renormalized ground state = ", np.sum( TD[state_j] ) * dLxyz[0]*dLxyz[1]*dLxyz[2])
                else:
                    if ( abs( norm - GS_NORM ) > 0.25 ): # Check within quarter of an electron...
                        is_QCHEM = True
                        print("I FOUND QCHEM ERROR ! Forcing non-ground state densities to be same as ground state density...")
                        TD[state_j,state_k,:,:,:] *= (GS_NORM/norm)
                        norm                       = GS_NORM
                        print("Skipping diagonal density normalization: norm =", norm)
                        if ( abs( norm - GS_NORM ) > 1e-2 ):
                            print("Total electrons are still broken...QCHEM + grid data issue ?")
                            print( "abs( norm - GS_NORM ) > 1e-2 --> True" )
                            exit()
                        #TD[state_j,state_k,:,:,:] /= norm # Normalize to 1.0 instead of NELECT
                        #print("Performing normalization of the diagonal density: norm =", norm)


            compute_Electrostatic_Moments( TD, state_j, state_k ) 
            if ( np.allclose(TD[state_j,state_k,:,:,:],np.zeros((Nxyz[0],Nxyz[1],Nxyz[2]))) ):
                print("ZEROS")
                exit()                  

    return TD


def get_TDM_Data():
    """
    Read Fragmented hgfdsTDM generated by Multiwfn.
    """
    #NFrags = int( len( np.loadtxt(f'TDMat/tmat_G_S1.txt') ) ** (1/2) )
    NFrags = int( len( np.loadtxt(f'TDMat/tmat_G_S1.txt') ) ** (1/2) )
    TDM = np.zeros(( NM,NM, NFrags, NFrags ))
    for j in range( 1,NM ):
        try:
            #TDM[0,j,:,:] = np.loadtxt(f'TDMat/tmat_G_S{j}.txt')[:,2].reshape(( NFrags, NFrags )) # CHECK TO MAKE SURE THIS GETS IT RIGHT
            TDM[0,j,:,:] = np.loadtxt(f'TDMat/tmat_G_S{j}.txt')[:,2].reshape(( NFrags, NFrags )) # CHECK TO MAKE SURE THIS GETS IT RIGHT
            TDM[j,0,:,:] = TDM[0,j,:,:]
            #print( np.shape(TDM[j,0,:,:].flatten()), NFrags )
        except OSError:
            print(f"\t\t**** tmat_G_S{j}.txt NOT FOUND ****")
            continue

        """
        print( f"Plotting original TDM for transition {j}." )
        plt.imshow( TDM[0,j,:,:] , origin='lower' )
        plt.xlabel( r"Hole ($\AA$)" ,fontsize=15)
        plt.ylabel( r"Electron ($\AA$)" ,fontsize=15)
        plt.title( r"TDM $S_0 \rightarrow S_{}$".format(j) ,fontsize=15)
        plt.savefig(f"data_TD/TDM_0_{j}.jpg",dpi=300)
        plt.clf()
        """
    return TDM

def getExansion(Upol):

    sp.call('mkdir -p data_expansion',shell=True)

    print ("Getting Projected Expansion Coefficients")

    matter_threshold = 1e-2 # Only plot contributions that are more than {threshold} at some point
    photon_threshold = 1e-2 # Only plot contributions that are more than {threshold} at some point
    U_0 = Upol[ : , :, 0 ]

    for pol in range( NPolCompute ):

        # First for polaritonic diagonal density elements: J --> J

        U_p = Upol[ : , :, pol ]

        # Trace Photon
        U0_MATTER = np.zeros(( NM, NM, len(A0_LIST) ))
        for A0IND in range( len(A0_LIST) ):
            for n in range( NF ):
                for alpha in range( NM ):
                    polIND1 = alpha * NF + n
                    for beta in range( NM ):
                        polIND2 = beta * NF + n
                        U0_MATTER[alpha, beta, A0IND] += U_p[A0IND, polIND1] * U_p[A0IND, polIND2 ]

        # Save matter matrix for all coupling strengths
        for A0IND, A0 in enumerate(A0_LIST):
            A0 = round(A0,6)
            np.savetxt( f"data_expansion/RHO_{pol}{pol}_MATTER_JK_{EVEC_OUT}_A0{A0}_WC{WC}_NM{NM}_NF{NF}.dat", U0_MATTER[:,:,A0IND], fmt="%1.5f" )
            np.savetxt( f"data_expansion/RHO_{pol}{pol}_MATTER_JK_{EVEC_OUT}_A0{A0}_WC{WC}_NM{NM}_NF{NF}_ABS.dat", np.abs(U0_MATTER[:,:,A0IND]), fmt="%1.5f" )
            #DATA = np.round( U0_MATTER[:,:,A0IND] / matter_threshold) * matter_threshold
            DATA = U0_MATTER[:,:,A0IND] * 1.0
            DATA[ np.abs(DATA) <= matter_threshold ] = 0.0
            plt.imshow( np.abs(DATA), origin='lower', cmap='brg', norm=mpl.colors.LogNorm(vmin=matter_threshold, vmax=1.00) )
            plt.colorbar(pad=0.01)
            plt.xlabel("Electronic State $\\alpha$",fontsize=15)
            plt.ylabel("Electronic State $\\beta$",fontsize=15)
            plt.title(f"Electronic Contributions to $P_{pol}$",fontsize=15)
            plt.savefig(f"data_expansion/RHO_{pol}{pol}_MATTER_JK_{EVEC_OUT}_A0{A0}_WC{WC}_NM{NM}_NF{NF}.jpg",dpi=600)
            plt.clf()

        ROW_MATTER  = U0_MATTER[0,:,:]
        DIAG_MATTER = np.array([ np.array(U0_MATTER[J,J,:]) for J in range( NM ) ])
        np.savetxt( f"data_expansion/RHO_{pol}{pol}_MATTER_JJ_{EVEC_OUT}_A0SCAN_WC{WC}_NM{NM}_NF{NF}.dat", DIAG_MATTER, fmt="%1.5f" )
        np.savetxt( f"data_expansion/RHO_{pol}{pol}_MATTER_0J_{EVEC_OUT}_A0SCAN_WC{WC}_NM{NM}_NF{NF}.dat", ROW_MATTER, fmt="%1.5f" )
        np.savetxt( f"data_expansion/RHO_{pol}{pol}_MATTER_0J_{EVEC_OUT}_A0SCAN_WC{WC}_NM{NM}_NF{NF}_ABS.dat", np.abs(ROW_MATTER), fmt="%1.5f" )
        # Plot diagonal ones
        states = []
        for j in range( NM ):
            if ( np.max(np.abs(DIAG_MATTER[j,:])) > matter_threshold ):
                states.append(j)
        #states = [ count for count,j in enumerate(DIAG_MATTER[:,-1] > matter_threshold) if j == True ]
        for count,alpha in enumerate(states):
            if ( alpha == 0 and pol == 0 ):
                plt.plot( A0_LIST, 1-DIAG_MATTER[alpha,:], "-o", c='black', label=f"1 - $\\xi$({alpha},{alpha})" )
            else:
                plt.plot( A0_LIST, DIAG_MATTER[alpha,:], "-o", label=f"$\\xi$({alpha},{alpha})" )
        plt.legend()
        plt.xlim(A0_LIST[0], A0_LIST[-1])
        plt.ylim(0)
        plt.savefig(f"data_expansion/RHO_{pol}{pol}_MATTER_JJ_{EVEC_OUT}_A0SCAN_WC{WC}_NM{NM}_NF{NF}.jpg",dpi=600)
        plt.clf()
        # Plot transition ones
        states = []
        for j in range( NM ):
            if ( np.max(np.abs(ROW_MATTER[j,:])) > matter_threshold ):
                states.append(j)
        #states = [ count for count,j in enumerate(ROW_MATTER[:,-1] > matter_threshold) if j == True ]
        for count,alpha in enumerate(states):
            if ( alpha == 0 and pol == 0 ):
                plt.plot( A0_LIST, 1-np.abs(ROW_MATTER[alpha,:]), "-o", c='black', label=f"1 - $\\xi$({0},{alpha})" )
            else:
                #plt.plot( A0_LIST, np.abs(ROW_MATTER[alpha,:]), "-o", label=f"$\\xi$({0},{alpha})" )
                plt.plot( A0_LIST, ROW_MATTER[alpha,:], "-o", label=f"$\\xi$({0},{alpha})" )
                plt.legend()
        plt.xlim(A0_LIST[0], A0_LIST[-1])
        plt.ylim(0)
        plt.savefig(f"data_expansion/RHO_{pol}{pol}_MATTER_0J_{EVEC_OUT}_A0SCAN_WC{WC}_NM{NM}_NF{NF}.jpg",dpi=600)
        plt.clf()

        # Trace matter
        U0_PHOTON = np.zeros(( NF, NF, len(A0_LIST) ))
        for A0IND in range( len(A0_LIST) ):
            for alpha in range( NM ):
                for n in range( NF ):
                    polIND1 = alpha * NF + n
                    for m in range( NF ):
                        polIND2 = alpha * NF + m
                        U0_PHOTON[n, m, A0IND] += U_p[A0IND, polIND1] * U_p[ A0IND, polIND2]

        ROW_PHOTON  = U0_PHOTON[0,:,:]
        DIAG_PHOTON = np.array([ np.array(U0_PHOTON[J,J,:]) for J in range( NF ) ])
        np.savetxt( f"data_expansion/RHO_{pol}{pol}_PHOTON_JJ_{EVEC_OUT}_A0SCAN_WC{WC}_NM{NM}_NF{NF}.dat", DIAG_PHOTON, fmt="%1.5f" )
        np.savetxt( f"data_expansion/RHO_{pol}{pol}_PHOTON_0J_{EVEC_OUT}_A0SCAN_WC{WC}_NM{NM}_NF{NF}.dat", np.abs(ROW_PHOTON), fmt="%1.5f" )
        # Plot diagonal ones
        states = []
        for j in range( NF ):
            if ( np.max(np.abs(DIAG_PHOTON[j,:])) > photon_threshold ):
                states.append(j)
        #states = [ count for count,j in enumerate(DIAG_PHOTON[:,-1] > photon_threshold) if j == True ]
        for count,alpha in enumerate(states):
            if ( alpha == 0 ):
                plt.plot( A0_LIST, DIAG_PHOTON[alpha,:], "-o", c='black', label=f"$\phi$({alpha},{alpha})" )
            else:
                plt.plot( A0_LIST, DIAG_PHOTON[alpha,:], "-o", label=f"$\phi$({alpha},{alpha})" )
        plt.legend()
        plt.xlim(A0_LIST[0], A0_LIST[-1])
        plt.ylim(0)
        plt.savefig(f"data_expansion/RHO_{pol}{pol}_PHOTON_JJ_{EVEC_OUT}_A0SCAN_WC{WC}_NM{NM}_NF{NF}.jpg",dpi=600)
        plt.clf()
        # Plot transition ones
        states = []
        for j in range( NF ):
            if ( np.max(np.abs(ROW_PHOTON[j,:])) > photon_threshold ):
                states.append(j)
        for count,alpha in enumerate(states):
            if ( alpha == 0 ):
                plt.plot( A0_LIST, np.abs(ROW_PHOTON[alpha,:]), "-o", c='black', label=f"$\phi$({0},{alpha})" )
            else:
                plt.plot( A0_LIST, np.abs(ROW_PHOTON[alpha,:]), "-o", label=f"$\phi$({0},{alpha})" )
                plt.legend()
        plt.xlim(A0_LIST[0], A0_LIST[-1])
        plt.ylim(0)
        plt.savefig(f"data_expansion/RHO_{pol}{pol}_PHOTON_0J_{EVEC_OUT}_A0SCAN_WC{WC}_NM{NM}_NF{NF}.jpg",dpi=600)
        plt.clf()

        # Plot contributions from specific states as functions of A0
        states_0J = []
        states_JJ = []
        for j in range( NM ):
            if ( np.max(np.abs(ROW_MATTER[j,:])) > matter_threshold ):
                states_0J.append(j)
        for j in range( NM ):
            if ( np.max(np.abs(DIAG_MATTER[j,:])) > matter_threshold ):
                states_JJ.append(j)

        output_0J = np.zeros(( len(states_0J), len(A0_LIST) ))
        output_JJ = np.zeros(( len(states_JJ), len(A0_LIST) ))
        for A0IND in range( len(A0_LIST) ):
            for count, state in enumerate(states_0J):
                output_0J[count, :] = ROW_MATTER[state,:]
        for A0IND in range( len(A0_LIST) ):
            for count, state in enumerate(states_JJ):
                output_JJ[count, :] = DIAG_MATTER[state,:]
        np.savetxt( f"data_expansion/RHO_{pol}{pol}_MATTER_JJ_{EVEC_OUT}_A0SCAN_WC{WC}_NM{NM}_NF{NF}_A0.dat", output_JJ.T, fmt="%1.5f", header=" ".join(map(str,states_JJ)) )
        np.savetxt( f"data_expansion/RHO_{pol}{pol}_MATTER_0J_{EVEC_OUT}_A0SCAN_WC{WC}_NM{NM}_NF{NF}_A0.dat", output_0J.T, fmt="%1.5f", header=" ".join(map(str,states_0J)) )
        np.savetxt( f"data_expansion/RHO_{pol}{pol}_MATTER_0J_{EVEC_OUT}_A0SCAN_WC{WC}_NM{NM}_NF{NF}_A0_ABS.dat", np.abs(output_0J).T, fmt="%1.5f", header=" ".join(map(str,states_0J)) )

        states = []
        for j in range( NF ):
            if ( np.max(np.abs(ROW_PHOTON[j,:])) > photon_threshold ):
                states.append(j)
        #states = [ count for count,j in enumerate(ROW_PHOTON[:,-1] > photon_threshold) if j == True ]
        output_JJ = np.zeros(( len(states), len(A0_LIST) ))
        output_0J = np.zeros(( len(states), len(A0_LIST) ))
        for A0IND in range( len(A0_LIST) ):
            for count, state in enumerate(states):
                output_JJ[count, :] = DIAG_PHOTON[state,:]
                output_0J[count, :] = ROW_PHOTON[state,:]
        np.savetxt( f"data_expansion/RHO_{pol}{pol}_PHOTON_JJ_{EVEC_OUT}_A0SCAN_WC{WC}_NM{NM}_NF{NF}_A0.dat", output_JJ.T, fmt="%1.5f", header=" ".join(map(str,states)) )
        np.savetxt( f"data_expansion/RHO_{pol}{pol}_PHOTON_0J_{EVEC_OUT}_A0SCAN_WC{WC}_NM{NM}_NF{NF}_A0.dat", np.abs(output_0J).T, fmt="%1.5f", header=" ".join(map(str,states)) )


        # Now again for polaritonic transition elements: 0 --> J

        # Trace Photon
        U0_MATTER = np.zeros(( NM, NM, len(A0_LIST) ))
        for A0IND in range( len(A0_LIST) ):
            for n in range( NF ):
                for alpha in range( NM ):
                    polIND1 = alpha * NF + n
                    for beta in range( NM ):
                        polIND2 = beta * NF + n
                        U0_MATTER[alpha, beta, A0IND] += U_0[A0IND, polIND1] * U_p[A0IND, polIND2]

        ROW_MATTER  = U0_MATTER[0,:,:]
        DIAG_MATTER = np.array([ np.array(U0_MATTER[J,J,:]) for J in range( NM ) ])
        np.savetxt( f"data_expansion/RHO_{0}{pol}_MATTER_JJ_{EVEC_OUT}_A0SCAN_WC{WC}_NM{NM}_NF{NF}.dat", DIAG_MATTER, fmt="%1.5f" )
        np.savetxt( f"data_expansion/RHO_{0}{pol}_MATTER_0J_{EVEC_OUT}_A0SCAN_WC{WC}_NM{NM}_NF{NF}.dat", np.abs(ROW_MATTER), fmt="%1.5f" )
        # Plot diagonal ones
        states = []
        for j in range( NM ):
            if ( np.max(np.abs(DIAG_MATTER[j,:])) > matter_threshold ):
                states.append(j)
        #states = [ count for count,j in enumerate(DIAG_MATTER[:,-1] > matter_threshold) if j == True ]
        for count,alpha in enumerate(states):
            if ( alpha == 0 and pol == 0 ):
                plt.plot( A0_LIST, 1-DIAG_MATTER[alpha,:], "-o", c='black', label=f"1 - $\\xi$({alpha},{alpha})" )
            else:
                plt.plot( A0_LIST, DIAG_MATTER[alpha,:], "-o", label=f"$\\xi$({alpha},{alpha})" )
        plt.legend()
        plt.xlim(A0_LIST[0], A0_LIST[-1])
        plt.ylim(0)
        plt.savefig(f"data_expansion/RHO_{0}{pol}_MATTER_JJ_{EVEC_OUT}_A0SCAN_WC{WC}_NM{NM}_NF{NF}.jpg",dpi=600)
        plt.clf()
        # Plot transition ones
        states = []
        for j in range( NM ):
            if ( np.max(np.abs(ROW_MATTER[j,:])) > matter_threshold ):
                states.append(j)
        states = [ count for count,j in enumerate(ROW_MATTER[:,-1] > matter_threshold) if j == True ]
        for count,alpha in enumerate(states):
            if ( alpha == 0 and pol == 0 ):
                plt.plot( A0_LIST, 1-np.abs(ROW_MATTER[alpha,:]), "-o", c='black', label=f"1 - $\\xi$({0},{alpha})" )
            else:
                plt.plot( A0_LIST, np.abs(ROW_MATTER[alpha,:]), "-o", label=f"$\\xi$({0},{alpha})" )
                plt.legend()
        plt.xlim(A0_LIST[0], A0_LIST[-1])
        plt.ylim(0)
        plt.savefig(f"data_expansion/RHO_{0}{pol}_MATTER_0J_{EVEC_OUT}_A0SCAN_WC{WC}_NM{NM}_NF{NF}.jpg",dpi=600)
        plt.clf()

        # Trace matter
        U0_PHOTON = np.zeros(( NF, NF, len(A0_LIST) ))
        for A0IND in range( len(A0_LIST) ):
            for alpha in range( NM ):
                for n in range( NF ):
                    polIND1 = alpha * NF + n
                    for m in range( NF ):
                        polIND2 = alpha * NF + m
                        U0_PHOTON[n, m, A0IND] += U_0[A0IND, polIND1] * U_p[A0IND, polIND2]

        ROW_PHOTON  = U0_PHOTON[0,:,:]
        DIAG_PHOTON = np.array([ np.array(U0_PHOTON[J,J,:]) for J in range( NF ) ])
        np.savetxt( f"data_expansion/RHO_{0}{pol}_PHOTON_JJ_{EVEC_OUT}_A0SCAN_WC{WC}_NM{NM}_NF{NF}.dat", DIAG_PHOTON, fmt="%1.5f" )
        np.savetxt( f"data_expansion/RHO_{0}{pol}_PHOTON_0J_{EVEC_OUT}_A0SCAN_WC{WC}_NM{NM}_NF{NF}.dat", np.abs(ROW_PHOTON), fmt="%1.5f" )
        # Plot diagonal ones
        states = []
        for j in range( NF ):
            if ( np.max(np.abs(DIAG_PHOTON[j,:])) > photon_threshold ):
                states.append(j)
        #states = [ count for count,j in enumerate(DIAG_PHOTON[:,-1] > photon_threshold) if j == True ]
        for count,alpha in enumerate(states):
            if ( alpha == 0 ):
                plt.plot( A0_LIST, DIAG_PHOTON[alpha,:], "-o", c='black', label=f"$\phi$({alpha},{alpha})" )
            else:
                plt.plot( A0_LIST, DIAG_PHOTON[alpha,:], "-o", label=f"$\phi$({alpha},{alpha})" )
        plt.legend()
        plt.xlim(A0_LIST[0], A0_LIST[-1])
        plt.ylim(0)
        plt.savefig(f"data_expansion/RHO_{0}{pol}_PHOTON_JJ_{EVEC_OUT}_A0SCAN_WC{WC}_NM{NM}_NF{NF}.jpg",dpi=600)
        plt.clf()
        # Plot transition ones
        states = []
        for j in range( NF ):
            if ( np.max(np.abs(ROW_PHOTON[j,:])) > photon_threshold ):
                states.append(j)
        #states = [ count for count,j in enumerate(ROW_PHOTON[:,-1] > photon_threshold) if j == True ]
        for count,alpha in enumerate(states):
            if ( alpha == 0 ):
                plt.plot( A0_LIST, np.abs(ROW_PHOTON[alpha,:]), "-o", c='black', label=f"$\phi$({0},{alpha})" )
            else:
                plt.plot( A0_LIST, np.abs(ROW_PHOTON[alpha,:]), "-o", label=f"$\phi$({0},{alpha})" )
                plt.legend()
        plt.xlim(A0_LIST[0], A0_LIST[-1])
        plt.ylim(0)
        plt.savefig(f"data_expansion/RHO_{0}{pol}_PHOTON_0J_{EVEC_OUT}_A0SCAN_WC{WC}_NM{NM}_NF{NF}.jpg",dpi=600)
        plt.clf()

        # Plot contributions from specific states as functions of A0
        states = []
        for j in range( NF ):
            if ( np.max(np.abs(ROW_PHOTON[j,:])) > photon_threshold ):
                states.append(j)
        #states = [ count for count,j in enumerate(ROW_PHOTON[:,-1] > photon_threshold) if j == True ]

        output_JJ = np.zeros(( len(states), len(A0_LIST) ))
        output_0J = np.zeros(( len(states), len(A0_LIST) ))
        for A0IND in range( len(A0_LIST) ):
            for count, state in enumerate(states):
                output_JJ[count, :] = DIAG_PHOTON[state,:]
                output_0J[count, :] = ROW_PHOTON[state,:]
        np.savetxt( f"data_expansion/RHO_{0}{pol}_PHOTON_JJ_A0_{EVEC_OUT}_A0SCAN_WC{WC}_NM{NM}_NF{NF}.dat", output_JJ.T, fmt="%1.5f", header=" ".join(map(str,states)) )
        np.savetxt( f"data_expansion/RHO_{0}{pol}_PHOTON_0J_A0_{EVEC_OUT}_A0SCAN_WC{WC}_NM{NM}_NF{NF}.dat", np.abs(output_0J).T, fmt="%1.5f", header=" ".join(map(str,states)) )




def getDipole_pow_1(Upol,MU):

    print("Starting ground to excited transition dipole moments.")

    eff_dipole_1 = np.zeros(( len(A0_LIST), NPolCompute))
    eff_dipole_1[:,:] += compute_dipole_1_fast( NF, Upol, MU, NPolCompute )
    
    for A0IND in range(len(A0_LIST)):
        dipFile = open( f'data_dipole/dipole_Polaritons_{EVEC_OUT}_A0{A0_LIST[A0IND]}.dat','w' )
        for p in range( NPolCompute ):
            dipFile.write( f'{eff_dipole_1[ A0IND, p ]}\n')
        dipFile.close()
        #eff_dipole_1[A0IND,:] = np.loadtxt(f'data_dipole/dipole_Polaritons_A0{round(A0_LIST[A0IND],8)}_WC{round(WC,4)}.dat')

    return eff_dipole_1

def plot_TD( TD_matter, TD_pol ):

    NMatPlot = [ 1 ]
    #NPolPlot = [ 6,7 ] #
    NPolPlot = np.arange(NPolCompute)

    #for j in NMatPlot: # Only 0-N transition density, Z-Dimension
    #    T = np.sum( TD_matter[0,j,:,:,:],axis=(1,2) )
    #    #plt.plot( np.arange(Nxyz[0]) * 0.529 * dLxyz[0], np.abs(T),'o', linewidth=6, alpha=0.5, label=f'$M_{j}^x$' )
    #    plt.plot( np.arange(Nxyz[0]) * 0.529 * dLxyz[0], np.abs(T),'-', linewidth=6, alpha=0.5, label=f'$M_{j}^x$' )
    for p in NPolPlot: # 0: P0 --> P0, Z-Dimensions
        for A0IND in range(NA0):
            A0 = round(A0_LIST[A0IND],8)
            Px = np.sum( TD_pol[A0IND,p,:,:,:],axis=(1,2) )
            plt.plot( np.arange(Nxyz[0]) * 0.529 * dLxyz[0], np.abs(Px),'--', linewidth=2, label=f'$P0{p}^x (\A0 = {A0})$' )
            
    plt.legend(loc='upper right')
    plt.xlabel('Length (A)',fontsize=15) 
    plt.ylabel('Transition Density',fontsize=15)
    plt.tight_layout()
    plt.savefig(f"data_TD/Transition_Density_Slices_POL_Matter_X_{EVEC_OUT}.jpg")
    plt.clf()


    for j in NMatPlot: #NM): # Only 0-N transition density
        T = np.sum( TD_matter[0,j,:,:,:],axis=(0,2) )
        #plt.plot( np.arange(Nxyz[1]) * 0.529 * dLxyz[1], np.abs(T),'o', linewidth=6, alpha=0.5, label=f'$M_{j}^y$' )
        plt.plot( np.arange(Nxyz[1]) * 0.529 * dLxyz[1], np.abs(T),'-', linewidth=6, alpha=0.5, label=f'$M_{j}^y$' )
    for p in NPolPlot: # 0: P0 --> P0
        for A0IND in range(NA0):
            A0 = round(A0_LIST[A0IND],8)
            Py = np.sum( TD_pol[A0IND,p,:,:,:],axis=(0,2) )
            plt.plot( np.arange(Nxyz[1]) * 0.529 * dLxyz[1], np.abs(Py),'--', linewidth=2, label=f'$P0{p}^y (\A0 = {A0})$' )
            
    plt.legend(loc='upper right')
    plt.xlabel('Length (A)',fontsize=15)
    plt.ylabel('Transition Density',fontsize=15)
    plt.tight_layout()
    plt.savefig(f"data_TD/Transition_Density_Slices_POL_Matter_Y_{EVEC_OUT}.jpg")
    plt.clf()

    for j in NMatPlot: #NM): # Only 0-N transition density
        T = np.sum( TD_matter[0,j,:,:,:],axis=(0,1) )
        #plt.plot( np.arange(Nxyz[2]) * 0.529 * dLxyz[2], np.abs(T),'o', linewidth=6, alpha=0.5, label=f'$M_{j}^z$' )
        plt.plot( np.arange(Nxyz[2]) * 0.529 * dLxyz[2], np.abs(T),'-', linewidth=6, alpha=0.5, label=f'$M_{j}^z$' )
    for p in NPolPlot: # 0: P0 --> P0
        for A0IND in range(NA0):
            A0 = round(A0_LIST[A0IND],8)
            Pz = np.sum( TD_pol[A0IND,p,:,:,:],axis=(0,1) )
            plt.plot( np.arange(Nxyz[2]) * 0.529 * dLxyz[2], np.abs(Pz),'--', linewidth=2, label=f'$P0{p}^z (\A0 = {A0})$' )
            
    plt.legend(loc='upper right')
    plt.xlabel('Length (A)',fontsize=15)
    plt.ylabel('Transition Density',fontsize=15)
    plt.tight_layout()
    plt.savefig(f"data_TD/Transition_Density_Slices_POL_Matter_Z_{EVEC_OUT}.jpg")
    plt.clf()

def compute_Transition_density_1r( Upol, TD_matter ):
    TD_pol = get_TD_fast_1r( Upol, TD_matter, NPolCompute ).reshape(( len(Upol), NPolCompute, Nxyz[0],Nxyz[1],Nxyz[2] ))

    sp.call('mkdir -p data_TD',shell=True)

    if ( write_TD_Files == True ):
        # Print TD Files in Gaussian cube format
        for p in range(NPolCompute): # p = 1 --> T_{01}(R)
            for A0IND in range(len(A0_LIST)):
                A0 = round(A0_LIST[A0IND],8)
                print (f"Writing transition density file. p: 0-->{p}, A0 = {A0}")
                f = open(f'data_TD/trans_{EVEC_OUT}_A0{A0}_WC{WC}_NM{NM}_NF{NF}_P0{p}.cube','w')
                f.write(f"P0{1} Transition Density\n")
                f.write(f"Totally {NGrid} grid points\n")
                f.write(f"{NAtoms} {-Lxyz[0]/0.529} {-Lxyz[1]/0.529} {-Lxyz[2]/0.529}\n")
                f.write(f"{Nxyz[0]} {dLxyz[0]}  0.000000   0.000000\n")
                f.write(f"{Nxyz[1]} 0.000000   {dLxyz[1]} 0.000000\n")
                f.write(f"{Nxyz[2]} 0.000000   0.000000   {dLxyz[2]} \n")
                for at in range(len(coords)):
                    f.write( coords[at] )
                for x in range(Nxyz[0]):
                    #print(f'X = {x}')
                    for y in range(Nxyz[1]):
                        outArray = []
                        for z in range(Nxyz[2]):
                            outArray.append( TD_pol[A0IND,p,x,y,z] )
                            if ( len(outArray) % 6 == 0 or z == Nxyz[2]-1 ):
                                #outArray.append('\n')
                                f.write( " ".join(map( str, np.round(outArray,8) )) + "\n" )
                                outArray = []
                f.close()


    #plot_TD( TD_matter, TD_pol )


def compute_Transition_density_matrix_1r( Upol, TDM_matter ):
    TDM_pol = get_TDM_fast_1r( Upol, TDM_matter, NPolCompute ).reshape(( len(Upol), NPolCompute, len(TDM_matter[0,0]), len(TDM_matter[0,0]) ))

    for A0IND in range( len(A0_LIST) ):
        for p in range( NPolCompute ):
            print( f"Plotting and printing polaritonic TDM for transition {p}." )
            np.savetxt(f'data_TD/TDM_P_0_{p}_A0{A0_LIST[A0IND]}.dat', TDM_pol[A0IND,p,:,:], header=r"TDM $P_0$ --> $P_{}$ $A_0$ = {} a.u.".format(p,A0_LIST[A0IND]) )
            plt.imshow( np.abs( TDM_pol[A0IND,p,:,:] ) , origin='lower', vmin=0, vmax=1.0 )
            plt.xlabel( r"Electron ($\AA$)" ,fontsize=15)
            plt.ylabel( r"Hole ($\AA$)" ,fontsize=15)
            plt.title( r"TDM $P_0$ $\rightarrow$ $P_{}$,   $A_0$ = {}  a.u.".format(p,A0_LIST[A0IND]) ,fontsize=15)
            plt.colorbar(pad=0.01)
            plt.savefig(f"data_TD/TDM_P_0_{p}_{EVEC_OUT}_A0{A0_LIST[A0IND]}.jpg",dpi=300)
            plt.clf()

            np.savetxt(f'data_TD/TDM_P_0_{p}_A0{A0_LIST[A0IND]}_diag.dat', (TDM_pol[A0IND,p])[np.diag_indices(len(TDM_pol[0,0]))], header=r"TDM (DIAG) $P_0$ --> $P_{}$ $A_0$ = {} a.u.".format(p,A0_LIST[A0IND]) )
            norm = np.sum( np.abs( (TDM_pol[A0IND,p])[np.diag_indices(len(TDM_pol[0,0]))] ) ** 2 )
            plt.plot( np.abs( (TDM_pol[A0IND,p])[np.diag_indices(len(TDM_pol[0,0]))] ) ** 2 / norm , "-", c="black" )
            plt.xlabel( r"R ($X_h = X_e$) Electron ($\AA$)" ,fontsize=15)
            plt.ylabel( r"$\rho^2$" ,fontsize=15)
            plt.title( r"TDM $P_0$ $\rightarrow$ $P_{}$,   $A_0$ = {}  a.u.".format(p,A0_LIST[A0IND]) ,fontsize=15)
            plt.savefig(f"data_TD/TDM_P_0_{p}_{EVEC_OUT}_A0{A0_LIST[A0IND]}_diag.jpg",dpi=300)
            plt.clf()

def compute_NTO_1r( Upol, NTO_matter ):
    NTO_pol = get_NTO_fast_1r( Upol, NTO_matter, NPolCompute ).reshape(( len(Upol), NPolCompute, 2, Nxyz[0],Nxyz[1],Nxyz[2] ))

    if ( write_TD_Files == True ):
        # Print TD Files in Gaussian cube format
        for p in range(NPolCompute): # p = 1 --> NTO_{01}(R)
            for A0IND in range(len(A0_LIST)):
                for ind,eh in enumerate( ['HOTO','LUTO'] ):
                    A0 = round(A0_LIST[A0IND],3)
                    print (f"Writing NTO file. p: 0-->{p}, A0 = {A0}")
                    f = open(f'data_TD/NTO_{eh}_{EVEC_OUT}_A0{A0}_WC{WC}_NM{NM}_NF{NF}_P0{p}.cube','w')
                    f.write(f"P0{p} NTO {eh}\n")
                    f.write(f"Totally {NGrid} grid points\n")
                    f.write(f"{NAtoms} -{Lxyz[0]} -{Lxyz[1]} -{Lxyz[2]}\n")
                    f.write(f"{Nxyz[0]} {dLxyz[0]}  0.000000   0.000000\n")
                    f.write(f"{Nxyz[1]} 0.000000   {dLxyz[1]} 0.000000\n")
                    f.write(f"{Nxyz[2]} 0.000000   0.000000   {dLxyz[2]} \n")
                    for at in range(len(coords)):
                        f.write( coords[at] )
                    for x in range(Nxyz[0]):
                        #print(f'X = {x}')
                        for y in range(Nxyz[1]):
                            outArray = []
                            for z in range(Nxyz[2]):
                                outArray.append( NTO_pol[A0IND,p,ind,x,y,z] )
                                if ( len(outArray) % 6 == 0 or z == Nxyz[2]-1 ):
                                    #outArray.append('\n')
                                    f.write( " ".join(map( "{:1.5f}".format, np.round(outArray,8) )) + "\n" )
                                    outArray = []
                    f.close()


def compute_diagonal_density_1r( Upol, TD_matter ):
    sp.call('mkdir -p data_diagonal_density',shell=True)

    DIAG_DENSITY = get_diag_density_fast_1r( Upol, TD_matter, NPolCompute ).reshape(( len(Upol), NPolCompute, Nxyz[0],Nxyz[1],Nxyz[2] ))


    if ( write_TD_Files == True ):
        # Print Files in Gaussian cube format
        for p in range(NPolCompute): # p = 1 --> T_{1-1}(R)
            for A0IND in range(len(A0_LIST)):
                if ( np.allclose(DIAG_DENSITY[A0IND,p,:,:,:],np.zeros((Nxyz[0],Nxyz[1],Nxyz[2]))) ):
                    print("Warning! Diagonal Density Zero: A0 =", A0)
                A0 = round(A0_LIST[A0IND],8)
                print (f"Writing diagonal density file. p: {p}-->{p}, A0 = {A0}")
                f = open(f'data_diagonal_density/diagonal_density_{EVEC_OUT}_A0{A0}_WC{WC}_NM{NM}_NF{NF}_P{p}{p}.cube','w')
                f.write(f"P{p}{p} Diagonal Density\n")
                f.write(f"Totally {NGrid} grid points\n")
                f.write(f"{NAtoms} {-Lxyz[0]/0.529} {-Lxyz[1]/0.529} {-Lxyz[2]/0.529}\n")
                f.write(f"{Nxyz[0]} {dLxyz[0]}  0.000000   0.000000\n")
                f.write(f"{Nxyz[1]} 0.000000   {dLxyz[1]} 0.000000\n")
                f.write(f"{Nxyz[2]} 0.000000   0.000000   {dLxyz[2]} \n")
                for at in range(len(coords)):
                    f.write( coords[at] )
                for x in range(Nxyz[0]):
                    #print(f'X = {x}')
                    for y in range(Nxyz[1]):
                        outArray = []
                        for z in range(Nxyz[2]):
                            outArray.append( DIAG_DENSITY[A0IND,p,x,y,z] )
                            if ( len(outArray) % 6 == 0 or z == Nxyz[2]-1 ):
                                #outArray.append('\n')
                                f.write( " ".join(map( str, np.round(outArray,8) )) + "\n" )
                                outArray = []
                f.close()



    return DIAG_DENSITY



def compute_Transition_density_1r1q( Upol, TD_matter ):
    dict_xyz = { 'x':0, 'y':1, 'z':2 }
    HO_WFNs = [ getHO_q(n) for n in range(NF) ]
    TD_projected = getTraceTD( np.abs(TD_matter) ) # state, state, plotDIM
    TD_pol = get_TD_fast_1r1q( Upol, TD_projected, NPolCompute, HO_WFNs )

    RGrid = np.linspace( 0, Lxyz[ dict_xyz[plotDIM] ], Nxyz[ dict_xyz[plotDIM] ] )
    
    for A0IND in range( len(A0_LIST) ):
        print ( f' Saving contour plot for A0 # {A0IND} ' )
        #for p in range( NPolCompute ):
        for p in [ 0, 1, 6, 7, 21, 22 ]: 
            plt.contourf( RGrid, QGrid, TD_pol[ A0IND, p, :, : ].T )
            plt.xlabel( r' r ($\AA$)', fontsize=15 )
            plt.ylabel( ' q$_c$ (a.u.)', fontsize=15 )
            plt.title( r' < P0 | $\hat\rho (r,q_c)$ | P'+f'{p} >', fontsize=15 )
            plt.colorbar()
            plt.tight_layout()
            plt.savefig( f'data_TD/TD_1r1q_A0{A0_LIST[A0IND]}_NM{NM}_NF{NF}_P0{p}.jpg' )
            plt.clf()


def getOscStr(Epol,eff_dipole_1):        
        
    print("Starting oscillator strength calculations.")
    eff_osc_str = compute_eff_osc_str_fast( NPolar, Epol, eff_dipole_1 )
    np.savetxt(f'data_dipole/osc_str.dat', np.c_[ np.arange(len(eff_osc_str)), eff_osc_str ] )
    return eff_osc_str

def plotSpectra(Epol,eff_osc_str,Char):

    sp.call('mkdir -p data_spectra',shell=True)

    EMin = 4 # eV
    EMax = 10 # eV
    Npts = 2000

    sig = 0.1 # eV

    dE = (EMax - EMin) / Npts
    energy = np.linspace( EMin, EMax, Npts )

    Epol_transition = np.zeros(( NA0, NPolar ))
    for j in range( NPolar ):
        Epol_transition[:,j] = (Epol[:,j] - Epol[:,0])


    # Make SPECTRA with gaussian width sig
    Spec = np.zeros(( NA0, Npts ))
    for A0IND in range( len(A0_LIST) ):
        if ( A0IND == 0 ): print(eff_osc_str[A0IND,:10])
        for k in range( len(energy) ):
            #Spec[A0IND,k] += np.sum( eff_osc_str[A0IND,:] * np.exp( -(energy[k] - Epol_transition[A0IND,:]) ** 2 / 2 / sig ** 2 ) )
            Spec[A0IND,k] += np.sum( eff_osc_str[A0IND,:] * (sig/4) * sig / ( (energy[k] - Epol_transition[A0IND,:])**2 + (0.5*sig)**2 ) )

    X, Y = np.meshgrid(energy, A0_LIST)
    #Spec[Spec < 0.0001] = -1 
    
    np.savetxt( f"data_spectra/polaritonic_SPECTRA_E{d}_WC{WC}_NM{NM}_NF{NF}_Ndipoles{NPolCompute}_sig{sig}.dat", Spec )



    # GET FRANK STYLE

    """
    # Add white to beginning using space from first color
    rainbow = mpl.colormaps['rainbow']#.resampled(256)
    violet = rainbow( [0] )
    white = np.array([256/256, 256/256, 256/256, 1])
    NTotal = 10000
    NShift = 2000
    neWColors = rainbow(np.linspace(0, 1, NTotal))
    neWColors[NShift:,:] = rainbow(np.linspace(0, 1, NTotal - NShift))
    for j in range( NShift ):
        frac = j / NShift
        neWColors[j, :] = np.array( (1-frac) * white + frac * violet )
    cmap = ListedColormap(neWColors)
    """

    # Add white to beginning
    rainbow = mpl.colormaps['rainbow']#.resampled(256)
    violet = rainbow( [0] )
    white = np.array([256/256, 256/256, 256/256, 1])
    NTotal = 10000
    NShift = 100
    NStart = 50
    neWColors = rainbow(np.linspace(0, 1, NTotal))
    neWColors[NShift:,:] = rainbow(np.linspace(0, 1, NTotal - NShift))
    for j in range( NShift ):
        if( j > NStart ):
            frac = (j-NStart) / (NShift) * (NShift/NStart)
        else:
            frac = 0
        neWColors[j, :] = np.array( (1-frac) * white + frac * violet )
    cmap = ListedColormap(neWColors)




    #cmap = mpl.cm.magma
    #cmap = mpl.cm.magma_r
    #cmap = mpl.cm.Purples
    #cmap = mpl.cm.terrain_r
    #cmap = mpl.cm.rainbow
    #cmap = mpl.cm.PuBu
    #cmap = mpl.cm.afmhot_r
    #plt.contourf( X, Y, Spec, cmap=cmap, norm=mpl.colors.Normalize(vmin=0, vmax=35)  )
    #plt.contourf( X, Y, Spec, cmap=cmap )
    
    
    #plt.imshow( Spec, origin='lower', interpolation='gaussian', extent=(np.amin(energy), np.amax(energy), np.amin(A0_LIST), np.amax(A0_LIST)), cmap=cmap, aspect='auto', norm=mpl.colors.Normalize(vmin=0, vmax=0.4) )
    plt.imshow( Spec, origin='lower', extent=(np.amin(energy), np.amax(energy), np.amin(A0_LIST), np.amax(A0_LIST)), cmap=cmap, aspect='auto', norm=mpl.colors.Normalize(vmin=0, vmax=0.35) )
    #plt.imshow( Spec, origin='lower', cmap=cmap )

    print(np.shape(Spec))

    #plt.colorbar()
    plt.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=0, vmax=0.35), cmap=cmap), pad=0.01)
    #plt.colorbar(mpl.cm.ScalarMappable(cmap=cmap), pad=0.01)

    plt.xlim(EMin,EMax)
    #plt.ylim(A0_LIST[0],A0_LIST[-1])
    plt.xlabel("Energy (eV)", fontsize=10)
    plt.ylabel("Coupling Strength, A$_0$ (a.u.)",fontsize=10)
    plt.title(f"Absorption (NExc:{NM} NF: {NF})",fontsize=10)
    #plt.colorbar()
    plt.tight_layout()
    plt.savefig(f"data_spectra/polaritonic_SPECTRA_E{d}_WC{WC}_NM{NM}_NF{NF}_Ndipoles{NPolCompute}.jpg", dpi=600)
    plt.clf()




    ############### Make DOS with gaussian width sig #################
    Npts = 1000
    sig = 0.05 # eV
    #sig_A0 = 0.0001 # eV
    energy = np.linspace( EMin, EMax, Npts )
    #A0_grid = np.linspace( A0_LIST[0], A0_LIST[-1], Npts )
    DOS = np.zeros(( NA0, Npts ))
    #DOS = np.zeros(( Npts, Npts ))
    for k in range( len(energy) ):
        #print(f"{k+1} of = {len(energy)}" )   
        #for ek in range( len(A0_grid) ):         
        for A0IND, A0 in enumerate( A0_LIST ):
            #DOS[A0IND,k] += np.sum( np.exp( -(energy[k] - Epol_transition[A0IND,:] ) ** 2 / 2 / sig ** 2 ) )
            #DOS[ek,k] += np.sum( np.exp( -(energy[k] - Epol_transition[A0IND,:] ) ** 2 / 2 / sig ** 2 ) * np.exp( -(A0_grid[ek] - A0_LIST[A0IND])**2 / 2 / sig_A0**2 ) )
            DOS[A0IND,k] += np.sum( np.sum( 1.0000 * (sig/4) * sig / ( (energy[k] - Epol_transition[A0IND,:])**2 + (0.5*sig)**2 ) ) )

    X, Y = np.meshgrid(energy, A0_LIST)
    #X, Y = np.meshgrid(energy, A0_grid)
    #DOS[DOS < 0.0001] = -1 
    



    #cmap = mpl.cm.magma
    #cmap = mpl.cm.Purples
    cmap = mpl.cm.terrain_r
    cmap = mpl.cm.rainbow
    #plt.contourf( X, Y, DOS, cmap=cmap )
    #plt.contourf( X, Y, DOS, cmap=cmap, norm=mpl.colors.Normalize(vmin=0, vmax=2)  )
    #plt.contourf( X, Y, np.abs(DOS), cmap=cmap, norm=mpl.colors.LogNorm()  )
    #plt.imshow( DOS , origin='lower')
    #plt.pcolormesh( X, Y, DOS, cmap=cmap, norm=mpl.colors.Normalize(vmin=0, vmax=4)  )
    plt.imshow( DOS, origin='lower', interpolation='gaussian', extent=(np.amin(energy), np.amax(energy), np.amin(A0_LIST), np.amax(A0_LIST)), cmap=cmap, aspect='auto', norm=mpl.colors.Normalize(vmin=0, vmax=1) )
    #plt.imshow( DOS, origin='lower', extent=(np.amin(energy), np.amax(energy), np.amin(A0_LIST), np.amax(A0_LIST)), cmap=cmap, aspect='auto', norm=mpl.colors.Normalize(vmin=0, vmax=1) )

    


    #plt.colorbar()
    plt.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=0, vmax=1), cmap=cmap))
    #plt.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.LogNorm(vmin=0.1), cmap=cmap))

    plt.xlim(EMin,EMax)
    plt.ylim(A0_LIST[0],A0_LIST[-1])
    plt.xlabel("Energy (eV)", fontsize=10)
    plt.ylabel("A0",fontsize=10)
    plt.title(f"DOS (NExc:{NM} NF: {NF})",fontsize=10)
    #plt.colorbar()
    plt.tight_layout()
    plt.savefig(f"data_spectra/polaritonic_DOS_E{d}_WC{WC}_NM{NM}_NF{NF}_Ndipoles{NPolCompute}.jpg", dpi=800)
    plt.clf()

    # Produce character-colored plot of polariton energies
    # Let's just save the data nicely and plot in origin
    np.savetxt(f"data_spectra/A0_LIST.dat", A0_LIST[:] )
    np.savetxt(f"data_spectra/EPol_Transition.dat", Epol_transition[:,:] )
    np.savetxt(f"data_spectra/Photon_Number.dat", Char[:,:] )


    NINTERP = 5000
    NSTATE_INTERP = 25

    A0_interp = np.linspace( A0_LIST[0], A0_LIST[-1], NINTERP )
    Epol_interp = np.zeros(( NSTATE_INTERP, NINTERP ))
    Char_interp = np.zeros(( NSTATE_INTERP, NINTERP ))

    for state in range( NSTATE_INTERP ):
        f_x = interp1d( A0_LIST, Epol_transition[:,state] )
        Epol_interp[ state, : ] = f_x( A0_interp )

        f_x = interp1d( A0_LIST, Char[:,state] )
        Char_interp[ state, : ] = f_x( A0_interp )

    np.savetxt(f"data_spectra/A0_LIST_interp.dat", A0_interp[:] )
    np.savetxt(f"data_spectra/EPol_Transition_interp.dat", Epol_interp[:,:].T )
    np.savetxt(f"data_spectra/Photon_Number_interp.dat", Char_interp[:,:].T )


def compute_difference_density_1r( TDM_matter, diag_density ):

    sp.call('mkdir -p data_difference_density',shell=True)
    
    # Let's start with the following options:
    # 0 (cavity) - 0 (no cavity)
    # 1 (cavity) - 1 (no cavity)
    # 2 (cavity) - 1 (no cavity)
    # 1 (cavity) - 0 (cavity)
    # 1 (no cavity) - 0 (no cavity)
    # 2 (cavity) - 0 (cavity)
    # 2 (cavity) - 1 (cavity)
    # 4 (no cavity) - 0 (no cavity)
    # 4 (cavity) - 0 (cavity)
    # 7 (no cavity) - 0 (no cavity)
    # 13 (no cavity) - 0 (no cavity)
    # 15 (no cavity) - 0 (no cavity)
    # 43 (no cavity) - 0 (no cavity)
    
    option_names = ["00_cavity_no-cavity", "11_cavity_no-cavity", "21_cavity_no-cavity", \
                    "10_cavity", "10_no-cavity", "20_cavity", "21_cavity", \
                    "40_no-cavity", "40_cavity","70_no-cavity","130_no-cavity",\
                    "150_no-cavity","430_no-cavity"] # For output files
    N_OPTIONS = 1 # Number of above conditions

    DIFF_DENSITY = np.zeros(( len(diag_density), N_OPTIONS, Nxyz[0],Nxyz[1],Nxyz[2] )) 

    # 0 (cavity) - 0 (no cavity)
    for A0IND, A0 in enumerate( A0_LIST ):
        DIFF_DENSITY[A0IND,0,:,:,:] = diag_density[A0IND,0,:,:,:] - TDM_matter[0,0,:,:,:]

    """
    # 1 (cavity) - 1 (no cavity)
    for A0IND, A0 in enumerate( A0_LIST ):
        DIFF_DENSITY[A0IND,1,:,:,:] = diag_density[A0IND,1,:,:,:] - TDM_matter[1,1,:,:,:]

    # 2 (cavity) - 1 (no cavity)
    for A0IND, A0 in enumerate( A0_LIST ):
        DIFF_DENSITY[A0IND,2,:,:,:] = diag_density[A0IND,2,:,:,:] - TDM_matter[1,1,:,:,:]

    # 1 (cavity) - 0 (cavity)
    for A0IND, A0 in enumerate( A0_LIST ):
        DIFF_DENSITY[A0IND,3,:,:,:] = diag_density[A0IND,1,:,:,:] - diag_density[A0IND,0,:,:,:]

    # 1 (no cavity) - 0 (no cavity)
    for A0IND, A0 in enumerate( A0_LIST ):
        DIFF_DENSITY[A0IND,4,:,:,:] = TDM_matter[1,1,:,:,:] - TDM_matter[0,0,:,:,:]

    # 2 (cavity) - 0 (cavity)
    for A0IND, A0 in enumerate( A0_LIST ):
        DIFF_DENSITY[A0IND,5,:,:,:] = diag_density[A0IND,2,:,:,:] - diag_density[A0IND,0,:,:,:]

    # 2 (cavity) - 1 (cavity)
    for A0IND, A0 in enumerate( A0_LIST ):
        DIFF_DENSITY[A0IND,6,:,:,:] = diag_density[A0IND,2,:,:,:] - diag_density[A0IND,1,:,:,:]

    # 4 (no cavity) - 0 (no cavity)
    for A0IND, A0 in enumerate( A0_LIST ):
        DIFF_DENSITY[A0IND,7,:,:,:] = TDM_matter[4,4,:,:,:] - TDM_matter[0,0,:,:,:]

    # 4 (cavity) - 0 (cavity)
    for A0IND, A0 in enumerate( A0_LIST ):
        DIFF_DENSITY[A0IND,8,:,:,:] = diag_density[A0IND,4,:,:,:] - diag_density[A0IND,0,:,:,:]

    # 7 (no cavity) - 0 (no cavity)
    for A0IND, A0 in enumerate( A0_LIST ):
        DIFF_DENSITY[A0IND,9,:,:,:] = TDM_matter[7,7,:,:,:] - TDM_matter[0,0,:,:,:]

    # 13 (no cavity) - 0 (no cavity)
    for A0IND, A0 in enumerate( A0_LIST ):
        DIFF_DENSITY[A0IND,10,:,:,:] = TDM_matter[13,13,:,:,:] - TDM_matter[0,0,:,:,:]

    # 15 (no cavity) - 0 (no cavity)
    for A0IND, A0 in enumerate( A0_LIST ):
        DIFF_DENSITY[A0IND,11,:,:,:] = TDM_matter[15,15,:,:,:] - TDM_matter[0,0,:,:,:]

    # 43 (no cavity) - 0 (no cavity)
    for A0IND, A0 in enumerate( A0_LIST ):
        DIFF_DENSITY[A0IND,12,:,:,:] = TDM_matter[43,43,:,:,:] - TDM_matter[0,0,:,:,:]
    """


    # Check the sum of the density difference --> Should this be zero ? Not sure actually.
    NORM = np.zeros(( N_OPTIONS, len(A0_LIST) ))
    for p in range(N_OPTIONS):
        for A0IND, A0 in enumerate( A0_LIST ):
            NORM[p,A0IND] = np.sum( DIFF_DENSITY[A0IND,p,:,:,:] ) * dLxyz[0] * dLxyz[1] * dLxyz[2]
    np.savetxt("data_difference_density/diff_dens_norm.dat", NORM*1000, header="(N_OPTIONS x NA0) m|e|", fmt="%1.6f")



    plot_ind = 0 * ("x" == plotDIM) + 1 * ("y" == plotDIM) + 2 * ("z" == plotDIM)
    R = np.arange(Nxyz[plot_ind]) * dLxyz[plot_ind] * 0.529 # Bohr --> Angstrom
    R_fine = np.linspace( R[0], R[-1], 500 )

    if ( write_TD_Files == True ):
        # Print TD Files in Gaussian cube format
        for p in range(N_OPTIONS):
            for A0IND in range(len(A0_LIST)):
                A0 = round(A0_LIST[A0IND],8)
                print (f"Writing difference density file. option = {option_names[p]}, A0 = {A0}")
                f = open(f'data_difference_density/difference_density_{option_names[p]}_{EVEC_OUT}_A0{A0}_WC{WC}_NM{NM}_NF{NF}.cube','w')
                f.write(f"Difference Density: {option_names[p]} \n")
                f.write(f"Totally {NGrid} grid points\n")
                f.write(f"{NAtoms} {-Lxyz[0]/0.529} {-Lxyz[1]/0.529} {-Lxyz[2]/0.529}\n")
                f.write(f"{Nxyz[0]} {dLxyz[0]}  0.000000   0.000000\n")
                f.write(f"{Nxyz[1]} 0.000000   {dLxyz[1]} 0.000000\n")
                f.write(f"{Nxyz[2]} 0.000000   0.000000   {dLxyz[2]} \n")
                for at in range(len(coords)):
                    f.write( coords[at] )
                for x in range(Nxyz[0]):
                    #print(f'X = {x}')
                    for y in range(Nxyz[1]):
                        outArray = []
                        for z in range(Nxyz[2]):
                            outArray.append( DIFF_DENSITY[A0IND,p,x,y,z] )
                            if ( len(outArray) % 6 == 0 or z == Nxyz[2]-1 ):
                                #outArray.append('\n')
                                f.write( " ".join(map( str, np.round(outArray,8) )) + "\n" )
                                outArray = []
                f.close()
                dA   = dLxyz[1]*dLxyz[2]*("x" == plotDIM) + dLxyz[0]*dLxyz[2]*("y" == plotDIM) + dLxyz[0]*dLxyz[1]*("z" == plotDIM)
                axis = (1,2)*("x" == plotDIM) + (0,2)*("y" == plotDIM) + (0,1)*("z" == plotDIM)
                func = np.sum( DIFF_DENSITY[A0IND,p,:,:,:] * dA, axis=axis )
                func_interp = interp1d(R,func,kind='cubic')
                #plt.plot( R, f ,label="A$_0$ = "+f"{round(A0_LIST[A0IND],3)} a.u.")
                print("Length of RGrid:", len(R), len(R_fine))
                print("Length of function:", len(func), len(func_interp(R_fine)))
                plt.plot( R_fine, func_interp(R_fine) ,label="A$_0$ = "+f"{round(A0_LIST[A0IND],5)} a.u.")
            plt.legend()
            plt.xlim(0,R[-1])
            plt.xlabel(f"Real-space Position Along '{plotDIM}' ($\AA$)",fontsize=15)
            plt.ylabel(f"Difference Density",fontsize=15)
            plt.title(f"Density Type: {option_names[p]}",fontsize=15)
            plt.tight_layout()
            plt.savefig(f'data_difference_density/difference_density_{option_names[p]}_{EVEC_OUT}_WC{WC}_NM{NM}_NF{NF}.jpg',dpi=600)
            plt.clf()


            # This will be useful when working with differently sized molecules
            RX_MIN = 1
            RX_MAX = 11            
            RY_MIN = 1
            RY_MAX = 11     
            INDS_X = ( ind for ind,r in enumerate(R_fine) if (r >= RX_MIN and r <= RX_MAX) )
            INDS_Y = ( ind for ind,r in enumerate(R_fine) if (r >= RY_MIN and r <= RY_MAX) )
            RX_fine_CUT = np.array([ R_fine[ind] for ind in INDS_X])
            RY_fine_CUT = np.array([ R_fine[ind] for ind in INDS_Y])
            xyz_dict = {0:"x",1:"y",2:"z"}
            for A0IND,A0 in enumerate(A0_LIST):

                print(f"Writing the difference density ({option_names[p]}) contour data to a file (A0 = {round(A0,6)}).")
                A0 = round(A0,5)
                axis = 0*((1,2) == plotDIM2D) + 1*((0,2) == plotDIM2D) + 2*((0,1) == plotDIM2D)
                dL   = dLxyz[0]*((1,2) == plotDIM2D) + dLxyz[1]*((0,2) == plotDIM2D) + dLxyz[2]*((0,1) == plotDIM2D)
                func = np.sum( DIFF_DENSITY[A0IND,p,:,:,:]*1000 * dL, axis=axis )
                VMIN = np.min( func )
                VMAX = np.max( func )
                VMAX = np.max([ -VMIN, VMAX  ])
                VMIN = -VMAX
                if ( VMAX > 30 ):
                    print ( "\n\tWARNING!!!! SOMETHING MAY BE WRONG WITH DIFFERENCE DENSITY!!!" )
                    print ( "\tWARNING!!!! VERY LARGE DIFFERENCE DENSITY!!!" )
                    print ( "\tWARNING!!!! USE WITH CAUTION!!!" )
                    print ( f"\t (MIN,MAX) = ({np.min( func )},{np.max( func )}) m|e|/A^2" )
                
                func_interp = interp2d(R,R,func,kind='cubic')
                np.savetxt( f'data_difference_density/difference_density_contour_{EVEC_OUT}_{option_names[p]}_A0{A0}_WC{WC}_NM{NM}_NF{NF}.dat', func_interp(RX_fine_CUT,RY_fine_CUT).T )
                np.savetxt( f'data_difference_density/difference_density_contour_RXGRID_NM{NM}_NF{NF}.dat', RX_fine_CUT )
                np.savetxt( f'data_difference_density/difference_density_contour_RYGRID_NM{NM}_NF{NF}.dat', RY_fine_CUT )

                X,Y = np.meshgrid( RX_fine_CUT,RY_fine_CUT )
                plt.contourf( X, Y, func_interp(RX_fine_CUT,RY_fine_CUT), cmap="seismic", levels=500, vmin=VMIN, vmax=VMAX)
                plt.colorbar(pad=0.01)
                plt.xlim(RX_MIN,RX_MAX)
                plt.ylim(RY_MIN,RY_MAX)
                plt.xlabel(f"Real-space Position Along '{xyz_dict[plotDIM2D[0]]}' ($\AA$)",fontsize=15)
                plt.ylabel(f"Real-space Position Along '{xyz_dict[plotDIM2D[1]]}' ($\AA$)",fontsize=15)
                plt.title(f"Difference Density ({option_names[p]}) m|e|/A^2",fontsize=15)
                plt.tight_layout()
                plt.savefig(f'data_difference_density/difference_density_contour_{EVEC_OUT}_{option_names[p]}_A0{A0}_WC{WC}_NM{NM}_NF{NF}.jpg',dpi=600)
                plt.clf()


def main():
    getGlobals()
    Epol, Upol, EAD, MU = get_HadMU()

    #### Density Analysis ####
    TDM_matter = get_TD_Data()
    #TD_pol_1r    = compute_Transition_density_1r( Upol, TDM_matter )
    #TD_pol_1r1q  = compute_Transition_density_1r1q( Upol, TDM_matter ) 
    diag_density = compute_diagonal_density_1r( Upol, TDM_matter )
    diff_density = compute_difference_density_1r( TDM_matter, diag_density )

    #### Density Matrix Analysis ####
    #TDM_matter = get_TDM_Data()
    #TDM_pol_1r = compute_Transition_density_matrix_1r( Upol, TDM_matter ) 

    #### NTO ANALYSIS ####
    #NTO_matter = get_NTO_Data()
    #NTO_pol_1r = compute_NTO_1r( Upol, NTO_matter ) 

    getExansion(Upol)

    #eff_dipole_1 = getDipole_pow_1(Upol,MU)
    #eff_osc_str = getOscStr(Epol,eff_dipole_1)
    #plotSpectra(Epol,eff_osc_str,Char)



    
    
    
    #make_movie()


if ( __name__ == '__main__'):
    main()
