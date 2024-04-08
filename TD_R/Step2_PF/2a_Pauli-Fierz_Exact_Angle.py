import numpy as np
from numpy import kron as kron_prod
import sys
import subprocess as sp

##### Modify from Braden M. Weight #####

# SYNTAX: python3 Pauli-Fierz_Exact_Angle.py A0 WC

def get_globals():
    global NM, NF, A0, wc_eV, wc_AU
    global EVEC_INTS, EVEC_NORM, EVEC_OUT
    global RPA, THETA, PHI

    ##### MAIN USER INPUT SECTION #####
    NM        = 50                  # Number of Electronic States (including ground state)
    NF        = 10                  # Number of Fock Basis States
    RPA       = True                # If True, look for TD-DFT/RPA data rather than TD-DFT/TDA data
    ##### END USER INPUT SECTION  #####
    
    A0    = float(sys.argv[1]) # a.u.
    wc_eV = float(sys.argv[2]) # eV
    
    THETA = 160.4 #80.2 #68.8 #80.2     #MAX or MIN Angles
    PHI   = 137.5 #68.8 #183.3 #91.7

    THETA_RAD = THETA / 180 * np.pi
    PHI_RAD   = PHI / 180 * np.pi

    wc_AU     = wc_eV / 27.2114
    Ex        = np.sin(THETA_RAD) * np.cos(PHI_RAD)
    Ey        = np.sin(THETA_RAD) * np.sin(PHI_RAD)
    Ez        = np.cos(THETA_RAD)
    EVEC = np.array([ Ex,Ey,Ez ])
    EVEC_NORM = EVEC / np.linalg.norm(EVEC)

    sp.call("mkdir -p data_PF", shell=True)

def get_a_op(nf):
    a = np.zeros((nf,nf))
    for m in range(1,nf):
        a[m,m-1] = np.sqrt(m)
    return a.T

def get_H_PF(EAD, MU):
    print (f"Dimension = {(NM*NF)}")

    I_ph = np.identity(NF)
    I_el = np.identity(NM)
    a_op = get_a_op(NF)
    q_op = a_op.T + a_op
    MU   = np.einsum("d,JKd->JK", EVEC_NORM[:], MU[:,:,:] )

    H_EL = np.diag( EAD )
    H_PH = np.diag( np.arange(NF) * wc_AU )

    H_PF   = kron_prod(H_EL, I_ph)                        # Matter
    H_PF  += kron_prod(I_el, H_PH)                        # Photon

    H_PF  += kron_prod(MU, q_op) * wc_AU * A0             # PH-EL Interaction
    H_PF  += kron_prod(MU @ MU, I_ph) * wc_AU * A0**2     # Dipole Self-Energy

    if ( H_PF.size * H_PF.itemsize * 10 ** -9 > 0.5 ): # If H_PF larger than 0.5 GB, tell the user.
        print(f"\tWARNING!!! Large Matrix" )
        print(f"\tMemory size of numpy array in (MB, GB): ({round(H_PF.size * H_PF.itemsize * 10 ** -6,2)},{round(H_PF.size * H_PF.itemsize * 10 ** -9,2)})" )

    return H_PF

def get_ADIABATIC_DATA():

    EAD  = np.zeros(( NM ))
    MU  = np.zeros(( NM,NM,3 ))

    if ( RPA == True ):
        EAD += np.loadtxt(f"../PLOTS_DATA/ADIABATIC_ENERGIES_RPA.dat")[:NM] # in AU
        MU  += np.load(f"../PLOTS_DATA/DIPOLE_RPA.dat.npy")[:NM,:NM] # in AU
    else:
        EAD += np.loadtxt(f"../PLOTS_DATA/ADIABATIC_ENERGIES_TDA.dat")[:NM] # in AU
        MU  += np.load(f"../PLOTS_DATA/DIPOLE_TDA.dat.npy")[:NM,:NM] # in AU

    return EAD, MU

def SolvePlotandSave(H_PF,EAD,MU):

        # Diagonalize Hamiltonian
        E, U = np.linalg.eigh( H_PF )
        
        # Save Data
        np.savetxt( f"data_PF/E_THETA_{round(THETA,2)}_PHI_{round(PHI,2)}_A0_{round(A0,6)}_WC_{round(wc_eV,6)}_NF_{NF}_NM_{NM}.dat", E * 27.2114 )
        np.save( f"data_PF/U_THETA_{round(THETA,2)}_PHI_{round(PHI,2)}_A0_{round(A0,6)}_WC_{round(wc_eV,6)}_NF_{NF}_NM_{NM}.dat", U )

        print ( A0, wc_eV )

        np.savetxt( f"data_PF/EAD.dat", EAD * 27.2114 )
        np.save( f"data_PF/MU.dat", MU )

def main():

    get_globals()

    EAD, MU = get_ADIABATIC_DATA() 
    H_PF    = get_H_PF( EAD, MU )
    SolvePlotandSave( H_PF, EAD, MU)

if __name__ == "__main__":
    main()
