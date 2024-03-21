**Title: Investigating Cavity Quantum Electrodynamics-Enabled Endo/Exo- Selectivities in a Diels-Alder Reaction**

**Wang J, Weight B, Huo P. _ChemRxiv_ (2024). doi:10.26434/chemrxiv-2024-6xsr6** This content is a preprint and has not been peer-reviewed.

**Abstract:** Coupling molecules to a quantized radiation field inside an optical cavity has shown great promise in modifying chemical reactivity. It was recently proposed that strong light-matter interactions are able to differentiate endo/exo products of a Diels-Alder reaction at the transition state. Using the recently developed parameterized quantum electrodynamic _ab initio_ polariton chemistry approach along with time-dependent density functional theory, we theoretically confirm that the ground state selectivity of a Diels-Alder reaction can be fundamentally changed by strongly coupling to the cavity, generating preferential endo or exo isomers which are formed with equal probability for the same reaction outside the cavity. This provides an important and necessary benchmark with the high-level self-consistent QED coupled cluster approach. In addition, by computing the ground state difference density, we show that the cavity induces a redistribution of electron density from intramolecular pi-bonding orbitals to intermolecular bonding orbitals, thus providing chemically relavent description of the cavity-induced changes to the ground state chemistry and thus changes to the molecular orbital theory inside the cavity. We extend this exploration to an arbitrary cavity polarization vector which leads to critical polarization angles that maximize the endo=/exo selectivity of the reaction. Finally, we decompose the energy contributions from the Hamiltonian and provide discussion relating to the dominent dipole self-energy effects on the ground state.

**Implementation Notes:**
1. The initial optimized geometries for reacants (R), transition states (TS), and products (P) are obtained from Ref 1. 
2. PF codes have been modified based on the work of author Braden. See Reference 2.
3. Example files can be found in the TD_R directory; the name and corresponding coordinates can be changed to TS or P as needed.
4. Within the TD_R directory, modify the polarization vectors in the input section of the Python codes to achieve the desired directions (X-, Y-, Z-polarizations or specific theta/phi angles).

**Codes Contact Information:**
jialongwang@rochester.edu; bweight@ur.rochester.edu; pengfei.huo@rochester.edu

**References:**
1. Pavošević, F., Smith, R.L. & Rubio, A. Computational study on the catalytic control of endo/exo Diels-Alder reactions by cavity quantum vacuum fluctuations. Nat Commun 14, 2766 (2023). https://doi.org/10.1038/s41467-023-38474-w.
   See also the corresponding GitHub repository: https://github.com/fabijan5/qed-diels-alder/blob/main/Supplementary_Data.txt
2. Weight, B. M. Ab_Initio_Polariton_Properties, GitHub repository (2024). https://github.com/bradenmweight/Ab_Initio_Polariton_Properties/tree/main

