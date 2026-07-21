# GrALfisika — RCWA simulation of rough Inconel 625 surfaces

This repository contains the Python implementation and notebooks developed for the Bachelor's Thesis:

**Simulation of the Optical Properties of Rough Metallic Surfaces Using RCWA**

The main objective of the project is to study the spectral emissivity of rough metallic surfaces of Inconel 625 using the Rigorous Coupled-Wave Analysis (RCWA) method. The numerical results are compared with experimental emissivity data in order to evaluate the applicability and limitations of RCWA for non-ideal, experimentally measured surfaces.

## Project overview

The code implements a two-dimensional RCWA formulation for periodic surface-relief structures. Experimentally measured rough profiles are preprocessed, discretized using a staircase approximation, and then introduced into the RCWA solver.

The study focuses on four surface profiles of Inconel 625:

- `nocoating_x1`
- `nocoating_x2`
- `nocoating_y1`
- `nocoating_y2`

The optical response is calculated for normal incidence and TM polarization over the infrared wavelength range. The complex refractive index of Inconel 625 is estimated using the Hagen-Rubens approximation from temperature-dependent electrical resistivity values.

## Main features

- 2D RCWA implementation in Python.
- TE and TM polarization solvers.
- Staircase discretization of rough surface profiles.
- Calculation of reflection, transmission, absorption and emissivity.
- Convergence analysis with respect to the number of Fourier harmonics and vertical slices.
- Comparison with experimental emissivity data.
- Analysis of roughness effects by comparison with a flat Inconel 625 surface.

## Repository structure

```text
GrALfisika/
│
├── src/                    # RCWA modules and auxiliary functions
│   ├── TE_matrixes.py
│   ├── TM_matrixes.py
│   ├── TE_solver.py
│   ├── TM_solver.py
│   └── Utils.py
│
├── notebooks/              # Jupyter notebooks used for simulations and analysis
│
├── csv/                    # Input profiles and numerical/experimental results
│   ├── Inconel_gratings/
│   └── results/
│
├── figures/                # Generated figures
│
├── LICENSE                 # GNU GPLv3 license
└── README.md

## License

This project is licensed under the GNU General Public License v3.0 or later.
See the LICENSE file for details.

Part of the implementation was developed taking as reference the RCWA MATLAB code
from Prof. Zhuomin Zhang's group at Georgia Tech, originally developed in 2007
and approved for free release and distribution in 2014.
