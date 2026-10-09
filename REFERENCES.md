# References

Software, methods and data used by **PBA Autonomous Workflow**. Please cite the relevant entries
when you use the corresponding part of the package. BibTeX for every entry with an identifier is in
[`docs/references.bib`](docs/references.bib); to cite this repository itself, see [`CITATION.cff`](CITATION.cff).

Versions are those used to produce the results in `runs/` and `docs/`. Tools used only to make
the project logo are credited in [`docs/logo/README.md`](docs/logo/README.md), not here.

## Machine-learned interatomic potentials

*MACE requires e3nn 0.4.4 and was run in a separate environment from UMA and PET-MAD.*

- **MACE architecture · mace-torch 0.3.16 · [MACE](https://github.com/ACEsuit/mace)** — Batatia, Kovacs, Simm *et al.* (2022). MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields. *Advances in Neural Information Processing Systems 35*, 11423-11436. https://doi.org/10.52202/068431-0830
- **MACE-MP-0 foundation model (`mace-mp-small`, `mace-mp-medium`)** — Batatia, Benner, Chiang *et al.* (2023). A foundation model for atomistic materials chemistry. arXiv:2401.00096
- **MPtrj training data of MACE-MP-0 (introduced with CHGNet)** — Deng, Zhong, Jun *et al.* (2023). CHGNet as a pretrained universal neural network potential for charge-informed atomistic modelling. *Nature Machine Intelligence* **5**, 1031-1041. https://doi.org/10.1038/s42256-023-00716-3
- **Materials Project** — Jain, Ong, Hautier *et al.* (2013). Commentary: The Materials Project: A materials genome approach to accelerating materials innovation. *APL Materials* **1**, 011002. https://doi.org/10.1063/1.4812323
- **PET-MAD · pet-mad 1.4.4 · [PET-MAD](https://github.com/lab-cosmo/pet-mad)** — Mazitov, Bigi, Kellner *et al.* (2025). PET-MAD as a lightweight universal interatomic potential for advanced materials modeling. *Nature Communications* **16**, 10653. https://doi.org/10.1038/s41467-025-65662-7
- **PET architecture** — Pozdnyakov, Ceriotti (2023). Smooth, exact rotational symmetrization for deep learning on point clouds. *Advances in Neural Information Processing Systems 36*, 79469-79501. https://doi.org/10.52202/075280-3478
- **MAD training dataset** — Mazitov, Chorna, Fraux *et al.* (2025). Massive Atomic Diversity: a compact universal dataset for atomistic machine learning. *Scientific Data* **12**, 1857. https://doi.org/10.1038/s41597-025-06109-y
- **UMA (`uma-s-1p1`, `omat` and `odac` task heads) · fairchem-core 2.22.0 · [fairchem (UMA)](https://github.com/facebookresearch/fairchem) · weights: [UMA weights](https://huggingface.co/facebook/UMA)** — Wood, Dzamba, Fu *et al.* (2025). UMA: A Family of Universal Models for Atoms. *Advances in Neural Information Processing Systems 38*, 143528-143564. https://doi.org/10.52202/085713-4310
- **OMat24 dataset (UMA `omat` head)** — Barros-Luque, Shuaibi, Fu *et al.* (2026). The Open Materials 2024 (OMat24) inorganic materials dataset and models. *Nature Computational Science* **6**, 642-652. https://doi.org/10.1038/s43588-026-00996-w
- **ODAC23 dataset (UMA `odac` head)** — Sriram, Choi, Yu *et al.* (2024). The Open DAC 2023 Dataset and Challenges for Sorbent Discovery in Direct Air Capture. *ACS Central Science* **10**, 923-941. https://doi.org/10.1021/acscentsci.3c01629
- **e3nn (used by MACE) · [e3nn](https://github.com/e3nn/e3nn)** — Geiger, Smidt (2022). e3nn: Euclidean Neural Networks. arXiv:2207.09453
- **PyTorch 2.10.0 · [PyTorch](https://github.com/pytorch/pytorch)** — Paszke, Gross, Massa *et al.* (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. arXiv:1912.01703 (NeurIPS 2019)

## Density functional theory

- **GPAW 26.7.0 (serial source build) · [GPAW](https://gitlab.com/gpaw/gpaw)** — Mortensen, Larsen, Kuisma *et al.* (2024). GPAW: An open Python package for electronic structure calculations. *The Journal of Chemical Physics* **160**, 092503. https://doi.org/10.1063/5.0182685
  - Enkovaara, Rostgaard, Mortensen *et al.* (2010). Electronic structure calculations with GPAW: a real-space implementation of the projector augmented-wave method. *Journal of Physics: Condensed Matter* **22**, 253202. https://doi.org/10.1088/0953-8984/22/25/253202
- **Projector augmented-wave method** — Blöchl (1994). Projector augmented-wave method. *Physical Review B* **50**, 17953-17979. https://doi.org/10.1103/PhysRevB.50.17953
- **PBE exchange–correlation functional** — Perdew, Burke, Ernzerhof (1996). Generalized Gradient Approximation Made Simple. *Physical Review Letters* **77**, 3865-3868. https://doi.org/10.1103/PhysRevLett.77.3865
- **libxc 7.1.2 · [libxc](https://gitlab.com/libxc/libxc)** — Lehtola, Steigemann, Oliveira *et al.* (2018). Recent developments in libxc — A comprehensive library of functionals for density functional theory. *SoftwareX* **7**, 1-5. https://doi.org/10.1016/j.softx.2017.11.002
- **FFTW 3.3.11 · [FFTW](https://github.com/FFTW/fftw3)** — Frigo, Johnson (2005). The Design and Implementation of FFTW3. *Proceedings of the IEEE* **93**, 216-231. https://doi.org/10.1109/JPROC.2004.840301
- **OpenBLAS 0.3.34 · [OpenBLAS](https://github.com/OpenMathLib/OpenBLAS)** — OpenBLAS: an optimized BLAS library (software). https://www.openmathlib.org/OpenBLAS/
- **GPAW PAW datasets 24.11.0** — Installed with `gpaw install-data --gpaw`; dataset list: https://gitlab.com/gpaw/gpaw/-/raw/master/doc/setups/setups.rst

## Atomistic simulation tools

- **ASE 3.29.0 (structures, LBFGS relaxation, `FrechetCellFilter`) · [ASE](https://gitlab.com/ase/ase)** — Hjorth Larsen, Jørgen Mortensen, Blomqvist *et al.* (2017). The atomic simulation environment—a Python library for working with atoms. *Journal of Physics: Condensed Matter* **29**, 273002. https://doi.org/10.1088/1361-648X/aa680e
- **L-BFGS optimiser** — Liu, Nocedal (1989). On the limited memory BFGS method for large scale optimization. *Mathematical Programming* **45**, 503-528. https://doi.org/10.1007/BF01589116
- **icet 3.2 (special quasirandom structures) · [icet](https://gitlab.com/materials-modeling/icet)** — Ångqvist, Muñoz, Rahm *et al.* (2019). ICET – A Python Library for Constructing and Sampling Alloy Cluster Expansions. *Advanced Theory and Simulations* **2**, 1900015. https://doi.org/10.1002/adts.201900015
- **Special quasirandom structures** — Zunger, Wei, Ferreira *et al.* (1990). Special quasirandom structures. *Physical Review Letters* **65**, 353-356. https://doi.org/10.1103/PhysRevLett.65.353

## Experiment planning and optimisation (implemented in `pba_autoworkflow.optimize`)

*These methods are re-implemented on NumPy/SciPy; no Bayesian-optimisation library is a dependency.*

- **Gaussian-process regression, Matérn-5/2 kernel with ARD** — Rasmussen, C. E. & Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning*. MIT Press. ISBN 978-0-262-18253-9.
- **q-Noisy Expected Hypervolume Improvement (qNEHVI)** — Daulton, Balandat, Bakshy (2021). Parallel Bayesian Optimization of Multiple Noisy Objectives with Expected Hypervolume Improvement. arXiv:2105.08195 (NeurIPS 34, 2021)
- **Kriging-believer batch selection** — Ginsbourger, Le Riche, Carraro (2010). Kriging Is Well-Suited to Parallelize Optimization. *Adaptation Learning and Optimization*, 131-162. https://doi.org/10.1007/978-3-642-10701-6_6
- **Dominated hypervolume indicator** — Zitzler, Thiele (1999). Multiobjective evolutionary algorithms: a comparative case study and the strength Pareto approach. *IEEE Transactions on Evolutionary Computation* **3**, 257-271. https://doi.org/10.1109/4235.797969
- **Sobol' sequences (seed design)** — Sobol' (1967). On the distribution of points in a cube and the approximate evaluation of integrals. *USSR Computational Mathematics and Mathematical Physics* **7**, 86-112. https://doi.org/10.1016/0041-5553(67)90144-9
- **Scrambling of Sobol' points** — Owen (1995). Randomly Permuted (t,m,s)-Nets and (t, s)-Sequences. *Lecture Notes in Statistics*, 299-317. https://doi.org/10.1007/978-1-4612-2552-2_19
- **L-BFGS-B (GP hyperparameter fit, via SciPy)** — Byrd, Lu, Nocedal *et al.* (1995). A Limited Memory Algorithm for Bound Constrained Optimization. *SIAM Journal on Scientific Computing* **16**, 1190-1208. https://doi.org/10.1137/0916069

## Analysis of instrument data (`pba_autoworkflow.analysis`)

- **Savitzky–Golay smoothing (XRD peak search)** — Savitzky, Golay (1964). Smoothing and Differentiation of Data by Simplified Least Squares Procedures.. *Analytical Chemistry* **36**, 1627-1639. https://doi.org/10.1021/ac60214a047
- **SNIP background subtraction (XRD)** — Ryan, Clayton, Griffin *et al.* (1988). SNIP, a statistics-sensitive background treatment for the quantitative analysis of PIXE spectra in geoscience applications. *Nuclear Instruments and Methods in Physics Research Section B: Beam Interactions with Materials and Atoms* **34**, 396-402. https://doi.org/10.1016/0168-583X(88)90063-8
- **Pseudo-Voigt peak profile** — Thompson, Cox, Hastings (1987). Rietveld refinement of Debye–Scherrer synchrotron X-ray data from Al <sub>2</sub> O <sub>3</sub>. *Journal of Applied Crystallography* **20**, 79-83. https://doi.org/10.1107/S0021889887087090
- **Scherrer equation** — Scherrer, P. (1918). Bestimmung der Größe und der inneren Struktur von Kolloidteilchen mittels Röntgenstrahlen. *Nachr. Ges. Wiss. Göttingen, Math.-Phys. Kl.* 1918, 98–100.
  - Patterson (1939). The Scherrer Formula for X-Ray Particle Size Determination. *Physical Review* **56**, 978-982. https://doi.org/10.1103/PhysRev.56.978
- **Williamson–Hall analysis (domain size and microstrain)** — Williamson, Hall (1953). X-ray line broadening from filed aluminium and wolfram. *Acta Metallurgica* **1**, 22-31. https://doi.org/10.1016/0001-6160(53)90006-6
- **Quantitative phase analysis from Rietveld scale factors (Hill–Howard relation)** — Hill, Howard (1987). Quantitative phase analysis from neutron powder diffraction data using the Rietveld method. *Journal of Applied Crystallography* **20**, 467-474. https://doi.org/10.1107/S0021889887086199
- **Saturation vapour pressure of water (drying-rate index)** — Buck (1981). New Equations for Computing Vapor Pressure and Enhancement Factor. *Journal of Applied Meteorology* **20**, 1527-1532. https://doi.org/10.1175/1520-0450(1981)020<1527:NEFCVP>2.0.CO;2 (coefficients of Buck's 1996 revision of this formula).
- **Drying-selected phase of zinc hexacyanoferrate (drying model, paper routes used for validation)** — Pilipavicius, Skarnulyte, Gece, Vilciauskas. Control of the Zinc Insertion Pathways in Zinc Hexacyanoferrate by Drying-Selected Phase and Structural Disorder. Manuscript.
- **Spearman rank correlation (force-field validation)** — Spearman (1904). The Proof and Measurement of Association between Two Things. *The American Journal of Psychology* **15**, 72. https://doi.org/10.2307/1412159

## Reference crystal structures (`pba_autoworkflow/data/phases`)

Taken from the Crystallography Open Database (COD; Gražulis *et al.* (2012). Crystallography Open Database (COD): an open-access collection of crystal structures and platform for world-wide collaboration. *Nucleic Acids Research* **40**, D420-D427. https://doi.org/10.1093/nar/gkr900). COD data are in the public domain.

- **Cubic Zn₃[Fe(CN)₆]₂·xH₂O (Fm-3m), COD 2020370** — Gravereau, Garnier (1984). Structure de la phase cubique de l'hexacyanoferrate(III) de zinc: Zn₃[Fe(CN)₆]₂·nH₂O. *Acta Crystallographica C* **40**, 1306-1309. https://doi.org/10.1107/S0108270184007757
- **Rhombohedral Na₂Zn₃[Fe(CN)₆]₂·xH₂O (R-3c), COD 2106959** — Garnier, Gravereau, Hardy (1982). Zeolitic iron cyanides: the structure of Na₂Zn₃[Fe(CN)₆]₂(H₂O)ₓ. *Acta Crystallographica B* **38**, 1401. (No DOI registered with Crossref.)
- **Monoclinic Na₂Mn[Fe(CN)₆]·2H₂O (P2₁/n), COD 7047887** — Oliver-Tolentino, Osiry *et al.* (2018). Electronic density distribution of Mn–N bonds by a tuning effect through partial replacement of Mn by Ni in sodium manganese hexacyanoferrate. *Dalton Transactions* **47**, 16492-16501. https://doi.org/10.1039/C8DT03595D
- **NaCl, COD 9003308** — Walker, Verma, Cranswick *et al.* (2004). Halite-sylvite thermoelasticity. *American Mineralogist* **89**, 204-210. https://doi.org/10.2138/am-2004-0124
- **Co(OH)₂ and Ni(OH)₂, COD 1548810 and 1548811** — Zhao, Kulik (2018). *Journal of Chemical Theory and Computation* **14**, 670. https://doi.org/10.1021/acs.jctc.7b01061
- **Mn(OH)₂ and Fe(OH)₂, COD 9009111 and 9009104** — Wyckoff, R. W. G. (1963). *Crystal Structures*, vol. 1, 2nd ed., p. 239. Interscience, New York.
- **CuO (tenorite), COD 7212242** — Volanti, Orlandi *et al.* (2010). *CrystEngComm* **12**, 1696. https://doi.org/10.1039/B922978G
- **ZnO (zincite), COD 2300450** — Schreyer, Guo *et al.* (2014). *Journal of Applied Crystallography* **47**, 659. https://doi.org/10.1107/S1600576714003379
- **pymatgen (structure factors when building the library; not needed at runtime)** — Ong, Richards, Jain *et al.* (2013). Python Materials Genomics (pymatgen): A robust, open-source python library for materials analysis. *Computational Materials Science* **68**, 314-319. https://doi.org/10.1016/j.commatsci.2012.10.028

## Scientific Python stack

- **NumPy · [NumPy](https://github.com/numpy/numpy)** — Harris, Millman, van der Walt *et al.* (2020). Array programming with NumPy. *Nature* **585**, 357-362. https://doi.org/10.1038/s41586-020-2649-2
- **SciPy · [SciPy](https://github.com/scipy/scipy)** — Virtanen, Gommers, Oliphant *et al.* (2020). SciPy 1.0: fundamental algorithms for scientific computing in Python. *Nature Methods* **17**, 261-272. https://doi.org/10.1038/s41592-019-0686-2
- **pandas · [pandas](https://github.com/pandas-dev/pandas)** — McKinney (2010). Data Structures for Statistical Computing in Python. *Proceedings of the Python in Science Conference*, 56-61. https://doi.org/10.25080/Majora-92bf1922-00a
- **Matplotlib · [Matplotlib](https://github.com/matplotlib/matplotlib)** — Hunter (2007). Matplotlib: A 2D Graphics Environment. *Computing in Science & Engineering* **9**, 90-95. https://doi.org/10.1109/MCSE.2007.55
- **pydantic · [pydantic](https://github.com/pydantic/pydantic)** — Colvin, S. *et al.* pydantic: data validation using Python type hints (software).
- **pytest, pytest-asyncio · [pytest](https://github.com/pytest-dev/pytest) · [pytest-asyncio](https://github.com/pytest-dev/pytest-asyncio)** — Krekel, H. *et al.* pytest (software).

## Related work (cited, not used in the code)

- **Thermodynamic interatomic potentials (TIP); code and weights not released at the time of writing** — Nam, Deng, Du *et al.* (2026). Universal Thermodynamic Interatomic Potentials for Crystalline Materials. arXiv:2608.14502
