The code used for the evaluations can be seen in the source code folder.
To run the code a folder structure looking like this is needed

Evaluations is done in Ubuntu 20.04 using python 3.8

- env (optional)
- GeodesicMotionSkills (See below how to install)
- riepybdlib
- s-vae-pytorch (Should be installed when installing GeodesicMotionSkills)
- stochman (Should be installed when installing GeodesicMotionSkills)
- source
--- Clusters
--- Demos_ur5
--- experiments
--- experiments_gmr
--- models
--- admittance.py
--- gmr.py
--- vae.py

All the folders/files in source should be in the zip file.

To install GeodesicMotionSkills follow guide on: https://github.com/boschresearch/GeodesicMotionSkills

To run the VAE code on a trained VAE copy the .pt file from one of the experiments folders in to the models folder.
GMR code should run without editing, but make sure to choose whether to run on UR demonstrations in the code.




