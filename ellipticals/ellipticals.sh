#!/bin/bash
#SBATCH --partition=starndard --time=5-00:00:00 --cpus-per-task=12 --output=out.log
source activate /scratch/nravi3/rotation-curves
python3 elliptical_virial_mass_main_multiprocess.py