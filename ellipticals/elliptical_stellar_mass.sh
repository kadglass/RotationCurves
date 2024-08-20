#!/bin/bash
#SBATCH --partition=standard --time=1-00:00:00 --output=out.log
source activate /scratch/nravi3/rotation-curves
python3 elliptical_stellar_mass_main.py