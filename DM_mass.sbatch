#!/bin/bash
#SBATCH -p standard
#SBATCH --mem=50gb
#SBATCH --time=18:00:00
#SBATCH -J DM_mass
#SBATCH -o /scratch/jsm171/DM_mass_log.log 
#SBATCH -e /scratch/jsm171/DM_mass_error.err

module load python3/3.5.2
srun python3 /home/jsm171/RotationCurves/dark_matter_mass_main.py
