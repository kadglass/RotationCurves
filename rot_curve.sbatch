#!/bin/bash
#SBATCH -p standard
#SBATCH --mem=30gb
#SBATCH --time=24:00:00
#SBATCH -J rot_curve_analysis
#SBATCH -o /home/jsm171/rot_curve_log.log
#SBATCH -e /home/jsm171/rot_curve_error.err

module load python3/3.5.2

srun python3 /home/jsm171/RotationCurves/rot_curve_main.py
