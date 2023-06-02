#!/bin/bash
#SBATCH --partition=standard --time=5-00:00:00 --cpus-per-task=10 --mem=40g --output=out.log
source activate /scratch/nravi3/rotation-curves
python3 disk_mass_main_multiprocess.py