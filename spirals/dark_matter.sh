#!/bin/bash
#SBATCH --partition=standard --time=5-00:00:00 --cpus-per-task=12 --mem=62g --output=out.log
source activate /scratch/nravi3/rotation-curves
python3 DRP_vel_map_main_multiprocess.py