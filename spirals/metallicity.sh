#!/bin/bash
#SBATCH --partition=standard --time=2-00:00:00  --mem=10g --output=out.log
source /software/anaconda3/2022.05-user/etc/profile.d/conda.sh
conda activate /scratch/nravi3/rotation-curves-v2
python3 metallicity_map_main.py
python3 stellar_metallicity_map_main.py