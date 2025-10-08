#!/bin/bash
#SBATCH -J RND
#SBATCH --nodes 1
#SBATCH --ntasks 31
#SBATCH --ntasks-per-node=31
#SBATCH --ntasks-per-core=1
#SBATCH --time=96:00:00
#SBATCH --mail-user=yuri.lavinas@ut-capitole.fr
#SBATCH --mail-type=ALL

export OMP_NUM_THREADS=1
export PATH="/users/p16043/lavinas/miniconda3/bin:$PATH"

# LOAD MODULES
module purge
module avail python
module load python/3.8.18
module load intelmpi chdb/1.0

# RUN CHDB
srun chdb --in-type "1 30" --command-line "../scripts/AL_onepluslambda_rnd.sh %name% >%out-dir%/loggs.out 2>&1" --out-dir output-${SLURM_JOB_ID} --report report.txt
