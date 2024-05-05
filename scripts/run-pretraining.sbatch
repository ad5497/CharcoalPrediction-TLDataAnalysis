## HPC; submit this with "sbatch run-pretraining.sbatch" either
## from the burst node or while in the interactive srun session
## script assumes pretraining.py is in /scratch/ad5497

#!/bin/bash
#SBATCH --account=ds_ua_301_004-2024sp
#SBATCH --partition=n1s8-v100-2
#SBATCH --gres=gpu:v100:2
#SBATCH --mail-type=END
#SBATCH --mail-user=ad5497@nyu.edu

cd /scratch/ad5497/pytorch-example

singularity exec --nv --bind /scratch/ad5497 \
        --overlay my_pytorch.ext3:rw /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif /bin/bash -c "

source /ext3/env.sh; cd /scratch/ad5497; conda activate; python pretraining.py"