## submit this with "sbatch run-pretraining-test.sbatch" either
## from the burst node or while in the interactive srun session
## script assumes pretraining-test.py is in /scratch/ad5497

#!/bin/bash
#SBATCH --account=ds_ua_301_004-2024sp
#SBATCH --partition=n1s16-v100-2
#SBATCH --gres=gpu:v100:2
#SBATCH --time=04:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=ad5497@nyu.edu

cd /scratch/ad5497/pytorch-example

singularity exec --nv --bind /scratch/ad5497 \
        --overlay my_pytorch.ext3:ro /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif /bin/bash -c "

source /ext3/env.sh; cd /scratch/ad5497; conda activate; python pretraining-test.py"