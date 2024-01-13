#!/bin/sh

#SBATCH --account=rrg-rgreiner
#SBATCH --time=00:05:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=1-80
#SBATCH --output=outputs/run_name.txt
#SBATCH --gres=gpu:1

source ~/tinystories/bin/activate
module load python/3.10
module load cuda

# rm -rf export.dat
# for beta in .1 1; do
    # echo export beta=$beta>> export.dat
# done



# `sed -n "${SLURM_ARRAY_TASK_ID}p" <export.dat`
# echo ${SLURM_ARRAY_TASK_ID}

echo "Current working directory is `pwd`"
echo "Running on hostname `hostname`"

echo "Starting run at: `date`"
python train.py --batch_size=100 --max_iters=10  --eval_iters=10 --gradient_accumulation_steps=1 --beta=$beta
echo "Program test finished with exit code $? at: `date`"
