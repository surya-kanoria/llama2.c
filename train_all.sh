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
for label_name in set_1 set_2 set_3 set_4 set_5 set_6 set_7 set_8 set_9 set_10 set_11 set_12 set_13 set_14 set_15 set_16 set_17 set_18; do
    for beta in 1.1 1; do
        python train.py --batch_size=100 --max_iters=10  --eval_iters=10 --gradient_accumulation_steps=1 --beta=$beta --set_type=$label_name
    done
done

for label_name in set_1 set_2 set_3 set_4 set_5 set_6 set_7 set_8 set_9 set_10 set_11 set_12 set_13 set_14 set_15 set_16 set_17 set_18; do
    for IPO_tau_parameter in 1.1 1; do
        python train.py --batch_size=100 --max_iters=10  --eval_iters=10 --gradient_accumulation_steps=1 --IPO_tau_parameter=$IPO_tau_parameter --set_type=$label_name --loss_type="IPO"
    done
done
echo "Program test finished with exit code $? at: `date`"
