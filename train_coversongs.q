#!/bin/bash
#qsub script to submit to nyu's scheduler

#PBS -N 
#PBS -m abe
#PBS -j oe
#PBS -M mss460@nyu.edu
#PBS -d /scratch/mss460/CoverSongs2
#PBS -l nodes=1:ppn=2:gpus=1:k80
#PBS -l mem=36GB
#PBS -l walltime=24:00:00

# max batchsize on tesla k80 = 10
# max batchsize on titan black = 5

# echo "import sys; sys.setdefaultencoding('utf-8')" > sitecustomize.py
export PYTHONIOENCODING=utf-8
module purge
module load tensorflow/python3.5.1/20161029
cd /scratch/mss460/CoverSongs2

python3.5 /scratch/mss460/CoverSongs2/train_coversongs.py --batch_size 16  --num_epochs 1000 --dropout_factor 0.5 --l2_reg_lambda .05 --dev_size_percent .1 --evaluate_every 100 --learning_rate 0.000005 --filters_per_layer '128,96,64,32'
