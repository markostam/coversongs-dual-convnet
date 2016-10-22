#!/bin/bash
#qsub script to submit to nyu's scheduler

#PBS -N 
#PBS -m abe
#PBS -j oe
#PBS -M mss460@nyu.edu
#PBS -d /scratch/mss460/CoverSongs2
#PBS -l nodes=1:ppn=8:gpus=1:k80
#PBS -l mem=32GB
#PBS -l walltime=96:00:00


module purge
module load tensorflow/python3.5.1/20160418
cd /scratch/mss460/CoverSongs2
python3.5 /scratch/mss460/CoverSongs2/train_coversongs.py --batch_size 1024 --num_epochs 10000 --dropout_factor 0.8 --l2_reg_lambda 0.1

 