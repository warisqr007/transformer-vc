#!/bin/bash
##ENVIRONMENT SETTINGS; CHANGE WITH CAUTION
#SBATCH --export=NONE                #Do not propagate environment
#SBATCH --get-user-env=L             #Replicate login environment

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=dl-hw2	#Set the job name to "JobExample5"
#SBATCH --time=8:00:00            #Set the wall clock limit to 1hr and 30min
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48                   #Request 1 task
#SBATCH --mem=32G                 #Request 2560MB (2.5GB) per node
#SBATCH --output=PINNs_5_500.%j     #Send stdout/err to "Example5Out.[jobID]"
#SBATCH --gres=gpu:rtx:1                #Request 2 GPU per node can be 1 or 2
#SBATCH --partition=gpu              #Request the GPU partition/queue

##OPTIONAL JOB SPECIFICATIONS
##SBATCH --account=132767164921             #Set billing account to 123456
##SBATCH --mail-type=ALL              #Send email on all job events
##SBATCH --mail-user=bhanu@tamu.edu    #Send all emails to email_address


#First Executable Line
module load GCC/10.2.0
module load CUDA/11.1.1
module load OpenMPI/4.0.5
module load PyTorch/1.10.0
module load Anaconda3/2021.11
module load Anaconda3/2021.11

cd /scratch/user/bhanu/dl_hw2/ResNet
python main.py

