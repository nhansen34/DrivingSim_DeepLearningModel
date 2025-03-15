#!/bin/bash
#SBATCH --job-name=ped_risk_resume # Changed job name to indicate resuming
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=ped_resume_%j.log # Updated log name
#SBATCH --error=ped_resume_%j.err # Updated error log name
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=nhansen3@uoregon.edu

# Load necessary modules
module load slurm
module load ondemand-jupyter
module load python/3.8.1
module load pytorch/1.10.0

# Change to your script directory
cd /home/nhansen3/DrivingSimProj/pretrained_scripts/

# Set environment variables for PyTorch
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

# Run your training script with checkpoint path
python train_script.py --resume_from_checkpoint pretrained_scripts/best_pedestrian_model.pth