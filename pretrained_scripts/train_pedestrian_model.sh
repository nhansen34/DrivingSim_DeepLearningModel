#!/bin/bash
#SBATCH --job-name=ped_risk_train    # Job name
#SBATCH --partition=gpu              # GPU partition since you're using PyTorch
#SBATCH --gres=gpu:1                 # Request 1 GPU
#SBATCH --ntasks=1                   # Run a single task
#SBATCH --cpus-per-task=4            # CPUs for data loading/processing
#SBATCH --mem=32G                    # More memory for dataset handling
#SBATCH --time=24:00:00              # Deep learning jobs often take longer
#SBATCH --output=ped_train_%j.log    # Standard output log
#SBATCH --error=ped_train_%j.err     # Standard error log
#SBATCH --mail-type=BEGIN,END,FAIL   # Mail events
#SBATCH --mail-user=nhansen3@uoregon.edu  # Where to send mail

# Load necessary modules
module load slurm
module load ondemand-jupyter
module load python/3.8.1
module load cuda/11.3             # Make sure this matches your PyTorch CUDA version
module load pytorch/1.10.0        # Adjust version as needed

# Change to your script directory
cd /home/nhansen3/DrivingSimProj/pretrained_scripts/train_script.py

# Set environment variables for PyTorch
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

# Run your training script
python train_script.py

# If your script needs arguments, add them here
# python train_script.py --batch_size 64 --max_epochs 100 --gpus 1