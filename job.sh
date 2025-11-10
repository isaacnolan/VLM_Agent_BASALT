#!/bin/bash
#SBATCH --account=pas3150
#SBATCH --job-name=auto_sbatch    # Job name
#SBATCH --nodes=1 --ntasks-per-node=16
#SBATCH --gpus-per-node=1
#SBATCH --time=1:00:00         # Time limit hrs:min:sec
#SBATCH --output=logs/jobs/minerl_run_%j.log  # Standard output and error log (%j = job ID)
#SBATCH --mail-type=BEGIN,FAIL

module load miniconda3/24.1.2-py310
conda activate qwen
python qwen_policy_server.py &

conda deactivate

bash setup_display.sh
conda activate minerl
python qwen_policy_client.py --record-dir Episode_Outputs --max-steps 50