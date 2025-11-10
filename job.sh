#!/bin/bash
#SBATCH --account=pas3150
#SBATCH --job-name=auto_sbatch    # Job name
#SBATCH --nodes=1 --ntasks-per-node=16
#SBATCH --gpus-per-node=1
#SBATCH --time=1:00:00         # Time limit hrs:min:sec
#SBATCH --output=minerl_run.log  # Standard output and error log
#SBATCH --mail-type=BEGIN,FAIL

conda activate qwen
python qwen_policy_server.py
python test_qwen_server.py
bash setup.sh
bash setup_display.sh
conda activate minerl
python qwen_policy_client.py --record-dir Episode_Outputs --max-steps 50