#!/bin/bash
#SBATCH --account=pas3150
#SBATCH --job-name=minerl_job    # Job name
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=1:00:00         # Time limit hrs:min:sec
#SBATCH --output=logs/jobs/minerl_run_%j.log  # Standard output and error log (%j = job ID)
#SBATCH --mail-type=BEGIN,FAIL

module load miniconda3/24.1.2-py310
source $(conda info --base)/etc/profile.d/conda.sh
conda activate qwen
which python

echo "================================"
echo "Starting QWEN Policy Server..."
echo "================================"
python -u policy_server/app.py &
SERVER_PID=$!
echo "Server started with PID: $SERVER_PID"

echo "Waiting for server to initialize..."
sleep 15

# Check if server is still running
if ps -p $SERVER_PID > /dev/null; then
   echo "✓ Server process is running"
else
   echo "✗ Server process died! Check logs above for errors."
   exit 1
fi

conda deactivate

echo "================================"
echo "Setting up display..."
echo "================================"
source setup_display.sh

conda activate minerl
which python

echo "================================"
echo "Testing server connection..."
echo "================================"
# Try to connect to server
for i in {1..5}; do
    if curl -s http://localhost:8001/health > /dev/null; then
        echo "✓ Server is responding!"
        break
    else
        echo "Attempt $i/5: Server not responding yet..."
        sleep 2
    fi
done

echo "================================"
echo "Starting client..."
echo "================================"
python client/run_agent.py --task MineRLBasaltCreateVillageAnimalPen-v0 --record-dir Episode_Outputs --max-steps 50

