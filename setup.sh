#!/bin/bash

# Setup script for MineRL environment
# Usage: 
#   ./setup.sh              # Install into existing/current environment
#   ./setup.sh --new-env    # Create new conda environment named 'minerl'

CREATE_NEW_ENV=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --new-env)
            CREATE_NEW_ENV=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--new-env]"
            echo "  --new-env    Create a new conda environment named 'minerl'"
            echo "  --help       Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

if [ "$CREATE_NEW_ENV" = true ]; then
    echo "Creating new conda environment 'minerl'..."
    conda create -n minerl python=3.7 pip py-opencv cudatoolkit=10.2 pytorch=1.9 -c pytorch -c conda-forge -y
    conda activate minerl
else
    echo "Installing into current environment..."
    echo "Current environment: $CONDA_DEFAULT_ENV"
fi

conda install -c conda-forge openjdk=8 -y
export JAVA_HOME=$(dirname $(dirname "$(readlink -f "$(which java)")"))
echo "JAVA_HOME=$JAVA_HOME"
pip install git+https://github.com/minerllabs/minerl
pip install coloredlogs matplotlib aicrowd-api aicrowd-gym gym3 attrs
pip install -e .

echo ""
echo "Setup complete!"
if [ "$CREATE_NEW_ENV" = true ]; then
    echo "Remember to activate the environment: conda activate minerl"
fi