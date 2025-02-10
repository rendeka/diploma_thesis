#!/bin/bash
#PBS -q gpu
#PBS -l select=1:ncpus=1:ngpus=1:mem=8gb:scratch_local=4gb

# Set a custom log directory
LOG_DIR="/home/rendeka/diploma_thesis/job_outputs"
mkdir -p "$LOG_DIR"

# Redirect stdout and stderr
#PBS -o $LOG_DIR/run.sh.o$PBS_JOBID
#PBS -e $LOG_DIR/run.sh.e$PBS_JOBID

# Load the necessary module for Python
module add python/python-3.10.4-intel-19.0.4-sc7snnf

# Activate venv
source /auto/vestec1-elixir/home/rendeka/venv/bin/activate

# Set up base path envronment variable
export SKYRMION_BASE_PATH="/auto/vestec1-elixir/home/rendeka/diploma_thesis"

# Now run the Python script
python3 $SKYRMION_BASE_PATH/run.py