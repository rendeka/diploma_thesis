#!/bin/bash
#PBS -q gpu
#PBS -l select=1:ncpus=1:ngpus=1:mem=8gb:scratch_local=4gb

# Redirect PBS output to /dev/null to prevent duplicate logs in PBS_O_WORKDIR
#PBS -o /dev/null
#PBS -e /dev/null

# Set a custom log directory
LOG_DIR="/home/rendeka/diploma_thesis/job_outputs"
mkdir -p "$LOG_DIR"

exec > /home/rendeka/diploma_thesis/job_outputs/run.sh.o$PBS_JOBID 2> /home/rendeka/diploma_thesis/job_outputs/run.sh.e$PBS_JOBID

# Redirect stdout and stderr
#PBS -o /home/rendeka/diploma_thesis/job_outputs/run.sh.o$PBS_JOBID
#PBS -e /home/rendeka/diploma_thesis/job_outputs/run.sh.e$PBS_JOBID

# Load the necessary module for Python
module add python/python-3.10.4-intel-19.0.4-sc7snnf

# Activate venv
source /auto/vestec1-elixir/home/rendeka/venv/bin/activate

# Set up base path envronment variable
export SKYRMION_BASE_PATH="/auto/vestec1-elixir/home/rendeka/diploma_thesis"

# Now run the Python script
python3 $SKYRMION_BASE_PATH/run.py
