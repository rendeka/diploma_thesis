#!/bin/bash
#PBS -q gpu
#PBS -l select=1:ncpus=1:ngpus=1:mem=8gb:scratch_local=4gb

# Set a custom log directory
LOG_DIR="/home/rendeka/diploma_thesis/job_outputs"
mkdir -p "$LOG_DIR"

# Redirect stdout and stderr to your logs
exec > $LOG_DIR/run.sh.o$PBS_JOBID 2> $LOG_DIR/run.sh.e$PBS_JOBID

echo "================================================="
echo " Hello from $(hostname)"
echo " PBS_WORKDIR: $PBS_O_WORKDIR"
echo " JOB ID: $PBS_JOBID"
echo "================================================="

# Load the module for Python
module add python/python-3.10.4-intel-19.0.4-sc7snnf

# Option A: Activate a *shared* venv in /home
source /home/rendeka/venv/bin/activate

# Option B (if no /home venv): skip venv, just use the module's Python
# (comment out the source line)

# Set base path (shared across all nodes)
export SKYRMION_BASE_PATH="/home/rendeka/diploma_thesis"

echo "SKYRMION_BASE_PATH=$SKYRMION_BASE_PATH"
ls -l $SKYRMION_BASE_PATH

# Run the Python script
python3 $SKYRMION_BASE_PATH/${SCRIPT:-run.py}