#!/bin/bash
#SBATCH --job-name=ED_tr
#SBATCH --mail-user=Mario.Beaulieu@crim.ca
#SBATCH --mail-type=ALL
#SBATCH --output=/misc/tmp/beaulima/%x.out.%j.txt

#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=32

#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16

#SBATCH --constraint=cgroup

CONFIG=${1}
BASE_DIR="/misc/voute1_ptl-bema1/visi/beaulima/projects/EvalDom"
CONFIGPATHDIR="${BASE_DIR}/deepmasw/configs"
MPLCONFIGDIR="/misc/data22-brs/AARISH/01/results/.cache/matplotlib"
CONDA_ENV=EvalDom
cd ${BASE_DIR}
echo $(pwd)
source ~/miniconda3/bin/activate ${CONDA_ENV}
APP=${BASE_DIR}/run.py
python ${APP} --config-name ${CONFIG}