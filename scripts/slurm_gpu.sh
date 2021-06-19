#!/bin/bash
#SBATCH --job-name=gpr_tr
#SBATCH --mail-user=Mario.Beaulieu@crim.ca
#SBATCH --mail-type=ALL
#SBATCH --output=/misc/tmp/beaulima/%x.out.%j.txt

#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=32

#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8

#SBATCH --constraint=cgroup

CONFIG=${1}
BASE_DIR="/misc/voute1_ptl-bema1/visi/beaulima/projects/DEEP_MASW"
CONFIGPATHDIR="${BASE_DIR}/deepmasw/configs"
MPLCONFIGDIR="/misc/data22-brs/AARISH/01/results/.cache/matplotlib"
CONDA_ENV=DEEP_MASW
cd ${BASE_DIR}
echo $(pwd)
source ~/miniconda3/bin/activate ${CONDA_ENV}
APP=${BASE_DIR}/deepmasw/train.py
python ${APP} --config-name ${CONFIG}