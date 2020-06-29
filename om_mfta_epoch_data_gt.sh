#!/bin/sh
#SBATCH --job-name=mftma_epoch
#SBATCH --output=mftma_epoch_%j.out
#SBATCH --error=mftma_epoch_%j.err
#SBATCH --mem=20G
#SBATCH --nodes=1
#SBATCH -t 04:00:00
#SBATCH -c 1
#SBATCH -w node[001-016]

# echo "Running training  ${1}"
# echo "Running epoch ${2}"

timestamp() {
  date +"%T"
}

filename="mftma_epoch_"$(date '+%Y%m%d%T')".txt"

cd /om/user/`whoami`/neural_manifolds/

python run_mftma_on_epoch_data_noargs.py > "$filename"

#python run_mftma_on_epoch_data.py "${1}" "${2}" > "$filename"
