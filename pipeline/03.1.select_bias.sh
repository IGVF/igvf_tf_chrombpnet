#!/bin/bash

datasets_path="/oak/stanford/groups/engreitz/Users/opushkar/igvf_tf_collab"
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

# Build --fold-bias args from fold_bias_suffix associative array (sourced from dataset_config.sh).
# Produces e.g.: "0:_1 1:_06 2:_1 3:_1 4:_09"
build_fold_bias_args() {
    local args=()
    for fold in "${!fold_bias_suffix[@]}"; do
        args+=( "${fold}:${fold_bias_suffix[$fold]}" )
    done
    echo "${args[@]}"
}

export DATASET_DIR="${datasets_path}/igvf11_h7_hesc"
source "${SCRIPT_DIR}/config.sh"
source "${CONDA_INIT}"
conda activate "${CONDA_ENV}"

python "${SCRIPT_DIR}/select_bias_model.py" \
    --core-path /oak/stanford/groups/engreitz/Users/opushkar/igvf_tf_collab \
    --biases 05 06 07 08 09 1 \
    --folds 0 1 2 3 4 \
    --dataset igvf11_h7_hesc \
    --peak-type all \
    --fold-bias $(build_fold_bias_args)

export DATASET_DIR="${datasets_path}/igvf_endothelial"
source "${SCRIPT_DIR}/config.sh"
source "${CONDA_INIT}"
conda activate "${CONDA_ENV}"

python "${SCRIPT_DIR}/select_bias_model.py" \
    --core-path /oak/stanford/groups/engreitz/Users/opushkar/igvf_tf_collab \
    --biases 05 06 07 08 \
    --folds 0 1 2 3 4 \
    --dataset igvf_endothelial \
    --peak-type all \
    --fold-bias $(build_fold_bias_args)

export DATASET_DIR="${datasets_path}/igvf3_cardiomyocyte"
source "${SCRIPT_DIR}/config.sh"
source "${CONDA_INIT}"
conda activate "${CONDA_ENV}"

python "${SCRIPT_DIR}/select_bias_model.py" \
    --core-path /oak/stanford/groups/engreitz/Users/opushkar/igvf_tf_collab \
    --biases 05 06 07 08 \
    --folds 0 1 2 3 4 \
    --dataset igvf3_cardiomyocyte \
    --peak-type all \
    --fold-bias $(build_fold_bias_args)

export DATASET_DIR="${datasets_path}/igvf6_definitive_endoderm"
source "${SCRIPT_DIR}/config.sh"
source "${CONDA_INIT}"
conda activate "${CONDA_ENV}"

python "${SCRIPT_DIR}/select_bias_model.py" \
    --core-path /oak/stanford/groups/engreitz/Users/opushkar/igvf_tf_collab \
    --biases 05 06 07 08 \
    --folds 0 1 2 3 4 \
    --dataset igvf6_definitive_endoderm \
    --peak-type all \
    --fold-bias $(build_fold_bias_args)
