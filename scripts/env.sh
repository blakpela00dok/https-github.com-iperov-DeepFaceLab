#!/usr/bin/env bash
conda activate deepfacelab
export DFL_PYTHON="python3.7"
export DFL_WORKSPACE="$DEEPFACELAB_WORKSPACE"

if [ ! -d "$DFL_WORKSPACE" ]; then
    mkdir "$DFL_WORKSPACE"
    mkdir "$DFL_WORKSPACE/data_src"
    mkdir "$DFL_WORKSPACE/data_src/aligned"
    mkdir "$DFL_WORKSPACE/data_src/aligned_debug"
    mkdir "$DFL_WORKSPACE/data_dst"
    mkdir "$DFL_WORKSPACE/data_dst/aligned"
    mkdir "$DFL_WORKSPACE/data_dst/aligned_debug"
    mkdir "$DFL_WORKSPACE/model"
fi

export DFL_ROOT="$DEEPFACELAB_PATH"
export DFL_SRC="$DEEPFACELAB_PATH"
