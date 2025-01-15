#!/bin/bash
model_dir=$(dirname $(readlink -f "$0"))
model_name="yolov8s_Xray_F32_1b"
echo "current path : $model_dir"

function mv_yolov8_seg(){
    mkdir -p ../../$model_name/datasets
    mkdir -p ../../$model_name/models
    mkdir -p ../../$model_name/models/bm1684
    sudo chmod -R 777 ../../$model_name
    cp -r ../models/bm1684/$model_name.bmodel ../../$model_name/models/bm1684/

    if [ -d "../python" ]; then
        cp -r ../python ../../$model_name/
    else
        echo "Warning: python directory does not exist, skipping this copy operation."
    fi
    if [ -d "../cpp" ]; then
        cp -r ../cpp ../../$model_name/
    else
        echo "Warning: cpp directory does not exist, skipping this copy operation."
    fi
    if [ -d "../datasets/test" ]; then
        cp -r ../datasets/test ../../$model_name/datasets
    else
        echo "Warning: datasets/test directory does not exist, skipping this copy operation."
    fi
    if [ -f "../datasets/coco.names" ]; then
        cp ../datasets/coco.names ../../$model_name/datasets
    else
        echo "Warning: datasets/coco.names file does not exist, skipping this copy operation."
    fi
    echo "move all files success!"
}

mv_yolov8_seg
