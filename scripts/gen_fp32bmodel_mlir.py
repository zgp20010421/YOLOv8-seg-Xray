import subprocess
import os
import shutil
import logging
import glob

def copy_files(src: str, dst: str, file_extension: str = '*'):
    if not os.path.exists(dst):
        os.makedirs(dst)
    try:
        for file_path in glob.glob(os.path.join(src, file_extension)):
            if os.path.isfile(file_path):
                shutil.copy(file_path, dst)
                print("Copied {} to {}".format(file_path, dst))
    except Exception as e:
        print("Error Copying {} to {}".format(file_path, dst))

def gen_fp32bmodel_mlir(model_name:str, type_name:str, batch_size:int, outdir:str, chip_name:str, img_size:int, quantize:str):
    """
    params:
        model_name: yolov8s
        type_name: Xray
        batch_size: 1
        outdir: ../models/bm1684
        chip_name: bm1684
        img_size: 320
        quantize: F32     
    """
    model_transform_cmd = [
        "model_transform.py",
        "--model_name={}".format(model_name),
        "--model_def=../models/onnx/{}-seg-{}-{}b.onnx".format(model_name, type_name, batch_size),
        "--input_shapes=[[{},3,{},{}]]".format(batch_size, img_size, img_size),
        "--mean=0.0,0.0,0.0",
        "--scale=0.0039216,0.0039216,0.0039216",
        "--keep_aspect_ratio",
        "--pixel_format=rgb",
        "--mlir={}_{}_{}b.mlir".format(model_name, type_name, batch_size),
    ]
    model_deploy_cmd = [
        "model_deploy.py",
        "--mlir={}_{}_{}b.mlir".format(model_name, type_name, batch_size),
        "--quantize={}".format(quantize),
        "--chip={}".format(chip_name),
        "--model={}_{}_{}_{}b.bmodel".format(model_name, type_name, quantize, batch_size)
    ]
    subprocess.run(model_transform_cmd)
    subprocess.run(model_deploy_cmd)

    # mv
    copy_files('./', '{}'.format(outdir), '{}_{}_{}b.bmodel'.format(model_name, quantize, batch_size))


if __name__ == "__main__":
    gen_fp32bmodel_mlir('yolov8s', 'Xray', 1, '../models/bm1684', 'bm1684', 320, 'F32')
    
