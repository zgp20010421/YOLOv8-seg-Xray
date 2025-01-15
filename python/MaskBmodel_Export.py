import torch
import torch.nn.functional as F
import numpy as np

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x0, x1):
        mask_info = x0
        ptotos = x1[0].view(32, -1)
        x = torch.matmul(mask_info, ptotos)
        return x


output1=torch.rand(1,32,160,160)
mask_info=torch.rand(1,10,32)
model = Model()


torch.onnx.export(
    model,    
    (mask_info,output1),
    "../models/onnx/yolov8s_getmask_32_fp32.onnx",
    verbose=True, 
    input_names=["mask_info", "output1"], 
    output_names=["output"], 
    opset_version=11,
    dynamic_axes={
        "mask_info": {0:"batch", 1:"num"},
        "output1": {0:"batch"},
        "output": {0:"batch", 1:"num"}
    }
)

print('Export MaskBmodel success!')