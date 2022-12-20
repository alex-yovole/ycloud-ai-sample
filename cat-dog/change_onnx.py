import torch
import torchvision
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-path', type=str, required=True, help='net type')
parser.add_argument('-onnx_name', type=str, required=True, help='net type')
args = parser.parse_args()



dummy_input = torch.randn(1, 3, 224, 224, device='cuda')
model = torch.load(args.path)

input_names = ["input_1"]
output_names = ["output_1"]

torch.onnx.export(model, dummy_input, args.onnx_name +'.onnx', verbose=True, input_names=input_names, output_names=output_names)


