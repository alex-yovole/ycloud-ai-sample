import torch.utils.data.distributed
import torchvision.transforms as transforms

from torch.autograd import Variable
import os
from PIL import Image
import argparse
from torchvision.utils import save_image

parser = argparse.ArgumentParser()
parser.add_argument('-path', type=str, required=True, help='net type')
parser.add_argument('-pro_name', type=str, required=True, help='net type')
args = parser.parse_args()


classes = ('cat', 'dog')

unloader = transforms.ToPILImage()

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torch.load("model.pth")
model.eval()
model.to(DEVICE)
path=args.path

img=Image.open(path)
img=transform_test(img)
save_image(img,args.pro_name + '.jpg')
img.unsqueeze_(0)
img = Variable(img).to(DEVICE)
out=model(img)
# Predict
_, pred = torch.max(out.data, 1)
print('Image Name:{},predict:{}'.format(path,classes[pred.data.item()]))
