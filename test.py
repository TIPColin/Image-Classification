import numpy as np
from PIL import Image
import os
import jittor as jt
import glob
import jittor.transform as transforms
from jittor import Var #The Variable class from PyTorch is equivalent to the Var class in Jittor. It represents a tensor variable with associated gradients for automatic differentiation.
import jittor.nn as F
from model import ScNet


# GPU ID
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

# The path of data and log
data_root = '/data/xshen/partition/test'
project_root = '/data/xshen/JCST_NIvsCG-master/log'

# Data size and Patch size
kPrcgNum = 1600
patch_size = 96

normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

transform = transforms.Compose([
        transforms.TenCrop(patch_size),
        transforms.Lambda(lambda crops: jt.stack([transforms.ToTensor()(crop) for crop in crops])),  # returns a 4D tensor
        transforms.Lambda(lambda crops: jt.stack([normalize(crop) for crop in crops]))
    ])


# instantiate model and initialize weights
model = ScNet()
model.cuda()
checkpoint = jt.load(project_root + '/checkpoint_1200.pth')
model.load_state_dict(checkpoint['state_dict'])
model.eval()

imageTmp = []
testTmp = []

testImageDir = data_root + '/NI'
testImageFile = list(glob.glob(testImageDir + '/*.jpg')) + list(glob.glob(testImageDir + '/*.png'))
testImageDir = data_root + '/CG'
testImageFile += list(glob.glob(testImageDir + '/*.jpg')) + list(glob.glob(testImageDir + '/*.png'))
for line in testImageFile:
    image_path = line
    lists = image_path.split('/')
    if lists[-2] == 'NI':
        testClass = 1
    else:
        testClass = 0

    with open(image_path, 'rb') as f:
        img = Image.open(f)
        img = img.convert('RGB')
        test_input = transform(img)
        test_input = test_input.cuda()
        input_var = Var(test_input, volatile=True)

        ncrops, c, h, w = input_var.size()
        # compute output
        output = model(input_var.view(-1, c, h, w))
        # _, pred = torch.max(output, 1)
        pred = F.softmax(output, dim=1)
        mean = jt.mean(pred, dim=0)
        label = 0
        if mean[1] > 0.5:
            label = 1
        testTmp.append(int(label))  # the predicted label
        imageTmp.append(testClass)

imageLabelNp = np.array(imageTmp)
testLabelNp = np.array(testTmp)

#  Computing average accuracy on patches
result = imageLabelNp == testLabelNp

cg_result = result[kPrcgNum:]
ni_result = result[:kPrcgNum]

print('NI accuracy is:', ni_result.sum()*100.0/len(ni_result))
print('CG accuracy is:', cg_result.sum()*100.0/len(cg_result))
print('The average accuracy is:', (ni_result.sum()*100.0/len(ni_result) + cg_result.sum()*100.0/len(cg_result))/ 2)
