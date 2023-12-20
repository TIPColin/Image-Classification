import jittor as jt
from jittor import transform
from jittor.dataset import Dataset, SequentialSampler, RandomSampler
import numpy as np
from jittor.dataset import DataLoader
#import torch.utils.data as data
#import torch
#from torch.utils.data.dataloader import DataLoader
from PIL import Image #import jittor.dataset.image as jimage
import os
import random

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG', '.TIF',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort(key=len)
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    print(classes)
    print(class_to_idx)
    return classes, class_to_idx

def make_dataset(dir, class_to_idx):
    from PIL import ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    images = []
    num_in_class = []  # the number of samples in each class
    images_txt = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir), key=len):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            num = 0
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)
                    images_txt.append(target + '/' + fname)
                    num += 1
            num_in_class.append(num)

    return images, num_in_class, images_txt

def pil_loader(path, mode='RGB'): #use PIL to open the image file and convert it to RGB format
    # open path as file to avoid ResourceWarning
    with open(path, 'rb') as f:
        img = Image.open(f)
        if mode == 'RGB':
            return img.convert('RGB')
        elif mode == 'HSV':
            return img.convert('HSV')
        
def accimage_loader(path):
    # In Jittor, we can directly use PIL.Image for image loading
    try:
        return Image.open(path)
    except IOError:
        # Potentially a decoding problem, fall back to default_loader
        return pil_loader(path)


def default_loader(path, mode='RGB'):
    # Jittor does not have an equivalent of torchvision's get_image_backend(),
    # so we assume PIL is always used for image loading
    if mode is not None:
        return accimage_loader(path)
    else:
        return pil_loader(path, mode)
  
class MyDataset(Dataset):
    
    def __init__(self, args, transform=None, target_transform=None, loader=default_loader):
        classes, class_to_idx = find_classes('/data/xshen/partition/train_new')
        imgs, num_in_class, images_txt = make_dataset('/data/xshen/partition/train_new', class_to_idx)
        if len(imgs) == 0:
            raise RuntimeError("Found 0 images in subfolders of: " + '/data/xshen/partition' + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS))

        self.mode = 'RGB'
        self.input_nc = 3
        self.imgs = imgs
        self.num_in_class = num_in_class
        self.images_txt = images_txt
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path, self.mode)
        if self.transform is not None:
            img = jt.array(np.array(self.transform(img)), dtype='float32') #self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
            
        img = jt.array(np.array(img))
        target = jt.array(np.array(target))
        return img, target

    def __len__(self):
        return len(self.imgs)
    
class RandomBalancedSampler:    
    def __init__(self, data_source):
        print('Using RandomBalancedSampler...')
        self.data_source = data_source
        self.num_in_class = data_source.num_in_class
        #self.batch_size = batch_size  # Add batch_size attribute

    def __iter__(self):
        num_in_class = self.num_in_class
        a_perm = jt.array(random.sample(range(num_in_class[0]), num_in_class[0]))
        b_perm = jt.array(random.sample(range(num_in_class[1]), num_in_class[1]))

        if num_in_class[0] > num_in_class[1]:
            a_perm = a_perm[:num_in_class[1]]
        elif num_in_class[0] < num_in_class[1]:
            b_perm = b_perm[:num_in_class[0]]

        assert len(a_perm) == len(b_perm)

        index = jt.contrib.concat([a_perm, b_perm])

        return iter(index)

    def __len__(self):
        return min(self.num_in_class) * 2

class PairedSampler:
    def __init__(self, data_source):
        print('Using PairedSampler...')
        self.data_source = data_source
        self.num_in_class = data_source.num_in_class

    def __iter__(self):
        num_in_class = self.num_in_class
        a_perm = jt.array(random.sample(range(num_in_class[0]), num_in_class[0]))
        b_perm = a_perm + num_in_class[0]

        index = jt.contrib.concat([a_perm, b_perm])

        return iter(index)

    def __len__(self):
        return min(self.num_in_class) * 2

def DataLoaderHalf(dataset, batch_size=1, shuffle=False, half_constraint=False, sampler_type='RandomBalancedSampler', drop_last=True, num_workers=0, pin_memory=False):
    if half_constraint:
        if sampler_type == 'PairedSampler':
            sampler = PairedSampler(dataset)
        else:
            sampler = RandomBalancedSampler(dataset)
    else:
        if shuffle:
            sampler = RandomSampler(dataset)
        else:
            sampler = SequentialSampler(dataset)

    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last)
