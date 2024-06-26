import os
import random
import numpy as np
import torch
import nibabel as nib
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
import cv2

def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        
        if len(image.shape) == 3:
            image = np.max(image,axis=2)
        
        label = label.squeeze(0)
        
        if len(label.shape) == 3:
            label = np.max(label,axis=2)
            
        #make image a square
        if image.shape[0] != image.shape[1]:
        # Make the image a square by resizing
            if image.shape[0] > image.shape[1]:
                image = cv2.resize(image, (image.shape[0], image.shape[0]))
            else:
                image = cv2.resize(image, (image.shape[1], image.shape[1]))
        
        if label.shape[0] != label.shape[1]:
            
            if label.shape[0] > image.shape[1]:
                label = cv2.resize(label, (label.shape[0], label.shape[0]))
            else:
                label = cv2.resize(label, (label.shape[1], label.shape[1]))
        
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
            
        if image.shape != self.output_size:
            image = cv2.resize(image,self.output_size)
        if label.shape != self.output_size:
            label = cv2.resize(label,self.output_size)
        
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample['image'] = image
        sample['label'] = (label/255.0).long()
        return sample

class transform_test(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        
        if len(image.shape) == 3:
            image = np.max(image,axis=2)
        
        label = label.squeeze(0)
        
        if len(label.shape) == 3:
            label = np.max(label,axis=2)
            
        #make image a square
        if image.shape[0] != image.shape[1]:
        # Make the image a square by resizing
            if image.shape[0] > image.shape[1]:
                image = cv2.resize(image, (image.shape[0], image.shape[0]))
            else:
                image = cv2.resize(image, (image.shape[1], image.shape[1]))
        
        if label.shape[0] != label.shape[1]:
            
            if label.shape[0] > image.shape[1]:
                label = cv2.resize(label, (label.shape[0], label.shape[0]))
            else:
                label = cv2.resize(label, (label.shape[1], label.shape[1]))
                       
        
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        
        if image.shape != self.output_size:
            image = cv2.resize(image,self.output_size)
        if label.shape != self.output_size:
            label = cv2.resize(label,self.output_size)
        
        
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample['image'] = image
        sample['label'] = (label/255.0).long()
        return sample


class Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, transform=None):
        self.transform = transform  # using transform in torch!
        self.sample_list = open(list_dir).readlines()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        slice_name = self.sample_list[idx].strip('\n')
        data_path_image = os.path.join(self.data_dir, slice_name)
        data_path_mask = os.path.join(self.data_dir,'mask',slice_name[0],slice_name[8:-4]+'_m'+'.nii')
        image = nib.load(data_path_image).get_fdata()
        label = nib.load(data_path_mask).get_fdata()

        sample = {'image': image, 'label': label, 'case_name': slice_name[0]+'/'+slice_name[8:-4]}
        if self.transform:
            sample = self.transform(sample)
        return sample

