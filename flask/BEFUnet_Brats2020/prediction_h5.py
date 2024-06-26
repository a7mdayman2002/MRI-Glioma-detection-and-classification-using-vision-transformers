import numpy as np
import torch
from torch import nn
from scipy.ndimage import zoom


class predictor_h5(nn.Module):
    def __init__(self,model,output_size):
        super().__init__()
        self.model = model.eval()
        self.output_size = output_size
        
    def forward(self,x):
        img, label = x['image'][:], x['mask'][:]

        image, label = np.max(img,axis = -1), np.max(label,axis = -1)

        h, w = image.shape

        if h != self.output_size[0] or w != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / h, self.output_size[1] / w), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / h, self.output_size[1] / w), order=0)

        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32)).long()


        image = image.expand(1,3,image.shape[1],image.shape[2])

        # forward path for image
        x = self.model(image)
        # give probability to each output class torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
        x = torch.softmax(x,dim=1)
        # choose the class with the highest probability
        x = torch.argmax(x,dim=1,keepdim=True)
        
        return x