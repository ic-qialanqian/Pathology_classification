import datetime
import os
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader,Dataset,random_split
from torchvision import transforms
import matplotlib.pyplot as plt
from dataset_sod import ImageFolder
import dataset_sod
from misc import AvgMeter, check_mkdir

from torch.backends import cudnn
import torch.nn.functional as functional
import numpy as np
import cv2
import torch.nn.functional as F
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
import math
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix,precision_recall_curve

import matplotlib.pyplot as plt
from conch.open_clip_custom import create_model_from_pretrained
from conch.downstream.zeroshot_path import zero_shot_classifier
import json

cudnn.benchmark = True

torch.manual_seed(2023)
torch.cuda.set_device(2)


train_data = './'
test_data='./'


class new_loader(Dataset):
    def __init__(self, txt_file, root_dir, transform=None,joint_transform=None):
        """
        Args:
            txt_file (string): Path to the text file containing image names and labels.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.labels = {}
        with open(root_dir+txt_file, 'r') as file:
            for line in file:
                #print(line.strip().split(','))
                if len(line.strip().split(','))>1: 
                    image_name, label = line.strip().split(',')
                    if int(label)>0:
                        real_label = 1
                    else:
                        real_label = 0
                    self.labels[image_name] = real_label
                
        self.root_dir = root_dir
        self.transform = transform
        self.joint_transform = joint_transform
        
        

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name = list(self.labels.keys())[idx]
        img_path = os.path.join(img_name)
        image = Image.open(img_path)
        label = self.labels[img_name]

        if self.transform:
            image = self.transform(image)
        
        if self.joint_transform:
            image = self.joint_transform(image)
        #print(img_name)
        #print(label)
        return image, label



##########################hyperparameters###############################
ckpt_path = './save'

args = {
    'crop_size': 256,
    
}
##########################data augmentation###############################

img_transform = transforms.Compose([
    transforms.Resize((args['crop_size'],args['crop_size'])),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
#target_transform = transforms.ToTensor()
target_transform = transforms.Compose([
            transforms.Resize((args['crop_size'],args['crop_size'])),
            transforms.ToTensor()
            ])


##########################################################################

def prompt_template( ): 
    path = "./CONCH/prompts/gleansongroups.json"
    with open(path) as f: 
        template = json.load(f) 
    class_name_d = template['0']['classnames']
    class_names = [  class_name_d[e] for e in class_name_d]
    class_template = template['0']['templates']
    basic_names = [e for e in class_name_d]
    return class_names,class_template,basic_names
def get_class_weights_crc(model,device): 
    class_names, class_template,basic_names   = prompt_template() 
    model_weights = zero_shot_classifier(model,class_names,class_template,device=device)
    
    return model_weights,basic_names


test_set = new_loader(txt_file= 'test.txt', root_dir='./', transform = img_transform,joint_transform =None )


val_loader = DataLoader(test_set, batch_size=1, num_workers=12, shuffle=True)



def loss_fn(x, y):
	x = F.normalize(x, dim=1, p=2)
	y = F.normalize(y, dim=1, p=2)
	return 2 - 2 * (x * y).sum(dim=1)
 

class IOUBCE_loss(nn.Module):
    def __init__(self):
        super(IOUBCE_loss, self).__init__()
        self.nll_lose = nn.BCEWithLogitsLoss()

    def forward(self, input_scale, taeget_scale):
        b,_,_,_ = input_scale.size()
        loss = []
        for inputs, targets in zip(input_scale, taeget_scale):
            bce = self.nll_lose(inputs,targets)
            pred = torch.sigmoid(inputs)
            inter = (pred * targets).sum(dim=(1, 2))
            union = (pred + targets).sum(dim=(1, 2))
            IOU = (inter + 1) / (union - inter + 1)
            loss.append(1- IOU + bce)
        total_loss = sum(loss)
        return total_loss / b

IOUBCE = IOUBCE_loss().cuda()



def main():
    foundation_model, preprocess = create_model_from_pretrained('conch_ViT-B-16', "../CONCH/checkpoints/pytorch_model.bin")
    foundation_model = foundation_model.cuda()
   
    
    test(foundation_model)



total = 0

def test(foundation_model):
    
    
    val_accuracy = 0.0
    y_true = []
    y_pred = []
    total_correct = 0
    total_samples = 0
                
    model_weights,basic_names = get_class_weights_crc(foundation_model,device='cuda:2')
    print(model_weights.shape)
    for val_data in val_loader:
        val_images, val_labels = (
                            val_data[0].cuda(),
                            val_data[1].cuda(),
                        )
                        
        
        with torch.inference_mode():
            img_feats = foundation_model.encode_image(val_images, proj_contrast=False, normalize=False)
            
            
            
            
            logits = (img_feats @ model_weights)#.cpu() 
            predicted = logits.argmax(dim=1)

                        
                        
            total_samples += val_labels.size(0)
            total_correct += (predicted == val_labels).sum().item()
                        
                    
            
            y_true.extend(val_labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    # Convert lists to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
                

               
    # Calculate confusion matrix
    con_matrix = confusion_matrix(y_true, y_pred)

    # Log confusion matrix
    print("Confusion Matrix:")
    print(con_matrix)
                
    accuracy = accuracy_score(y_true, y_pred)
    print("Accuracy:", accuracy)
    
    # Precision
    precision = precision_score(y_true, y_pred, average='weighted')
    print("Precision:", precision)
    
    # Recall
    recall = recall_score(y_true, y_pred, average='weighted')
    print("Recall:", recall)
    
    # F1 Score
    f1 = f1_score(y_true, y_pred, average='weighted')
    print("F1 Score:", f1)
        


        
if __name__ == '__main__':
    main()
