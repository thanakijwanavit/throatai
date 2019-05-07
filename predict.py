# Imports here
import numpy as np
from torchvision import datasets,transforms
import torch
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pylab
from PIL import Image
from sklearn import preprocessing
from PIL import Image, ImageOps
from torch import nn
from torch import optim
import torch.nn.functional as F
import torchvision
import json
import seaborn as sb
import matplotlib.pyplot as plt
import pandas as pd
from collections import OrderedDict
import time
import operator
import get_input_args2
from choosemodel import choosemodel
import pprint
pp = pprint.PrettyPrinter(indent=4)

in_arg=get_input_args2.get_input_args()
checkpoint_path=in_arg.model_directory



def load_checkpoint(filepath):
    checkpoint=torch.load(filepath)
    model=choosemodel(checkpoint['arch'])
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    newdict=checkpoint['cat_to_name_dict']
    #return {'model':model,'newdict':newdict}
    return [model,newdict]

#### load model and newdict
model,newdict = load_checkpoint(checkpoint_path)#['model','newdict']
#print('model class to idx is ',model.class_to_idx)
#print(model.__dict__)

##### check if alternative dict is input, in that case, assign to alternative dict
if in_arg.category_names != 'None':
    print('provide alternative dictionary',in_arg.category_names)
    with open(in_arg.category_names, 'r') as f:
        cat_to_name = json.load(f)
    newdict={int(i):j for i,j in cat_to_name.items()}
else:
    print('default dictionary name is used')



def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    size=(256,256)
    pilim=ImageOps.fit(image, size, Image.ANTIALIAS)
    width, height = pilim.size   # Get dimensions
    new_width=224
    new_height=224
    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2
    croppedIm=pilim.crop((left, top, right, bottom))
    numpyim= np.array(croppedIm)/255

    mean = numpyim.mean(axis=(0,1))
    std= numpyim.std(axis=(0,1))
    nmean = np.array([0.485, 0.456, 0.406])
    nstd = np.array([0.229, 0.224, 0.225])
    #numpyim = ((numpyim - mean)/std)*nstd+nmean
    numpyim = (numpyim - mean)/std

    return numpyim.transpose((2,0,1))
    # TODO: Process a PIL image for use in a PyTorch model
def imshow(image,df, ax=None, title=None):
    if ax is None:
        a4_dims = (4.5, 8.27)
        fig, (ax,ax1) = plt.subplots(2,1,figsize=a4_dims)
        #ax1 = plt.subplots(1,2)
    #else:
        #fig, axs = plt.subplots(1,2)
        #ax1=sb.barplot(data=df,x='category',y='probability')
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))


    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    ax.imshow(image)
    #sb.set(rc={'figure.figsize':(11.7,8.27)})
    sb.barplot(data=df,y='category',x='probability',ax=ax1)
    #print(resultdic)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_title(df['category'].iloc[0])
    return ax

def predict(image_path, model, topk=5):

    im=Image.open(image_path)
    im=process_image(im)
    imagetensor=torch.from_numpy(im).float()

    # Add the extra select dimension for unsqueezing
    unsqueezedtensor=imagetensor.unsqueeze_(0)
    #print(unsqueezedtensor.shape)
    #get output
    if in_arg.gpu==False:
        model.to('cpu')
        output=model(unsqueezedtensor)
        print('using cpu')
    else:
        model.to('cuda')
        output=model(unsqueezedtensor.cuda())
        print('using cuda')
    # TODO: Implement the code to predict the class from an image file
    ## get idx to class  dictionary
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}






    probability_output,categorynumber=torch.topk(output.data,topk,1)
    probability=torch.exp(probability_output)
    resultdic={'category':[],'probability':[]}
    for i,j in zip(categorynumber[0],probability[0]):
        category_id=int(idx_to_class[i.item()])
        resultdic['category'].append(newdict[category_id])
        resultdic['probability'].append(j.item())
    #df=pd.DataFrame(resultdic)
    #print(df.to_string())
    #imshow(im,df)
    #resultdic['probs'] = df.pop('probability')
    #resultdic['classes'] = df.pop('category')
    return resultdic

def display(image_path, model, topk=5):
    result=predict(image_path,model,topk)
    df=pd.DataFrame(result)
    print(df.to_string())
    #pp.pprint(result)

topk=in_arg.top_k
display(in_arg.image_dir,model,topk)
