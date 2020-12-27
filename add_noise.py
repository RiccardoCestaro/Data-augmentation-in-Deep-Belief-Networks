from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from skimage.util import random_noise
import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import argparse
import sys
from scipy.io import savemat

ap = argparse.ArgumentParser()
ap.add_argument(
    '-d', '--dataset', type=str, 
    help='dataset to use'
)
args = vars(ap.parse_args())
BATCH_SIZE = 60000

if args['dataset'] == 'mnist' or args['dataset'] == 'fashionmnist':  
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    if args['dataset'] == 'mnist':
        trainset = datasets.MNIST(
            root='./data',
            train=True,
            download=True, 
            transform=transform
        )
        testset = datasets.MNIST(
            root='./data',
            train=False,
            download=True,
            transform=transform
        )
    elif args['dataset'] == 'fashionmnist':
        trainset = datasets.FashionMNIST(
            root='./data',
            train=True,
            download=True, 
            transform=transform
        )
        testset = datasets.FashionMNIST(
            root='./data',
            train=False,
            download=True,
            transform=transform
        )
    
trainloader = torch.utils.data.DataLoader(
    trainset, 
    batch_size=BATCH_SIZE,
    shuffle=True
)
testloader = torch.utils.data.DataLoader(
    testset, 
    batch_size=BATCH_SIZE,
    shuffle=False
)


def test_data():
    for data in testloader:
        img, _ = data[0], data[1]
    return img, _  
        
def no_noise():
    for data in trainloader:
        img, _ = data[0], data[1]
    return img, _    
        
def gaussian_noise(magnitude):
    for data in trainloader:
        img, _ = data[0], data[1]
        gauss_img = torch.tensor(random_noise(img, mode='gaussian', mean=0, var=0.05+magnitude, clip=True))
    return gauss_img, _    
    
def salt_pepper_noise(magnitude):
    for data in trainloader:
        img, _ = data[0], data[1]
        s_and_p = torch.tensor(random_noise(img, mode='s&p', salt_vs_pepper=0.2+magnitude*5, clip=True))
    return s_and_p, _  
    
def speckle_noise(magnitude):
    for data in trainloader:
        img, _ = data[0], data[1]
        speckle_noise = torch.tensor(random_noise(img, mode='speckle', mean=0, var=0.05+magnitude, clip=True))
    return speckle_noise, _  
    
def reshape(data,label):
    data = np.array(data)
    label = np.array(label)
    data = np.einsum('ijky->ijyk', data)
    t_label = np.zeros((BATCH_SIZE,10))
    for l in range(label.size):
        for i in range(t_label[l].size):
            t_label[l][i] = (label[l] == i) 
    label = t_label
    data = np.einsum('ijk->ikj',(np.reshape(data,(125,480,784))))
    label = np.reshape(label,(125,480,10))   
    label = np.einsum('ijk->ikj', label)
    return data,label
  
  
def reshape_test(data,label):
    data = np.array(data)
    label = np.array(label)
    data = np.einsum('ijky->ijyk', data)
    t_label = np.zeros((10000,10))
    for l in range(label.size):
        for i in range(t_label[l].size):
            t_label[l][i] = (label[l] == i) 
    label = t_label
    data = np.reshape(data,(100,100,784))
    label = np.reshape(label,(100,100,10))    
    data = np.einsum('ijk->ikj', data)
    label = np.einsum('ijk->ikj', label)
    return data,label
    
print("--- Creating Original .mat ---")
[img,label] = no_noise()
[testimg,testlabel] = test_data()
[img,label] = reshape(img,label)
[testimg,testlabel] = reshape_test(testimg,testlabel)
mat = {"batchdata": img, "batchtargets": label, "testbatchdata": testimg, "testbatchtargets": testlabel}
savemat("MNIST_125.mat", mat)
print("--- Saved Original .mat ---")
print("\n\n")

print("--- Start creating noise .mat's ---")

round = 1
for magnitude in np.arange(0.15,0.15,0.01):
    
    print("--- round "+str(round)+" ---")
    round = round+1
    [gauss_img,label] = gaussian_noise(magnitude)
    [gauss_img,label] = reshape(gauss_img,label)
    mat = {"batchdata": gauss_img, "batchtargets": label, "testbatchdata": testimg, "testbatchtargets": testlabel}
    savemat("MNIST_GAUSS_NOISE_125_" + str(0.05+magnitude) + ".mat", mat)

    [s_and_p,label] = salt_pepper_noise(magnitude)
    [s_and_p,label] = reshape(s_and_p,label)
    mat = {"batchdata": s_and_p, "batchtargets": label, "testbatchdata": testimg, "testbatchtargets": testlabel}
    savemat("MNIST_SALT_PEPPER_NOISE_125_" + str(0.5+magnitude*10) + ".mat", mat)

    [speckle_noise,label] = salt_pepper_noise(magnitude)
    [speckle_noise,label] = reshape(speckle_noise,label)
    mat = {"batchdata": speckle_noise, "batchtargets": label, "testbatchdata": testimg, "testbatchtargets": testlabel}
    savemat("MNIST_SPECKLE_NOISE_125_" + str(0.05+magnitude) + ".mat", mat)

print("--- Finished saving .mat's ---")
print("\n\n")

#call deeptrain_GPU.py for training a DBN
#--------------using python2--------------
import subprocess
subprocess.run('python -V', shell=True)
print("\n\n")



print("\n\n")
print("--- training original training set --- (May requires some minutes)")



print(subprocess.run('python deeptrain_GPU.py -i "MNIST_125.mat" -o "DN_MNIST_125.mat"', shell=True, capture_output=True))

print("\n\n")
print("--- training noise training sets --- (May requires a lot of time)")
round = 0
for magnitude in np.arange(0.00,0.15,0.01):
    
    print("--- round "+str(round)+" ---")
    round = round+1
    print("\n\n")
    subprocess.run('python -V', shell=True)
    
    print(subprocess.run('python deeptrain_GPU.py -i "MNIST_GAUSS_NOISE_125_' + str(0.05+magnitude) + '.mat" -o "DN_MNIST_GAUSS_NOISE_125_' + str(0.05+magnitude) + '.mat"', shell=True, capture_output=True))
    print("\n\n")
    
    print(subprocess.run('python deeptrain_GPU.py -i "MNIST_SALT_PEPPER_NOISE_125_' + str(0.5+magnitude*5) + '.mat" -o "DN_MNIST_SALT_PEPPER_NOISE_125_' + str(0.5+magnitude*5) + '.mat"', shell=True, capture_output=True))
    print("\n\n")
    
    print(subprocess.run('python deeptrain_GPU.py -i "MNIST_SPECKLE_NOISE_125_' + str(0.05+magnitude) + '.mat" -o "DN_MNIST_SPECKLE_NOISE_125_' + str(0.05+magnitude) + '.mat"', shell=True, capture_output=True))
    print("\n\n")

