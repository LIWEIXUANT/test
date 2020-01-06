import torch  #包含多维数据结构以及基于其上的多种数学操作
import torchvision  #包含了目前流行的数据集，模型结构和常用的图片转换工具
import torch.nn as nn  #所有网络的基类
import numpy as np
import torchvision.transforms as transforms

# x = torch.tensor(1.,requires_grad=True)
# w = torch.tensor(2.,requires_grad=True)
# b = torch.tensor(3.,requires_grad=True)
#
# y = x*w+b
# y.backward()
# print(y)
# print(x.grad)  #求偏导时对其他元看做常数

''' 一次前向传播经过一次优化后的结果'''
# x=torch.randn(10,3)  #标准正太分布
# y=torch.randn(10,2)
#
# #build a fully connected layer
# linear = nn.Linear(3,2) #input （N,3）out (N,2) linear is a class
# print ('w: ', linear.weight) # w is (2,3) b is (1,2)
# print ('b: ', linear.bias)
#
# #build loss function and optimizer
# criterion = nn.MSELoss()  #均方差误差
# optimizer = torch.optim.SGD(linear.parameters(), lr=0.01)
#
# #forward pass
# pred = linear(x)
#
# #compute loss
# loss = criterion(pred, y)
# print('loss: ', loss.item())
#
# #backwrad loss
# loss.backward()
# # Print out the gradients.
# print('dL/dw: ', linear.weight.grad)
# print('dL/db: ', linear.bias.grad)
#
# # 1-step gradient descent.
# optimizer.step()
# pred = linear(x)
# loss = criterion(pred, y)
# print('loss after 1 step optimization: ', loss.item())
'''load data from numpy'''
# #create a numpy array
# x = np.array([[1,2],[3,4]])
# y =torch.from_numpy(x)
# print(y)
# print(y.numpy())
'''input your custom dataset'''
# You should build your custom dataset as below.
# class CustomDataset(torch.utils.data.Dataset):
#     def __init__(self):
#         # TODO
#         # 1. Initialize file paths or a list of file names.
#         pass
#     def __getitem__(self, index):
#         # TODO
#         # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
#         # 2. Preprocess the data (e.g. torchvision.Transform).
#         # 3. Return a data pair (e.g. image and label).
#         pass
#     def __len__(self):
#         # You should change 0 to the total size of your dataset.
#         return 0
#
#
# # You can then use the prebuilt data loader.
# custom_dataset = CustomDataset()
# train_loader = torch.utils.data.DataLoader(dataset=custom_dataset,
#                                            batch_size=64,
#                                            shuffle=True)
'''pretrained model'''
images = torch.randn(64, 3, 224, 224)
#resnet = torchvision.models.resnet18(pretrained=True)
#resnet.fc = nn.Linear(resnet.fc.in_features, 100)
#outputs = resnet(images)
# print(outputs.size())     # (64, 100)
# Save and load the entire model.
#torch.save(resnet, 'model.ckpt')
#load trained model
# model = torch.load('model.ckpt')
# outputs = model(images)
# print(outputs.size())     # (64, 100)
# Save and load only the model parameters (recommended).
#torch.save(resnet.state_dict(), 'params.ckpt')
#使用只保存参数的方法加载模型
resnet = torchvision.models.resnet18(pretrained=True)
resnet.fc = nn.Linear(resnet.fc.in_features, 100)
resnet.load_state_dict(torch.load('params.ckpt'))
outputs = resnet(images)
print(outputs.size())


