#coding=utf-8
import numpy as np
import pandas as pd
import os
import glob
import torch

os.chdir(os.getcwd())

xls = 'dataset.xls'
df =pd.read_excel(xls)
batch_size = 3
#print(df.loc[0].values)

train_dataset = torch.tensor(df.loc[:99].values)   

test_dataset = torch.tensor(df.loc[100:].values)   


train_feature = train_dataset[:,2:].float() 
train_label = train_dataset[:,1].float()  
test_feature = test_dataset[:,2:].float()
test_label = test_dataset[:,1].float()

import torch.utils.data as Data

dataset = Data.TensorDataset(train_feature,train_label)
data_iter = Data.DataLoader(dataset,batch_size,shuffle=True)

dataset_test = Data.TensorDataset(test_feature,test_label)
data_test_iter = Data.DataLoader(dataset_test,batch_size,shuffle=True)

#continue
from torch import nn

D_in = 10  ##cell of input layer
H1 = 9 ## cell of hidden layer
H2=9
H3=9
D_out = 1 ## cell of output layer
lerning_rate = 1e-4   

	
		
class MyModel(nn.Module):
	def __init__(self,D_in,H1,H2,H3,D_out):
		super(MyModel,self).__init__()
		self.net = nn.Sequential(nn.Linear(D_in,H1),nn.ReLU(),nn.Linear(H1,H2),nn.ReLU(),nn.Linear(H2,H3),nn.ReLU(),nn.Linear(H3,D_out))  		                                                                              		                                                                           
	def forward(self,x):
		y_pred = self.net(x)
		return y_pred
		
model = MyModel(D_in,H1,H2,H3,D_out)

optimizer = torch.optim.Adam(model.parameters(),lr=lerning_rate) 
loss_fn = nn.MSELoss(reduction='mean') 	

epoch = 2000  
txt = open('result.txt','w+')
train_loss_list = []
test_loss_list = []
epoch_list = []
for ii in range(epoch):
	train_loss,n = 0,0
	epoch_list.append(ii)
	for x,y in data_iter:
		#print(x.shape)
		y = y.view(-1,1)
		
		#print(y.shape)
		y_pred = model(x)
		#print(y_pred.shape)
		loss = loss_fn(y_pred,y)
		optimizer.zero_grad()
		loss.backward()  
		optimizer.step()
		train_loss = train_loss + loss.item()
		n = n + 1
	
	test_loss,m = 0,0
	for x_test,y_test in data_test_iter:
		y_test = y_test.view(-1,1)
		y_test_pred = model(x_test)
		l = loss_fn(y_test_pred,y_test)
		test_loss = test_loss + l.item()
		m = m + 1
	train_loss = train_loss/float(n)
	test_loss = test_loss/float(m)
	train_loss_list.append(train_loss)
	test_loss_list.append(test_loss)
	txt.write('epoch: {},train loss:{},test loss:{}\n'.format(str(ii),train_loss,test_loss))
	
	
import matplotlib.pyplot as plt

plt.figure()
plt.plot(epoch_list,train_loss_list,linestyle='-.',marker='o',label='train')
plt.plot(epoch_list,train_loss_list,linestyle='-.',marker='>',label='test')
plt.xlabel('epoch')
plt.ylabel('loss')

plt.legend()
#plt.show()
plt.savefig("result_train.png",dpi=800, bbox_inches='tight')
plt.clf()	
txt_result = open('result_nn.txt','w+')
#y_true_list = []
#y_prediction_list = []


cycle_num = []
true_list = []
prediction_list = []
for cc in range(len(test_feature)):
	
	y_prediction = model(test_feature[cc,:])
	txt_result.write(str(cc+1)+'	'+str(test_label[cc].item())+'	'+str(y_prediction.data.item())+'\n')
	cycle_num.append(cc+1)
	true_list.append(test_label[cc].item())
	prediction_list.append(y_prediction.data.item())
plt.figure()
plt.plot(cycle_num,true_list,linestyle='-',label='True')
plt.plot(cycle_num,prediction_list,linestyle='-',label='Prediction')
plt.xlabel('Circles')
plt.ylabel('capacity')
plt.legend()
#plt.show()
plt.savefig("result_prediction.png",dpi=800, bbox_inches='tight')
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
