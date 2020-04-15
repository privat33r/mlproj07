import torch
import pandas as pd
import numpy as np
import argparse,os
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

def prepareSample(filename):
  data=pd.read_csv(filename,low_memory=False)
  sample=pd.read_csv('sample.txt')

  data = pd.concat([sample,data])

  data=data.replace(np.float64('-inf'),-1) #to replace -inf from taking log of 0 with -1

#getting rid of string inputs which is not useful for our purporses
  data = data.replace(':0','')
  data = data.drop('State',1)
  data = data.drop('County',1)

#changing the way dates are represented, year is not considered here
  data['month'] = data['date']
  for i, row in data.iterrows():
    l = row['month'].split('-')
    data.at[i,'month'] = int(l[1])
    data.at[i,'date'] = int(l[2])

#getting rid of faulty data with a tailing ':0'
    if ':0' in str(row['Dentist Ratio']):
      data.at[i,'Dentist Ratio'] = row['Dentist Ratio'].replace(':0','')
    if ':0' in str(row['Primary Care Physicians Ratio']):
      data.at[i,'Primary Care Physicians Ratio'] = row['Primary Care Physicians Ratio'].replace(':0','')
    if ':0' in str(row['Mental Health Provider Ratio']):
      data.at[i,'Mental Health Provider Ratio'] = row['Mental Health Provider Ratio'].replace(':0','')

    if row['Presence of Water Violation'] == "No":
      data.at[i,'Presence of Water Violation'] = 0
    else:
      data.at[i,'Presence of Water Violation'] = 1

    if '(' in str(row['Population']):
      data.at[i,'Population'] = row['Population'].split('(')[0]

  data = data.astype('float')

#There are missing values that are represented by NaNs.
#These NaNs were originally causing my training and test losses to turn out as NaNs.
#My decision was to fill the NaNs as the mean values of each column.
  means = data.mean()
  for col in data.columns:
    data[col]=data[col].fillna(means[col])

#making sure the inputs are normalized, greatly reduced the loss
  x=preprocessing.scale(data.values[:,:-2])
  # x=data.to_numpy()[:,:-2]#make it a numpy array,
  y=data.to_numpy()[:,-2:]
  N,D_in=x.shape
  y=np.reshape(y,(N,2)) #needs to be Nx1, not just of length N

  row=torch.from_numpy(x[-1,:]) #make them pytorch tensors
  row=row.float() #make it single precision

  return row

parser=argparse.ArgumentParser()
parser.add_argument("gpu",type=int,help="Which GPU to use")
parser.add_argument("filename",help="Which CSV file to pull data from")
parser.add_argument("--lr",type=float,default=1e-3,help="Learning rate")
args=parser.parse_args()

filename=args.filename
gpu=args.gpu
learning_rate=args.lr

sample = prepareSample(filename)

D_in=46
H=1000
D_out=2

model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.PReLU(),
    torch.nn.Linear(H, H),
    torch.nn.PReLU(),
    torch.nn.Linear(H, H),
    torch.nn.PReLU(),
    torch.nn.Linear(H, H),
    torch.nn.PReLU(),
    torch.nn.Linear(H, H),
    torch.nn.PReLU(),
    torch.nn.Linear(H, H),
    torch.nn.PReLU(),
    torch.nn.Linear(H, H),
    torch.nn.PReLU(),
    torch.nn.Linear(H, D_out)
    )

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

if torch.cuda.is_available():
  sample=sample.cuda()
  model=model.cuda()

checkpoint = torch.load("covid_model.pickle")
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epochs = checkpoint['epochs']
loss = checkpoint['loss']

prediction = model(sample)
cpu_pred = prediction.cpu()
result = cpu_pred.data.numpy()
print(result)

# Unable successfully use model to obtain predictions for D+14 on a datapoint, Anne Arundel county on 15Apr20.
# Reason: lack of familiarity with the library and its syntax.
