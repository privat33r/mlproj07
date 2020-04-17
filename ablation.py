#Yixin Ye SI486L
#Project07

#The purpose of this project is to utilize deep neural networks to
#build a model to predict COVID-19 cases and deaths two weeks out.
#We will be using a GPU to handle the processing.

import torch
import pandas as pd
import numpy as np
import argparse,os
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

#buildData is a function to read the input file and does the preprocessing of
#the raw csv file into a tensor and then a cuda object for GPU processing
def buildData(filename,drops=[]):
  data=pd.read_csv(filename,low_memory=False)
  data=data.replace(np.float64('-inf'),-1) #to replace -inf from taking log of 0 with -1

#getting rid of string inputs which is not useful for our purporses
  data = data.replace(':0','')
  data = data.drop('State',1)
  data = data.drop('County',1)

#changing the way dates are represented, year is not considered here
  data['month'] = data['date']

  data = data[['FIPS','% Fair or Poor Health','Average Number of Physically Unhealthy Days','Average Number of Mentally Unhealthy Days','% Low Birthweight','% Smokers','% Adults with Obesity','Food Environment Index','% Physically Inactive','% With Access to Exercise Opportunities',
 '% Excessive Drinking','% Driving Deaths with Alcohol Involvement','Chlamydia Rate','Teen Birth Rate','% Uninsured','Primary Care Physicians Ratio','Dentist Ratio','Mental Health Provider Ratio',
 'Preventable Hospitalization Rate','% With Annual Mammogram','% Flu Vaccinated','High School Graduation Rate','% Some College','% Unemployed','% Children in Poverty',
 'Income Ratio','% Single-Parent Households','Social Association Rate','Violent Crime Rate','Injury Death Rate','Average Daily PM2.5','Presence of Water Violation','% Severe Housing Problems','% Drive Alone to Work',
 '% Long Commute - Drives Alone','Population','Density per square mile of land area - Population','date','month','logDCases','logDDeaths','% Increase D-2 Cases','% Increase D-2 Deaths','% Increase D-4 Cases',
 '% Increase D-4 Deaths','Days under SAH','log D+14 Cases','log D+14 Deaths']]

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

  for col in drops:
    data = data.drop(col,1)

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

# # This is to add noise but it didn't help bring down the test loss.
#   noise = np.random.normal(0, 0.1, [x.shape[0],x.shape[1]])
#   x = x + noise

#code to find NaNs in the input array
  # print((x_train != x_train).any())
  # nans = np.argwhere(np.isnan(x))
  # print(nans,len(nans))

  x_train,x_test,y_train,y_test=train_test_split(x,y)

  x_train=torch.from_numpy(x_train) #make them pytorch tensors
  y_train=torch.from_numpy(y_train)
  x_test=torch.from_numpy(x_test)
  y_test=torch.from_numpy(y_test)

#single precision values work faster/better than doubles
  x_train=x_train.float() #make it single precision
  y_train=y_train.float() #make it single precision
  x_test=x_test.float() #make it single precision
  y_test=y_test.float() #make it single precision
  return x_train,x_test,y_train,y_test

parser=argparse.ArgumentParser()
parser.add_argument("gpu",type=int,help="Which GPU to use")
parser.add_argument("filename",help="Which CSV file to pull data from")
parser.add_argument("--lr",type=float,default=1e-3,help="Learning rate")
parser.add_argument("--epochs",type=int,default=1e4,help="Epochs to train")
args=parser.parse_args()

gpu=args.gpu
assert gpu>=0 and gpu<4
os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu)

filename=args.filename

learning_rate=args.lr

epochs=int(args.epochs)

for col in ['FIPS','% Fair or Poor Health','Average Number of Physically Unhealthy Days','Average Number of Mentally Unhealthy Days','% Low Birthweight','% Smokers','% Adults with Obesity','Food Environment Index','% Physically Inactive','% With Access to Exercise Opportunities',
 '% Excessive Drinking','% Driving Deaths with Alcohol Involvement','Chlamydia Rate','Teen Birth Rate','% Uninsured','Primary Care Physicians Ratio','Dentist Ratio','Mental Health Provider Ratio',
 'Preventable Hospitalization Rate','% With Annual Mammogram','% Flu Vaccinated','High School Graduation Rate','% Some College','% Unemployed','% Children in Poverty',
 'Income Ratio','% Single-Parent Households','Social Association Rate','Violent Crime Rate','Injury Death Rate','Average Daily PM2.5','Presence of Water Violation','% Severe Housing Problems','% Drive Alone to Work',
 '% Long Commute - Drives Alone','Population','Density per square mile of land area - Population','date','month','logDCases','logDDeaths','% Increase D-2 Cases','% Increase D-2 Deaths','% Increase D-4 Cases',
 '% Increase D-4 Deaths','Days under SAH']:

  x_train,x_test,y_train,y_test=buildData(filename,[col])

  # N is batch size; D_in is input dimension;
  # H is hidden dimension; D_out is output dimension.
  D_in=x_train.shape[1]
  H=1000
  D_out=y_train.shape[1]

  #x = torch.randn(N, D_in)
  #y = torch.randn(N, D_out)

  # Use the nn package to define our model and loss function.
  #I used PReLU instead of ReLU in the example.
  #Originally intended to manage gradient explosions but found out that it
  # consistently performs better than ReLU or LeakyReLU.
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

  if torch.cuda.is_available():
    x_train=x_train.cuda()
    y_train=y_train.cuda()
    x_test=x_test.cuda()
    y_test=y_test.cuda()
    model=model.cuda()

  loss_fn = torch.nn.MSELoss(reduction='sum').cuda()

  # Use the optim package to define an Optimizer that will update the weights of
  # the model for us. Here we will use Adam; the optim package contains many other
  # optimization algoriths. The first argument to the Adam constructor tells the
  # optimizer which Tensors it should update.
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

  try: # if another model exists
    checkpoint = torch.load("covid_model.pickle")
    saved_min_testloss = checkpoint['testloss'].item()
    # print("Earlier model found, saved mininum test loss:",saved_min_testloss)
  except:
    saved_min_testloss = float("inf")
    # print("Earlier model not found.")

  # Early stop variable init
  min_testloss = float("inf")
  early_stopcount = 0
  early_stopped = False

  for t in range(epochs):
    # Forward pass: compute predicted y by passing x to the model.
    y_train_pred = model(x_train)
    # Compute and print loss.
    loss = loss_fn(y_train_pred, y_train)

    y_test_pred = model(x_test)
    testloss = loss_fn(y_test_pred, y_test)
    # Before the backward pass, use the optimizer object to zero all of the
    # gradients for the variables it will update (which are the learnable
    # weights of the model). This is because by default, gradients are
    # accumulated in buffers( i.e, not overwritten) whenever .backward()
    # is called. Checkout docs of torch.autograd.backward for more details.
    optimizer.zero_grad()

    # Backward pass: compute gradient of the loss with respect to model
    # parameters
    loss.backward()

    # read online to prevent exploding gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), .25)
    for p in model.parameters():
      p.data.add_(-learning_rate, p.grad.data)

    # Calling the step function on an Optimizer makes an update to its
    # parameters
    optimizer.step()
    if t % 100 == 99:
      # print(t, loss.item(),testloss.item())
      # print(t,end=';')
      pass

  #HERE IS THE EARLY STOPPING ALGORITHM
  #Stops and saves when testloss doesn't improve in 1000 epochs (instead of 10)
    if testloss.item() < min_testloss:
      min_testloss = testloss.item()
      if min_testloss < saved_min_testloss:
        torch.save({
                'epochs': epochs,
                'model_sequence': str(model),
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'testloss': testloss},"covid_model.pickle")
        saved_min_testloss = min_testloss
      early_stopcount = 0

    else:
      early_stopcount += 1
      if early_stopcount >= 1000:
        # print("Early stopping... Training stopped and best model saved to covid_model.pickle.")
        print("Minimum Test Loss:",min_testloss,col)
        early_stopped = True
        break

  if not early_stopped:
    # print("Epochs limit reached, training stopped and best model saved to covid_model.pickle.")
    print("Minimum Test Loss:",min_testloss,col)
