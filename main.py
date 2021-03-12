#%cd "/content/drive/MyDrive/Hybrid Optimization"

import numpy as np
import pandas as pd
from MantaRayOA import MantaRayOA, metrics
from OBHSA import OBHSA
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--csv_name', type=str, required = True, help='Name of csv file- Example: SpectEW.csv')
parser.add_argument('--csv_header', type=str, default = 'no', help='Does csv file have header?: yes/no')
parser.add_argument('--generations', type=int, default = 20, help='Number of Generations to run the Genetic Algorithm')
parser.add_argument('--popSize', type=int, default = 20, help='Population Size to be used in MRFO and OBHSA')
args = parser.parse_args()


root = "./"
if root[-1]!='/':
    root+='/'
csv_path = args.csv_name

if args.csv_header=='yes':
    df = np.asarray(pd.read_csv(root+csv_path))
else:
    df = np.asarray(pd.read_csv(root+csv_path,header=None))
data = df[:,0:-1]
target = df[:,-1]

pop_size = args.popSize
num_gen = args.generations

pop1,matrix1 = MantaRayOA(data,target, popSize = pop_size, num_generations=num_gen)
pop2,matrix2 = OBHSA(data,target, popSize = pop_size, num_generations=num_gen)

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
Fold = 5
kfold = KFold(Fold, True, random_state = 1)
train_id = []
test_id = []
for train_index, test_index in kfold.split(data):
  train_id.append(train_index)
  test_id.append(test_index)

import sklearn.svm
clf = sklearn.svm.SVC(kernel='rbf',gamma='scale',C=5000)

unique=np.unique(target)
num_classes=unique.shape[0]
classes=[]
for i in range(num_classes):
    classes.append('Class'+str(i+1))

for fold in range(5):
  pop1_f = pop1[:,:,fold]
  pop2_f = pop2[:,:,fold]
  new_pop = np.concatenate((pop1_f,pop2_f),axis=0)
  mat1_f = matrix1[:,:,fold]
  mat2_f = matrix2[:,:,fold]
  new_mat = np.concatenate((mat1_f,mat2_f),axis=0)

  score = np.zeros(shape=(new_mat.shape[0],))
  for i,m in enumerate(new_mat):
    sum = 0
    for e in m:
      sum+=2**(e)
    score[i]=sum
  score = score/new_mat.shape[0]

  scored_pop = np.zeros(shape=new_pop.shape)
  for i,p in enumerate(new_pop):
    p[np.where(p==1)[0]]=score[i]
    scored_pop[i] = p

  feat_imp = np.sum(scored_pop,axis=0)
  mean = np.mean(feat_imp)
  final_feat = np.zeros(shape = feat_imp.shape)
  final_feat[np.where(feat_imp>mean)[0]] = 1
  cols = np.flatnonzero(final_feat)

  trainX, trainy= np.asarray(data[train_id[fold]]), np.asarray(target[train_id[fold]])
  testX, testy  = np.asarray(data[test_id[fold]]), np.asarray(target[test_id[fold]])
  X_train = trainX[:,cols]
  clf.fit(X_train,trainy)
  X_test = testX[:,cols]
  preds = clf.predict(X_test)
  print("Fold {}:\n".format(fold+1))
  metrics(testy,preds,classes)
