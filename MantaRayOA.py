import numpy as np
import pandas as pd
import random
import math,time,sys
from matplotlib import pyplot
from datetime import datetime
import sklearn.svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import csv
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import *
from sklearn.neural_network import MLPClassifier
import time
import warnings

tic=time.time()
warnings.filterwarnings("ignore")

def metrics(labels,predictions,classes):
    print("Classification Report:")
    print(classification_report(labels, predictions, target_names = classes))
    matrix = confusion_matrix(labels, predictions)
    print("Confusion matrix:")
    print(matrix)
    print("Classwise Accuracy :{}".format(matrix.diagonal()/matrix.sum(axis = 1)))
    print("Balanced Accuracy Score: ",balanced_accuracy_score(labels,predictions))
    
#ELM CLASSIFIER
class ELM (BaseEstimator, ClassifierMixin):

    """
    3 step model ELM
    """

    def __init__(self,hid_num,a=1):
        """
        Args:
        hid_num (int): number of hidden neurons
        a (int) : const value of sigmoid funcion
        """
        self.hid_num = hid_num
        self.a = a

    def _sigmoid(self, x):
        """
        sigmoid function
        Args:
        x (float): input
        Returns:
        float: output of sigmoid
        """
        sigmoid_range = 34.538776394910684
        x = np.clip(x, -sigmoid_range, sigmoid_range)
        return 1 / (1 + np.exp(-self.a * x))

    def _add_bias(self, X):
        """add bias to list
        Args:
        x_vs [[float]] Array: vec to add bias
        Returns:
        [float]: added vec
        Examples:
        >>> e = ELM(10, 3)
        >>> e._add_bias(np.array([[1,2,3], [1,2,3]]))
        array([[1., 2., 3., 1.],
               [1., 2., 3., 1.]])
        """

        return np.c_[X, np.ones(X.shape[0])]

    def _ltov(self, n, label):
        """
        trasform label scalar to vector
        Args:
        n (int) : number of class, number of out layer neuron
        label (int) : label
        Exmples:
        >>> e = ELM(10, 3)
        >>> e._ltov(3, 1)
        [1, -1, -1]
        >>> e._ltov(3, 2)
        [-1, 1, -1]
        >>> e._ltov(3, 3)
        [-1, -1, 1]
        """
        return [-1 if i != label else 1 for i in range(1, n + 1)]

    def fit(self, X, y):
        """
        learning
        Args:
        X [[float]] array : feature vectors of learnig data
        y [[float]] array : labels of leanig data
        """
        # number of class, number of output neuron
        self.out_num = max(y)

        if self.out_num != 1:
            y = np.array([self._ltov(self.out_num, _y) for _y in y])

        # add bias to feature vectors
        X = self._add_bias(X)

        # generate weights between input layer and hidden layer
        np.random.seed()
        self.W = np.random.uniform(-1., 1.,
                                   (self.hid_num, X.shape[1]))

        # find inverse weight matrix
        _H = np.linalg.pinv(self._sigmoid(np.dot(self.W, X.T)))

        self.beta = np.dot(_H.T, y)

        return self

    def predict(self, X):
        """
        predict classify result
        Args:
        X [[float]] array: feature vectors of learnig data
        Returns:
        [int]: labels of classification result
        """
        _H = self._sigmoid(np.dot(self.W, self._add_bias(X).T))
        y = np.dot(_H.T, self.beta)

        if self.out_num == 1:
            return np.sign(y)
        else:
            return np.argmax(y, 1) + np.ones(y.shape[0])



        
################################################################################################################3
def sigmoid1(gamma):     #convert to probability
	if gamma < 0:
		return 1 - 1/(1 + math.exp(gamma))
	else:
		return 1/(1 + math.exp(-gamma))

def sigmoid2(gamma):
	gamma /= 2
	if gamma < 0:
		return 1 - 1/(1 + math.exp(gamma))
	else:
		return 1/(1 + math.exp(-gamma))
		
def sigmoid3(gamma):
	gamma /= 3
	if gamma < 0:
		return 1 - 1/(1 + math.exp(gamma))
	else:
		return 1/(1 + math.exp(-gamma))

def sigmoid4(gamma):
	gamma *= 2
	if gamma < 0:
		return 1 - 1/(1 + math.exp(gamma))
	else:
		return 1/(1 + math.exp(-gamma))


def Vfunction1(gamma):
	return abs(np.tanh(gamma))

def Vfunction2(gamma):
	val = (math.pi)**(0.5)
	val /= 2
	val *= gamma
	val = math.erf(val)
	return abs(val)

def Vfunction3(gamma):
	val = 1 + gamma*gamma
	val = math.sqrt(val)
	val = gamma/val
	return abs(val)

def Vfunction4(gamma):
	val=(math.pi/2)*gamma
	val=np.arctan(val)
	val=(2/math.pi)*val
	return abs(val)


def fitness(position,classifier,omega,trainX,testX,trainy,testy):
    cols=np.flatnonzero(position)
    val=1
    if np.shape(cols)[0]==0:
            return val

    classifier=classifier.upper()
    # clf = RandomForestClassifier(n_estimators=300)

    # clf=MLPClassifier( alpha=0.01, max_iter=1000) #hidden_layer_sizes=(1000,500,100)
    #cross=3
    #test_size=(1/cross)
    #X_train, X_test, y_train, y_test = train_test_split(trainX, trainy,  stratify=trainy,test_size=test_size)

    if classifier == 'MLP':
        clf = MLPClassifier(activation = 'tanh',solver = 'lbfgs', alpha=0.01, hidden_layer_sizes=(1000, 500, 100), max_iter=1000, random_state=1)
    elif classifier == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=5)
    elif classifier == 'ELM':
        clf = ELM(hid_num = 50)
    else:
        clf = sklearn.svm.SVC(kernel='rbf',gamma='scale',C=5000)

    train_data=trainX[:,cols]
    test_data=testX[:,cols]
    clf.fit(train_data,trainy)
    val=1-clf.score(test_data,testy)

    #in case of multi objective  []
    set_cnt=sum(position)
    set_cnt=set_cnt/np.shape(position)[0]
    val=omega*val+(1-omega)*set_cnt
    return val

def onecount(position):
	cnt=0
	for i in position:
		if i==1.0:
			cnt+=1
	return cnt


def allfit(population,classifier,omega,trainX,testX,trainy,testy):
	x=np.shape(population)[0]
	acc=np.zeros(x)
	for i in range(x):
		acc[i]=fitness(population[i],classifier,omega,trainX,testX,trainy,testy)     
		#print(acc[i])
	return acc

def initialize(popSize,dim):
	population=np.zeros((popSize,dim))
	minn = 1
	maxx = math.floor(0.8*dim)
	if maxx<minn:
		minn = maxx
	
	for i in range(popSize):
		random.seed(i**3 + 10 + time.time() ) 
		no = random.randint(minn,maxx)
		if no == 0:
			no = 1
		random.seed(time.time()+ 100)
		pos = random.sample(range(0,dim-1),no)
		for j in pos:
			population[i][j]=1
		
		#print(population[i])  
		
	return population

def toBinary(population,popSize,dimension,oldPop):

	for i in range(popSize):
		for j in range(dimension):
			temp = Vfunction3(population[i][j])

			# if temp > 0.5: # sfunction
			# 	population[i][j] = 1
			# else:
			# 	population[i][j] = 0

			if temp > 0.5: # vfunction
				population[i][j] = (1 - oldPop[i][j])
			else:
				population[i][j] = oldPop[i][j]
	return population
    
def classification_accuracy(labels, predictions):
    correct = np.where(labels == predictions)[0]
    accuracy = correct.shape[0]/labels.shape[0]
    return accuracy

def get_vector(labels,pop_preds):
    vector = np.zeros(shape=(pop_preds.shape[0],4))
    for i in range(pop_preds.shape[0]):
        preds = pop_preds[i]
        acc = classification_accuracy(labels,preds)
        pre = precision_score(labels,preds,average="macro")
        rec = recall_score(labels,preds,average="macro")
        f1 = f1_score(labels,preds,average="macro")
        vector[i] = np.array([acc,pre,rec,f1])
    return vector

def check_pop(pop):
    com = np.zeros(shape=pop.shape[0])
    arr = (pop==com)
    if False in arr:
        return False
    else:
        return True
    
def MantaRayOA(data,label,omega = 0.85, #weightage for no of features and accuracy
               popSize = 5,
               max_iter = 30,
               S = 2,
               num_generations = 1,
               classifier = 'SVM',
               ):
    
    print("\nMANTA RAY FORAGING OPTIMIZATION:\n")
    population_output = np.zeros(shape=(popSize,data.shape[1],5))
    vector_output = np.zeros(shape=(popSize,4,5)) #4 because acc,pre,rec,f1; and 5 because folds=5
    (a,b)=np.shape(data)
    print("Number of images: ",a)
    print("Number of features: ",b)
    dimension = np.shape(data)[1]

    ##NO OF CLASSES
    unique=np.unique(label)
    num_classes=unique.shape[0]
    classes=[]
    for i in range(num_classes):
        classes.append('Class'+str(i+1))

    global_count = 0
    accuracy_list = []
    features_list = []

    if classifier == 'MLP':
        clf = MLPClassifier(activation = 'tanh',solver = 'lbfgs', alpha=0.01, hidden_layer_sizes=(1000, 500, 100), max_iter=1000, random_state=1)
    elif classifier == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=5)
    elif classifier == 'ELM':
        clf = ELM(hid_num = 50)
    else:
        clf = sklearn.svm.SVC(kernel='rbf',gamma='scale',C=5000)


    ##### KFold Validations ########

    from numpy import array
    from sklearn.model_selection import KFold
    from sklearn.model_selection import train_test_split
    Fold = 5

    kfold = KFold(Fold, True, random_state = 1)

    f = 0

    for train_index, test_index in kfold.split(data):
      best_accuracy = -1
      best_no_features = -1
      average_accuracy = 0
      trainX, trainy= np.asarray(data[train_index]), np.asarray(label[train_index])
      testX, testy  = np.asarray(data[test_index]), np.asarray(label[test_index])
      for train_iteration in range(num_generations):
        print("\nFold: {}, Iteration: {}".format(f+1,train_iteration+1))

        clf.fit(trainX,trainy)
        val=clf.score(testX,testy)
        whole_accuracy = val
        #print("Total Acc: ",val)

        # for population_iteration in range(2):
        global_count += 1
        #print('global: ',global_count)

        x_axis = []
        y_axis = []

        population = initialize(popSize,dimension)
        # print(population)

        start_time = datetime.now()
        fitList = allfit(population,classifier,omega,trainX,testX,trainy,testy)
        bestInx = np.argmin(fitList)
        fitBest = min(fitList)
        Mbest = population[bestInx].copy()
        for currIter in range(max_iter):
          popnew = np.zeros((popSize,dimension))
          x_axis.append(currIter)
          y_axis.append(min(fitList))
          for i in range(popSize):
            random.seed(time.time() + 10.01)
            randNo = random.random()
            if randNo<0.5:
              #chain foraging
              random.seed(time.time())
              r = random.random()
              alpha = 2*r*(abs(math.log(r))**0.5)
              if i == 1:
                popnew[i] = population[i] + r * (Mbest - population[i]) + alpha*(Mbest - population[i])
              else:
                popnew[i] = population[i] + r * (population[i-1] - population[i]) + alpha*(Mbest - population[i])
            else:
              #cyclone foraging
              cutOff = random.random()
              r = random.random()
              r1 = random.random()
              beta = 2 * math.exp(r1 * (max_iter - currIter + 1) / max_iter) * math.sin(2 * math.pi * r1)
              if currIter/max_iter < cutOff:
                # exploration
                Mrand = np.zeros(np.shape(population[0]))
                no = random.randint(1,max(int(0.1*dimension),2))
                random.seed(time.time()+ 100)
                pos = random.sample(range(0,dimension-1),no)
                for j in pos:
                  Mrand[j] = 1

                if i==1 :
                  popnew[i] = Mrand + r * (Mrand - population[i]) + beta * (Mrand - population[i])
                else:
                  popnew[i] = Mrand + r * (population[i-1] - population[i]) + beta * (Mrand - population[i])
              else:
                # exploitation
                if i == 1:
                  popnew[i] = Mbest + r * (Mbest - population[i]) + beta * (Mbest - population[i])
                else:
                  popnew[i] = Mbest + r * (population[i-1] - population[i]) + beta * (Mbest - population[i])

          # print(popnew)
          
          popnew = toBinary(popnew,popSize,dimension,population)
          popnewTemp = popnew.copy()
          #compute fitness for each individual
          fitList = allfit(popnew,classifier,omega,trainX,testX,trainy,testy)
          if min(fitList)<fitBest :
            bestInx = np.argmin(fitList)
            fitBest = min(fitList)
            Mbest = popnew[bestInx].copy()
          # print(fitList,fitBest)

          #somersault foraging
          for i in range(popSize):
            r2 = random.random()
            random.seed(time.time())
            r3 = random.random()
            popnew[i] = popnew[i] + S * (r2*Mbest - r3*popnew[i])

          popnew = toBinary(popnew,popSize,dimension,popnewTemp)
          #compute fitness for each individual
          fitList = allfit(popnew,classifier,omega,trainX,testX,trainy,testy)
          if min(fitList)<fitBest :
            bestInx = np.argmin(fitList)
            fitBest = min(fitList)
            Mbest = popnew[bestInx].copy()
          # print(fitList,fitBest)

          population = popnew.copy()


        time_required = datetime.now() - start_time

        # pyplot.plot(x_axis,y_axis)
        # pyplot.xlim(0,max_iter)
        # pyplot.ylim(max(0,min(y_axis)-0.1),min(max(y_axis)+0.1,1))
        # pyplot.show()


        output = Mbest.copy()
        #print("This is output:",output)

        #test accuracy
        cols=np.flatnonzero(output)
        #print("This is cols:",cols)
        X_test=testX[:,cols]
        X_train=trainX[:,cols]
        #print("X_train.shape:",X_train.shape)

        clf.fit(X_train,trainy)
        val=clf.score(X_test, testy )
        print(val,onecount(output))

        accuracy_list.append(val)
        features_list.append(onecount(output))
        gen_preds = np.zeros(shape=(popSize,testy.shape[0]))
        if ( val == best_accuracy ) and ( onecount(output) < best_no_features ):
          best_accuracy = val
          best_no_features = onecount( output )
          best_time_req = time_required
          best_whole_accuracy = whole_accuracy
          for i in range(popnew.shape[0]):
              if check_pop(popnew[i]):
                  popnew[i] = initialize(popSize,dimension)[i]
              cols = np.flatnonzero(popnew[i])
              X_train = trainX[:,cols]
              clf.fit(X_train,trainy)
              X_test = testX[:,cols]
              preds = clf.predict(X_test)
              gen_preds[i] = preds

        if val > best_accuracy :
          best_accuracy = val
          best_no_features = onecount( output )
          best_time_req = time_required
          best_whole_accuracy = whole_accuracy
          for i in range(popnew.shape[0]):
              if check_pop(popnew[i]):
                  popnew[i] = initialize(popSize,dimension)[i]
              cols = np.flatnonzero(popnew[i])
              X_train = trainX[:,cols]
              clf.fit(X_train,trainy)
              X_test = testX[:,cols]
              preds = clf.predict(X_test)
              gen_preds[i] = preds

        predictions = clf.predict(X_test)

        print('Best Accuracy: {}, Best number of Features: {}'.format(best_accuracy, best_no_features))
                      
      vector_output[:,:,f] = get_vector(testy,gen_preds)
      population_output[:,:,f]=popnew.copy()
      f += 1
    return population_output, vector_output
