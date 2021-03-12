import numpy as np
import pandas as pd
import sklearn
from sklearn import datasets,svm,metrics
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import *

def reduce_features(solution, features):
    selected_elements_indices = np.where(solution ==1)[0]
    reduced_features = features[:, selected_elements_indices]
    return reduced_features

def classification_accuracy(labels, predictions):
    correct = np.where(labels == predictions)[0]
    accuracy = correct.shape[0]/labels.shape[0]
    return accuracy

def cal_pop_fitness(pop, features, labels, train_indices,val_indices,classifier):
    test_accuracies = np.zeros(pop.shape[0])
    val_accuracies = np.zeros(pop.shape[0])
    idx = 0

    val_pop_pred = np.zeros(shape=(pop.shape[0],val_indices.shape[0]))
    for i,curr_solution in enumerate(pop):

        reduced_features = reduce_features(curr_solution, features)
        train_data = reduced_features[train_indices, :]
        val_data=reduced_features[val_indices,:]

        train_labels = labels[train_indices]
        val_labels=labels[val_indices]
        if classifier=='SVM':
          SV_classifier = sklearn.svm.SVC(kernel='rbf',gamma='scale',C=5000)
          SV_classifier.fit(X=train_data, y=train_labels)
          val_predictions = SV_classifier.predict(val_data)
          val_accuracies[idx] = classification_accuracy(val_labels, val_predictions)
          idx = idx + 1
        elif classifier == 'KNN':
          knn=KNeighborsClassifier(n_neighbors=8)
          knn.fit(train_data,train_labels)
          val_predictions=knn.predict(val_data)
          val_accuracies[idx]=classification_accuracy(val_labels,predictions)
          idx = idx + 1
        else :
          mlp = MLPClassifier()
          mlp.fit(train_data,train_labels)
          val_predictions=mlp.predict(val_data)
          val_accuracies[idx]=classification_accuracy(val_labels,predictions)
          idx = idx + 1
        val_pop_pred[i] = val_predictions
       
    return val_accuracies,val_pop_pred

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

def OBHSA(data_inputs,data_outputs,
          popSize=20,
          HMCR=0.9,
          PAR=0.35,
          classifier="SVM",
          num_generations = 10 #Number of generations in each fold
          ):
    print("\nOPPOSITION-BASED HARMONY SEARCH:\n")
    population_output = np.zeros(shape = (popSize,data_inputs.shape[1],5))
    vector_output = np.zeros(shape=(popSize,4,5)) #4 because acc,pre,rec,f1; and 5 because folds=5
    
    num_samples = data_inputs.shape[0]
    num_feature_elements = data_inputs.shape[1]
    HM_shape=(popSize,num_feature_elements)
    harmony_memory=np.random.randint(low=0,high=2,size=HM_shape)
    NCHV = np.ones((1, num_feature_elements))
    best_outputs = []
    best_opp_outputs = []
    
    kf=KFold(5,True,random_state=1)
    fold=0
    for train_indices,test_val_indices in kf.split(data_inputs):
      print("Fold : ",fold+1)
      val_indices,test_indices=train_test_split(test_val_indices,test_size=0.5,shuffle=True,random_state=8)
      best_test_outputs=[]

      harmony_memory=np.random.randint(low=0,high=2,size=HM_shape)
      opposite_memory=1-harmony_memory
      total_memory=np.concatenate((harmony_memory,opposite_memory),axis=0)
      total_fitness,_ = cal_pop_fitness(total_memory,data_inputs,data_outputs,train_indices,val_indices,classifier)
      fit_ind = np.argpartition(total_fitness, -popSize)[-popSize:]
      harmony_memory=total_memory[fit_ind,:]

      gen_fit = np.array([-1])      
      for currentIteration in range(num_generations):
          NCHV = np.ones((1, num_feature_elements))
          print("Generation : ", currentIteration+1)
        
          fitness,val_pop_preds=cal_pop_fitness(harmony_memory,data_inputs,data_outputs,train_indices,val_indices,classifier)
          best_outputs.append(np.max(fitness))
          print("Best validation result : ", max(best_outputs))

          if max(fitness)>max(gen_fit):
              gen_fit = fitness
              gen_labels = data_outputs[val_indices]
              gen_preds = val_pop_preds   

          for i in range(num_feature_elements):
              ran = np.random.rand()
              if ran < HMCR:
                  index = np.random.randint(0, popSize)
                  NCHV[0, i] = harmony_memory[index, i]
                  pvbran = np.random.rand()
                  if pvbran < PAR:
                      pvbran1 = np.random.rand()
                      result = NCHV[0, i]
                      if pvbran1 < 0.5:
                          result =1-result

              else:
                  NCHV[0, i] = np.random.randint(low=0,high=2,size=1)
          
          new_fitness,_ = cal_pop_fitness(NCHV, data_inputs, data_outputs, train_indices, val_indices,classifier)
          if new_fitness > min(fitness):
              min_fit_idx = np.where(fitness == min(fitness))
              harmony_memory[min_fit_idx, :] = NCHV
              fitness[min_fit_idx] = new_fitness

          opp_NCHV=1-NCHV
          new_opp_fitness,_=cal_pop_fitness(opp_NCHV,data_inputs, data_outputs, train_indices, val_indices,classifier)
          if new_opp_fitness > min(fitness):
              min_fit_idx = np.where(fitness == min(fitness))
              harmony_memory[min_fit_idx, :] = opp_NCHV
              fitness[min_fit_idx] = new_opp_fitness
      
      fitness,_ = cal_pop_fitness(harmony_memory, data_inputs, data_outputs, train_indices,val_indices,classifier)

      best_match_idx = np.where(fitness == np.max(fitness))[0]
      best_match_idx = best_match_idx[0]

      best_solution = harmony_memory[best_match_idx, :]
      best_solution_indices = np.where(best_solution == 1)[0]
      best_solution_num_elements = best_solution_indices.shape[0]
      best_solution_fitness = np.max(fitness)

      #print("best_match_idx : ", best_match_idx)
      #print("best_solution : ", best_solution)
      #print("Selected indices : ", best_solution_indices)
      print("Number of selected elements : ", best_solution_num_elements)

      vector_output[:,:,fold] = get_vector(gen_labels,gen_preds)
      population_output[:,:,fold] = harmony_memory
      
      fold=fold+1
    return population_output,vector_output
