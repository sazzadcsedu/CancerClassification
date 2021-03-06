#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 00:35:03 2020

@author: russell
"""

import pandas as pd
import numpy as np
from SupervisedAlgorithm import Logistic_Regression_Classifier
from SupervisedAlgorithm import Ridge_Regression_Classifier
from SupervisedAlgorithm  import RandomForest_Classifier
from SupervisedAlgorithm  import MultinomialNB_Classifier
from SupervisedAlgorithm  import SVM_Classifier
from SupervisedAlgorithm  import LDA_Classifier
from SupervisedAlgorithm  import KNN_Classifier
from SupervisedAlgorithm  import TF_IDF
from SupervisedAlgorithm import  Performance
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt
from sklearn.svm import SVC

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import chi2
from sklearn import preprocessing

import random

number_of_features = 150 # 10 #7129 # 5


def write_to_CSV(X_data, Y_label):
    import csv
     #persons=[('Lata',22,45),('Anil',21,56),('John',20,60)]
    csvfile=open("/Users/russell/Downloads/hello.csv",'w', newline='')
   
    obj=csv.writer(csvfile)
    
    
    data= []
    label = []
    '''
    for i in range (len(X_data)):
        row_data= []
        for j in range(len(X_data[i])):
            row_data.append(X_data[i][j])
            label.append(Y_label[value])
    '''
        
    for element in zip((i for i in X_data), Y_label):
        obj.writerow(element)
        
    csvfile.close()
    
    
    
def write_to_text(data, label, filename):
    f = open(filename,"w+")
    
    for i in range(len(data)):
        row = ''
        for j in range(len(data[i])): 
            value = data[i][j];
            #review = data[i].replace("\n","")
            row = row + str(round(value,4)) + ","
        row = row + str(label[i])
        
        f.write(row)
        f.write("\n")
    
    
def pre_process_data(data):
    
    min_max_scaler = preprocessing.MinMaxScaler()
    data = min_max_scaler.fit_transform(data)
    return data


def slice_data(data,indices):
    
    new_data = []
    for i in range(len(data)):
        temp = [0] * len(indices)
        new_data.append(temp)
    
    for i in range(len(data)):
        k = 0
        for j in indices:
            #print("+ ", i,j)
            new_data[i][k] = data[i][j]
            k +=1
    
    
    return new_data


#---------FeatureSelection------
def feature_selection_chi2(data, label):
    
    print("Chi2 Feature Selection: ")
    
    features = SelectKBest(score_func=chi2, k=number_of_features)
    features.fit(data, label)
    
    gene_indices = []
    mask = features.get_support() #list of booleans

    for i in range(len(mask)):
        if mask[i] == 1:
            gene_indices.append(i)
            
    '''
    for i in gene_indices:
        print(i)
    '''
    
            
    return features, gene_indices
    
    
def feature_selection_f_classif(data, label):
    features = SelectKBest(score_func=f_classif, k=number_of_features)
    features.fit(data, label)
    
    gene_indices = []
    mask = features.get_support() #list of booleans

    for i in range(len(mask)):
        if mask[i] == 1:
            gene_indices.append(i)
    
    print("\nANOVA f_classif Feature Selection")  
    
    '''
    for i in gene_indices:
        print(i)    
    '''
        
    return features, gene_indices

def feature_selection_mutual_info_classif(data, label):
    features = SelectKBest(score_func=mutual_info_classif, k=number_of_features)
    features.fit(data, label)
    
    gene_indices = []
    mask = features.get_support() #list of booleans

    for i in range(len(mask)):
        if mask[i] == 1:
            gene_indices.append(i)
            
    print("\nMutual_info feature selection")  
    
    
    return features, gene_indices



#------------RC-------
def feature_selection_classification(data, label):
    
    from scipy.stats import pearsonr
    from scipy import stats
    from scipy.stats import kendalltau
    
    
    #--------- Feature Selection Step --------
    
    # Chi-square feature selection 
    
    '''
    features_chi2, gene_indices_chi2 = feature_selection_chi2(data, label)
    features  = features_chi2.transform(data)
    gene_indices = gene_indices_chi2
    '''
    
    
    # ANOVA feature selection 
    features_f_classif, gene_indices_f_classif = feature_selection_f_classif(data, label)
    features  = features_f_classif.transform(data)
    gene_indices = gene_indices_f_classif
    
    
    # Mutual Info feature selection 
    '''
    features_mutual_info_classif, gene_indices_mutual_info_classif = feature_selection_mutual_info_classif(data, label)
    features  = features_mutual_info_classif.transform(data)
    gene_indices = gene_indices_mutual_info_classif
    '''
    
    print(len(features))
    
    #print(features_chi2)
    
    
    attributes = []
    
    for i in range(len(features[0])):
        attribute = [0] * len(features)
        attributes.append(attribute)
        
    
    for i in range(len(features)):
        for j in range(len(features[0])):
            attributes[j][i] = features[i][j]
        
        
    
    print(len(attributes))
   
    
    
    #--------- Feature Reduction Step --------
 
    remove_list_pearson = {}
    remove_list_spearman = {}
    remove_list_kendalltau = {}
    
    for i in range(len(attributes)):
        
        if i in remove_list_pearson or i in remove_list_spearman or i in remove_list_kendalltau:
            continue
            
        for j in range(i + 1, len(attributes)):
            
            if i == j:
                continue
            #print(i,j)
            
                 
            c, p = pearsonr(attributes[i],attributes[j])
            
            if abs(c) > 0.47: # .40: # > 0.70:
                remove_list_pearson[j] = 1
         
         
            #print("Pearson: ", c, p )
         
            rho, pval = stats.spearmanr(attributes[i],attributes[j])
         
            if abs(rho) > 0.50: # .40: # > 0.70:
                remove_list_spearman[j] = 1
                
            coef, p = kendalltau(attributes[i], attributes[j])
            
            if abs(coef) > 0.34: # .40: # > 0.70:
                remove_list_kendalltau[j] = 1
            
            

    #--------- Classification-Pearson --------  
    non_redundant_attributes = []
    for i in range(len(attributes)):
        
        if i not in remove_list_pearson:
            non_redundant_attributes.append(gene_indices[i])

    
    print("\n\nPearson: ")
    print("Non_redundant_attributes: ", len(non_redundant_attributes))
    
    pearsonr_data = slice_data(data,non_redundant_attributes)
    classify(pearsonr_data, label)
    
    
     #--------- Classification-Spearman --------  
    non_redundant_attributes = []
    for i in range(len(attributes)):
        
        if i not in remove_list_spearman:
            non_redundant_attributes.append(gene_indices[i])
    
    print("\n\n Spearman: ")
    print("non_redundant_attributes: ", len(non_redundant_attributes))
    
    spearman_data = slice_data(data,non_redundant_attributes)
    
    write_to_text(spearman_data, label, "/Users/russell/Downloads/hello_spearman.txt")
    
    classify(spearman_data, label)
    



    #--------- Classification-kendal --------  
    non_redundant_attributes = []
    for i in range(len(attributes)):
        
        if i not in remove_list_kendalltau:
            non_redundant_attributes.append(gene_indices[i])

    
    print("\n\nKendall tau: ")
    print("non_redundant_attributes: ", len(non_redundant_attributes))
    
    kendalltau_data = slice_data(data,non_redundant_attributes)
    classify(kendalltau_data, label)



#-------------KBest-----------
def feature_selection(data, label):
    #data = pre_process_data(data)
    
    features_chi2, gene_indices_chi2 = feature_selection_chi2(data, label)
    features_f_classif, gene_indices_f_classif = feature_selection_f_classif(data, label)
    features_mutual_info_classif, gene_indices_mutual_info_classif = feature_selection_mutual_info_classif(data, label)
    
    features_chi2  = features_chi2.transform(data)
    features_f_classif  = features_f_classif.transform(data)
    features_mutual_info_classif  = features_mutual_info_classif.transform(data)
    
    get_top_features_ensembles(data, label, gene_indices_chi2, gene_indices_f_classif, gene_indices_mutual_info_classif)
    
    #return 
    
    print("\nChi2 :" , gene_indices_chi2)
    classify(features_chi2, label)

    #return
    
    print("\nf_classif :", gene_indices_f_classif)
    classify(features_f_classif, label)
    
    print("\nmutual_info_classif :", gene_indices_mutual_info_classif)
    classify(features_mutual_info_classif, label)
       

'''
def SelectKBest_feature(data, label):
    #min_max_scaler = preprocessing.MinMaxScaler()
    #data = min_max_scaler.fit_transform(data)
    
    data = pre_process_data(data)
    
    #test_1 = SelectKBest(score_func=chi2, k=5)
    test_1 = SelectKBest(score_func=f_classif, k=6)
    #test_1 = SelectKBest(score_func= mutual_info_classif, k=5)
    model = test_1.fit(data, label)
    mask = model.get_support() #list of booleans
    new_features = [] # The list of your K best features

    for i in range(len(mask)):
        if mask[i] == 1:
            new_features.append(i)
    
    #test_2 = SelectKBest(score_func=f_classif, k=5)
    #test = SelectKBest(score_func= mutual_info_classif, k=100)
    
    #print(test_1[:4])
    
    print(new_features, len(new_features))
    
    #return
    #test = test_1  +  test_2
    
    #print(test)
    
    #f_regression
    model = test_1.fit(data, label)
    
    np.set_printoptions(precision=3)
    print(model.scores_)
    
    features = model.transform(data)
#    Summarize selected features
    print(features[0:5,:])
    
    classify(features, label)
    
'''

def classify(data, label):
    
    #print(len(data))
    
    c = list(zip(data, label))
    random.shuffle(c)
    data, label  = zip(*c)
   # print("Total Data all directory: ",len(label), type(label), label[1])
    
    
    
    label = np.asarray(label)
    
    data = np.asarray(data)
    
    num_of_fold = 10
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=num_of_fold)
    
    #X_train_english, Y_train_english  = get_sampled_data(X_train_english, Y_train_english)
    
    
    
    total_bengali_f1 = 0
    total_english_f1 = 0
    total_bengali_acc = 0
    total_english_acc = 0
    total_bengali_precison = 0
    total_english_precison = 0
    total_bengali_recall = 0
    total_english_recall = 0
    
    #prediction_english_all = []
    #prediction_bengali_all = []

    

    for train_index, test_index in kf.split(data, label):

        classifier = Logistic_Regression_Classifier() 
        
        #classifier = Ridge_Regression_Classifier()
        
        
        #classifier = RandomForest_Classifier()
        
        classifier = SVM_Classifier()
        #classifier = KNN_Classifier()
        
        #classifier = SGD_Classifier()
        
        
        #classifier = LDA_Classifier()
        
        #classifier = MultinomialNB_Classifier()
        
        
        label_train, label_test = label[train_index], label[test_index]
        data_train, data_test = data[train_index], data[test_index]
       
        prediction = classifier.predict(data_train, label_train, data_test)
    
        performance = Performance() 
    
        
        
        conf_matrix,precision,  recall, f1_score, acc = performance.get_results(label_test, prediction)
    
        #print("Bengali F1#",f1_score)
        total_bengali_f1 += f1_score
        total_bengali_acc += acc
        total_bengali_precison  += precision
        total_bengali_recall += recall
        
        #print
    
    
    print("Overall")
    #print("\n\nEnglish P-R-F1-Acc: ",round(total_english_precison/num_of_fold,3), round(total_english_recall/num_of_fold,3)   , round(total_english_f1/num_of_fold, 3), round(total_english_acc/num_of_fold,3))
    print("\Total: P:R:F1-Acc", round(total_bengali_precison/num_of_fold,3), round(total_bengali_recall/num_of_fold,3) , round(total_bengali_f1/num_of_fold,3), round(total_bengali_acc/num_of_fold, 3))
    #print("\n\n------")
    
    
    
def custom_data():
    ##feature = "1 1 1 1 1 0 0 0 0 1 1 1 0 0 1 1 0 1 1 1 1 0 1 0 1 1 1 0 1"
    feature = "[1 1 1 1 1 0 1 1 1 1 0 1 1 1 0 1 1 1 0 0 1 1 1 0 1 0 1 0 0]"
    feature = "[1 1 1 1 0 1 1 0 1 0 1 0 0 0 1 1 1 0 1 1 1 0 0 0 1 1 0 0 1]"
    #feature = "[1 1 1 0 0 0 0 0 0 1 0 1 1 1 0 0 1 1 1 1 0 0 1 1 0 1 0 0 0]"
    #feature = "[0 0 1 0 1 1 1 1 0 0 1 0 1 0 1 1 0 1 1 0 1 0 1 1 0 1 1 1 1]"
    
    
    #feature = "[1 1 0 0 0 1 0 1 1 0 0 0 1 1 0 0 1 1 1 1 1 0 1 0 1 1 0 0 1]"
    ##feature = "[1 0 1 0 1 0 0 1 0 1 0 1 1 0 1 0 0 1 1 0 0 0 0 1 1 0 1 0 1]"
    feature = "[1 1 0 0 0 1 0 1 0 1 0 1 0 0 1 0 1 0 1 0 1 0 1 1 0 0 1 1 1]"
    feature = "[1 1 0 1 1 0 0 0 0 1 0 1 0 0 0 1 1 0 1 1 0 1 1 1 1 0 1 0 0]"
    feature = "[1 1 1 1 0 1 1 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 0 1 1 1 1 0]"
   
    tokens = feature.split()
    
    print (tokens)
    
    input_data = pd.read_csv('/Users/russell/Downloads/Alon-Spearman.csv')
    
    
    data = []
    label = []
    for i in range (len(input_data)):#(181):
        col = []
        data.append(col)
        
    print("\n----\n")
        
    for i in range (len(input_data)):#(181):cle
        all_features = input_data.iloc[i].values.tolist()
        #print("#: ", all_features[:-1])
        class_label = int(all_features[-1])
        all_features = all_features[:-1]
        selected_features = []
        for j in range(len(all_features)):
            if tokens[j] == '1':
                selected_features.append(all_features[j])
        data[i] = selected_features
        
    #for i in range (len(input_data)):
        #label.append(data[i][-1])
        
        label.append(class_label)
    print("Num of feature---", len(data[0]))
    ##print(label)
    #return
     
    '''
    for i in range (len(input_data)):#(181):cle
        data[i] = data[i][:-1]
    
    for i in range(len(input_data)):
        row = []
        for j in range(len(tokens)):
            if tokens[j] == "1":
                row.append(input_data[i][j])
        
        data.append(row)
        label.append(int(input_data[i][-1]))
    '''
    
    classify(data, label)
    

def main():
  
    #input_data = pd.read_csv('/Users/russell/microarray-data/csv/alon/alon_inputs.csv')
    #label = pd.read_csv('/Users/russell/microarray-data/csv/alon/alon_outputs.csv')
    
    
    #input_data = pd.read_csv('/Users/russell/microarray-data/csv/alon/alon_inputs.csv')
    #label = pd.read_csv('/Users/russell/microarray-data/csv/alon/alon_outputs.csv')

    #input_data = pd.read_csv('/Users/russell/microarray-data/csv/pomeroy/pomeroy_inputs.csv')
    #label = pd.read_csv('/Users/russell/microarray-data/csv/pomeroy/pomeroy_outputs.csv')
    
    
    #input_data = pd.read_csv('/Users/russell/microarray-data/csv/shipp/shipp_inputs.csv')
    #label = pd.read_csv('/Users/russell/microarray-data/csv/shipp/shipp_outputs.csv')
    
     
    input_data = pd.read_csv('/Users/russell/microarray-data/csv/golub/golub_inputs.csv')
    label = pd.read_csv('/Users/russell/microarray-data/csv/golub/golub_outputs.csv')
   
    
    
    #input_data = pd.read_csv('/Users/russell/microarray-data/csv/gordon/gordon_inputs.csv')
    #label = pd.read_csv('/Users/russell/microarray-data/csv/gordon/gordon_outputs.csv')
    

    #input_data = pd.read_csv('/Users/russell/microarray-data/csv/singh/singh_inputs.csv')
    #label = pd.read_csv('/Users/russell/microarray-data/csv/singh/singh_outputs.csv')

    
    #input_data = pd.read_csv('/Users/russell/microarray-data/csv/khan/khan_inputs.csv')
    #label = pd.read_csv('/Users/russell/microarray-data/csv/khan/khan_outputs.csv')
    

    #input_data = pd.read_csv('/Users/russell/microarray-data/csv/shipp/shipp_inputs.csv')
    #label = pd.read_csv('/Users/russell/microarray-data/csv/shipp/shipp_outputs.csv')
    
    
    #input_data = pd.read_csv('/Users/russell/microarray-data/csv/chin/chin_inputs.csv')
    #label = pd.read_csv('/Users/russell/microarray-data/csv/chin/chin_outputs.csv')
    
    
     
    #input_data = pd.read_csv('/Users/russell/microarray-data/csv/west/west_inputs.csv')
    #label = pd.read_csv('/Users/russell/microarray-data/csv/west/west_outputs.csv')
    
    
    #input_data = pd.read_csv('/Users/russell/microarray-data/csv/chowdary/chowdary_inputs.csv')
    #label = pd.read_csv('/Users/russell/microarray-data/csv/chowdary/chowdary_outputs.csv')
    
    label = label.values.tolist() 
    label = [ int(value[0]) for value in label]
    
    print("---",len(input_data), len(label))
    
    
    data = []
    for i in range (len(label)):#(181):
        col = []
        data.append(col)
        
    for i in range (len(label)):#(181):cle
        data[i] = input_data.iloc[i].values.tolist()
        #data[i] = data[i][:10000]
    
   
    data = pre_process_data(data)
    
    feature_selection_classification(data, label)
    
    

if __name__ == main():
    main()
