#!/usr/bin/python3

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


#########################################################
### your code goes here ###
print('='*10, ' SVM ', '='*10)

features_train = features_train[:len(features_train)/100]
labels_train = labels_train[:len(labels_train)/100]

from sklearn import svm
clf = svm.SVC()
clf = svm.SVC(kernel="linear", C=0.9, gamma=1)

tfit = time()
clf.fit(features_train, labels_train)
print ("training time:", round(time()-tfit, 3), "s")

## prediction
tpred=time()
pred = clf.predict(features_test)
print("prediction time: ", round(time()-tpred, 3), "s")
#########################################################
accuracy = clf.score(features_test, labels_test)
print( "Accuracy: ", accuracy)
#########################################################
'''
You'll be Provided similar code in the Quiz
But the Code provided in Quiz has an Indexing issue
The Code Below solves that issue, So use this one
'''

# features_train = features_train[:int(len(features_train)/100)]
# labels_train = labels_train[:int(len(labels_train)/100)]

#########################################################
