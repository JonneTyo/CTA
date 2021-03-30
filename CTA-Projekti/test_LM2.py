#!/usr/bin/env python
# coding: utf-8

# In[332]:


###
# Version check
###

# Python version
import sys
print('Python: {}'.format(sys.version))
# scipy
import scipy
print('scipy: {}'.format(scipy.__version__))
# numpy
import numpy
print('numpy: {}'.format(numpy.__version__))
# matplotlib
import matplotlib
print('matplotlib: {}'.format(matplotlib.__version__))
# pandas
import pandas
print('pandas: {}'.format(pandas.__version__))
# scikit-learn
import sklearn
print('sklearn: {}'.format(sklearn.__version__))


# In[359]:


###
# Import libraries
###

from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import numpy as np
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
# from sklearn.linear_model import Lasso
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier


# In[334]:


###
# Load data
###

names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
# dataset = pandas.read_csv('iris.data', names=names)
# dataset = pandas.read_csv('Seafile/python/iris.data', names=names)
from sklearn import datasets
iris = datasets.load_iris()


# In[335]:


###
# Initialise models
###
seed = 123

# Spot Check Algorithms
models = []
models.append(('LogitRidge  ', LogisticRegression(penalty='l2', solver='liblinear', multi_class='ovr'))) # logit Ridge
models.append(('LinDiscrAnal', LinearDiscriminantAnalysis()))
models.append(('KNearNeigh  ', KNeighborsClassifier()))
models.append(('DecisTree   ', DecisionTreeClassifier()))
models.append(('GaussNB     ', GaussianNB()))
models.append(('LogitLasso  ', LogisticRegression(penalty='l1', solver='liblinear', multi_class='ovr'))) # logit Lasso
models.append(('SVM         ', SVC(gamma='auto')))
models.append(('NeurNet     ', MLPClassifier()))
models.append(('RandFor     ', RandomForestClassifier()))
models.append(('Boosting    ', GradientBoostingClassifier()))


# In[336]:


###
# Separate data
###

# Split-out validation dataset
X = iris.data
Y = iris.target
validation_size = 0.33

X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)


# In[337]:


###
# Build models
###

scoring = 'accuracy'


# In[371]:


# CV: evaluate each model in turn
results = []
names = []
print("Method      : mean AUC  sd AUC")
print("---------------------------------")
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)


# In[339]:


# AUROC: Evaluate each model in turn
AUROC_train = []
AUROC_test = []
for name, model in models:
    # build model
    tmp_fit=model.fit(X_train,Y_train)
    res=tmp_fit.predict(X_validation)
    AUROC_test.append([sklearn.metrics.roc_auc_score(1*(Y_validation==0),1*(res==0)),
                       sklearn.metrics.roc_auc_score(1*(Y_validation==1),1*(res==1)),
                       sklearn.metrics.roc_auc_score(1*(Y_validation==2),1*(res==2))])
    res=tmp_fit.predict(X_train)
    AUROC_train.append([sklearn.metrics.roc_auc_score(1*(Y_train==0),1*(res==0)),
                       sklearn.metrics.roc_auc_score(1*(Y_train==1),1*(res==1)),
                       sklearn.metrics.roc_auc_score(1*(Y_train==2),1*(res==2))])


# In[378]:


import pandas
tmp_train=pandas.DataFrame(AUROC_train)
tmp_test=pandas.DataFrame(AUROC_test)


# In[380]:


print("Train dataset")
for k in range(10):
    msg = "%s: %f" % (names[k], tmp_train.mean(axis=1).tolist()[k])
    print(msg)

print("")
print("Test dataset")
for k in range(10):
    msg = "%s: %f" % (names[k], tmp_test.mean(axis=1).tolist()[k])
    print(msg)


# In[ ]:




