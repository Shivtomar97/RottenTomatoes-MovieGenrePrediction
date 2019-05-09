"""
A simple script that demonstrates how we can use grid search to set the parameters of a classifier
"""
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from nltk.corpus import stopwords
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

animation = pd.read_csv("/Users/jaynanda/Desktop/Assignments/660/Project/All Data/animation_all.csv")
animation['Genre']=1

all_movie = pd.read_csv("/Users/jaynanda/Desktop/Assignments/660/Project/All Data/all_movie.csv")
all_movie['Genre']=0

all_movie= all_movie.sample(len(animation))
train= pd.concat([animation,all_movie])
train= train.drop_duplicates()
train = train.sample(frac=1)
Y = pd.DataFrame(train['Genre'])
train = train.drop('Genre',axis=1)

train = train.drop('Unnamed: 0',axis=1)

from sklearn.model_selection import train_test_split
rev_train, rev_test, labels_train, labels_test = train_test_split(train, Y, test_size=0.20, random_state=42)

#Build a counter based on the training dataset
rev_train = rev_train.values.T.tolist()
rev_test = rev_test.values.T.tolist()
labels_train = labels_train.values.T.tolist()
labels_test = labels_test.values.T.tolist()
rev_train1=[]
for sublist in rev_train:
    for item in sublist:
        rev_train1.append(item)

rev_test1=[]
for sublist in rev_test:
    for item in sublist:
        rev_test1.append(item)
rev_train2=[]
rev_test2=[]
for item in rev_train1:
    rev_train2.append(item)


for item in rev_test1:
    rev_test2.append(item)
#print(type(rev_train2))

counter = CountVectorizer()
counter.fit(rev_train2)



#count the number of times each term appears in a document and transform each doc into a count vector
counts_train = counter.transform(rev_train2)#transform the training data
counts_test = counter.transform(rev_test2)#transform the testing data
labels_train1=[]
for sublist in labels_train:
    for item in sublist:
        labels_train1.append(item)


labels_test1=[]
for sublist in labels_test:
    for item in sublist:
        labels_test1.append(item)

#print(labels_test1)

KNN_classifier=KNeighborsClassifier()
LREG_classifier=LogisticRegression()
DT_classifier = DecisionTreeClassifier()

predictors=[('knn',KNN_classifier),('lreg',LREG_classifier),('dt',DT_classifier)]

VT=VotingClassifier(predictors)



#=======================================================================================
#build the parameter grid
KNN_grid = [{'n_neighbors': [1,3,5,7,9,11,13,15,17], 'weights':['uniform','distance']}]

#build a grid search to find the best parameters
gridsearchKNN = GridSearchCV(KNN_classifier, KNN_grid, cv=5)

#run the grid search
gridsearchKNN.fit(counts_train,labels_train1)

#=======================================================================================
#build the parameter grid
DT_grid = [{'max_depth': [3,4,5,6,7,8,9,10,11,12],'criterion':['gini','entropy']}]

#build a grid search to find the best parameters
gridsearchDT  = GridSearchCV(DT_classifier, DT_grid, cv=5)

#run the grid search
gridsearchDT.fit(counts_train,labels_train1)

#=======================================================================================

#build the parameter grid
LREG_grid = [ {'C':[0.5,1,1.5,2],'penalty':['l1','l2']}]

#build a grid search to find the best parameters
gridsearchLREG  = GridSearchCV(LREG_classifier, LREG_grid, cv=5)

#run the grid search
gridsearchLREG.fit(counts_train,labels_train1)

#=======================================================================================

VT.fit(counts_train,labels_train1)

#use the VT classifier to predict
predicted=VT.predict(counts_test)

#print the accuracy
print (accuracy_score(predicted,labels_test1))




