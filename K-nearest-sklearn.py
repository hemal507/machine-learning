import pandas as pd
import numpy as np
from sklearn import preprocessing, neighbors
from sklearn.model_selection import train_test_split

accuracies = []
for i in range(25):
    df = pd.read_csv('classification-inp1.csv',sep=',')
    df.replace('?',-99999,inplace=True)
    df.drop(['id'],1,inplace=True)

    X = np.array(df.drop(['class'],1))
    y = np.array(df['class'])

    X_train, X_test, y_train , y_test = train_test_split(X,y,test_size=0.2)
    ##print(len(X_train),len(y_train))

    clf = neighbors.KNeighborsClassifier()
    clf.fit(X_train,y_train)
    accuracy = clf.score(X_test, y_test)
    accuracies.append(accuracy)
    print(accuracy)

##    example_measure = np.array([[4,2,1,1,1,2,3,2,1],[4,2,2,2,2,2,3,2,1]])
##    example_measure = example_measure.reshape(len(example_measure),-1)
##    predict = clf.predict(example_measure)

##    print(predict)
    
    
print(sum(accuracies)/len(accuracies))
