# 2016.06.14 19:26:53 CDT
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn import pipeline
from sklearn.grid_search import GridSearchCV

def svm_acc(mds, kernel):
    mkt_dummies = pd.get_dummies(mds['Market'])
    mds = pd.concat([mds, mkt_dummies], axis=1)
    mds = mds.drop('Market', axis=1)
    mds = mds.dropna()
    train = mds[(mds['DayIndex'] >= 400)]
    test = mds[(mds['DayIndex'] < 400)]
    y_train = train['Up']
    X_train = train.drop('Up', axis=1)
    y_test = test['Up']
    X_test = test.drop('Up', axis=1)
    clf = svm.SVC(kernel=kernel)
    clf.fit(X_train, y_train)
    y_guess = clf.predict(X_test)
    return accuracy_score(y_test, y_guess)



def batch_svm_acc(mds, kernel):
    mkt_names = list(mds.Market.unique())
    acc_scores = []
    for mkt_name in mkt_names:
        mkt_mds = mds[(mds['Market'] == mkt_name)]
        acc_scores.append(svm_acc(mkt_mds, kernel))

    return acc_scores



def gridsearch_svm(mds):
    mkt_names = list(mds.Market.unique())
    acc_scores = []
    mds = mds.dropna()
    print 'in func'
    for mkt_name in mkt_names:
        mkt_mds = mds[(mds['Market'] == mkt_name)]
        mkt_mds = mkt_mds.copy(deep=True)
        mkt_mds.drop('Market', axis=1, inplace=True)
        print 'done copy'
        tuned_parameters = [{'estimator__C': [1, 2, 3]}]
        train = mkt_mds[(mkt_mds['DayIndex'] >= 400)]
        test = mkt_mds[(mkt_mds['DayIndex'] < 400)]
        y_train = train['Up']
        X_train = train.drop('Up', axis=1)
        y_test = test['Up']
        X_test = test.drop('Up', axis=1)
        selector = RFE(SVC(kernel='linear'))
        clf = GridSearchCV(selector, tuned_parameters, cv=3, scoring='accuracy')
        clf.fit(X_train, y_train)
        print 'done fit'
        y_guess = clf.predict(X_test)
        acc_scores.append(accuracy_score(y_test, y_guess))

    return acc_scores
