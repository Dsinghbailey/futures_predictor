# 2016.06.14 19:26:53 CDT
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.grid_search import GridSearchCV


def svm_acc(mds, kernel):
    mkt_dummies = pd.get_dummies(mds['Market'])
    mds = pd.concat([mds, mkt_dummies], axis=1)
    mds = mds.drop('Market', axis=1)
    mds = mds.dropna()
    train = mds[(mds['DayIndex'] >= 400)]
    validation = mds[(mds['DayIndex'] < 400)]
    y_train = train['Up']
    X_train = train.drop('Up', axis=1)
    y_validation = validation['Up']
    X_validation = validation.drop('Up', axis=1)
    clf = svm.SVC(kernel=kernel)
    clf.fit(X_train, y_train)
    guess_train = clf.predict(X_train)
    guess_validation = clf.predict(X_validation)
    train_acc = accuracy_score(guess_train, y_train)
    validation_acc = accuracy_score(guess_validation, y_validation)
    print "svm train accuracy is {}".format(train_acc)
    print "svm validation accuracy is {}".format(validation_acc)
    return train_acc, validation_acc


def batch_svm_acc(mds, kernel):
    mkt_names = list(mds.Market.unique())
    acc_scores = []
    for mkt_name in mkt_names:
        print mkt_name
        mkt_mds = mds[(mds['Market'] == mkt_name)]
        train_acc, validation_acc = svm_acc(mkt_mds, 'rbf')
        acc_scores.append([train_acc, validation_acc])
    return acc_scores


def gridsearch_svm(mds):
    mkt_names = list(mds.Market.unique())
    acc_scores = []
    mds = mds.dropna()
    print "BATCH DATA"
    for mkt_name in mkt_names:
        print mkt_name
        mkt_mds = mds[(mds['Market'] == mkt_name)]
        mkt_mds = mkt_mds.copy(deep=True)
        mkt_mds.drop('Market', axis=1, inplace=True)
        train = mkt_mds[(mkt_mds['DayIndex'] >= 400)]
        validation = mkt_mds[(mkt_mds['DayIndex'] < 400)]
        y_train = train['Up']
        X_train = train.drop('Up', axis=1)
        y_validation = validation['Up']
        X_validation = validation.drop('Up', axis=1)
        selector = SVC()
        tuned_parameters = [{'C': [1, 2, 3]}]
        clf = GridSearchCV(selector,
                           tuned_parameters,
                           cv=3,
                           scoring='accuracy')
        clf.fit(X_train, y_train)
        guess_train = clf.predict(X_train)
        guess_validation = clf.predict(X_validation)
        train_acc = accuracy_score(guess_train, y_train)
        validation_acc = accuracy_score(guess_validation, y_validation)
        print "PCA svm train accuracy is {}".format(train_acc)
        print "PCA svm validation accuracy is {}".format(validation_acc)
        acc_scores.append([train_acc, validation_acc])
    return acc_scores
