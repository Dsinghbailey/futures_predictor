from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline


def pca_svm_acc(mds):
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
    pre = PCA(n_components=8, whiten=True)
    X_train_pca = pre.fit_transform(X_train)
    X_test_pca = pre.fit_transform(X_test)
    clf = SVC()
    clf.fit(X_train_pca, y_train)
    guess_train = clf.predict(X_train_pca)
    guess_test = clf.predict(X_test_pca)
    train_acc = accuracy_score(guess_train, y_train)
    test_acc = accuracy_score(guess_test, y_test)
    print "svm train accuracy is {}".format(train_acc)
    print "svm test accuracy is {}".format(test_acc)
    return train_acc, test_acc



def batch_pca_svm_acc(mds):
    mkt_names = list(mds.Market.unique())
    acc_scores = []
    print "BATCH DATA"
    for mkt_name in mkt_names:
        print mkt_name
        mkt_mds = mds[(mds['Market'] == mkt_name)]
        train_acc, test_acc = pca_svm_acc(mkt_mds)
        acc_scores.append([train_acc, test_acc])
    return acc_scores


def pca_svm_cv_acc(mds):
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
    pipe = Pipeline(steps=[('pca', PCA()),
                           ('svc', SVC())])
    tuned_parameters = [{'pca__n_components': [4, 5, 7, 8, 10, 15]}]
    clf = GridSearchCV(pipe,
                       tuned_parameters,
                       cv=4,
                       scoring='accuracy')
    clf.fit(X_train, y_train)
    guess_train = clf.predict(X_train)
    guess_test = clf.predict(X_test)
    train_acc = accuracy_score(guess_train, y_train)
    test_acc = accuracy_score(guess_test, y_test)
    print "cv svm train accuracy is {}".format(train_acc)
    print "cv svm test accuracy is {}".format(test_acc)
