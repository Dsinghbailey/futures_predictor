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
    clf = PCA(n_components=8, whiten=True)
    X_train_pca = clf.fit_transform(X_train)
    clf = SVC()
    clf.fit(X_train_pca, y_train)
    X_test_pca = clf.fit_transform(X_test)
    y_guess = clf.predict(X_test_pca)
    return accuracy_score(y_test, y_guess)


def batch_pca_svm_acc(mds):
    mkt_names = list(mds.Market.unique())
    acc_scores = []
    for mkt_name in mkt_names:
        mkt_mds = mds[(mds['Market'] == mkt_name)]
        acc_scores.append(pca_svm_acc(mkt_mds))
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
    y_guess = clf.predict(X_test)
    return accuracy_score(y_test, y_guess)
