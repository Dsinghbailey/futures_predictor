# 2016.06.14 19:26:53 CDT
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.grid_search import GridSearchCV
from Preprocessor import validation_split


# Implements an SVM with and rbf kernel on market data
# Returns train and validation set accuracy
def svm_acc(mds, kernel):
    train, validation = validation_split(mds)
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


# Implements an SVM with and rbf kernel on each market
# Returns train and validation set accuracies in a list
def batch_svm_acc(mds, kernel):
    mkt_names = list(mds.Market.unique())
    acc_scores = []
    for mkt_name in mkt_names:
        print mkt_name
        mkt_mds = mds[(mds['Market'] == mkt_name)]
        train_acc, validation_acc = svm_acc(mkt_mds, 'rbf')
        acc_scores.append([train_acc, validation_acc])
    return acc_scores


# Implements an SVM with and rbf kernel on each market
# Includes grid search of complexity
# Returns train and validation set accuracies in a list
def gridsearch_svm(mds):
    mkt_names = list(mds.Market.unique())
    acc_scores = []
    mds = mds.dropna()
    print "BATCH DATA"
    for mkt_name in mkt_names:
        print mkt_name
        train, validation = validation_split(mds)
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
