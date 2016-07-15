from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from Preprocessor import validation_split


# Implements an SVM with an rbf kernel on market data
# PCA is performed on data beforehand
# Returns train and validation set accuracy
def pca_svm_acc(mds):
    train, validation = validation_split(mds)
    y_train = train['Up']
    X_train = train.drop('Up', axis=1)
    y_validation = validation['Up']
    X_validation = validation.drop('Up', axis=1)
    pre = PCA(n_components=8, whiten=True)
    X_train_pca = pre.fit_transform(X_train)
    X_validation_pca = pre.fit_transform(X_validation)
    clf = SVC()
    clf.fit(X_train_pca, y_train)
    guess_train = clf.predict(X_train_pca)
    guess_validation = clf.predict(X_validation_pca)
    train_acc = accuracy_score(guess_train, y_train)
    validation_acc = accuracy_score(guess_validation, y_validation)
    print "svm train accuracy is {}".format(train_acc)
    print "svm validation accuracy is {}".format(validation_acc)
    return train_acc, validation_acc


# Implements an SVM with and rbf kernel on each market
# PCA is performed on data beforehand
# Returns train and validation set accuracies in a list
def batch_pca_svm_acc(mds):
    mkt_names = list(mds.Market.unique())
    acc_scores = []
    print "BATCH DATA"
    for mkt_name in mkt_names:
        print mkt_name
        mkt_mds = mds[(mds['Market'] == mkt_name)]
        train_acc, validation_acc = pca_svm_acc(mkt_mds)
        acc_scores.append([train_acc, validation_acc])
    return acc_scores


# Implements an SVM with an rbf kernel on market data
# PCA is performed on data with gridsearch 
# Returns train and validation set accuracy
def pca_svm_cv_acc(mds):
    train, validation = validation_split(mds)
    y_train = train['Up']
    X_train = train.drop('Up', axis=1)
    y_validation = validation['Up']
    X_validation = validation.drop('Up', axis=1)
    pipe = Pipeline(steps=[('pca', PCA()),
                           ('svc', SVC())])
    tuned_parameters = [{'pca__n_components': [4, 5, 7, 8, 10, 15]}]
    clf = GridSearchCV(pipe,
                       tuned_parameters,
                       cv=4,
                       scoring='accuracy')
    clf.fit(X_train, y_train)
    guess_train = clf.predict(X_train)
    guess_validation = clf.predict(X_validation)
    train_acc = accuracy_score(guess_train, y_train)
    validation_acc = accuracy_score(guess_validation, y_validation)
    print "cv svm train accuracy is {}".format(train_acc)
    print "cv svm validation accuracy is {}".format(validation_acc)
