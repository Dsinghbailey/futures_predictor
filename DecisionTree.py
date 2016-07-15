from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from Preprocessor import validation_split


# Implements a random forest on marketdata
# returns accuracy score for train and validation sets
def random_forest_acc(mds):
    train, validation = validation_split(mds)
    y_train = train['Up']
    X_train = train.drop('Up', axis=1)
    y_validation = validation['Up']
    X_validation = validation.drop('Up', axis=1)
    # fit and predict
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    guess_train = clf.predict(X_train)
    guess_validation = clf.predict(X_validation)
    train_acc = accuracy_score(guess_train, y_train)
    validation_acc = accuracy_score(guess_validation, y_validation)
    print "random Forest train accuracy is {}".format(train_acc)
    print "random Forest validation accuracy is {}".format(validation_acc)
    return train_acc, validation_acc


# Implements a random forest on each individual markets 
# returns accuracy scores for train and validation sets in a list
def batch_random_forest_acc(mds):
    mkt_names = list(mds.Market.unique())
    acc_scores = []
    print "BATCH DATA"
    for mkt_name in mkt_names:
        print mkt_name
        mkt_mds = mds[mds['Market'] == mkt_name]
        train_acc, validation_acc = random_forest_acc(mkt_mds)
        acc_scores.append([train_acc, validation_acc])
    return acc_scores
