from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd


def random_forest_acc(mds):
    # Add dummies for market
    mkt_dummies = pd.get_dummies(mds['Market'])
    mds = pd.concat([mds, mkt_dummies], axis=1)
    mds = mds.drop('Market', axis=1)

    # Split train and test
    mds = mds.dropna()
    train = mds[mds['DayIndex'] >= 400]
    test = mds[mds['DayIndex'] < 400]
    y_train = train['Up']
    X_train = train.drop('Up', axis=1)
    y_test = test['Up']
    X_test = test.drop('Up', axis=1)
    # fit and predict
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    guess_train = clf.predict(X_train)
    guess_test = clf.predict(X_test)
    train_acc = accuracy_score(guess_train, y_train)
    test_acc = accuracy_score(guess_test, y_test)
    print "random Forest train accuracy is {}".format(train_acc)
    print "random Forest test accuracy is {}".format(test_acc)
    return train_acc, test_acc


def batch_random_forest_acc(mds):
    mkt_names = list(mds.Market.unique())
    acc_scores = []
    print "BATCH DATA"
    for mkt_name in mkt_names:
        print mkt_name
        mkt_mds = mds[mds['Market'] == mkt_name]
        train_acc, test_acc = random_forest_acc(mkt_mds)
        acc_scores.append([train_acc, test_acc])
    return acc_scores
