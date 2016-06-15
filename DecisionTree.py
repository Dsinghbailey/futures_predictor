from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from Preprocessor import get_full_mds
import pandas as pd
import numpy as np

def random_forest_acc(mds):
    # Add dummies for market
    mkt_dummies = pd.get_dummies(mds['Market'])
    mds = pd.concat([mds, mkt_dummies], axis=1)
    mds = mds.drop('Market', axis=1)

    # Split train and test
    mds = mds.dropna()
    train = mds[mds['DayIndex']>=400]
    test = mds[mds['DayIndex']<400]

    y_train= train['Up']
    X_train = train.drop('Up', axis=1)
    y_test = test['Up']
    X_test = test.drop('Up',axis =1)

    # fit and predict
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)

    y_guess = clf.predict(X_test)
    return accuracy_score(y_test,y_guess)


def batch_random_forest_acc(mds):
    mkt_names = list(mds.Market.unique())
    acc_scores = []
    for mkt_name in mkt_names:
        mkt_mds = mds[mds['Market'] == mkt_name]
        acc_scores.append(random_forest_acc(mkt_mds))
    return acc_scores
