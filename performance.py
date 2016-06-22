import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from nn import create_model
from keras.utils.np_utils import to_categorical


# Helper
def draw_eq_curve(mkt, eq_curve):
    # Graph
    fig = plt.figure()
    fig.subplots_adjust(bottom=0.1)
    ax = fig.add_subplot(111)
    plt.title(mkt, fontsize=18)
    plt.xlabel('Day Index', fontsize=14)
    plt.ylabel('Assets', fontsize=14)
    ax.autoscale_view()
    plt.plot(eq_curve)
    plt.show()
    pnl = eq_curve.iloc[-1] - eq_curve[0]
    print ("\npnl = %s" % pnl)
    pnl_min = 1 - eq_curve.min()/1000
    print ("max loss since inception = {:04.2f}%".format(pnl_min)) 
    return pnl


# Equity curve for long only strategy
# Takes test mds
def draw_long_only(mds):
    mkt_names = list(mds.Market.unique())
    pnls = []
    for mkt_name in mkt_names:
        mkt_mds = mds[(mds['Market'] == mkt_name)]
        mkt_mds.reset_index(inplace=True)
        mkt_mds = mkt_mds[(mkt_mds['DayIndex'] < 400)]
        # Sums for managed account
        mkt_mds.eq_change = mkt_mds['Change']*1000
        mkt_mds.eq_curve = 1000 + mkt_mds.eq_change.rolling(window=400, min_periods=1).sum()
        pnls.append(draw_eq_curve(mkt_name, mkt_mds.eq_curve))
    print "average PNL %s" % np.mean(pnls)


# Equity curve for short only strategy
# Takes test mds
def draw_short_only(mds):
    mkt_names = list(mds.Market.unique())
    pnls = []
    for mkt_name in mkt_names:
        mkt_mds = mds[(mds['Market'] == mkt_name)]
        mkt_mds.reset_index(inplace=True)
        mkt_mds = mkt_mds[(mkt_mds['DayIndex'] < 400)]
        # Sums for managed account
        mkt_mds.eq_change = mkt_mds['Change']*-1000
        mkt_mds.eq_curve = 1000 + mkt_mds.eq_change.rolling(window=400, min_periods=1).sum()
        pnls.append(draw_eq_curve(mkt_name, mkt_mds.eq_curve))
    print "\naverage PNL %s" % np.mean(pnls)


# Helper for Equity curve for pca_svm
# Takes full mds
def pca_svm_cv_guess(mds):
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
    print "accuracy is {}".format(accuracy_score(y_test, y_guess))
    return y_guess


# Equity curve for pca_svm
# Takes full mds and test mds
def draw_pca_svm_cv(train_mds, test_mds):
    mkt_names = list(train_mds.Market.unique())
    pnls = []
    test_mds = test_mds[(test_mds['DayIndex'] < 400)].copy(deep=True)
    y_guess = pca_svm_cv_guess(train_mds)
    test_mds['y_guess'] = y_guess
    # Convert 0 to -1 for shorting
    test_mds['y_guess'].replace(to_replace=0, value=-1, inplace=True)
    print mkt_names
    for mkt_name in mkt_names:
        mkt_mds = test_mds[(test_mds['Market'] == mkt_name)]
        mkt_mds.reset_index(inplace=True)
        # Sums for managed account
        mkt_mds.eq_change = mkt_mds['Change']*1000*mkt_mds.y_guess
        mkt_mds.eq_curve = 1000 + mkt_mds.eq_change.rolling(window=400, min_periods=1).sum()
        pnls.append(draw_eq_curve(mkt_name, mkt_mds.eq_curve))
    print "\naverage PNL %s" % np.mean(pnls)


# Helper for Equity curve for 2 layer nn
# Takes full mds
def nn_guess(mds, epoch=5, batch=16):
    mkt_dummies = pd.get_dummies(mds['Market'])
    mds = pd.concat([mds, mkt_dummies], axis=1)
    mds = mds.drop('Market', axis=1)
    mds = mds.dropna()
    train = mds[(mds['DayIndex'] >= 400)]
    test = mds[(mds['DayIndex'] < 400)]
    # Multiply by 1 to convert to bool
    y_train = train['Up'] * 1
    X_train = train.drop('Up', axis=1)
    y_test = test['Up'] * 1
    X_test = test.drop('Up', axis=1)

    model = create_model(X_train.shape[1], '2layer')

    # Convert to Keras format
    X_train = (X_train).as_matrix()
    X_test = (X_test).as_matrix()
    y_train = to_categorical(y_train.values)
    y_test = to_categorical(y_test.values)
    model.fit(X_train, y_train, nb_epoch=epoch, batch_size=batch)
    y_guess = model.predict_classes(X_test)
    y_guess = to_categorical(y_guess)
    print "accuracy is {}".format(accuracy_score(y_test, y_guess))
    return y_guess


# 2layer nn
def draw_nn(train_mds, test_mds):
    mkt_names = list(train_mds.Market.unique())
    pnls = []
    test_mds = test_mds[(test_mds['DayIndex'] < 400)].copy(deep=True)
    y_guess = nn_guess(train_mds)
    test_mds['y_guess'] = y_guess[:, 1]
    # Convert 0 to -1 for shorting
    test_mds['y_guess'].replace(to_replace=0, value=-1, inplace=True)
    for mkt_name in mkt_names:
        mkt_mds = test_mds[(test_mds['Market'] == mkt_name)].copy(deep=True)
        mkt_mds.reset_index(inplace=True)
        # Sums for managed account
        mkt_mds.eq_change = mkt_mds['Change']*1000*mkt_mds.y_guess
        mkt_mds.eq_curve = 1000 + mkt_mds.eq_change.rolling(window=400, min_periods=1).sum()
        pnls.append(draw_eq_curve(mkt_name, mkt_mds.eq_curve))
    print "\naverage PNL %s" % np.mean(pnls)
