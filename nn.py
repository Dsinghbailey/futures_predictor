from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
import pandas as pd
from sklearn.metrics import accuracy_score
from keras.utils.np_utils import to_categorical
from sklearn.decomposition import PCA
import time


def create_model(input_size, type):
    # Simple 2 layer nn
    if type == '2layer':
        # set_trace()
        model = Sequential()
        model.add(Dense(15, input_dim=input_size))
        model.add(Activation("sigmoid"))
        model.add(Dense(output_dim=2, input_dim=15))
        model.add(Activation("softmax"))
        model.compile(loss='categorical_crossentropy',
                      optimizer='sgd', metrics=['accuracy'])
    # 4 layer nn
    if type == '4layer':
        # set_trace()
        model = Sequential()
        model.add(Dense(15, input_dim=input_size))
        model.add(Activation("sigmoid"))
        model.add(Dense(output_dim=20, input_dim=40))
        model.add(Activation("sigmoid"))
        model.add(Dense(output_dim=2, input_dim=15))
        model.add(Activation("softmax"))
        model.compile(loss='categorical_crossentropy',
                      optimizer='sgd', metrics=['accuracy'])
    # 4 layer nn with dropout and regularization
    if type == '4layerdrop':
        # set_trace()
        model = Sequential()
        model.add(Dense(15, input_dim=input_size))
        model.add(Activation("sigmoid"))
        model.add(Dropout(0.15))
        model.add(Dense(output_dim=20, input_dim=40))
        model.add(Activation("sigmoid"))
        model.add(Dropout(0.15))
        model.add(Dense(output_dim=2, input_dim=15))
        model.add(Activation("softmax"))
        model.compile(loss='categorical_crossentropy',
                      optimizer='sgd', metrics=['accuracy'])
    return model


def nn_acc(mds, type, epoch=5, batch=16):
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

    model = create_model(X_train.shape[1], type)

    # Convert to Keras format
    X_train = (X_train).as_matrix()
    X_test = (X_test).as_matrix()
    y_train = to_categorical(y_train.values)
    y_test = to_categorical(y_test.values)
    model.fit(X_train, y_train, nb_epoch=epoch, batch_size=batch)
    y_guess = model.predict_classes(X_test)
    y_guess = to_categorical(y_guess)
    return accuracy_score(y_test, y_guess)


def pca_nn_acc(mds, type):
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

    pre = PCA(n_components=8, whiten=True)
    X_train_pca = pre.fit_transform(X_train)
    X_test_pca = pre.fit_transform(X_test)

    model = create_model(X_train_pca.shape[1], type)

    # Convert to Keras format
    y_train = to_categorical(y_train.values)
    y_test = to_categorical(y_test.values)
    model.fit(X_train_pca, y_train, nb_epoch=5, batch_size=16)
    y_guess = model.predict_classes(X_test_pca)
    y_guess = to_categorical(y_guess)
    return accuracy_score(y_test, y_guess)


def feature_scaled_nn_acc(mds, type):
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

    pre = PCA(n_components=19, whiten=True)
    X_train_pca = pre.fit_transform(X_train)
    X_test_pca = pre.fit_transform(X_test)

    model = create_model(X_train_pca.shape[1], type)

    # Convert to Keras format
    y_train = to_categorical(y_train.values)
    y_test = to_categorical(y_test.values)
    model.fit(X_train_pca, y_train, nb_epoch=5, batch_size=16)
    time.sleep(0.1)
    y_guess = model.predict_classes(X_test_pca)
    y_guess = to_categorical(y_guess)
    return accuracy_score(y_test, y_guess)
