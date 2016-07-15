from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from sklearn.metrics import accuracy_score
from keras.utils.np_utils import to_categorical
from sklearn.decomposition import PCA
import time
from Preprocessor import validation_split


# Creates models for NN's using keras functional api
# Returns Model
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
    # 3 layer nn
    if type == '3layer':
        # set_trace()
        model = Sequential()
        model.add(Dense(15, input_dim=input_size))
        model.add(Activation("sigmoid"))
        model.add(Dense(output_dim=15, input_dim=15))
        model.add(Activation("sigmoid"))
        model.add(Dense(output_dim=2, input_dim=15))
        model.add(Activation("softmax"))
        model.compile(loss='categorical_crossentropy',
                      optimizer='sgd', metrics=['accuracy'])
    # 3 layer nn with tanh
    if type == '3layertanh':
        # set_trace()
        model = Sequential()
        model.add(Dense(15, input_dim=input_size))
        model.add(Activation("tanh"))
        model.add(Dense(output_dim=20, input_dim=40))
        model.add(Activation("tanh"))
        model.add(Dense(output_dim=2, input_dim=15))
        model.add(Activation("softmax"))
        model.compile(loss='categorical_crossentropy',
                      optimizer='sgd', metrics=['accuracy'])
    # 3 layer nn with dropout
    if type == '3layerdrop':
        model = Sequential()
        model.add(Dense(15, input_dim=input_size))
        model.add(Activation("sigmoid"))
        model.add(Dropout(0.15))
        model.add(Dense(output_dim=15, input_dim=15))
        model.add(Activation("sigmoid"))
        model.add(Dropout(0.15))
        model.add(Dense(output_dim=2, input_dim=15))
        model.add(Activation("softmax"))
        model.compile(loss='categorical_crossentropy',
                      optimizer='sgd', metrics=['accuracy'])
    # 3 layer nn with dropout
    if type == 'op3layerdrop':
        model = Sequential()
        model.add(Dense(15, input_dim=input_size))
        model.add(Activation("tanh"))
        model.add(Dropout(0.15))
        model.add(Dense(output_dim=30, input_dim=15))
        model.add(Activation("tanh"))
        model.add(Dropout(0.15))
        model.add(Dense(output_dim=2, input_dim=30))
        model.add(Activation("softmax"))
        model.compile(loss='categorical_crossentropy',
                      optimizer='sgd', metrics=['accuracy'])
    return model


# Train NN on dataset
# Print accuracy, return guess
def nn_acc(mds, type, epoch=5, batch=16):
    train, validation = validation_split(mds)
    # Multiply by 1 to convert to bool
    y_train = train['Up'] * 1
    X_train = train.drop('Up', axis=1)
    y_validation = validation['Up'] * 1
    X_validation = validation.drop('Up', axis=1)
    # Create Model
    model = create_model(X_train.shape[1], type)
    # Convert to Keras format
    X_train = (X_train).as_matrix()
    X_validation = (X_validation).as_matrix()
    y_train = to_categorical(y_train.values)
    y_validation = to_categorical(y_validation.values)
    # Fit and guess
    model.fit(X_train, y_train, nb_epoch=epoch, batch_size=batch)
    guess_train = model.predict_classes(X_train)
    guess_train = to_categorical(guess_train)

    guess_validation = model.predict_classes(X_validation)
    guess_validation = to_categorical(guess_validation)

    train_acc = accuracy_score(y_train, guess_train)
    validation_acc = accuracy_score(y_validation, guess_validation)
    print "\n neural net train accuracy is {}".format(train_acc)
    print "\n neural net validation accuracy is {}".format(validation_acc)
    return guess_validation


# Train NN on dataset after whitening and performing PCA
# with components = 8, the ideal number from svms
# Print accuracy, return guess
def pca_nn_acc(mds, type):
    train, validation = validation_split(mds)
    # Multiply by 1 to convert to bool
    y_train = train['Up'] * 1
    X_train = train.drop('Up', axis=1)
    y_validation = validation['Up'] * 1
    X_validation = validation.drop('Up', axis=1)
    pre = PCA(n_components=8, whiten=True)
    X_train_pca = pre.fit_transform(X_train)
    X_validation_pca = pre.fit_transform(X_validation)
    model = create_model(X_train_pca.shape[1], type)
    # Convert to Keras format
    y_train = to_categorical(y_train.values)
    y_validation = to_categorical(y_validation.values)
    model.fit(X_train_pca, y_train, nb_epoch=5, batch_size=16)
    time.sleep(0.1)
    # Fit and guess
    guess_train = model.predict_classes(X_train_pca)
    guess_train = to_categorical(guess_train)

    guess_validation = model.predict_classes(X_validation_pca)
    guess_validation = to_categorical(guess_validation)

    train_acc = accuracy_score(y_train, guess_train)
    validation_acc = accuracy_score(y_validation, guess_validation)
    print "\n neural net train accuracy is {}".format(train_acc)
    print "\n neural net validation accuracy is {}".format(validation_acc)
    return guess_validation


# Train NN on dataset after whitening and performing PCA
# with components = features so data is not lost
# Print accuracy return guess
def feature_scaled_nn_acc(mds, type):
    train, validation = validation_split(mds)
    # Multiply by 1 to convert to bool
    y_train = train['Up'] * 1
    X_train = train.drop('Up', axis=1)
    y_validation = validation['Up'] * 1
    X_validation = validation.drop('Up', axis=1)
    pre = PCA(n_components=19, whiten=True)
    X_train_pca = pre.fit_transform(X_train)
    X_validation_pca = pre.fit_transform(X_validation)
    model = create_model(X_train_pca.shape[1], type)
    # Convert to Keras format
    y_train = to_categorical(y_train.values)
    y_validation = to_categorical(y_validation.values)
    model.fit(X_train_pca, y_train, nb_epoch=5, batch_size=16)
    time.sleep(0.1)
    # Fit and guess
    guess_train = model.predict_classes(X_train_pca)
    guess_train = to_categorical(guess_train)

    guess_validation = model.predict_classes(X_validation_pca)
    guess_validation = to_categorical(guess_validation)

    train_acc = accuracy_score(y_train, guess_train)
    validation_acc = accuracy_score(y_validation, guess_validation)
    print "\n neural net train accuracy is {}".format(train_acc)
    print "\n neural net validation accuracy is {}".format(validation_acc)
    return guess_validation
