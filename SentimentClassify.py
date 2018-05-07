import keras
from keras import Input
from keras.layers import TimeDistributed
from keras import regularizers
from keras.engine import Model
from keras.optimizers import Adam
from keras.models import load_model
from WordsEmbedding import  *
from keras.layers import BatchNormalization
from sklearn.naive_bayes import  GaussianNB, MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from lime import lime_text
from lime.lime_text import  LimeTextExplainer
from sklearn.pipeline import make_pipeline
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Reshape, MaxPool2D, Concatenate
from keras.layers import Embedding
from keras.layers import Flatten
from keras.preprocessing import sequence
from keras.layers import LSTM
from keras.layers import SimpleRNN
import fasttext as ft
import pickle


def random_forest():
    trees = [1000]
    for num in trees:
        rfc = RandomForestClassifier(n_estimators= num)
        print("start fit")
        rfc = rfc.fit(train_embed, train_labels)
        print("finish fit")
        with open("model/random_forest.pk","wb") as f:
            pickle.dump(rfc,f)
        with open("model/vectorizer.pk","wb") as f:
            pickle.dump(vectorizer,f)
        print("finish dump")
        rfc_predicted = rfc.predict(test_embed)
        print(accuracy_score(test_labels, rfc_predicted))

def explain_result():
    class_names = ['positive', 'negative', 'neutral']
    with open("model/random_forest.pk", "rb") as f:
        rf = pickle.load(f)
    c = make_pipeline(vectorizer, rf)

    explainer = LimeTextExplainer(class_names=class_names)
    text = "Không "
    print(c.predict_proba([text]))
    exp = explainer.explain_instance(text, c.predict_proba, num_features=6)
    print(exp.as_list())


def predict_sentiment(text):
    with open("model/random_forest.pk", "rb") as f:
        rfc = pickle.load(f)
    with open("model/vectorizer.pk", "rb") as f:
        vectorizer = pickle.load(f)
    c = make_pipeline(vectorizer, rfc)
    return rfc.predict(vectorizer.transform([text]))[0],c.predict_proba([text])

def svm():
    cp = [1.0]
    for c in cp:
        svc_clf = SVC(C = c)
        svc_clf = svc_clf.fit(train_embed, train_labels)
        svc_predicted = svc_clf.predict(test_embed)
        print(accuracy_score(test_labels, svc_predicted))

def naive_bayes():
    clf = GaussianNB().fit(train_embed, train_labels)
    nb_predicted = clf.predict(test_embed)
    print(accuracy_score(test_labels, nb_predicted))

def knn():
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(train_embed, train_labels)
    knn_predicted = neigh.predict(test_embed)
    print(accuracy_score(test_labels, knn_predicted))

# def FastText():
#     #ftmodel = ft.supervised('data/trainprocess.txt', 'model/train', label_prefix='__label__')
#     #ftmodel = ft.load_model('model/model_sentiment.bin', encoding = 'utf-8', label_prefix='__label__')
#     ftmodel = ft.skipgram('data/trainprocess.txt', 'skip_gram', dim = 300)
#     print(len(ftmodel['langgg']))
#     print(ftmodel.words)
#     # ftpredicted = []
#     # for text in test_data:
#     #     lb = ftmodel.predict(text)
#     #     ftpredicted.append(int(lb[0][0]))
#     # print(accuracy_score(test_labels, ftpredicted))
#     #labels = ftmodel.predict(text)


def convert_one_hot(y_train, y_test):
    lab = np.amax(y_train)+1
    train_enc = np.zeros((y_train.shape[0],lab))
    train_enc[np.arange(y_train.shape[0]),y_train] = 1
    test_enc = np.zeros((y_test.shape[0], lab))
    test_enc[np.arange(y_test.shape[0]), y_test] = 1
    # for j in range(10,30,1):
    #     print(test_enc[j])
    #     print(y_test[j])
    # print(train_enc)
    # print(train_enc.shape)
    # print(test_enc)
    # print(test_enc.shape)
    return train_enc, test_enc

def reverse_one_hot(y):
    res = np.argmax(y, axis=1)
    # for i in range(y.shape[0]):
    #     for j in range(y.shape[1]):
    #         if (y[i,j] == 1):
    #             res[i] = j
    #             brea
    return res



def fit_lstm():
    X_train = train_embed
    y_train = np.array(train_labels)
    X_test = test_embed
    y_test = test_labels
    # X_train = sequence.pad_sequences(X_train, maxlen= max_words)
    # X_test  = sequence.pad_sequences(X_test, maxlen = max_words)
    y_train, y_test = convert_one_hot(y_train, y_test)
    # create the model
    model = Sequential()
    model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(3, activation="softmax"))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs= 200, batch_size= 32, verbose=2, shuffle=False)
    # score = model.evaluate(X_test, y_test, verbose = 0)
    # print("Accuracy : %.2f%%" %(score[1]*100))


def fit_cnn():
    X_train = train_embed
    y_train = np.array(train_labels)
    X_test = test_embed
    y_test = test_labels
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2],1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2], 1))
    y_train, y_test = convert_one_hot(y_train, y_test)
    sequence_length = train_embed.shape[1]  # 60
    embedding_dim = train_embed.shape[2]
    filter_sizes = [2, 3, 4, 5, 6, 7, 8]
    num_filters = 64
    drop = 0.6
    input_shape = train_embed[0].shape
    epochs = 300
    batch_size = 32
    inputs = Input(shape=(sequence_length,embedding_dim,1), dtype='float32')
    #batch_norm = BatchNormalization(input_shape = input_shape)(inputs)
    conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], embedding_dim), padding='valid',
                    kernel_initializer='normal', activation='relu', input_shape= input_shape)(inputs)
    conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], embedding_dim), padding='valid',
                    kernel_initializer='normal', activation='relu', input_shape= input_shape)(inputs)
    conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], embedding_dim), padding='valid',
                    kernel_initializer='normal', activation='relu', input_shape=input_shape)(inputs)
    conv_3 = Conv2D(num_filters, kernel_size=(filter_sizes[3], embedding_dim), padding='valid',
                    kernel_initializer='normal', activation='relu', input_shape=input_shape)(inputs)
    conv_4 = Conv2D(num_filters, kernel_size=(filter_sizes[4], embedding_dim), padding='valid',
                    kernel_initializer='normal', activation='relu', input_shape=input_shape)(inputs)
    conv_5 = Conv2D(num_filters, kernel_size=(filter_sizes[5], embedding_dim), padding='valid',
                    kernel_initializer='normal', activation='relu', input_shape=input_shape)(inputs)
    conv_6 = Conv2D(num_filters, kernel_size=(filter_sizes[6], embedding_dim), padding='valid',
                    kernel_initializer='normal', activation='relu', input_shape=input_shape)(inputs)


    maxpool_0 = MaxPool2D(pool_size=(sequence_length - filter_sizes[0] + 1, 1), strides=(1, 1), padding='valid')(conv_0)
    maxpool_1 = MaxPool2D(pool_size=(sequence_length - filter_sizes[1] + 1, 1), strides=(1, 1), padding='valid')(conv_1)
    maxpool_2 = MaxPool2D(pool_size=(sequence_length - filter_sizes[2] + 1, 1), strides=(1, 1), padding='valid')(conv_2)
    maxpool_3 = MaxPool2D(pool_size=(sequence_length - filter_sizes[3] + 1, 1), strides=(1, 1), padding='valid')(conv_3)
    maxpool_4 = MaxPool2D(pool_size=(sequence_length - filter_sizes[4] + 1, 1), strides=(1, 1), padding='valid')(conv_4)
    maxpool_5 = MaxPool2D(pool_size=(sequence_length - filter_sizes[5] + 1, 1), strides=(1, 1), padding='valid')(conv_5)
    maxpool_6 = MaxPool2D(pool_size=(sequence_length - filter_sizes[6] + 1, 1), strides=(1, 1), padding='valid')(conv_6)

    concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2, maxpool_3])
    flatten = Flatten()(concatenated_tensor)
    dropout = Dropout(drop)(flatten)
    output = Dense(units=3, activation='softmax')(dropout)

    # this creates a model that includes
    model = Model(inputs=inputs, outputs=output)
    adam = Adam(lr=1e-4, beta_1 = 0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    print("Traning Model...")
    model.fit(X_train, y_train, batch_size= batch_size, epochs=epochs, verbose=1,
              validation_data = (X_test, y_test))
    model.save("model/cnn.h5")
    return model

def fit_cnn_lstm():
    # prepare data
    X_train = train_embed
    y_train = np.array(train_labels)
    X_test = test_embed
    y_test = test_labels
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2], 1))
    y_train, y_test = convert_one_hot(y_train, y_test)

    # config parameter model
    sequence_length = train_embed.shape[1]
    embedding_dim = train_embed.shape[2]
    filter_size = 32
    kernel_size = 3
    pool_size = 2
    drop = 0.5
    batch_size = 128
    # define CNN model
    cnn = Sequential()
    cnn.add(Conv2D(filters=filter_size, kernel_size= kernel_size, padding='valid', activation='relu', input_shape=(sequence_length,embedding_dim,1)))
    cnn.add(MaxPooling2D(pool_size=pool_size, strides=(1,1)))
    #cnn.add(Dropout(drop))
    # define LSTM model
    model = Sequential()
    model.add(Conv2D(filters=filter_size, kernel_size=kernel_size, padding='valid', activation='relu',
                   input_shape=(sequence_length, embedding_dim, 1)))
    model.add(MaxPooling2D(pool_size=pool_size, strides=(1, 1)))
    model.add(Reshape(target_shape = (sequence_length, None)))
    model.add(LSTM(50))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=batch_size, verbose=2, shuffle=False)
    # score = model.evaluate(X_test, y_test, verbose=0)
    # print("Accuracy : %.2f%%" % (score[1] * 100))

    #fit
def predict_cnn(string):
    # X_test = test_embed
    X_test = embedding_for_test([string], max_words, dimens)
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2], 1))
    model = load_model("model/cnn.h5")
    predicted = model.predict(X_test)
    labels = reverse_one_hot(predicted)
    return labels[0], predicted[0]


max_words = 80
dimens = 100

if __name__ =="__main__":
    # train_embed, train_labels, test_embed, test_labels = fasttext_cnn(max_words, dimens)
    train_embed, train_labels, test_embed, test_labels = fasttext_tfidf(max_words, dimens)
    # train_embed, train_labels, test_embed, test_labels, vectorizer = tf_idf()
    # # # # # #explain_result()
    # random_forest()
    # svm()
    # knn()
    # # # #FastText()
    #fit_lstm()
    #model  = fit_cnn()
    fit_lstm()
    # predict_cnn()