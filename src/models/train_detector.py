import numpy as np
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump, load
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import keras


def train_detectors():
    pass



if __name__ == '__main__':
    shap_org = np.load('data/SHAP_signatures/normal/lstm_bae_sst2_org.npy')
    shap_adv = np.load('data/SHAP_signatures/adversarial/lstm_bae_sst2_adv.npy')

    print(f'Org shape: {shap_org.shape}')
    print(f'Adv shape: {shap_adv.shape}')



    data = np.concatenate((shap_org, shap_adv))
    org_labels = np.zeros((shap_org.shape[0],), dtype=np.int16)
    adv_labels = np.ones((shap_adv.shape[0],), dtype=np.int16)
    gt = np.concatenate((org_labels, adv_labels))


    x_train, x_test, y_train, y_test = train_test_split(data, gt, random_state=0, shuffle=True, train_size=0.9)

    print(f'Size: {x_train.shape}')

    randomF = RandomForestClassifier(random_state=42)
    randomF.fit(x_train, y_train)
    preds = randomF.predict(x_test)
    print(f'Random Forest: {accuracy_score(y_test, preds):.3f}')

    svc = SVC(random_state=42)
    svc.fit(x_train, y_train)
    preds = svc.predict(x_test)
    print(f'SVC: {accuracy_score(y_test, preds):.3f}')


    input_shape = x_train.shape[1]


    model = keras.Sequential([
        keras.layers.Dense(400, input_shape=(input_shape,), activation='relu', kernel_regularizer=keras.regularizers.l1(0.00001),
                              ),
        keras.layers.Dropout(0.5, seed=42),
        keras.layers.Dense(400, activation='relu', kernel_regularizer=keras.regularizers.l1(0.00001),
                              ),
        keras.layers.Dense(1, activation='sigmoid',)
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=10)
    model.evaluate(x_test, y_test)

    preds = model.predict(x_test)
    preds = preds.flatten()
    preds[preds < 0.5] = 0
    preds[preds >= 0.5] = 1
    print(accuracy_score(y_test, preds))