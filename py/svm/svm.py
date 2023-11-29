from csv_reader import open_csv
from cache_model import save_model, check_cache, load_model
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from gensim.models import KeyedVectors
from sklearn.metrics import accuracy_score, classification_report
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

class SVM:
    def __init__(self):
        self.accuracy = 0
        self.report = 0
        self.svm_classifier = 0
        self.w2v_model = 0
        self.result = 0
        pass

    def get_word2vec_embeddings(self, text):
        words = text.split()
        embeddings = [self.w2v_model[word] for word in words if word in self.w2v_model]
        if embeddings:
            return sum(embeddings) / len(embeddings)
        return np.zeros(self.w2v_model.vector_size)

    def train_model(self):
        df = open_csv('./dataset/Phishing_Email.csv')

        # impute missing values with an empty string
        df['Email Text'].fillna('', inplace=True)

        X = df['Email Text']
        y = df['Email Type']

        # split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # download pre-trained Word2Vec model (example: GoogleNews-vectors-negative300.bin)
        # load the model
        self.w2v_model = KeyedVectors.load_word2vec_format('./dataset/GoogleNews-vectors-negative300.bin', binary=True)

        # obtain Word2Vec embeddings for training and testing data
        X_train_word2vec = np.vstack(X_train.apply(self.get_word2vec_embeddings))
        X_test_word2vec = np.vstack(X_test.apply(self.get_word2vec_embeddings))

        # undersampling the majority class
        under_sampler = RandomUnderSampler(sampling_strategy='majority')
        X_resampled, y_resampled = under_sampler.fit_resample(X_train_word2vec, y_train)

        # using SMOTE for oversampling the minority class
        smote = SMOTE(sampling_strategy='minority')
        X_resampled, y_resampled = smote.fit_resample(X_resampled, y_resampled)

        # SVM classifier with class weights
        self.svm_classifier = SVC(kernel='linear', C=1, class_weight='balanced')

        # train the SVM model on the resampled data
        self.svm_classifier.fit(X_resampled, y_resampled)

        # make predictions on the test set
        predictions_svm = self.svm_classifier.predict(X_test_word2vec)

        # print("Classification Report:\n", classification_report(y_test, predictions_svm))
        if check_cache("./cache/svm_classifier") == False:
            save_model("./cache/svm_classifier", self.svm_classifier)


    def predict(self, user_input):
        self.svm_classifier = load_model("./cache/svm_classifier")
        
        user_input_word2vec = self.get_word2vec_embeddings(user_input)

        prediction = self.svm_classifier.predict(user_input_word2vec.reshape(1, -1))[0]

        self.result = str(prediction)

        pass
