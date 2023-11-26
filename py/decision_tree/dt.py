from csv_reader import open_csv
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import RandomOverSampler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

class DecisionTree:
    def __init__(self):
        self.accuracy = 0
        self.report = 0
        self.vectorizer = 0
        self.dt_classifier = 0
        self.result = False

        pass

    def train_model(self):
        df = open_csv("./dataset/Phishing_Email.csv")

        # impute missing values with an empty string
        df['Email Text'].fillna('', inplace=True)

        X = df['Email Text']
        y = df['Email Type']

        # split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # convert the text data to TF-IDF features
        self.vectorizer = TfidfVectorizer()
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)

        # apply oversampling to the training set using imbalanced-learn
        ros = RandomOverSampler(random_state=42)
        X_resampled, y_resampled = ros.fit_resample(X_train_tfidf, y_train)

        # create and train the Decision Tree classifier
        self.dt_classifier = DecisionTreeClassifier(class_weight='balanced')
        self.dt_classifier.fit(X_resampled, y_resampled)

        # make predictions on the test set
        predictions_dt = self.dt_classifier.predict(X_test_tfidf)

        # evaluate the model
        self.accuracy = accuracy_score(y_test, predictions_dt)
        self.report = classification_report(y_test, predictions_dt)

        pass

    def predict(self, user_input):

        user_input_vectorized = self.vectorizer.transform([user_input])

        prediction = self.dt_classifier.predict(user_input_vectorized)
        
        self.result = str(prediction[0])

        # pass

    