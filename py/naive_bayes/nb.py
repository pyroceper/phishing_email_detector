from csv_reader import open_csv
from cache_model import save_model, check_cache, load_model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
#debug
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

class NaiveBayes:
    def __init__(self):
        self.accuracy = 0
        self.report = 0
        self.vectorizer = 0
        self.nb_classifier = 0
        self.result = 0


    def train_model(self): 
        df = open_csv("./dataset/Phishing_Email.csv")

        # TF-IDF
        self.vectorizer = TfidfVectorizer()
        X = self.vectorizer.fit_transform(df['Email Text'])

        # split data into training data (80%) and testing data (20%)
        X_train, X_test, y_train, y_test = train_test_split(X, df['Email Type'], test_size=0.2, random_state=42)

        # debug using multinomial naive bayes classifier
        self.nb_classifier = MultinomialNB()

        # train the classifier on the TF-IDF features
        self.nb_classifier.fit(X_train, y_train)

        # make predictions on the testing dataset
        y_pred = self.nb_classifier.predict(X_test)

        # evalaute the model
        self.accuracy = accuracy_score(y_test, y_pred)
        self.report = classification_report(y_test, y_pred)

        if check_cache("./cache/nb_classifier") == False:
            save_model("./cache/nb_classifier", self.nb_classifier)


    def predict(self, user_input):
        self.nb_classifier = load_model("./cache/nb_classifier")
       
        user_input = user_input.lower()
        user_input_tfidf = self.vectorizer.transform([user_input])

        prediction = self.nb_classifier.predict(user_input_tfidf)

        self.result = str(prediction[0])

        