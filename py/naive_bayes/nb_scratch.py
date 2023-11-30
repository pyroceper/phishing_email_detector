from csv_reader import open_csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import numpy as np

class NaiveBayesScratch:
    def __init__(self):
        self.accuracy = 0
        self.report = 0
        self.vectorizer = 0
        self.prior_probabilities = 0
        self.conditional_probabilities = 0
        self.result = 0

    def train_model(self): 
        df = open_csv("./dataset/Phishing_Email.csv")

        separated_data = defaultdict(list)
        for i in range(len(data)):
            separated_data[data[target_column].iloc[i]].append(data.iloc[i])

        self.prior_probabilities = {target: len(separated_data[target]) / len(data) for target in separated_data}

        self.conditional_probabilities = defaultdict(dict)
        for target, target_data in separated_data.items():
            target_data_frame = pd.DataFrame(target_data)
            for feature in target_data_frame.columns:
                if feature != target_column:
                    self.conditional_probabilities[target][feature] = defaultdict(float)
                    for value in target_data_frame[feature].unique():
                        self.conditional_probabilities[target][feature][value] = (
                            (target_data_frame[feature] == value).sum() + 1) / (len(target_data_frame) + len(target_data_frame[feature].unique()))

    
    def predict(self, user_input):
        best_class = None
        max_posterior = float('-inf')

        for target in self.prior_probabilities:
            posterior = np.log(self.prior_probabilities[target])

        for feature, value in input_data.items():
            if feature in self.conditional_probabilities[target] and value in self.conditional_probabilities[target][feature]:
                posterior += np.log(self.conditional_probabilities[target][feature][value])

        if posterior > max_posterior:
            max_posterior = posterior
            best_class = target

        return best_class
        

    def calculate_accuracy(self, predictions, true_labels):
        correct_predictions = np.sum(np.array(predictions) == np.array(true_labels))
        total_predictions = len(predictions)
        accuracy = correct_predictions / total_predictions
        return accuracy 