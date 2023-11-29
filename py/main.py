from flask import Flask, request, jsonify
from naive_bayes.nb import NaiveBayes
from decision_tree.dt import DecisionTree
from svm.svm import SVM
from cache_model import check_cache

app = Flask(__name__)
nb = 0
dt = 0
svm = 0

@app.route('/train', methods=['POST'])
def train_model():
    try:
        data = request.get_json()
        test = data.get('test')
        if check_cache("./cache/nb_classifier") == False:
            nb.train_model()
        if check_cache("./cache/dt_classifier") == False:
            dt.train_model()
        if check_cache("./cache/svm_classifier") == False:
            svm.train_model()
        return jsonify({"message": "nb_trained", "accuracy" : nb.accuracy})
    except Exception as e:
        error_msg = str(e)
        return jsonify({'error': error_msg})

@app.route('/predict', methods=['POST'])
def predict_email():
    try:
        data = request.get_json()
        email = data.get('email')
        nb.predict(email)
        dt.predict(email)
        svm.predict(email)
        return jsonify({'email': email, 'nb_phish_prob': nb.accuracy, 'nb_result': nb.result, 'dt_phish_prob': dt.accuracy, 'dt_result': dt.result, 'svm_phish_prob': nb.accuracy, 'svm_result': svm.result})
    except Exception as e:
        error_msg = str(e)
        return jsonify({'error': error_msg})


if __name__ == '__main__':
    nb = NaiveBayes()
    dt = DecisionTree()
    svm = SVM()
    print("[!] Training models.....")
    nb.train_model()
    print("[+] Naive Bayes classifier created")
    dt.train_model()
    print("[+] Decision Tree classifier created")
    svm.train_model()
    print("[+] SVM classifier created")
    print("[+] Done!")
    app.run(debug=False)