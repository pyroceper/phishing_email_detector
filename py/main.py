from flask import Flask, request, jsonify
from naive_bayes.nb import NaiveBayes

app = Flask(__name__)
nb = 0

@app.route('/nb_train', methods=['POST'])
def nb_train_model():
    try:
        data = request.get_json()
        test = data.get('test')
        nb.train_model()
        return jsonify({"message": "nb_trained", "accuracy" : nb.accuracy})
    except Exception as e:
        error_msg = str(e)
        return jsonify({'error': error_msg}) 

@app.route('/nb_predict', methods=['POST'])
def nb_predict_email():
    try:
        data = request.get_json()
        email = data.get('email')
        nb.predict(email)
        return jsonify({'email': email, 'phish_prob': nb.accuracy, 'result': nb.result})
    except Exception as e:
        error_msg = str(e)
        return jsonify({'error': error_msg})

if __name__ == '__main__':
    nb = NaiveBayes()
    app.run(debug=True)