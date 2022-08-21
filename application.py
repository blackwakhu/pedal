import numpy as np
from flask import Flask, request, render_template
import pickle

application = Flask(__name__) 
model = pickle.load(open('randomforest.pkl', 'rb'))

@application.route('/')
def home():
    return render_template('index.html')

@application.route('/predict',methods=['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    return render_template('index.html', prediction_text='the amount is {}'.format(prediction[0]))

if __name__ == "__main__":
    application.run(debug=True)

