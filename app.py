import pickle
from flask import Flask,request,app,jsonify,render_template,url_for
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the model
regmodel = pickle.load(open('regmodel.pkl','rb'))
scalar = pickle.load(open('scailing.pkl','rb'))



@app.route('/') # Creating the Home Page
def home():
    return render_template('home.html')

@app.route('/predict_api',methods = ['POST'])

def predict_api():
    data = request.json['data']
    print(data) 
    print(np.array(list(data.values())).reshape(1,-1))
    new_data = scalar.transform(np.array(list(data.values())).reshape(1,-1))
    output = regmodel.predict(new_data)
    print(output[0])
    return jsonify(output[0])


@app.route('/predict',methods = ['POST'])
def predict():
    data = [float(x) for x in  request.form.values()] # From Html form
    final_input = scalar.transform(np.array(data).reshape(1,-1)) # Standardize into one d array
    print(final_input)
    
    
    # Prediciting the model
    output = regmodel.predict(final_input)[0] # As it is in array we need to get the result so we use indexing important step
    
    return render_template('home.html',prediction_text = f'The Predicited House Price is {output}')      # This Predciiton text is in the html file    # The result will get showed in the h













if __name__ == '__main__':
    app.run(debug=True) 