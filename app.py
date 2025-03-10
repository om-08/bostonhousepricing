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
    data = request.json['data'] # Once predict_api is hit the data will get stored in the form of json in the data variable
    print(data)
    print(np.array(list(data.values())).reshape(1,-1)) # This will save the data's value in the form of list of every feature  # Note that we would want to reshape the 2 d to one d array also # by adding the list keyword we change tfrom dict obj to list type data
    # Applying Standarad Scaler
    
    new_data = scalar.transform(np.array(list(data.values())).reshape(1,-1))  # Fit_transform only for the new data
    
    # Now Prediciting
    output = regmodel.predict(new_data)
    
    print(output[0]) # Since Output will be in two-dimensional 
    return jsonify(output[0])

if __name__ == '__main__':
    app.run(debug=True) 