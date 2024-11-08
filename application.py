# This is the application file that will be uploaded into the AWS BeanStalk 
# So application.py is the replica of the app.py
# Where app..py  is the main application file for the flask deployement
# But the application.py is the entry point for the AWS

from flask import Flask,request,render_template
import numpy as np
import pandas as pd



from src.pipeline.predict_pipeline import CustomData,PredictPipeline
application = Flask(__name__)

app = application

# Route to Home page

@app.route('/')

def index():
    return render_template('home.html')

@app.route('/predict data',methods = ['GET','POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            title=request.form.get('title'),
            text=request.form.get('text'),
            BloodPressure=request.form.get('BloodPressure'),
            
        )
        pred_df = data.get_data_as_df()
        print(pred_df)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        score = int(results[0])

        if score == 0:
            tag = "FAKE"
        
        else:
            tag = "Real"

        
        return render_template('home.html',results = tag)
    

if __name__ == "__main__":
    app.run(host = "0.0.0.0") #(,debug = True)


# In the application.py file.....app.run(.... ,debug = True)....debug =TRUE needs to be ommited


