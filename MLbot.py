#import required libraries
import numpy as np
from flask import Flask, request, make_response,render_template,jsonify
import json
import pickle
from flask_cors import cross_origin


#make flask app
app = Flask(__name__)

#model loading
model = pickle.load(open('lr_trained_model.sav', 'rb'))
@app.route('/')
@app.route('/home')
def home():
    return "Welcome User"

#receive and send response to dialogflow
@app.route('/webhook', methods=['POST'])

def webhook():
    req = request.get_json(silect=True, force=True)
    res = processRequest(req)
    res = json.dumps(res, indent=4)
    r = make_response(res)
    r.headers['Content-Type'] = 'application/json'
    print (r)
    return r

#processing request from dialogflow

def processRequest(req):
    result = req.get("queryResults")
    #fetching parameters
    parameters = result.get("parameters")
    gender = parameters.get("gender")
    age = parameters.get("age")
    education_level = parameters.get("education_level")
    household_income = parameters.get("household_income")
    trouble_sleeping_history = parameters.get("trouble_sleeping_history")
    sleep_hours = parameters.get("sleep_hours")
    sedentary_time = parameters.get("sedentary_time")
    cant_work = parameters.get("cant_work")
    limited_work = parameters.get("limited_work")
    memory_problems = parameters.get("memory_problems")
    prescription_count = parameters.get("prescription_count")

    features = [gender,age,education_level,household_income,trouble_sleeping_history,sleep_hours,sedentary_time,cant_work,limited_work,memory_problems,prescription_count]

    #dumping data into array
    final_features = [np.array(features)]

    #getting intent with fullfilment enabled
    intent = result.get("intent").get('displayName')

    #fitting out model with the data
    if (intent=='Default Welcom Intent - yes'):
        prediction = model.predict(final_features)
        print(prediction)

        if(prediction=='0'):
            status = 'You are not Depressed'
        else:
            status = 'Seems you have Depression'

        fulfillmentText = status

        print(fulfillmentText)
        print(prediction)
        return {
    "fulfillmentText": fulfillmentText
}    

if __name__ == '__main__':
    app.run()
