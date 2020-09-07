import numpy as np
from flask import Flask, request, jsonify, render_template
from joblib import load
app = Flask(__name__)
model= load('logisticregressor.save')
trans1=load('transform1')
trans2=load('transform2')
trans3=load('transform3')
trans4=load('transform4')
scaler = load('scaler')
              
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/y_predict',methods=['POST'])
def y_predict():
    '''
    For rendering results on HTML GUI
    '''
    
    x_test = [[x for x in request.form.values()]]
    print(x_test)
    
    test=trans1.transform(x_test)
    test=test[:,1:]
    
    test=trans2.transform(test)
    test=test[:,1:]
    
    test=trans3.transform(test)
    test=test[:,1:]
    
    test=trans4.transform(test)
    test=test[:,1:]
    
    print(test)
    #test = scaler.transform(test)
    prediction = model.predict(scaler.transform(test))
    print(prediction)
    if prediction[0] == 1:
        output = 'Congrats,you are eligible for loan'
    else:
        output = 'sorry,you are not eligible for loan'
    
    
    return render_template('index.html', prediction_text='Loan_Status:{}'.format(output))

    
'''@app.route('/predict_api',methods=['POST'])
def predict_api():
    
    #For direct API calls trought request
    
    data = request.get_json(force=True)
    prediction = model.y_predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)'''

if __name__ == "__main__":
    app.run(debug=True)
