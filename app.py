from flask import Flask, render_template, request
import joblib

app = Flask(__name__)
model = joblib.load('heart_disease_prediction.pkl')
scaler = joblib.load('standard_scaler.pkl')

@app.route('/', methods = ['GET'])
def hello_world():
    return render_template('index.html')

@app.route('/prediction', methods = ['POST'])
def prediction():

    age = request.form['age']
    gender = request.form['gender']
    cp = request.form['cp']
    trestbps = request.form['trestbps']
    chol = request.form['chol']
    fbs = request.form['fbs']
    restecg = request.form['restecg']
    thalach = request.form['thalach']
    exang = request.form['exang']
    oldpeak = request.form['oldpeak']
    slope = request.form['slope']
    ca = request.form['ca']
    thal = request.form['thal']
    
    age, gender, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal = int(age), int(gender), int(cp), int(trestbps), int(chol), int(fbs), int(restecg), int(thalach), int(exang), float(oldpeak), int(slope), int(ca), int(thal)  
    scaled_values = scaler.transform([[age, gender, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    
    result = model.predict(scaled_values)

    if result[0] == 1:
        string = 'Suffering from heart disease'
    else:
        string = 'Not be suffering from heart disease'
    return render_template('index.html', prediction_text = string)

# running the app
if __name__ == '__main__':
    app.run(port =5000, threaded = True)