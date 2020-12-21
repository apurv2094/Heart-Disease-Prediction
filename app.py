from flask import Flask, render_template, request
import joblib

app = Flask(__name__)
model = joblib.load('project.pkl')
scaler = joblib.load('standard_scalar.pkl')

# you are routing your web page to go through this app so that we can connect it with python

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
    # converting string form values to floating point
    age, gender, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal = float(age), float(bmi), float(cp), float(trestbps), float(chol), float(fbs), float(restecg), float(thalach), float(exang), float(oldpeak), float(slope), float(ca), float(thal)  
    scaled_values = scaler.transform([[age, trestbps, chol, thalach, oldpeak]])
    result = model.predict(scaled_values)
    string = 'See a doctor ' + str(result[0])
    return render_template('index.html', prediction_text = string)

# running the app
if __name__ == '__main__':
    app.run(port =5000, threaded = True)