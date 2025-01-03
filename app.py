from flask import Flask, request, render_template
import pandas
import numpy as np
import pickle

model = pickle.load(open("breastCancerModel.pkl",'rb'))
#flask app
app = Flask(__name__)


@app.route('/')
def index():
    return render_template("index.html")

print("welcome")
@app.route("/predict", methods=['POST'])

def predict():
    features = request.form['feature']
    feature_list = features.split(",")
    np_features = np.asarray(feature_list, dtype = np.float32)
    print(np_features)
    pred = model.predict(np_features.reshape(1, -1))

    output = ['cancerous' if pred[0] == 1 else 'not cancerous']
    
    return render_template('index.html', message = output)


#python main
if __name__ == "__main__":
    app.run(debug = True)

    
