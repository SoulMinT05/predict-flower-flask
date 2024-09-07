import joblib
from flask import Flask, render_template, request

app = Flask(__name__)

# Load mô hình
model = joblib.load('model_iris.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Lấy dữ liệu từ người dùng
    try:
        sepal_length = float(request.form['sepal_length'])
        sepal_width = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width = float(request.form['petal_width'])
        
        data = [[sepal_length, sepal_width, petal_length, petal_width]]
        prediction = model.predict(data)
        prediction_text = f'Loài hoa dự đoán: {prediction[0]}'
    except Exception as e:
        prediction_text = f'Error: {e}'

    return render_template('index.html', prediction_text=prediction_text)

if __name__ == "__main__":
    app.run(debug=True)
