from flask import Flask, request, jsonify
import h2o
import pandas as pd

app = Flask(__name__)
h2o.init()

MODEL_PATH = "../models/GBM_1_AutoML_3_20241128_24147"
model = h2o.load_model(MODEL_PATH)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.get_json()

        if not input_data:
            return jsonify({"error": "No input data"}), 400

        input_df = pd.DataFrame([input_data])
        input_h2o = h2o.H2OFrame(input_df)
        prediction = model.predict(input_h2o).as_data_frame().values.flatten().tolist()

        return jsonify({"prediction": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)