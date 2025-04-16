from flask import Flask, request, jsonify
import joblib
import numpy as np
from pytorch_tabnet.tab_model import TabNetClassifier
import pandas as pd

# 初始化 Flask 应用
app = Flask(__name__)

# 加载已保存的模型和标准化器
model = TabNetClassifier()
model.load_model('models/tabnet_classifier.zip')
scaler = joblib.load('models/scaler.pkl')


@app.route('/predict', methods=['POST'])
def predict():
    # 获取请求的 JSON 数据
    data = request.get_json()

    # 提取特征
    temperature = data.get('temperature')
    pressure = data.get('pressure')
    vibration = data.get('vibration')
    humidity = data.get('humidity')

    features = pd.DataFrame(
        [[temperature, pressure, vibration, humidity]],
        columns=['temperature', 'pressure', 'vibration', 'humidity']
    )
    scaled_features = scaler.transform(features)  # 传递DataFrame

    if temperature is None or pressure is None or vibration is None or humidity is None:
        return jsonify({'error': 'Missing required features'}), 400

    # 将输入特征组合成一个数组
    # features = np.array([[temperature, pressure, vibration, humidity]])
    #
    # # 标准化输入特征
    # scaled_features = scaler.transform(features)

    # 使用模型进行预测
    prediction = model.predict(scaled_features)

    # 返回预测结果
    return jsonify({'faulty': int(prediction[0])})


if __name__ == '__main__':
    app.run(debug=False)
