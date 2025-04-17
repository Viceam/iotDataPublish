import grpc
from concurrent import futures
import prediction_service_pb2_grpc
import prediction_service_pb2
from pytorch_tabnet.tab_model import TabNetClassifier
import joblib
import pandas as pd
import numpy as np
import consul  # 用于服务注册
import uuid


class PredictionService(prediction_service_pb2_grpc.PredictionServiceServicer):
    def __init__(self):
        # 加载模型和标准化器
        self.model = TabNetClassifier()
        self.model.load_model('models/tabnet_classifier.zip')
        self.scaler = joblib.load('models/scaler.pkl')

    def Predict(self, request, context):
        try:
            # 转换为DataFrame
            features = pd.DataFrame({
                'temperature': [request.temperature],
                'pressure': [request.pressure],
                'vibration': [request.vibration],
                'humidity': [request.humidity]
            })

            # 标准化
            scaled_features = self.scaler.transform(features)

            # 预测
            prediction = self.model.predict(scaled_features)
            return prediction_service_pb2.PredictionResponse(faulty=int(prediction[0]))

        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return prediction_service_pb2.PredictionResponse()


def register_to_consul():
    c = consul.Consul(host='localhost', port=8500)
    service_id = f"prediction-service-{uuid.uuid4()}"
    c.agent.service.register(
        name="prediction-service",
        service_id=service_id,
        address="localhost",
        port=50053,
        check=consul.Check.tcp("localhost", 50053, "10s")
    )


def serve():
    register_to_consul()  # 注册到Consul
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    prediction_service_pb2_grpc.add_PredictionServiceServicer_to_server(PredictionService(), server)
    server.add_insecure_port('[::]:50053')
    server.start()
    print("Prediction service running on port 50053")
    server.wait_for_termination()


if __name__ == '__main__':
    serve()
