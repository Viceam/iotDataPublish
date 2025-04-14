import pandas as pd
from sklearn.preprocessing import StandardScaler
from pytorch_tabnet.tab_model import TabNetClassifier
import joblib

# 步骤 1：加载待测数据
test_data = pd.read_csv('data/100.csv')

# 查看数据结构
print(test_data.head())

# 步骤 2：数据预处理
# 提取特征
test_features = test_data[['temperature', 'pressure', 'vibration', 'humidity', 'equipment', 'location']]

# 将类别变量（equipment 和 location）转化为数值类型（One-Hot Encoding）
test_features = pd.get_dummies(test_features, columns=['equipment', 'location'], drop_first=True)

# 加载标准化器并标准化待测数据
scaler = joblib.load('models/scaler.pkl')
test_features = scaler.transform(test_features)

# 步骤 3：加载模型
loaded_clf = TabNetClassifier()
loaded_clf.load_model('models/tabnet_classifier.zip')

# 步骤 4：进行预测
predictions = loaded_clf.predict(test_features)

# 将预测结果添加到待测数据中
test_data['predicted_faulty'] = predictions

# 输出待测数据的预测结果
print(test_data[['temperature', 'pressure', 'vibration', 'humidity', 'equipment', 'location', 'predicted_faulty']])

# 保存预测结果
test_data.to_csv('data/predicted_new_equipment_data.csv', index=False)
