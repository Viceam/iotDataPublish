import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import accuracy_score
import joblib

# 步骤 1：加载训练数据
train_data = pd.read_csv('data/equipment_anomaly_data.csv')

# 查看数据结构
print(train_data.head())

# 步骤 2：数据预处理
# 只保留数值型特征
features = train_data[['temperature', 'pressure', 'vibration', 'humidity']]
target = train_data['faulty']

# 标准化特征
scaler = StandardScaler()
X = scaler.fit_transform(features)

# 标签
y = target

# 步骤 3：划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 步骤 4：训练TabNet模型
model = TabNetClassifier()

# 训练模型
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], max_epochs=100, patience=10, batch_size=1024, virtual_batch_size=128)

# 步骤 5：评估模型
# 预测验证集结果
y_pred = model.predict(X_val)

# 计算准确率
accuracy = accuracy_score(y_val, y_pred)
print(f"Accuracy on validation set: {accuracy:.4f}")

# 保存模型和标准化器
model.save_model('models/tabnet_classifier')
joblib.dump(scaler, 'models/scaler.pkl')
