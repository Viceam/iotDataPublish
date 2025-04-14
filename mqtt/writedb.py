import pandas as pd
from sqlalchemy import create_engine

# 读取 CSV 前 100 行
df = pd.read_csv('data/equipment_anomaly_data.csv', nrows=100)

# 生成 equipment_id 列（sensor_0 到 sensor_99）
df['equipment_id'] = [f'sensor_{i}' for i in range(len(df))]

# 选择需要的列并重命名
df = df.rename(columns={
    'location': 'location',
    'equipment': 'type'
})[['location', 'type', 'equipment_id']]

# 创建数据库连接
engine = create_engine(
    'postgresql://postgres:1006@127.0.0.1:5432/iotdb'
)

# 写入数据库（id 自增无需指定）
df.to_sql(
    'equipments',
    con=engine,
    if_exists='append',  # 使用 append 模式追加数据
    index=False        # 禁用 Pandas 索引
)

print("数据写入成功！")
