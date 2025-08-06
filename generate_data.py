import pandas as pd
import numpy as np
import argparse
from datetime import datetime, timedelta
import random

class SensorDataGenerator:
    """
    感測器數據生成器
    生成模擬的感測器數據，用於測試和驗證異常檢測算法
    """
    
    def __init__(self):
        # 感測器參數定義
        self.sensor_params = {
            'temp': {
                'normal_range': (45, 50),
                'abnormal_high': (52, 60),
                'abnormal_low': (35, 43),
                'unit': '°C'
            },
            'pressure': {
                'normal_range': (1.00, 1.05),
                'abnormal_high': (1.08, 1.15),
                'abnormal_low': (0.90, 0.97),
                'unit': ''
            },
            'vibration': {
                'normal_range': (0.02, 0.04),
                'abnormal_high': (0.07, 0.12),
                'abnormal_low': (0.001, 0.015),
                'unit': ''
            }
        }
        
        # 時間戳記起始時間
        self.start_time = datetime(2024, 6, 3, 19, 0, 0)
    
    def generate_sensor_value(self, sensor_name, is_normal_label, normal_prob, abnormal_prob):
        """
        生成單個感測器的數值
        
        Args:
            sensor_name (str): 感測器名稱
            is_normal_label (bool): 是否為normal標籤
            normal_prob (float): normal label時在正常值域的機率
            abnormal_prob (float): abnormal label時在正常值域的機率
            
        Returns:
            float: 生成的感測器數值
        """
        params = self.sensor_params[sensor_name]
        
        # 根據標籤決定使用哪個機率
        prob_in_normal = normal_prob if is_normal_label else abnormal_prob
        
        # 決定是否在正常範圍內
        if random.random() < prob_in_normal:
            # 在正常範圍內
            min_val, max_val = params['normal_range']
        else:
            # 在異常範圍內
            if random.random() < 0.5:
                min_val, max_val = params['abnormal_high']
            else:
                min_val, max_val = params['abnormal_low']
        
        # 生成數值
        value = random.uniform(min_val, max_val)
        
        # 根據感測器類型進行適當的數值處理
        if sensor_name == 'temp':
            value = round(value, 1)  # 溫度保留一位小數
        elif sensor_name == 'pressure':
            value = round(value, 2)  # 壓力保留兩位小數
        elif sensor_name == 'vibration':
            value = round(value, 3)  # 振動保留三位小數
        
        return value
    
    def generate_row(self, row_num, normal_prob, abnormal_prob, null_prob):
        """
        生成單行數據
        
        Args:
            row_num (int): 行號
            normal_prob (float): normal label時在正常值域的機率
            abnormal_prob (float): abnormal label時在正常值域的機率
            null_prob (float): 產生空值的機率
            
        Returns:
            dict: 包含一行數據的字典
        """
        # 生成時間戳記
        timestamp = self.start_time + timedelta(minutes=row_num)
        
        # 決定標籤
        is_normal_label = random.random() < 0.7  # 70% 機率為normal
        label = 'normal' if is_normal_label else 'abnormal'
        
        # 生成感測器數據
        row_data = {
            'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'label': label
        }
        
        # 為每個感測器生成數值
        for sensor_name in ['temp', 'pressure', 'vibration']:
            # 檢查是否產生空值
            if random.random() < null_prob:
                row_data[sensor_name] = ''  # 空值
            else:
                value = self.generate_sensor_value(sensor_name, is_normal_label, normal_prob, abnormal_prob)
                row_data[sensor_name] = value
        
        return row_data
    
    def generate_dataset(self, num_rows, normal_prob, abnormal_prob, null_prob):
        """
        生成完整的數據集
        
        Args:
            num_rows (int): 數據行數
            normal_prob (float): normal label時在正常值域的機率
            abnormal_prob (float): abnormal label時在正常值域的機率
            null_prob (float): 產生空值的機率
            
        Returns:
            pd.DataFrame: 生成的數據集
        """
        data = []
        
        for i in range(num_rows):
            row_data = self.generate_row(i, normal_prob, abnormal_prob, null_prob)
            data.append(row_data)
        
        return pd.DataFrame(data)
    
    def print_statistics(self, df, normal_prob, abnormal_prob, null_prob):
        """
        印出數據統計資訊
        
        Args:
            df (pd.DataFrame): 數據集
            normal_prob (float): normal label時在正常值域的機率
            abnormal_prob (float): abnormal label時在正常值域的機率
            null_prob (float): 產生空值的機率
        """
        print("="*60)
        print("📊 數據生成統計資訊")
        print("="*60)
        
        print(f"📈 總行數: {len(df)}")
        print(f"📋 標籤分布:")
        label_counts = df['label'].value_counts()
        for label, count in label_counts.items():
            percentage = (count / len(df)) * 100
            print(f"   {label}: {count} 行 ({percentage:.1f}%)")
        
        print(f"\n🔧 使用的參數設定:")
        print(f"   normal_prob: {normal_prob} (normal label時每個感測器在正常值域的機率)")
        print(f"   abnormal_prob: {abnormal_prob} (abnormal label時每個感測器在正常值域的機率)")
        print(f"   null_prob: {null_prob} (產生空值的機率)")
        
        print(f"\n📊 各欄位空值統計:")
        for col in ['temp', 'pressure', 'vibration']:
            null_count = df[col].isna().sum() + (df[col] == '').sum()
            null_percentage = (null_count / len(df)) * 100
            print(f"   {col}: {null_count} 個空值 ({null_percentage:.1f}%)")
        
        print(f"\n📋 數據預覽 (前10行):")
        print(df.head(10).to_string(index=False))
        
        print(f"\n📋 數據預覽 (後10行):")
        print(df.tail(10).to_string(index=False))
        
        print("="*60)

def main():
    parser = argparse.ArgumentParser(description='感測器數據生成器')
    parser.add_argument('-n', '--num_rows', type=int, default=300, help='生成數據行數 (默認: 300)')
    parser.add_argument('-o', '--output', default='testing.csv', help='輸出檔案名稱 (默認: testing.csv)')
    parser.add_argument('--normal_prob', type=float, default=0.95, help='normal label時每個感測器在正常值域的機率 (默認: 0.95)')
    parser.add_argument('--abnormal_prob', type=float, default=0.3, help='abnormal label時每個感測器在正常值域的機率 (默認: 0.3)')
    parser.add_argument('--null_prob', type=float, default=0.05, help='產生空值的機率 (默認: 0.05)')
    
    args = parser.parse_args()
    
    # 創建生成器
    generator = SensorDataGenerator()
    
    # 生成數據
    print(f"🚀 開始生成 {args.num_rows} 行感測器數據...")
    df = generator.generate_dataset(args.num_rows, args.normal_prob, args.abnormal_prob, args.null_prob)
    
    # 儲存數據
    df.to_csv(args.output, index=False)
    print(f"✅ 數據已儲存至: {args.output}")
    
    # 印出統計資訊
    generator.print_statistics(df, args.normal_prob, args.abnormal_prob, args.null_prob)

if __name__ == "__main__":
    main()