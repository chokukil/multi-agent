#!/usr/bin/env python3
"""
🍒 CherryAI Phase 3: E2E 테스트용 데이터셋 준비

다양한 도메인의 테스트 데이터를 생성하여 
LLM First 원칙과 범용성을 검증할 수 있는 데이터셋 구성
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class TestDatasetGenerator:
    """테스트 데이터셋 생성기"""
    
    def __init__(self):
        self.output_dir = Path("test_datasets")
        self.output_dir.mkdir(exist_ok=True)
        
    def generate_all_datasets(self):
        """모든 테스트 데이터셋 생성"""
        print("🗂️ E2E 테스트용 데이터셋 생성 시작...")
        
        datasets = [
            ("classification_employees.csv", self.create_employee_classification_data),
            ("regression_housing.csv", self.create_housing_regression_data),
            ("eda_iris_variant.csv", self.create_iris_variant_data),
            ("timeseries_sales.csv", self.create_sales_timeseries_data),
            ("text_reviews.json", self.create_text_reviews_data),
            ("financial_stocks.xlsx", self.create_financial_stocks_data),
            ("sensor_iot.csv", self.create_iot_sensor_data),
            ("marketing_campaigns.json", self.create_marketing_campaign_data)
        ]
        
        created_files = []
        for filename, generator_func in datasets:
            try:
                filepath = self.output_dir / filename
                generator_func(filepath)
                created_files.append(str(filepath))
                print(f"✅ {filename} 생성 완료")
            except Exception as e:
                print(f"❌ {filename} 생성 실패: {e}")
        
        print(f"\n📊 총 {len(created_files)}개 테스트 데이터셋 생성 완료")
        return created_files
    
    def create_employee_classification_data(self, filepath: Path):
        """직원 분류 데이터 (타이타닉과 유사하지만 범용적)"""
        np.random.seed(42)
        n_samples = 800
        
        # 범용적인 직원 데이터 생성 (특정 도메인에 제한되지 않음)
        data = {
            'employee_id': range(1, n_samples + 1),
            'age': np.random.normal(35, 10, n_samples).astype(int),
            'years_experience': np.random.exponential(5, n_samples),
            'education_level': np.random.choice(['Bachelor', 'Master', 'PhD', 'High School'], n_samples, p=[0.4, 0.3, 0.15, 0.15]),
            'department': np.random.choice(['Engineering', 'Sales', 'Marketing', 'HR', 'Finance'], n_samples),
            'salary': np.random.normal(70000, 20000, n_samples),
            'performance_score': np.random.uniform(1, 5, n_samples),
            'remote_work': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
            'training_hours': np.random.poisson(40, n_samples),
            'promoted': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])  # 타겟 변수
        }
        
        df = pd.DataFrame(data)
        
        # 현실적인 관계 추가 (LLM이 발견할 수 있는 패턴)
        df.loc[df['education_level'] == 'PhD', 'salary'] *= 1.3
        df.loc[df['performance_score'] > 4, 'promoted'] = np.random.choice([0, 1], sum(df['performance_score'] > 4), p=[0.3, 0.7])
        
        df.to_csv(filepath, index=False)
        
    def create_housing_regression_data(self, filepath: Path):
        """주택 가격 회귀 데이터 (보스턴 주택과 유사하지만 현대적)"""
        np.random.seed(123)
        n_samples = 600
        
        data = {
            'property_id': range(1, n_samples + 1),
            'area_sqft': np.random.normal(2000, 500, n_samples),
            'bedrooms': np.random.poisson(3, n_samples),
            'bathrooms': np.random.uniform(1, 4, n_samples),
            'garage_spaces': np.random.poisson(2, n_samples),
            'year_built': np.random.randint(1970, 2023, n_samples),
            'lot_size': np.random.exponential(0.3, n_samples),
            'neighborhood_score': np.random.uniform(1, 10, n_samples),
            'crime_rate': np.random.exponential(0.1, n_samples),
            'school_rating': np.random.uniform(1, 10, n_samples),
            'distance_to_downtown': np.random.exponential(10, n_samples),
            'has_pool': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
            'has_garden': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
            'energy_efficiency': np.random.choice(['A', 'B', 'C', 'D'], n_samples, p=[0.2, 0.3, 0.3, 0.2])
        }
        
        df = pd.DataFrame(data)
        
        # 타겟 변수: 주택 가격 (현실적인 관계)
        df['price'] = (
            df['area_sqft'] * 150 +
            df['bedrooms'] * 10000 +
            df['bathrooms'] * 15000 +
            df['neighborhood_score'] * 8000 +
            df['school_rating'] * 5000 +
            (2023 - df['year_built']) * -200 +
            df['has_pool'] * 25000 +
            np.random.normal(0, 30000, n_samples)
        )
        
        df.to_csv(filepath, index=False)
        
    def create_iris_variant_data(self, filepath: Path):
        """아이리스 변형 데이터 (기본 EDA용)"""
        np.random.seed(456)
        n_samples = 450
        
        # 3가지 꽃 종류 (아이리스와 유사하지만 범용적)
        species = ['Rosa', 'Lily', 'Orchid']
        species_data = []
        
        for i, sp in enumerate(species):
            n_sp = n_samples // 3
            base_values = [4 + i, 2 + i * 0.5, 5 + i * 0.8, 1.5 + i * 0.3]
            
            sp_data = {
                'flower_id': range(i * n_sp + 1, (i + 1) * n_sp + 1),
                'petal_length': np.random.normal(base_values[0], 0.5, n_sp),
                'petal_width': np.random.normal(base_values[1], 0.3, n_sp),
                'sepal_length': np.random.normal(base_values[2], 0.7, n_sp),
                'sepal_width': np.random.normal(base_values[3], 0.4, n_sp),
                'color_intensity': np.random.uniform(0, 10, n_sp),
                'fragrance_level': np.random.poisson(3 + i, n_sp),
                'bloom_season': np.random.choice(['Spring', 'Summer', 'Fall'], n_sp),
                'species': [sp] * n_sp
            }
            species_data.append(pd.DataFrame(sp_data))
        
        df = pd.concat(species_data, ignore_index=True)
        df.to_csv(filepath, index=False)
        
    def create_sales_timeseries_data(self, filepath: Path):
        """매출 시계열 데이터"""
        np.random.seed(789)
        
        # 2년간의 일일 데이터
        start_date = datetime(2022, 1, 1)
        end_date = datetime(2023, 12, 31)
        dates = pd.date_range(start_date, end_date, freq='D')
        
        # 시계열 패턴 생성
        trend = np.linspace(100, 150, len(dates))
        seasonal = 20 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25)
        weekly = 10 * np.sin(2 * np.pi * np.arange(len(dates)) / 7)
        noise = np.random.normal(0, 5, len(dates))
        
        sales = trend + seasonal + weekly + noise
        
        data = {
            'date': dates,
            'daily_sales': sales,
            'transactions': np.random.poisson(50, len(dates)),
            'avg_transaction_value': sales / np.random.poisson(50, len(dates)) * np.random.uniform(0.8, 1.2, len(dates)),
            'marketing_spend': np.random.exponential(1000, len(dates)),
            'weather_score': np.random.uniform(1, 10, len(dates)),
            'is_weekend': [1 if d.weekday() >= 5 else 0 for d in dates],
            'is_holiday': np.random.choice([0, 1], len(dates), p=[0.95, 0.05]),
            'region': np.random.choice(['North', 'South', 'East', 'West'], len(dates))
        }
        
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
        
    def create_text_reviews_data(self, filepath: Path):
        """텍스트 리뷰 데이터 (자연어 처리용)"""
        np.random.seed(101112)
        
        # 다양한 도메인의 리뷰 템플릿
        positive_templates = [
            "This {product} is absolutely amazing! Great {feature} and excellent {quality}.",
            "I love the {feature} of this {product}. Really impressed with the {quality}.",
            "Outstanding {product}! The {feature} exceeded my expectations.",
            "Perfect {product} with incredible {feature}. Highly recommend!",
            "Excellent {quality} and wonderful {feature}. Very satisfied."
        ]
        
        negative_templates = [
            "Disappointed with this {product}. The {feature} is poor and {quality} is lacking.",
            "Not happy with the {feature} of this {product}. {quality} could be better.",
            "This {product} has issues with {feature}. Not worth the price.",
            "Poor {quality} and problematic {feature}. Would not recommend.",
            "The {feature} failed and {quality} is substandard."
        ]
        
        products = ['smartphone', 'laptop', 'headphones', 'camera', 'tablet', 'smartwatch', 'speaker']
        features = ['battery life', 'design', 'performance', 'camera quality', 'sound quality', 'display', 'build quality']
        qualities = ['durability', 'user experience', 'value for money', 'reliability', 'functionality']
        
        reviews = []
        for i in range(1000):
            is_positive = np.random.choice([True, False], p=[0.6, 0.4])
            template = np.random.choice(positive_templates if is_positive else negative_templates)
            
            review_text = template.format(
                product=np.random.choice(products),
                feature=np.random.choice(features),
                quality=np.random.choice(qualities)
            )
            
            reviews.append({
                'review_id': i + 1,
                'product_category': np.random.choice(['Electronics', 'Home', 'Sports', 'Books', 'Fashion']),
                'review_text': review_text,
                'rating': np.random.randint(4, 6) if is_positive else np.random.randint(1, 3),
                'sentiment': 'positive' if is_positive else 'negative',
                'review_date': (datetime.now() - timedelta(days=np.random.randint(0, 365))).isoformat(),
                'verified_purchase': bool(np.random.choice([True, False], p=[0.8, 0.2])),
                'helpful_votes': np.random.poisson(5),
                'reviewer_location': np.random.choice(['US', 'UK', 'CA', 'AU', 'DE'])
            })
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(reviews, f, indent=2, ensure_ascii=False)
            
    def create_financial_stocks_data(self, filepath: Path):
        """금융 주식 데이터 (Excel 형식)"""
        np.random.seed(131415)
        
        # 여러 주식의 데이터
        stocks = ['TECH_A', 'HEALTH_B', 'ENERGY_C', 'FINANCE_D', 'RETAIL_E']
        all_data = []
        
        for stock in stocks:
            dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
            
            # 주식 가격 시뮬레이션
            initial_price = np.random.uniform(50, 200)
            returns = np.random.normal(0.001, 0.02, len(dates))
            prices = [initial_price]
            
            for ret in returns[1:]:
                prices.append(prices[-1] * (1 + ret))
            
            for i, date in enumerate(dates):
                all_data.append({
                    'Date': date,
                    'Stock': stock,
                    'Open': prices[i] * np.random.uniform(0.98, 1.02),
                    'High': prices[i] * np.random.uniform(1.01, 1.05),
                    'Low': prices[i] * np.random.uniform(0.95, 0.99),
                    'Close': prices[i],
                    'Volume': np.random.randint(10000, 1000000),
                    'Market_Cap': prices[i] * np.random.randint(1000000, 10000000),
                    'P_E_Ratio': np.random.uniform(10, 30),
                    'Dividend_Yield': np.random.uniform(0, 5)
                })
        
        df = pd.DataFrame(all_data)
        df.to_excel(filepath, index=False)
        
    def create_iot_sensor_data(self, filepath: Path):
        """IoT 센서 데이터"""
        np.random.seed(161718)
        
        # 24시간 * 30일 데이터 (시간당 측정)
        timestamps = pd.date_range('2024-01-01', periods=24*30, freq='H')
        
        data = {
            'timestamp': timestamps,
            'sensor_id': [f'SENSOR_{i%10:03d}' for i in range(len(timestamps))],
            'temperature': 20 + 5 * np.sin(2 * np.pi * np.arange(len(timestamps)) / 24) + np.random.normal(0, 2, len(timestamps)),
            'humidity': 50 + 20 * np.sin(2 * np.pi * np.arange(len(timestamps)) / 24 + np.pi/4) + np.random.normal(0, 5, len(timestamps)),
            'pressure': 1013 + 10 * np.sin(2 * np.pi * np.arange(len(timestamps)) / (24*7)) + np.random.normal(0, 3, len(timestamps)),
            'air_quality': np.random.uniform(0, 100, len(timestamps)),
            'noise_level': 30 + 20 * (np.arange(len(timestamps)) % 24 > 6) * (np.arange(len(timestamps)) % 24 < 22) + np.random.normal(0, 5, len(timestamps)),
            'motion_detected': np.random.choice([0, 1], len(timestamps), p=[0.8, 0.2]),
            'battery_level': 100 - (np.arange(len(timestamps)) / len(timestamps)) * 80 + np.random.normal(0, 2, len(timestamps)),
            'location': [f'Zone_{i%5}' for i in range(len(timestamps))]
        }
        
        df = pd.DataFrame(data)
        
        # 이상치 추가 (LLM이 탐지할 수 있도록)
        anomaly_indices = np.random.choice(len(df), size=20, replace=False)
        df.loc[anomaly_indices, 'temperature'] += np.random.uniform(15, 25, 20)
        
        df.to_csv(filepath, index=False)
        
    def create_marketing_campaign_data(self, filepath: Path):
        """마케팅 캠페인 데이터 (JSON 형식)"""
        np.random.seed(192021)
        
        campaigns = []
        campaign_types = ['Email', 'Social Media', 'Search Ads', 'Display Ads', 'Video Ads']
        channels = ['Google', 'Facebook', 'Instagram', 'LinkedIn', 'Twitter', 'YouTube']
        
        for i in range(100):
            start_date = datetime(2023, 1, 1) + timedelta(days=np.random.randint(0, 300))
            duration = np.random.randint(7, 60)
            
            campaign = {
                'campaign_id': f'CAMP_{i+1:03d}',
                'campaign_name': f'Campaign {i+1}',
                'campaign_type': np.random.choice(campaign_types),
                'channel': np.random.choice(channels),
                'start_date': start_date.isoformat(),
                'end_date': (start_date + timedelta(days=duration)).isoformat(),
                'budget': np.random.uniform(1000, 50000),
                'target_audience': {
                    'age_range': f"{np.random.randint(18, 45)}-{np.random.randint(46, 65)}",
                    'interests': np.random.choice(['Technology', 'Sports', 'Travel', 'Food', 'Fashion'], size=np.random.randint(1, 3)).tolist(),
                    'location': np.random.choice(['US', 'Europe', 'Asia', 'Global'])
                },
                'metrics': {
                    'impressions': np.random.randint(10000, 1000000),
                    'clicks': np.random.randint(100, 50000),
                    'conversions': np.random.randint(10, 2000),
                    'cost_per_click': np.random.uniform(0.5, 5.0),
                    'click_through_rate': np.random.uniform(0.01, 0.1),
                    'conversion_rate': np.random.uniform(0.01, 0.15),
                    'return_on_ad_spend': np.random.uniform(1.5, 8.0)
                },
                'creative_elements': {
                    'headline': f'Amazing Product {i+1}',
                    'description': f'Discover the best {np.random.choice(["quality", "value", "experience"])} with our product.',
                    'call_to_action': np.random.choice(['Learn More', 'Shop Now', 'Sign Up', 'Download'])
                }
            }
            campaigns.append(campaign)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(campaigns, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    generator = TestDatasetGenerator()
    created_files = generator.generate_all_datasets()
    
    print(f"\n📁 생성된 파일들:")
    for file in created_files:
        print(f"  - {file}")
    
    print(f"\n🚀 E2E 테스트 준비 완료! Playwright로 테스트할 수 있습니다.") 