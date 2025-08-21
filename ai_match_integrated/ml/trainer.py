import json
import joblib
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler, OneHotEncoder
import pathlib
from app.models import Farmland, Buyer

# --- 경로 설정 ---
BASE_DIR = pathlib.Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "ml" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_DATA_PATH = DATA_DIR / "sample_data.json"
MODEL_PATH = MODEL_DIR / "kmeans_model.joblib"
PREPROCESSOR_PATH = MODEL_DIR / "preprocessors.joblib"

def train_and_save_model(n_clusters=2):
    print("AI 모델 학습을 시작합니다...")

    with open(SAMPLE_DATA_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    farmlands = [Farmland(**f) for f in data['farmlands']]
    buyers = [Buyer(**b) for b in data['buyers']]

    # --- 데이터 변환기(Preprocessor) 학습 ---
    # 클러스터링에 사용할 작물 변환기만 학습
    all_crops = [f.landCrop for f in farmlands] + [c for b in buyers for c in b.trustProfile.interestCrop]
    crop_mlb = MultiLabelBinarizer().fit([all_crops])

    # 다른 변환기들은 점수계산 등 다른 로직에서 사용될 수 있으므로 일단 유지
    all_tools = [t for b in buyers for t in b.trustProfile.equipment]
    tool_mlb = MultiLabelBinarizer().fit([all_tools])
    all_soils = [[f.soiltype] for f in farmlands]
    soil_ohe = OneHotEncoder(handle_unknown='ignore').fit(all_soils)
    all_water_sources = [[f.waterSource] for f in farmlands]
    water_ohe = OneHotEncoder(handle_unknown='ignore').fit(all_water_sources)
    numerical_features = [[f.landArea] for f in farmlands]
    scaler = StandardScaler().fit(numerical_features)

    # --- 전체 데이터 벡터화 (오직 '작물' 기준) ---
    processed_entities = []
    for farm in farmlands:
        vector = crop_mlb.transform([[farm.landCrop]])[0]
        processed_entities.append(vector)

    # --- K-means 모델 학습 및 저장 ---
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(processed_entities)

    joblib.dump(kmeans, MODEL_PATH)
    # preprocessors.joblib에는 다른 로직에서 필요할 수 있는 모든 변환기를 저장
    joblib.dump({
        'crop_mlb': crop_mlb, 'tool_mlb': tool_mlb, 'soil_ohe': soil_ohe, 
        'water_ohe': water_ohe, 'scaler': scaler
    }, PREPROCESSOR_PATH)

    print(f"모델 학습 완료! 모델은 '{MODEL_PATH}'에 저장되었습니다.")
    return {"message": "Model training successful", "clusters_created": n_clusters, "vector_size": len(processed_entities[0])}
