import os
import json
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict

# --- 로직 및 데이터 임포트 ---
from .ml_logic.trainer import train_and_save_model, MODEL_PATH, PREPROCESSOR_PATH, UPDATED_SAMPLE_DATA_PATH
from ..matching_engine import find_best_matches, load_data

# --- FastAPI 앱 및 데이터 모델 정의 ---
app = FastAPI(
    title="Re:Local AI Hybrid Matching Server",
    description="K-means 클러스터링과 점수 기반 랭킹을 결합한 하이브리드 AI 매칭 시스템"
)

# --- 입력 데이터 모델 ---
class ApplicantFacilities(BaseModel):
    soil: List[str] = Field(description="선호 토양")
    water_source: List[str] = Field(description="선호 용수")
    agri_water: bool
    electricity: bool
    warehouse: bool
    greenhouse: bool
    fence: bool
    paved_road: bool
    car_access: bool
    public_transport: bool
    machine_access: bool
    road_adjacent: bool

class ApplicantTransaction(BaseModel):
    trade_type: str = Field(description="희망 거래 형태 (임대, 매매)")
    budget: int = Field(description="희망 예산")
    preferred_sell_period: str = Field(description="희망 매도 시기")

class Applicant(BaseModel):
    id: int
    home_lat: float = Field(description="거주지 위도")
    home_lng: float = Field(description="거주지 경도")
    tools: List[str] = Field(description="보유 장비 목록")
    interested_crops: List[str] = Field(description="관심 작물 목록")
    preferences: ApplicantFacilities
    transaction: ApplicantTransaction

# --- 출력 데이터 모델 ---
class MatchResult(BaseModel):
    matches: List[Dict]
    cluster_info: Dict

# --- API 엔드포인트 정의 ---

@app.get("/")
def read_root():
    return {"message": "Re:Local AI 하이브리드 매칭 서버에 오신 것을 환영합니다."}

@app.post("/train", tags=["Model Training"])
def train_model_endpoint():
    """K-Means 모델을 학습시키고 최신 데이터로 업데이트합니다."""
    try:
        model_dir = os.path.dirname(MODEL_PATH)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        result = train_and_save_model(n_clusters=3)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"모델 학습 중 오류 발생: {e}")

@app.post("/match", response_model=MatchResult, tags=["AI Matching"])
def match_by_hybrid_model(applicant: Applicant):
    """
    하이브리드 매칭 엔진을 사용하여 신청자에게 최적의 농지를 추천합니다.
    1. K-Means로 유사한 농지 그룹을 1차 필터링합니다.
    2. 거래 조건(예산 등)으로 2차 필터링합니다.
    3. 최종 후보군에 대해 정밀 점수를 계산하여 순위를 매깁니다.
    """
    try:
        # --- 모델 및 데이터 로드 ---
        kmeans = joblib.load(MODEL_PATH)
        preprocessors = joblib.load(PREPROCESSOR_PATH)
        with open(UPDATED_SAMPLE_DATA_PATH, 'r', encoding='utf-8') as f:
            all_data = json.load(f)
        
        all_farmlands = [item for item in all_data if item.get('type') == 'farmland']
        all_user_labels = kmeans.labels_

    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="모델 파일이 없습니다. 먼저 /train 엔드포인트를 호출하여 모델을 학습시켜주세요.")
    
    # --- 1단계: K-Means 클러스터 예측 ---
    pref = applicant.preferences
    vector = []
    vector.extend(preprocessors['crop_mlb'].transform([applicant.interested_crops])[0])
    vector.extend(preprocessors['tool_mlb'].transform([applicant.tools])[0])
    vector.extend(preprocessors['pref_soil_mlb'].transform([pref.soil])[0])
    vector.extend(preprocessors['pref_water_mlb'].transform([pref.water_source])[0])
    vector.extend(np.zeros(len(preprocessors['soil_ohe'].categories_[0])))
    vector.extend(np.zeros(len(preprocessors['water_ohe'].categories_[0])))
    vector.extend(preprocessors['scaler'].transform([[0, applicant.home_lat, applicant.home_lng]])[0])
    vector.extend([int(getattr(pref, key)) for key in ['agri_water', 'electricity', 'warehouse', 'greenhouse', 'fence', 'paved_road', 'car_access', 'public_transport', 'machine_access', 'road_adjacent']])
    
    applicant_vector = np.array(vector).reshape(1, -1)
    predicted_cluster = kmeans.predict(applicant_vector)[0]

    # --- 2단계: 클러스터 및 거래 조건으로 후보군 필터링 ---
    candidate_farmlands = []
    applicant_trade = applicant.transaction
    
    # all_data에서 농지만 필터링하여 인덱스를 유지
    farmland_data_with_indices = [(i, user) for i, user in enumerate(all_data) if user['type'] == 'farmland']

    for i, farm in farmland_data_with_indices:
        # 클러스터 일치 여부 확인
        if all_user_labels[i] == predicted_cluster:
            # 거래 조건 일치 여부 확인
            farm_trade = farm['transaction']
            if (farm_trade['trade_type'] == applicant_trade.trade_type and
                farm_trade['price'] <= applicant_trade.budget and
                farm_trade['sell_period'] == applicant_trade.preferred_sell_period):
                candidate_farmlands.append(farm)

    # --- 3단계: 최종 후보군 점수 계산 및 랭킹 ---
    applicant_dict = applicant.dict()
    best_matches = find_best_matches(applicant_dict, candidate_farmlands)

    return {
        "matches": best_matches,
        "cluster_info": {
            "predicted_cluster": int(predicted_cluster),
            "candidate_count_in_cluster": len(candidate_farmlands)
        }
    }
