import json
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Optional

# 로직 파일 임포트
from .ml_logic.trainer import train_and_save_model, MODEL_PATH, PREPROCESSOR_PATH, UPDATED_SAMPLE_DATA_PATH

# --- FastAPI 앱 및 데이터 모델 정의 ---
app = FastAPI(title="Re:Local AI 매칭 서버 (개선판)", description="모든 농지 정보를 반영하여 개선된 K-means 클러스터링 매칭 시스템")

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
    applicant_cluster: int
    matched_aimatch: List[Dict]
    filtered_aimatch: List[Dict]

# --- API 엔드포인트 정의 ---

@app.get("/")
def read_root():
    return {"message": "안녕하세요! Re:Local AI 매칭 서버 (개선판) 입니다."}

@app.post("/train_v2")
def train_model_endpoint():
    try:
        import os
        os.makedirs("ai_app/model", exist_ok=True)
        result = train_and_save_model(n_clusters=3)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"모델 학습 중 오류 발생: {e}")

@app.post("/match_v2", response_model=MatchResult)
def match_applicant_to_farmland(applicant: Applicant):
    try:
        kmeans = joblib.load(MODEL_PATH)
        preprocessors = joblib.load(PREPROCESSOR_PATH)
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="모델 파일이 없습니다. 먼저 /train_v2 엔드포인트를 호출하여 모델을 학습시켜주세요.")

    # 1. 신청자 데이터 벡터화
    pref = applicant.preferences
    vector = []
    # 작물, 장비, 토양(선호), 용수(선호)
    vector.extend(preprocessors['crop_mlb'].transform([applicant.interested_crops])[0])
    vector.extend(preprocessors['tool_mlb'].transform([applicant.tools])[0])
    vector.extend(preprocessors['pref_soil_mlb'].transform([pref.soil])[0])
    vector.extend(preprocessors['pref_water_mlb'].transform([pref.water_source])[0])
    # 토양/용수(실제값없음), 숫자, 시설(선호도 boolean)
    vector.extend(np.zeros(len(preprocessors['soil_ohe'].categories_[0])))
    vector.extend(np.zeros(len(preprocessors['water_ohe'].categories_[0])))
    vector.extend(preprocessors['scaler'].transform([[0, applicant.home_lat, applicant.home_lng]])[0]) # 면적 0
    vector.extend([int(getattr(pref, key)) for key in ['agri_water', 'electricity', 'warehouse', 'greenhouse', 'fence', 'paved_road', 'car_access', 'public_transport', 'machine_access', 'road_adjacent']])
    
    applicant_vector = np.array(vector).reshape(1, -1)

    # 2. 클러스터 예측
    predicted_cluster = kmeans.predict(applicant_vector.reshape(1, -1))[0]

    # 3. 클러스터 내 농지 필터링
    with open(UPDATED_SAMPLE_DATA_PATH, 'r', encoding='utf-8') as f:
        all_users = json.load(f)
    
    all_user_labels = kmeans.labels_

    matched_farmlands = []
    for i, user in enumerate(all_users):
        if user['type'] == 'farmland' and all_user_labels[i] == predicted_cluster:
            matched_farmlands.append(user)
    
    # 4. 조건으로 최종 필터링
    filtered_farmlands = []
    for farm in matched_farmlands:
        if (farm['transaction']['trade_type'] == applicant.transaction.trade_type and
            farm['transaction']['price'] <= applicant.transaction.budget and
            farm['transaction']['sell_period'] == applicant.transaction.preferred_sell_period):
            filtered_farmlands.append(farm)

    return {
        "applicant_cluster": int(predicted_cluster),
        "matched_aimatch": matched_farmlands, # AI가 찾은 그룹
        "filtered_aimatch": filtered_farmlands # 최종 필터링된 결과
    }