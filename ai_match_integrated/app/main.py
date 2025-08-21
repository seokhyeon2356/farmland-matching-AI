import os
import json
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from app.models import (
    Buyer, MatchResult, Farmland, Seller, License, MatchingStatus, 
    ProfitInformation, RecommenderInformation, FavoriteFarmland
)
from ml.trainer import train_and_save_model, MODEL_PATH, PREPROCESSOR_PATH, SAMPLE_DATA_PATH
from core.matching_logic import find_best_matches
from typing import List

# --- FastAPI 앱 정의 ---
app = FastAPI(
    title="Re:Local AI Hybrid Matching Server (ERD-based)",
    description="K-means 클러스터링과 점수 기반 랭킹을 결합한 하이브리드 AI 매칭 시스템 (ERD 연동 버전)"
)

# --- 전역 변수 ---
kmeans_model = None
preprocessors = None
farmlands_data: List[Farmland] = []
sellers_data: List[Seller] = []
licenses_data: List[License] = []
matching_statuses_data: List[MatchingStatus] = []
profit_informations_data: List[ProfitInformation] = []
recommender_informations_data: List[RecommenderInformation] = []
favorite_farmlands_data: List[FavoriteFarmland] = []

@app.on_event("startup")
def load_model_and_data():
    """서버 시작 시 모델과 데이터를 로드합니다."""
    global kmeans_model, preprocessors, farmlands_data, sellers_data, licenses_data, matching_statuses_data, profit_informations_data, recommender_informations_data, favorite_farmlands_data
    if os.path.exists(MODEL_PATH) and os.path.exists(PREPROCESSOR_PATH):
        kmeans_model = joblib.load(MODEL_PATH)
        preprocessors = joblib.load(PREPROCESSOR_PATH)
        print("모델과 전처리기를 성공적으로 로드했습니다.")
    else:
        print("경고: 모델 또는 전처리기 파일이 없습니다. /train 엔드포인트를 호출하여 생성해주세요.")

    if os.path.exists(SAMPLE_DATA_PATH):
        with open(SAMPLE_DATA_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
            farmlands_data = [Farmland(**farm) for farm in data.get('farmlands', [])]
            sellers_data = [Seller(**seller) for seller in data.get('sellers', [])]
            licenses_data = [License(**lic) for lic in data.get('licenses', [])]
            matching_statuses_data = [MatchingStatus(**status) for status in data.get('matching_statuses', [])]
            profit_informations_data = [ProfitInformation(**info) for info in data.get('profit_informations', [])]
            recommender_informations_data = [RecommenderInformation(**info) for info in data.get('recommender_informations', [])]
            favorite_farmlands_data = [FavoriteFarmland(**fav) for fav in data.get('favorite_farmlands', [])]
        print(f"{len(farmlands_data)}개의 농지 데이터를 로드했습니다.")
        print(f"{len(sellers_data)}개의 판매자 데이터를 로드했습니다.")
        print(f"{len(licenses_data)}개의 자격증 데이터를 로드했습니다.")
        print(f"{len(matching_statuses_data)}개의 매칭 현황 데이터를 로드했습니다.")
        print(f"{len(profit_informations_data)}개의 수익 정보 데이터를 로드했습니다.")
        print(f"{len(recommender_informations_data)}개의 추천인 정보를 로드했습니다.")
        print(f"{len(favorite_farmlands_data)}개의 찜한 농지 정보를 로드했습니다.")
    else:
        print("경고: 샘플 데이터 파일이 없습니다.")

@app.get("/")
def read_root():
    return {"message": "Re:Local AI 하이브리드 매칭 서버 (ERD 연동 버전)에 오신 것을 환영합니다."}

@app.post("/train", tags=["Model Training"])
def train_model_endpoint():
    """K-Means 모델을 학습시키고 최신 데이터로 업데이트합니다."""
    try:
        result = train_and_save_model(n_clusters=2)
        load_model_and_data()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"모델 학습 중 오류 발생: {e}")

@app.post("/match", response_model=MatchResult, tags=["AI Matching"])
def match_by_hybrid_model(buyer: Buyer):
    """
    하이브리드 매칭 엔진을 사용하여 신청자에게 최적의 농지를 추천합니다.
    """
    if not kmeans_model or not preprocessors or not farmlands_data:
        raise HTTPException(status_code=500, detail="모델 또는 데이터가 로드되지 않았습니다. 먼저 /train 엔드포인트를 호출해주세요.")

    trust_profile = buyer.trustProfile
    
    # --- 1단계: K-Means 클러스터 예측 (오직 '작물' 기준) ---
    applicant_vector = preprocessors['crop_mlb'].transform([trust_profile.interestCrop])
    predicted_cluster = kmeans_model.predict(applicant_vector)[0]

    # --- 2단계: 클러스터 및 거래 조건으로 후보군 필터링 ---
    all_farmland_labels = kmeans_model.labels_
    candidate_farmlands = []
    for i, farm in enumerate(farmlands_data):
        if all_farmland_labels[i] == predicted_cluster:
            # 2-1. 기본 거래 조건 필터링
            if (farm.landTrade in trust_profile.wantTrade and
                farm.landPrice <= trust_profile.budget and
                farm.landWhen == trust_profile.wantPeriod):
                candidate_farmlands.append(farm)

    # --- 3단계: 최종 후보군 점수 계산 및 랭킹 ---
    best_matches = find_best_matches(
        buyer, 
        candidate_farmlands,
        sellers_data,
        profit_informations_data,
        licenses_data
    )

    return {
        "matches": best_matches,
        "cluster_info": {
            "predicted_cluster": int(predicted_cluster),
            "candidate_count_in_cluster": len(candidate_farmlands)
        }
    }

# --- New GET Endpoints ---
@app.get("/sellers", response_model=List[Seller], tags=["Data Retrieval"])
def get_sellers():
    return sellers_data

@app.get("/licenses", response_model=List[License], tags=["Data Retrieval"])
def get_licenses():
    return licenses_data

@app.get("/matching_statuses", response_model=List[MatchingStatus], tags=["Data Retrieval"])
def get_matching_statuses():
    return matching_statuses_data

@app.get("/profit_informations", response_model=List[ProfitInformation], tags=["Data Retrieval"])
def get_profit_informations():
    return profit_informations_data

@app.get("/recommender_informations", response_model=List[RecommenderInformation], tags=["Data Retrieval"])
def get_recommender_informations():
    return recommender_informations_data

@app.get("/favorite_farmlands", response_model=List[FavoriteFarmland], tags=["Data Retrieval"])
def get_favorite_farmlands():
    return favorite_farmlands_data
