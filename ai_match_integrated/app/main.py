import os
import json
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from app.models import Buyer, MatchResult, Farmland
from ml.trainer import train_and_save_model, MODEL_PATH, PREPROCESSOR_PATH, SAMPLE_DATA_PATH
from core.matching_logic import find_best_matches
from typing import List

# --- FastAPI 앱 정의 ---
app = FastAPI(
    title="AI Hybrid Matching Server",
    description="K-means 클러스터링과 점수 기반 랭킹을 결합한 하이브리드 AI 매칭 시스템"
)

# --- 전역 변수 ---
kmeans_model = None
preprocessors = None
farmlands_data: List[Farmland] = []
buyers_data: List[Buyer] = []

@app.on_event("startup")
def load_model_and_data():
    """서버 시작 시 모델과 데이터를 로드합니다."""
    global kmeans_model, preprocessors, farmlands_data, buyers_data
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
            buyers_data = [Buyer(**buyer) for buyer in data.get('buyers', [])]
        print(f"{len(farmlands_data)}개의 농지 데이터를 로드했습니다.")
        print(f"{len(buyers_data)}개의 구매자 데이터를 로드했습니다.")
    else:
        print("경고: 샘플 데이터 파일이 없습니다.")

@app.get("/")
def read_root():
    return {"message": "AI 하이브리드 매칭 서버에 오신 것을 환영합니다."}


@app.post("/farmlands-batch", tags=["Data Management"])
def create_farmlands_batch(new_farmlands: List[Farmland]):
    """새로운 농지 정보 리스트를 JSON 형식으로 받아 등록합니다."""
    added_count = 0
    existing_ids = {farm.landId for farm in farmlands_data}
    
    for new_farm in new_farmlands:
        if new_farm.landId not in existing_ids:
            farmlands_data.append(new_farm)
            existing_ids.add(new_farm.landId)
            added_count += 1

    # JSON 파일에 변경 사항 저장
    with open(SAMPLE_DATA_PATH, 'r+', encoding='utf-8') as f:
        # 현재 메모리에 있는 전체 데이터로 파일을 다시 씁니다.
        current_data = {
            "farmlands": [farm.dict() for farm in farmlands_data],
            "buyers": [buyer.dict() for buyer in buyers_data]
        }
        f.seek(0)
        json.dump(current_data, f, ensure_ascii=False, indent=4)
        f.truncate()

    return {"message": f"{added_count} farmlands created successfully"}


@app.post("/train", tags=["AI Model"])
def train_model_endpoint():
    """K-Means 모델을 학습시키고 최신 데이터로 업데이트합니다."""
    try:
        result = train_and_save_model(n_clusters=2)
        load_model_and_data() # 학습 후 모델과 데이터 다시 로드
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

    # --- 1단계: K-Means 클러스터 예측 ---
    trust_profile = buyer.trustProfile
    applicant_vector = preprocessors['crop_mlb'].transform([trust_profile.interestCrop])
    predicted_cluster = kmeans_model.predict(applicant_vector)[0]

    # --- 2단계: 클러스터로 후보군 필터링 ---
    all_farmland_labels = kmeans_model.labels_
    candidate_farmlands = []
    for i, farm in enumerate(farmlands_data):
        if all_farmland_labels[i] == predicted_cluster:
            candidate_farmlands.append(farm)

    # --- 3단계: 최종 후보군 점수 계산 및 랭킹 ---
    best_matches = find_best_matches(buyer, candidate_farmlands)

    return {
        "matches": best_matches,
        "cluster_info": {
            "predicted_cluster": int(predicted_cluster),
            "candidate_count_in_cluster": len(candidate_farmlands)
        }
    }