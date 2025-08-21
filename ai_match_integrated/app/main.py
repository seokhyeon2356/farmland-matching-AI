import os
import json
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Form
from app.models import (
    Buyer, MatchResult, Farmland, Seller, License, MatchingStatus, 
    ProfitInformation, Recommender, FavoriteFarmland
)
from ml.trainer import train_and_save_model, MODEL_PATH, PREPROCESSOR_PATH, SAMPLE_DATA_PATH
from core.matching_logic import find_best_matches
from typing import List, Optional
from datetime import datetime

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
recommender_data: List[Recommender] = []
favorite_farmlands_data: List[FavoriteFarmland] = []

@app.on_event("startup")
def load_model_and_data():
    """서버 시작 시 모델과 데이터를 로드합니다."""
    global kmeans_model, preprocessors, farmlands_data, sellers_data, licenses_data, matching_statuses_data, profit_informations_data, recommender_data, favorite_farmlands_data
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
            recommender_data = [Recommender(**info) for info in data.get('recommender_informations', [])]
            favorite_farmlands_data = [FavoriteFarmland(**fav) for fav in data.get('favorite_farmlands', [])]
        print(f"{len(farmlands_data)}개의 농지 데이터를 로드했습니다.")
        print(f"{len(sellers_data)}개의 판매자 데이터를 로드했습니다.")
        print(f"{len(licenses_data)}개의 자격증 데이터를 로드했습니다.")
        print(f"{len(matching_statuses_data)}개의 매칭 현황 데이터를 로드했습니다.")
        print(f"{len(profit_informations_data)}개의 수익 정보 데이터를 로드했습니다.")
        print(f"{len(recommender_data)}개의 추천인 정보를 로드했습니다.")
        print(f"{len(favorite_farmlands_data)}개의 찜한 농지 정보를 로드했습니다.")
    else:
        print("경고: 샘플 데이터 파일이 없습니다.")

@app.get("/")
def read_root():
    return {"message": "Re:Local AI 하이브리드 매칭 서버 (ERD 연동 버전)에 오신 것을 환영합니다."}

# --- Data Management Endpoints ---
@app.post("/farmlands", tags=["Data Management"])
def create_farmland(
    landId: int = Form(...),
    landName: str = Form(...),
    landAddress: str = Form(...),
    landRoadAddress: Optional[str] = Form(None),
    landNumber: str = Form(...),
    landCrop: str = Form(...),
    landArea: int = Form(...),
    soiltype: str = Form(...),
    waterSource: str = Form(...),
    ownerName: str = Form(...),
    ownerAge: int = Form(...),
    ownerAddress: str = Form(...),
    landRegisterDate: datetime = Form(...),
    landWater: str = Form(...),
    landElec: str = Form(...),
    landMachine: str = Form(...),
    landStorage: Optional[str] = Form(None),
    landHouse: Optional[str] = Form(None),
    landFence: Optional[str] = Form(None),
    landRoad: str = Form(...),
    landWellRoad: str = Form(...),
    landBus: str = Form(...),
    landCar: str = Form(...),
    landTrade: str = Form(...),
    landMatch: Optional[str] = Form(None),
    landPrice: Optional[int] = Form(None),
    landWhen: Optional[str] = Form(None),
    landWhy: Optional[str] = Form(None),
    landComent: Optional[str] = Form(None),
    landRegister: Optional[str] = Form(None),
    landCadastre: Optional[str] = Form(None),
    landCertification: Optional[str] = Form(None),
    landImage: Optional[str] = Form(None),
    sellerFarmland: int = Form(...),
    lat: float = Form(...),
    lng: float = Form(...)
):
    """새로운 농지 정보를 form-data 형식으로 받아 등록합니다."""
    new_farmland = Farmland(
        landId=landId, landName=landName, landAddress=landAddress, landRoadAddress=landRoadAddress,
        landNumber=landNumber, landCrop=landCrop, landArea=landArea, soiltype=soiltype, waterSource=waterSource,
        ownerName=ownerName, ownerAge=ownerAge, ownerAddress=ownerAddress, landRegisterDate=landRegisterDate,
        landWater=landWater, landElec=landElec, landMachine=landMachine, landStorage=landStorage, landHouse=landHouse,
        landFence=landFence, landRoad=landRoad, landWellRoad=landWellRoad, landBus=landBus, landCar=landCar,
        landTrade=landTrade, landMatch=landMatch, landPrice=landPrice, landWhen=landWhen, landWhy=landWhy,
        landComent=landComent, landRegister=landRegister, landCadastre=landCadastre, landCertification=landCertification,
        landImage=landImage, sellerFarmland=sellerFarmland, lat=lat, lng=lng
    )
    farmlands_data.append(new_farmland)

    with open(SAMPLE_DATA_PATH, 'r+', encoding='utf-8') as f:
        data = json.load(f)
        data['farmlands'] = [farm.dict() for farm in farmlands_data]
        f.seek(0)
        json.dump(data, f, ensure_ascii=False, indent=4)
        f.truncate()

    return {"message": "Farmland created successfully", "farmland_id": new_farmland.landId}

@app.post("/sellers", tags=["Data Management"])
def create_sellers(new_sellers: List[Seller]):
    """새로운 판매자 정보를 JSON 형식으로 받아 등록합니다."""
    sellers_data.extend(new_sellers)

    with open(SAMPLE_DATA_PATH, 'r+', encoding='utf-8') as f:
        data = json.load(f)
        data['sellers'] = [seller.dict() for seller in sellers_data]
        f.seek(0)
        json.dump(data, f, ensure_ascii=False, indent=4)
        f.truncate()

    return {"message": f"{len(new_sellers)} sellers created successfully"}


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
    """하이브리드 매칭 엔진을 사용하여 신청자에게 최적의 농지를 추천합니다."""
    if not kmeans_model or not preprocessors or not farmlands_data:
        raise HTTPException(status_code=500, detail="모델 또는 데이터가 로드되지 않았습니다. 먼저 /train 엔드포인트를 호출해주세요.")

    trust_profile = buyer.trustProfile
    
    applicant_vector = preprocessors['crop_mlb'].transform([trust_profile.interestCrop])
    predicted_cluster = kmeans_model.predict(applicant_vector)[0]

    all_farmland_labels = kmeans_model.labels_
    candidate_farmlands = []
    for i, farm in enumerate(farmlands_data):
        if all_farmland_labels[i] == predicted_cluster:
            if (farm.landTrade in trust_profile.wantTrade and
                farm.landPrice <= trust_profile.budget and
                farm.landWhen == trust_profile.wantPeriod):
                candidate_farmlands.append(farm)

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

# --- Data Retrieval Endpoints ---
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

@app.get("/recommenders", response_model=List[Recommender], tags=["Data Retrieval"])
def get_recommenders():
    return recommender_data

@app.get("/favorite_farmlands", response_model=List[FavoriteFarmland], tags=["Data Retrieval"])
def get_favorite_farmlands():
    return favorite_farmlands_data
