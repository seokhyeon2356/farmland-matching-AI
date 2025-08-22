import os
import json
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from typing import List

from app.models import Buyer, MatchResult, Farmland
from ml.trainer import train_and_save_model, MODEL_PATH, PREPROCESSOR_PATH, SAMPLE_DATA_PATH
from core.matching_logic import find_best_matches
# === main.py 상단 import들 아래 어딘가에 추가 ===
import re

_num_re = re.compile(r"-?\d+(?:\.\d+)?")
_TRUE_SET  = {"true","t","1","y","yes","on","예","가능","있음"}
_FALSE_SET = {"false","f","0","n","no","off","아니오","불가","없음"}

def _to_int_like(x):
    if isinstance(x, bool):  # bool은 int로 캐스팅 금지
        return int(x)
    if isinstance(x, (int, float)):
        return int(x)
    if isinstance(x, str):
        s = x.replace(",", "")
        m = _num_re.search(s)
        if m:
            return int(float(m.group()))
    raise ValueError(f"cannot coerce to int: {x!r}")

def _to_float_like(x):
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        s = x.replace(",", "")
        m = _num_re.search(s)
        if m:
            return float(m.group())
    raise ValueError(f"cannot coerce to float: {x!r}")

def _to_bool_like(x):
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(x)
    if isinstance(x, str):
        s = x.strip().lower()
        if s in _TRUE_SET:  return True
        if s in _FALSE_SET: return False
        m = _num_re.search(s)
        if m: return bool(int(float(m.group())))
    raise ValueError(f"cannot coerce to bool: {x!r}")

def _coerce_farmland_dict(d: dict) -> dict:
    """JSON에서 읽은 farmland 레코드를 Farmland가 먹을 수 있게 정규화"""
    out = dict(d)
    # ints
    for k in ("landId","ownerAge","landArea","landPrice"):
        if k in out:
            out[k] = _to_int_like(out[k])
    # floats
    for k in ("landLat","landLng"):
        if k in out:
            out[k] = _to_float_like(out[k])
    # bools
    for k in ("landWater","landElec","landMachine","landStorage","landHouse",
              "landFence","landRoad","landWellRoad","landBus","landCar"):
        if k in out:
            out[k] = _to_bool_like(out[k])
    return out

def _coerce_buyer_dict(d: dict) -> dict:
    out = dict(d)
    if "buyerId" in out:  out["buyerId"] = _to_int_like(out["buyerId"])
    if "buyerLat" in out: out["buyerLat"] = _to_float_like(out["buyerLat"])
    if "buyerLng" in out: out["buyerLng"] = _to_float_like(out["buyerLng"])
    tp = out.get("trustProfile") or {}
    if "budget" in tp: tp["budget"] = _to_int_like(tp["budget"])
    out["trustProfile"] = tp
    return out


# --- FastAPI 앱 정의 ---
app = FastAPI(
    title="AI Hybrid Matching Server",
    description="K-means 클러스터링과 점수 기반 랭킹을 결합한 하이브리드 AI 매칭 시스템"
)

# --- 전역 상태 ---
kmeans_model = None
preprocessors = None
farmlands_data: List[Farmland] = []
buyers_data: List[Buyer] = []


# ---------- 유틸: 현재 메모리 상태를 sample JSON에 저장 ----------
def _persist_state_to_file():
    os.makedirs(os.path.dirname(SAMPLE_DATA_PATH), exist_ok=True)
    with open(SAMPLE_DATA_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "farmlands": [farm.model_dump() for farm in farmlands_data],
            "buyers":    [buyer.model_dump() for buyer in buyers_data]
        }, f, ensure_ascii=False, indent=4)


# ---------- 기동 시 모델/데이터 로드 ----------
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
            farmlands_raw = data.get('farmlands', [])
            buyers_raw    = data.get('buyers', [])

# 문제 레코드 추적을 위해 1건씩 정규화 + 생성
            farmlands_tmp = []
            for i, farm in enumerate(farmlands_raw):
                try:
                    farm_norm = _coerce_farmland_dict(farm)
                    farmlands_tmp.append(Farmland(**farm_norm))
                except Exception as e:
                    print(f"[startup] farmland[{i}] 정규화/검증 실패: {e}. 원본={farm}")

            buyers_tmp = []
            for i, buyer in enumerate(buyers_raw):
                try:
                    buyer_norm = _coerce_buyer_dict(buyer)
                    buyers_tmp.append(Buyer(**buyer_norm))
                except Exception as e:
                    print(f"[startup] buyer[{i}] 정규화/검증 실패: {e}. 원본={buyer}")

            farmlands_data = farmlands_tmp
            buyers_data    = buyers_tmp

            print(f"{len(farmlands_data)}개의 농지 데이터를 로드했습니다.")
            print(f"{len(buyers_data)}개의 구매자 데이터를 로드했습니다.")
    else:
        print("경고: 샘플 데이터 파일이 없습니다.")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def read_root():
    return {"message": "AI 하이브리드 매칭 서버에 오신 것을 환영합니다."}


# ---------- (기존) 배치 추가: 보존 (개발용) ----------
@app.post("/farmlands-batch", tags=["Data Management"])
def create_farmlands_batch(new_farmlands: List[Farmland]):
    """새로운 농지 리스트를 append 방식으로 등록 (개발용)."""
    added_count = 0
    existing_ids = {farm.landId for farm in farmlands_data}

    for new_farm in new_farmlands:
        if new_farm.landId not in existing_ids:
            farmlands_data.append(new_farm)
            existing_ids.add(new_farm.landId)
            added_count += 1

    _persist_state_to_file()
    return {"message": f"{added_count} farmlands created successfully"}


# ---------- (신규) 전체 덮어쓰기: 운영용 ----------
@app.post("/farmlands-replace", tags=["Data Management"])
def replace_farmlands(new_farmlands: List[Farmland]):
    """
    DB의 '전체 농지 스냅샷'으로 교체(덮어쓰기).
    BE는 항상 이 엔드포인트로 전체 목록을 밀어주고 → /train → /match 순서로 호출.
    """
    global farmlands_data
    farmlands_data = list(new_farmlands)  # 깊은 복사 성격
    _persist_state_to_file()
    return {"message": f"replaced with {len(farmlands_data)} farmlands"}

# ---------- 학습 ----------
@app.post("/train", tags=["AI Model"])
def train_model_endpoint():
    """
    K-Means 모델 학습. 현재 메모리/파일에 저장된 최신 farmlands_data를 기준으로 학습.
    """
    try:
        if not farmlands_data:
            raise HTTPException(status_code=412, detail="no farmlands loaded. call /farmlands-replace first.")
        # trainer는 SAMPLE_DATA_PATH를 읽어 학습한다면, 현재 메모리를 파일로 먼저 동기화
        _persist_state_to_file()

        result = train_and_save_model(n_clusters=2)
        # 학습 후 모델/전처리/데이터 재로드
        load_model_and_data()
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"모델 학습 중 오류 발생: {e}")


# ---------- 매칭 ----------
@app.post("/match", response_model=MatchResult, tags=["AI Matching"])
def match_by_hybrid_model(buyer: Buyer):
    """
    하이브리드 매칭 엔진으로 신청자에게 최적의 농지를 추천.
    BE는 buyerId/buyerLat/buyerLng/trustProfile 형태로 JSON 전송.
    """
    if not kmeans_model or not preprocessors or not farmlands_data:
        raise HTTPException(status_code=500, detail="모델 또는 데이터가 로드되지 않았습니다. 먼저 /train 엔드포인트를 호출해주세요.")

    # 1) K-Means 클러스터 예측 (작물 기반 예시)
    trust_profile = buyer.trustProfile
    applicant_vector = preprocessors['crop_mlb'].transform([trust_profile.interestCrop])
    predicted_cluster = int(kmeans_model.predict(applicant_vector)[0])

    # 2) 해당 클러스터의 후보군 필터링
    all_farmland_labels = kmeans_model.labels_
    candidate_farmlands: List[Farmland] = [
        farm for i, farm in enumerate(farmlands_data) if all_farmland_labels[i] == predicted_cluster
    ]

    # 3) 점수 계산 및 랭킹 (상위 N개는 BE에서 저장 시 제한해도 됨)
    best_matches = find_best_matches(buyer, candidate_farmlands)

    return {
        "matches": best_matches,
        "cluster_info": {
            "predicted_cluster": predicted_cluster,
            "candidate_count_in_cluster": len(candidate_farmlands)
        }
    }
