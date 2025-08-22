from math import radians, sin, cos, sqrt, atan2
from typing import List, Dict, Any
from app.models import Farmland, Buyer
import re

# --- 작물 정규화 헬퍼
def _normalize_crops(text: str) -> set[str]:
    """
    '사과/배, 포도' 같은 문자열을 소문자/트림 후 set으로 변환.
    구분자는 콤마/슬래시/공백을 모두 허용.
    """
    if not text:
        return set()
    items = re.split(r"[,\s/]+", str(text))
    return {it.strip().lower() for it in items if it and it.strip()}

# --- 점수 계산 헬퍼 함수 ---
def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """두 지점 간의 거리를 하버사인 공식을 이용해 km 단위로 계산합니다."""
    R = 6371.0  # 지구 반지름 (km)
    
    lat1_rad, lon1_rad = radians(lat1), radians(lon1)
    lat2_rad, lon2_rad = radians(lat2), radians(lon2)
    
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    
    a = sin(dlat / 2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    
    return R * c

# --- 핵심 매칭 로직 ---
def calculate_distance_score(farmland: Farmland, buyer: Buyer, max_score: int = 25) -> float:
    """거리 점수를 계산합니다. 5km 이내 만점, 50km 초과 0점."""
    distance = haversine_distance(farmland.landLat, farmland.landLng, buyer.buyerLat, buyer.buyerLng)
    
    if distance <= 5:
        return max_score
    elif distance > 50:
        return 0
    else:
        # 5km에서 50km 사이를 선형적으로 감소
        return max_score * (1 - (distance - 5) / 45)

def _normalize_crops(text: str) -> set[str]:
    """
    '사과/배, 포도  ' 처럼 섞인 문자열을 소문자, 공백제거해서 set으로 변환.
    구분자는 콤마/슬래시/공백을 모두 허용.
    """
    if not text:
        return set()
    items = re.split(r"[,\s/]+", str(text))
    return {it.strip().lower() for it in items if it and it.strip()}

def calculate_crop_score(farmland: Farmland, buyer: Buyer, max_score: int = 25) -> float:
    applicant_crops = {c.strip().lower() for c in buyer.trustProfile.interestCrop if c and str(c).strip()}
    farm_crops = _normalize_crops(farmland.landCrop)

    if not applicant_crops:
        return 0.0
    inter = applicant_crops & farm_crops
    if not inter:
        return 0.0
    return (len(inter) / len(applicant_crops)) * max_score

def calculate_facility_score(farmland: Farmland, max_score: int = 25) -> float:
    """시설 점수를 계산합니다."""
    score = 0
    if farmland.landWater: score += 2.5
    if farmland.landElec: score += 2.5
    if farmland.landStorage: score += 2.5
    if farmland.landHouse: score += 2.5
    if farmland.landRoad: score += 2.5
    if farmland.landWellRoad: score += 2.5
    if farmland.landBus: score += 2.5
    if farmland.landCar: score += 2.5
    if farmland.landMachine: score += 2.5
    if farmland.landFence: score += 2.5
    return min(score, max_score)

def calculate_area_score(farmland: Farmland, buyer: Buyer, max_score: int = 25) -> float:
    """면적 적합도 점수를 계산합니다. 신청자 정보를 기반으로 이상적인 면적을 추정."""
    area = farmland.landArea
    tools = set(buyer.trustProfile.equipment)

    if '트렉터' in tools or '트랙터' in tools:
        ideal_area = 10000
    elif '경운기' in tools:
        ideal_area = 5000
    else:
        ideal_area = 1000

    diff_ratio = 1 - abs(area - ideal_area) / ideal_area
    return max(0, diff_ratio * max_score)

# --- 최종 매칭 점수 계산 ---
def calculate_total_match_score(
    farmland: Farmland, 
    buyer: Buyer,
    weights: Dict[str, float] = None
) -> Dict[str, Any]:
    if weights is None:
        weights = {
            'distance': 0.40,
            'crop': 0.30,
            'facility': 0.15,
            'area': 0.15
        }

    # 먼저 작물 점수만 계산
    crop_score = calculate_crop_score(farmland, buyer)
    if crop_score <= 0:
        return {
            "total_score": 0.0,                      # ✅ 작물 미스매치면 전체 0
            "details": {'distance': 0, 'crop': 0, 'facility': 0, 'area': 0},
        }

    # 나머지 점수 계산
    scores = {
        'distance': calculate_distance_score(farmland, buyer),
        'crop': crop_score,
        'facility': calculate_facility_score(farmland),
        'area': calculate_area_score(farmland, buyer)
    }
    total_score = sum(scores[key] * weights[key] for key in scores)

    return {
        "total_score": round(total_score, 2),
        "details": scores
    }


def find_best_matches(
    buyer: Buyer, 
    candidate_farmlands: List[Farmland]
) -> List[Dict[str, Any]]:
    """
    주어진 후보 농지 리스트에 대해 신청자와의 매칭 점수를 계산하고 정렬하여 반환합니다.
    """
    # 관심 작물 집합
    wanted = {c.strip().lower() for c in buyer.trustProfile.interestCrop if c and str(c).strip()}

    # ---- 작물 필터 (원하면 유지, 아니면 이 블록 삭제) ----
    def farm_has_wanted(f: Farmland) -> bool:
        return bool(wanted & _normalize_crops(f.landCrop))

    # 일치하는 게 있으면 그 농지만 남기고, 하나도 없으면 기존 후보 유지
    filtered = [f for f in candidate_farmlands if farm_has_wanted(f)]
    candidate_farmlands = filtered or candidate_farmlands
    # -----------------------------------------------------

    match_results = []
    for farm in candidate_farmlands:
        scores = calculate_total_match_score(farm, buyer)
        result = {
            "farmland_id": farm.landId,
            "farmland_name": farm.landName,
            "total_score": scores['total_score'],
            "score_details": scores['details'],
            # pydantic v2를 쓰면 model_dump(), v1이면 dict()
            "farmland_info": getattr(farm, "model_dump", getattr(farm, "dict"))()
        }
        match_results.append(result)

    match_results.sort(key=lambda x: x['total_score'], reverse=True)
    return match_results

