from math import radians, sin, cos, sqrt, atan2
from typing import List, Dict, Any
from app.models import Farmland, Buyer, Seller, ProfitInformation, License
import re
from datetime import datetime

# --- 점수 계산 헬퍼 함수 ---
def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    lat1_rad, lon1_rad = radians(lat1), radians(lon1)
    lat2_rad, lon2_rad = radians(lat2), radians(lon2)
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = sin(dlat / 2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

# --- 4대 핵심 매칭 로직 ---
def calculate_distance_score(farmland: Farmland, buyer: Buyer, max_score: int = 25) -> float:
    distance = haversine_distance(farmland.lat, farmland.lng, buyer.home_lat, buyer.home_lng)
    if distance <= 5:
        return max_score
    elif distance > 50:
        return 0
    else:
        return max_score * (1 - (distance - 5) / 45)

def calculate_crop_score(farmland: Farmland, buyer: Buyer, max_score: int = 25) -> float:
    applicant_crops = set(buyer.trustProfile.interestCrop)
    farmland_crops = set([farmland.landCrop])
    intersection = applicant_crops.intersection(farmland_crops)
    score = (len(intersection) / len(applicant_crops)) * max_score if applicant_crops else 0
    applicant_tools = set(buyer.trustProfile.equipment)
    grain_veg_crops = {'쌀', '보리', '콩', '옥수수', '감자', '고구마', '채소'}
    heavy_machinery = {'트랙터', '관리기', '경운기'}
    if applicant_tools.intersection(heavy_machinery):
        if farmland_crops.intersection(grain_veg_crops):
            score *= 1.2
    return min(score, max_score)

def calculate_facility_score(farmland: Farmland, buyer: Buyer, max_score: int = 25) -> float:
    score = 0
    if farmland.landWater: score += 5
    if farmland.landElec: score += 5
    if farmland.landStorage: score += 5
    if farmland.landHouse: score += 5
    if farmland.landRoad: score += 5
    return min(score, max_score)

def calculate_area_score(farmland: Farmland, buyer: Buyer, max_score: int = 25) -> float:
    area = farmland.landArea
    tools = set(buyer.trustProfile.equipment)
    if '트랙터' in tools:
        ideal_area = 20000
    elif '경운기' in tools:
        ideal_area = 10000
    elif '관리기' in tools:
        ideal_area = 5000
    else:
        ideal_area = 2000
    diff_ratio = 1 - abs(area - ideal_area) / ideal_area
    return max(0, diff_ratio * max_score)

# --- 신규 추가 매칭 로직 ---
def calculate_profitability_score(farmland: Farmland, profit_infos: List[ProfitInformation], max_score: int = 15) -> float:
    profit_info = next((p for p in profit_infos if p.landId == farmland.landId), None)
    if not profit_info or not profit_info.netProfit:
        return 0
    
    try:
        net_profit_str = re.sub(r'[^0-9]', '', profit_info.netProfit)
        net_profit = int(net_profit_str) * 10000 if '만' in profit_info.netProfit else int(net_profit_str)
        
        if net_profit >= 5000000:
            return max_score
        elif net_profit <= 0:
            return 0
        else:
            return max_score * (net_profit / 5000000)
    except (ValueError, TypeError):
        return 0

def calculate_seller_reputation_score(farmland: Farmland, sellers: List[Seller], max_score: int = 10) -> float:
    # ERD에 따라 Farmland와 Seller를 연결하는 명확한 FK가 필요합니다. 
    # 현재는 임시로 seller_land 필드가 farmland.landId와 같다고 가정합니다.
    seller = next((s for s in sellers if s.seller_land == farmland.landId), None)
    if not seller or not seller.sellerYear:
        return 0
    
    experience_years = datetime.now().year - seller.sellerYear
    if experience_years >= 30:
        return max_score
    elif experience_years < 5:
        return 0
    else:
        return max_score * ((experience_years - 5) / 25)

def calculate_license_score(buyer: Buyer, licenses: List[License], max_score: int = 5) -> float:
    buyer_licenses = [lic for lic in licenses if lic.buyerLicense == buyer.buyerId]
    # 자격증 하나당 2.5점, 최대 5점
    return min(len(buyer_licenses) * 2.5, max_score)

# --- 최종 매칭 점수 계산 ---
def calculate_total_match_score(
    farmland: Farmland, 
    buyer: Buyer,
    sellers: List[Seller],
    profit_infos: List[ProfitInformation],
    licenses: List[License],
    weights: Dict[str, float] = None
) -> Dict[str, Any]:
    if weights is None:
        weights = {
            'distance': 0.25,
            'crop': 0.25,
            'facility': 0.15,
            'area': 0.15,
            'profitability': 0.10,
            'seller_reputation': 0.05,
            'license': 0.05
        }
    
    scores = {
        'distance': calculate_distance_score(farmland, buyer),
        'crop': calculate_crop_score(farmland, buyer),
        'facility': calculate_facility_score(farmland, buyer),
        'area': calculate_area_score(farmland, buyer),
        'profitability': calculate_profitability_score(farmland, profit_infos),
        'seller_reputation': calculate_seller_reputation_score(farmland, sellers),
        'license': calculate_license_score(buyer, licenses)
    }
    
    total_score = sum(scores[key] * weights[key] for key in scores)
    
    return {
        "total_score": round(total_score, 2),
        "details": scores
    }

def find_best_matches(
    buyer: Buyer, 
    candidate_farmlands: List[Farmland],
    sellers: List[Seller],
    profit_infos: List[ProfitInformation],
    licenses: List[License]
) -> List[Dict[str, Any]]:
    match_results = []
    for farm in candidate_farmlands:
        scores = calculate_total_match_score(farm, buyer, sellers, profit_infos, licenses)
        result = {
            "farmland_id": farm.landId,
            "farmland_name": farm.landName,
            "total_score": scores['total_score'],
            "score_details": scores['details'],
            "farmland_info": farm.dict()
        }
        match_results.append(result)
        
    match_results.sort(key=lambda x: x['total_score'], reverse=True)
    
    return match_results