
import json
from math import radians, sin, cos, sqrt, atan2
from typing import List, Dict, Any

# --- 데이터 로딩 ---
def load_data(file_path: str) -> List[Dict[str, Any]]:
    """JSON 파일에서 농지 및 신청자 데이터를 로드합니다."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

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

# --- 4대 핵심 매칭 로직 ---

def calculate_distance_score(farmland: Dict[str, Any], applicant: Dict[str, Any], max_score: int = 25) -> float:
    """거리 점수를 계산합니다. 5km 이내 만점, 50km 초과 0점."""
    distance = haversine_distance(farmland['lat'], farmland['lng'], applicant['home_lat'], applicant['home_lng'])
    
    if distance <= 5:
        return max_score
    elif distance > 50:
        return 0
    else:
        # 5km에서 50km 사이를 선형적으로 감소
        return max_score * (1 - (distance - 5) / 45)

def calculate_crop_score(farmland: Dict[str, Any], applicant: Dict[str, Any], max_score: int = 25) -> float:
    """작물 적합도 점수를 계산합니다."""
    applicant_crops = set(applicant['interested_crops'])
    farmland_crops = set(farmland['crops_recommended'])
    
    # 1. 직접적인 작물 일치 점수
    intersection = applicant_crops.intersection(farmland_crops)
    score = (len(intersection) / len(applicant_crops)) * max_score if applicant_crops else 0
    
    # 2. 장비-작물 가중치 보너스
    applicant_tools = set(applicant.get('tools', []))
    
    # 곡물/채소류와 트랙터/관리기 조합
    grain_veg_crops = {'쌀', '보리', '콩', '옥수수', '감자', '고구마', '채소'}
    heavy_machinery = {'트랙터', '관리기', '경운기'}
    
    if applicant_tools.intersection(heavy_machinery):
        if farmland_crops.intersection(grain_veg_crops):
            score *= 1.2 # 20% 보너스
            
    return min(score, max_score) # 최대 점수 초과 방지

def calculate_facility_score(farmland: Dict[str, Any], applicant: Dict[str, Any], max_score: int = 25) -> float:
    """시설 일치도 점수를 계산합니다."""
    applicant_prefs = applicant['preferences']
    farmland_facilities = farmland['facilities']
    
    total_prefs = 0
    matched_prefs = 0
    
    for key, preferred_value in applicant_prefs.items():
        # boolean 값으로 된 선호도만 계산
        if isinstance(preferred_value, bool):
            total_prefs += 1
            if farmland_facilities.get(key) == preferred_value:
                matched_prefs += 1
                
    return (matched_prefs / total_prefs) * max_score if total_prefs > 0 else 0

def calculate_area_score(farmland: Dict[str, Any], applicant: Dict[str, Any], max_score: int = 25) -> float:
    """면적 적합도 점수를 계산합니다. 신청자 정보를 기반으로 이상적인 면적을 추정."""
    # 현재 신청자 데이터에 희망 면적이 없으므로, 보유 장비 기반으로 추정
    # 예: 트랙터 보유 시 대규모 면적 선호, 관리기만 보유 시 중소규모 선호
    area = farmland['area']
    tools = set(applicant.get('tools', []))

    if '트랙터' in tools:
        ideal_area = 20000  # 2ha
    elif '경운기' in tools:
        ideal_area = 10000 # 1ha
    elif '관리기' in tools:
        ideal_area = 5000 # 0.5ha
    else:
        ideal_area = 2000 # 0.2ha (소규모)

    # 이상적인 면적과의 차이가 적을수록 높은 점수
    diff_ratio = 1 - abs(area - ideal_area) / ideal_area
    return max(0, diff_ratio * max_score)


# --- 최종 매칭 점수 계산 ---

def calculate_total_match_score(
    farmland: Dict[str, Any], 
    applicant: Dict[str, Any],
    weights: Dict[str, float] = None
) -> Dict[str, float]:
    """
    한 쌍의 농지-신청자에 대한 최종 매칭 점수를 계산합니다.
    """
    if weights is None:
        weights = {
            'distance': 0.30,
            'crop': 0.30,
            'facility': 0.20,
            'area': 0.20
        }
    
    scores = {
        'distance': calculate_distance_score(farmland, applicant),
        'crop': calculate_crop_score(farmland, applicant),
        'facility': calculate_facility_score(farmland, applicant),
        'area': calculate_area_score(farmland, applicant)
    }
    
    total_score = (
        scores['distance'] * weights['distance'] +
        scores['crop'] * weights['crop'] +
        scores['facility'] * weights['facility'] +
        scores['area'] * weights['area']
    )
    
    return {
        "total_score": round(total_score, 2),
        "details": scores
    }

def find_best_matches(
    applicant: Dict[str, Any], 
    candidate_farmlands: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    주어진 후보 농지 리스트에 대해 신청자와의 매칭 점수를 계산하고 정렬하여 반환합니다.
    """
    match_results = []
    for farm in candidate_farmlands:
        scores = calculate_total_match_score(farm, applicant)
        result = {
            "farmland_id": farm['id'],
            "farmland_name": farm['name'],
            "total_score": scores['total_score'],
            "score_details": scores['details'],
            "farmland_info": farm # 전체 농지 정보 포함
        }
        match_results.append(result)
        
    # 점수가 높은 순으로 정렬
    match_results.sort(key=lambda x: x['total_score'], reverse=True)
    
    return match_results

