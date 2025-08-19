# ai_app/main.py
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Tuple
from math import radians, sin, cos, sqrt, atan2
import numpy as np
import joblib
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

app = FastAPI(title="Farmland-Applicant Matching (KMeans + Rules, Area-Weighted Tools)")

# ------------------------------
# 저장 경로
# ------------------------------
MODEL_DIR = Path(__file__).parent / "models"
MODEL_DIR.mkdir(exist_ok=True, parents=True)
F_MODEL_PATH = MODEL_DIR / "kmeans_farmland.joblib"
A_MODEL_PATH = MODEL_DIR / "kmeans_applicant.joblib"
F_SCALER_PATH = MODEL_DIR / "scaler_farmland.joblib"
A_SCALER_PATH = MODEL_DIR / "scaler_applicant.joblib"

# ------------------------------
# 하이퍼파라미터 & 가중치
# ------------------------------
N_CLUSTERS_FARMLAND = 4
N_CLUSTERS_APPLICANT = 4

WEIGHTS = {
    "cert_per_item": 0.0,            # ✅ 자격증은 매칭 점수에서 완전 제외
    "tool_per_item": 0.0,            # 장비 개수 자체 가점도 제거(이중 계산 방지)
    "trade_match": 3.0,              # 선호 거래방식 있으면 +3 (원하면 0으로)
    "crop_match": 0.0,               # (구) 경험 기반 매칭은 비활성화, 새 로직 사용
    "area_good": 5.0,                # (보조) 면적 > 1000㎡ 기본 가점
    "distance_penalty_per_km": 0.05, # 거리 km당 -0.05 (최대 -10)
    "cluster_bonus_same": 8.0,       # 군집 동일 +8
    "cluster_bonus_similar": 4.0     # 군집 인접 +4
}

# --- 필수 4지표용 설정 ---
CROP_INTEREST_BONUS = 8  # 관심 작물 교집합 1개당 가점

# 작물-장비 시너지 (팀 합의로 수시 튜닝)
CROP_TOOL_WEIGHT = {
    "양파": {"관리기": 2, "소형트랙터": 2},
    "감자": {"소형트랙터": 3, "예취기": 1},
    "딸기": {"하우스자재": 3, "관수장비": 2},
}

# 작업-필요장비 매핑 (장비 커버리지)
TASK_TOOLS = {
    "경운": ["관리기", "소형트랙터", "경운기"],
    "정식": ["이식기", "육묘트레이"],
    "수확": ["수확기", "운반수레", "예취기"],
    "운반": ["소형트랙터", "경운기", "화물차"],
}

# 군집 보너스
def cluster_bonus(f_label: int, a_label: int) -> float:
    if f_label == a_label:
        return WEIGHTS["cluster_bonus_same"]
    if abs(f_label - a_label) == 1:
        return WEIGHTS["cluster_bonus_similar"]
    return 0.0

# ------------------------------
# 스키마 (필수 4지표 필드 포함)
# ------------------------------
class Farmland(BaseModel):
    id: Optional[int] = None
    area: Optional[float] = None          # ㎡
    address: Optional[str] = None
    crop_history: List[str] = Field(default_factory=list)  # 과거 재배 이력(참고용)
    lat: Optional[float] = None
    lng: Optional[float] = None

    # 필수 지표용
    crops_recommended: List[str] = Field(default_factory=list)  # 권장/최근 작물
    required_tasks: List[str] = Field(default_factory=list)     # 요구 작업(경운/정식/수확/운반 등)

class Applicant(BaseModel):
    id: Optional[int] = None
    age: Optional[int] = None
    certificates: List[str] = Field(default_factory=list)       # 매칭 점수엔 미반영(신뢰 점수용)
    tools: List[str] = Field(default_factory=list)
    preferred_trade: List[str] = Field(default_factory=list)    # ["매입","임대","공유농","기타"]
    crop_experience: List[str] = Field(default_factory=list)    # (구)경험 — 현재 미사용

    # 필수 지표용
    interested_crops: List[str] = Field(default_factory=list)   # 관심 작물
    home_lat: Optional[float] = None
    home_lng: Optional[float] = None

class MatchRequest(BaseModel):
    farmland: Farmland
    applicant: Applicant

class TrainPayload(BaseModel):
    farmlands: List[Farmland] = Field(default_factory=list)
    applicants: List[Applicant] = Field(default_factory=list)
    n_clusters_farmland: Optional[int] = None
    n_clusters_applicant: Optional[int] = None

# ------------------------------
# 유틸
# ------------------------------
def haversine_km(lat1, lon1, lat2, lon2) -> Optional[float]:
    if None in (lat1, lon1, lat2, lon2):
        return None
    R = 6371.0
    p1, p2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)
    a = sin(dphi/2)**2 + cos(p1)*cos(p2)*sin(dlambda/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c

def area_multiplier(area: Optional[float]) -> float:
    """
    면적이 클수록 '장비 관련' 가점이 커지도록 보정.
    - ≤ 1,000㎡: x1.00
    - 1,000 ~ 10,000㎡: 선형으로 x1.00 → x1.60
    - ≥ 10,000㎡: x1.80 (상한)
    """
    if not area:
        return 1.0
    if area <= 1000:
        return 1.0
    if area >= 10000:
        return 1.80
    return 1.0 + 0.6 * ((area - 1000) / 9000.0)

# 피처 (KMeans용)
def farmland_features(f: Farmland) -> np.ndarray:
    area = f.area if f.area is not None else 0.0
    crop_len = len(f.crop_history)
    lat = f.lat if f.lat is not None else 0.0
    lng = f.lng if f.lng is not None else 0.0
    return np.array([area, crop_len, lat, lng], dtype=float)

def applicant_features(a: Applicant) -> np.ndarray:
    age = a.age if a.age is not None else 0.0
    cert_len = len(a.certificates)
    tool_len = len(a.tools)
    pref_len = len(a.preferred_trade)
    return np.array([age, cert_len, tool_len, pref_len], dtype=float)

def load_or_fit_defaults():
    # 모델/스케일러가 없으면, 더미 데이터로 초기 학습
    if not (F_MODEL_PATH.exists() and F_SCALER_PATH.exists()):
        farmlands_dummy = np.array([
            [800, 1, 36.77, 126.96],
            [1500, 2, 36.72, 126.95],
            [300, 0, 36.79, 126.98],
            [2500, 3, 36.70, 126.93],
            [1200, 1, 36.76, 126.97],
        ], dtype=float)
        f_scaler = StandardScaler().fit(farmlands_dummy)
        f_X = f_scaler.transform(farmlands_dummy)
        f_kmeans = KMeans(n_clusters=N_CLUSTERS_FARMLAND, n_init=10, random_state=42).fit(f_X)
        joblib.dump(f_scaler, F_SCALER_PATH)
        joblib.dump(f_kmeans, F_MODEL_PATH)

    if not (A_MODEL_PATH.exists() and A_SCALER_PATH.exists()):
        applicants_dummy = np.array([
            [24, 1, 1, 1],
            [31, 2, 3, 2],
            [27, 0, 1, 1],
            [40, 3, 2, 2],
            [22, 1, 0, 1],
        ], dtype=float)
        a_scaler = StandardScaler().fit(applicants_dummy)
        a_X = a_scaler.transform(applicants_dummy)
        a_kmeans = KMeans(n_clusters=N_CLUSTERS_APPLICANT, n_init=10, random_state=42).fit(a_X)
        joblib.dump(a_scaler, A_SCALER_PATH)
        joblib.dump(a_kmeans, A_MODEL_PATH)

def get_models():
    load_or_fit_defaults()
    f_scaler = joblib.load(F_SCALER_PATH)
    f_kmeans = joblib.load(F_MODEL_PATH)
    a_scaler = joblib.load(A_SCALER_PATH)
    a_kmeans = joblib.load(A_MODEL_PATH)
    return f_scaler, f_kmeans, a_scaler, a_kmeans

# ------------------------------
# 규칙 기반(경량) + 필수 4지표
# ------------------------------
def rule_score(f: Farmland, a: Applicant) -> Tuple[float, List[str]]:
    score = 50.0
    reasons: List[str] = []

    # 거래 형태(선택)
    if a.preferred_trade:
        score += WEIGHTS["trade_match"]
        reasons.append(f"거래 형태 가점 +{WEIGHTS['trade_match']:.1f}")
    else:
        reasons.append("거래 형태 가점 +0.0")

    # 면적(보조 규칙)
    if (f.area or 0) > 1000:
        score += WEIGHTS["area_good"]
        reasons.append(f"면적 가점 +{WEIGHTS['area_good']:.1f} (>1000㎡)")
    else:
        reasons.append("면적 가점 +0.0")

    # (구) 경험 기반 매칭은 비활성화
    reasons.append("작물 경험 매칭 +0.0 (비활성화)")
    reasons.append("장비 개수 가점 +0.0 (면적 가중 시너지/커버리지로 반영)")
    return score, reasons

def crop_compatibility_score(f_crops: List[str], a_interests: List[str],
                             a_tools: List[str], area_for_multiplier: Optional[float]) -> Tuple[float, List[str]]:
    score = 0.0
    reasons: List[str] = []

    # 1) 관심 작물 교집합 가점
    if f_crops and a_interests:
        match = set(f_crops) & set(a_interests)
        if match:
            gain = len(match) * CROP_INTEREST_BONUS
            score += gain
            reasons.append(f"작물 관심 일치 +{gain} ({', '.join(match)})")
        else:
            reasons.append("작물 관심 일치 +0")
    else:
        reasons.append("작물/관심 작물 정보 부족")

    # 2) 작물-장비 시너지 (면적 보정 적용)
    tool_gain_raw = 0
    for crop in f_crops or []:
        tool_map = CROP_TOOL_WEIGHT.get(crop, {})
        tool_gain_raw += sum(tool_map.get(t, 0) for t in (a_tools or []))

    if tool_gain_raw > 0:
        mul = area_multiplier(area_for_multiplier)
        tool_gain = round(tool_gain_raw * mul, 1)
        score += tool_gain
        reasons.append(f"작물 장비 시너지 +{tool_gain} (면적 보정 x{mul:.2f})")
    else:
        reasons.append("작물 장비 시너지 +0")
    return score, reasons

def equipment_coverage_score(required_tasks: List[str], tools: List[str],
                             area_for_multiplier: Optional[float]) -> Tuple[float, List[str]]:
    covered = 0
    reasons: List[str] = []
    for task in required_tasks or []:
        needs = TASK_TOOLS.get(task, [])
        if any(t in (tools or []) for t in needs):
            covered += 1
            reasons.append(f"작업[{task}] 충족")

    if required_tasks:
        rate = covered / len(required_tasks)
        base = round(rate * 12, 1)  # 최대 12점
        mul = area_multiplier(area_for_multiplier)
        s = round(base * mul, 1)
        if covered == 0:
            reasons.append("작업 충족 없음")
        else:
            reasons.append(f"작업 충족률 {int(rate*100)}% (+{s}, 면적 보정 x{mul:.2f})")
        return s, reasons
    else:
        return 0.0, ["요구 작업 없음"]

def scale_fit_score(area: Optional[float], tools: List[str]) -> Tuple[float, str]:
    if not area:
        return 0.0, "면적 정보 없음"
    has_tractor = any(x in (tools or []) for x in ["소형트랙터", "트랙터", "경운기"])
    if area > 10000:
        if has_tractor:
            return 5.0, "대면적 + 트랙터 보유 +5"
        return -4.0, "대면적 대비 장비 부족 -4"
    if area > 1000:
        return 3.0, "중대면적 가점 +3"
    return 1.0, "소면적 가점 +1"

def distance_penalty_score(f_lat: Optional[float], f_lng: Optional[float],
                           a_lat: Optional[float], a_lng: Optional[float]) -> Tuple[float, str]:
    d = haversine_km(f_lat, f_lng, a_lat, a_lng)
    if d is None:
        return 0.0, "거리 정보 없음"
    pen = min(10.0, d * WEIGHTS["distance_penalty_per_km"])
    return -pen, f"거리 패널티 -{pen:.1f} ({d:.1f}km)"

def label_from_score(s: float) -> str:
    if s >= 80: return "매우 높음"
    if s >= 70: return "높음"
    if s >= 55: return "보통"
    if s >= 40: return "낮음"
    return "매우 낮음"

# ------------------------------
# 엔드포인트
# ------------------------------
@app.post("/score")
def score(req: MatchRequest) -> Dict[str, Any]:
    f_scaler, f_kmeans, a_scaler, a_kmeans = get_models()

    # 1) 경량 규칙 기반(기본 50점 + 거래/면적 보조)
    base_score, reasons = rule_score(req.farmland, req.applicant)

    # 1-추가) 필수 4지표 합산 (면적 가중 반영)
    # (a) 작물 적합도
    crop_score, crop_reasons = crop_compatibility_score(
        req.farmland.crops_recommended, req.applicant.interested_crops,
        req.applicant.tools, req.farmland.area
    )
    base_score += crop_score
    reasons += crop_reasons

    # (b) 장비 커버리지
    equip_score, equip_reasons = equipment_coverage_score(
        req.farmland.required_tasks, req.applicant.tools, req.farmland.area
    )
    base_score += equip_score
    reasons += equip_reasons

    # (c) 면적·규모 적합도
    scale_score, scale_reason = scale_fit_score(req.farmland.area, req.applicant.tools)
    base_score += scale_score
    reasons.append(scale_reason)

    # (d) 거리/접근성
    dist_pen, dist_reason = distance_penalty_score(
        req.farmland.lat, req.farmland.lng, req.applicant.home_lat, req.applicant.home_lng
    )
    base_score += dist_pen
    reasons.append(dist_reason)

    # 2) K-Means 군집 보너스
    f_vec = farmland_features(req.farmland).reshape(1, -1)
    a_vec = applicant_features(req.applicant).reshape(1, -1)
    f_label = int(f_kmeans.predict(f_scaler.transform(f_vec))[0])
    a_label = int(a_kmeans.predict(a_scaler.transform(a_vec))[0])

    bonus = cluster_bonus(f_label, a_label)
    total = np.clip(base_score + bonus, 0, 100)
    reasons.append(f"군집 보너스 +{bonus:.1f} (농지군집 {f_label}, 청년군집 {a_label})")

    return {
        "score": float(total),
        "label": label_from_score(float(total)),
        "reasons": reasons,
        "clusters": {"farmland_label": f_label, "applicant_label": a_label},
        "model_version": "hybrid-kmeans-rules-v0.3.0"
    }

@app.post("/train")
def train(payload: TrainPayload) -> Dict[str, Any]:
    n_f = payload.n_clusters_farmland or N_CLUSTERS_FARMLAND
    n_a = payload.n_clusters_applicant or N_CLUSTERS_APPLICANT

    if payload.farmlands:
        f_mat = np.vstack([farmland_features(f) for f in payload.farmlands])
        f_scaler = StandardScaler().fit(f_mat)
        f_X = f_scaler.transform(f_mat)
        f_kmeans = KMeans(n_clusters=n_f, n_init=10, random_state=42).fit(f_X)
        joblib.dump(f_scaler, F_SCALER_PATH)
        joblib.dump(f_kmeans, F_MODEL_PATH)

    if payload.applicants:
        a_mat = np.vstack([applicant_features(a) for a in payload.applicants])
        a_scaler = StandardScaler().fit(a_mat)
        a_X = a_scaler.transform(a_mat)
        a_kmeans = KMeans(n_clusters=n_a, n_init=10, random_state=42).fit(a_X)
        joblib.dump(a_scaler, A_SCALER_PATH)
        joblib.dump(a_kmeans, A_MODEL_PATH)

    return {
        "status": "ok",
        "n_clusters_farmland": n_f,
        "n_clusters_applicant": n_a,
        "message": "KMeans models updated (if datasets provided)."
    }
