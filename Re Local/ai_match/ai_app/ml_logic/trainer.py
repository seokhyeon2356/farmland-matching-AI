
import json
import joblib
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler, OneHotEncoder
import pathlib

# --- 경로 설정 ---
# 코드가 어디서 실행되든 항상 이 파일의 위치를 기준으로 경로를 설정합니다.
# 이렇게 하면 경로 문제로 인한 오류를 방지할 수 있습니다.
BASE_DIR = pathlib.Path(__file__).parent.parent.parent # ai_match 폴더
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "ai_app" / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

UPDATED_SAMPLE_DATA_PATH = DATA_DIR / "sample_data_v3.json"
MODEL_PATH = MODEL_DIR / "kmeans_model_v2.joblib"
PREPROCESSOR_PATH = MODEL_DIR / "preprocessors_v2.joblib"

def train_and_save_model(n_clusters=3):
    print("AI 모델(v2) 학습을 시작합니다...")

    with open(UPDATED_SAMPLE_DATA_PATH, 'r', encoding='utf-8') as f:
        all_users = json.load(f)

    # --- 데이터 변환기(Preprocessor) 학습 ---
    
    # 1. 다중 라벨 데이터 (작물, 장비, 선호 토양/용수)
    crop_mlb = MultiLabelBinarizer().fit([u.get('crops_recommended', []) or u.get('interested_crops', []) for u in all_users])
    tool_mlb = MultiLabelBinarizer().fit([u.get('tools', []) for u in all_users if u['type'] == 'applicant'])
    pref_soil_mlb = MultiLabelBinarizer().fit([u.get('preferences', {}).get('soil', []) for u in all_users if u['type'] == 'applicant'])
    pref_water_mlb = MultiLabelBinarizer().fit([u.get('preferences', {}).get('water_source', []) for u in all_users if u['type'] == 'applicant'])

    # 2. 단일 카테고리 데이터 (농지 토양, 용수)
    soil_ohe = OneHotEncoder(handle_unknown='ignore').fit([[u['facilities']['soil']] for u in all_users if u['type'] == 'farmland'])
    water_ohe = OneHotEncoder(handle_unknown='ignore').fit([[u['facilities']['water_source']] for u in all_users if u['type'] == 'farmland'])

    # 3. 숫자 데이터 (면적, 위도, 경도)
    scaler = StandardScaler().fit([[u['area'], u['lat'], u['lng']] for u in all_users if u['type'] == 'farmland'])

    # --- 전체 사용자 데이터 벡터화 ---
    processed_users = []
    boolean_features = ['agri_water', 'electricity', 'warehouse', 'greenhouse', 'fence', 'paved_road', 'car_access', 'public_transport', 'machine_access', 'road_adjacent']
    for user in all_users:
        vector = []
        if user['type'] == 'farmland':
            fac = user['facilities']
            vector.extend(crop_mlb.transform([user['crops_recommended']])[0])
            vector.extend(np.zeros(len(tool_mlb.classes_)))
            vector.extend(np.zeros(len(pref_soil_mlb.classes_)))
            vector.extend(np.zeros(len(pref_water_mlb.classes_)))
            vector.extend(soil_ohe.transform([[fac['soil']]]).toarray()[0])
            vector.extend(water_ohe.transform([[fac['water_source']]]).toarray()[0])
            vector.extend(scaler.transform([[user['area'], user['lat'], user['lng']]])[0])
            vector.extend([int(fac[key]) for key in boolean_features])
        else: # applicant
            pref = user['preferences']
            vector.extend(crop_mlb.transform([user['interested_crops']])[0])
            vector.extend(tool_mlb.transform([user['tools']])[0])
            vector.extend(pref_soil_mlb.transform([pref['soil']])[0])
            vector.extend(pref_water_mlb.transform([pref['water_source']])[0])
            vector.extend(np.zeros(len(soil_ohe.categories_[0])))
            vector.extend(np.zeros(len(water_ohe.categories_[0])))
            vector.extend(scaler.transform([[0, user['home_lat'], user['home_lng']]])[0]) # 면적 0
            vector.extend([int(pref[key]) for key in boolean_features])
        processed_users.append(vector)

    processed_users = np.array(processed_users)

    # --- K-means 모델 학습 및 저장 ---
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(processed_users)

    joblib.dump(kmeans, MODEL_PATH)
    joblib.dump({
        'crop_mlb': crop_mlb, 'tool_mlb': tool_mlb, 'pref_soil_mlb': pref_soil_mlb, 
        'pref_water_mlb': pref_water_mlb, 'soil_ohe': soil_ohe, 'water_ohe': water_ohe, 'scaler': scaler
    }, PREPROCESSOR_PATH)

    print(f"모델(v2) 학습 완료! 모델은 '{MODEL_PATH}'에 저장되었습니다.")
    return {"message": "Model training successful (v2)", "clusters_created": n_clusters, "vector_size": processed_users.shape[1]}
