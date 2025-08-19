# Re:Local AI 하이브리드 매칭 서버

이 프로젝트는 Re:Local 서비스를 위한 AI 기반 농지-청년 매칭 시스템입니다.
**K-means 클러스터링**으로 유사한 농지 그룹을 빠르게 필터링하고, **점수 기반 랭킹**으로 개별 농지의 적합도를 정밀하게 계산하는 하이브리드 방식을 사용합니다.

## 주요 기능: 하이브리드 매칭 로직

1.  **1단계: K-means 클러스터링 필터링**
    *   전체 농지 데이터를 바탕으로 비슷한 특성의 농지들을 '그룹'(클러스터)으로 묶습니다.
    *   매칭 요청이 오면, 신청자의 특성을 분석하여 가장 적합한 농지 그룹을 예측하고, 해당 그룹의 농지만을 1차 후보로 선별합니다.

2.  **2단계: 점수 기반 정밀 랭킹**
    *   1차 선별된 후보군을 대상으로, 아래 4가지 핵심 요소를 종합 분석하여 개별 점수를 계산합니다.
        *   **거리/접근성**: 신청자의 거주지와 농지 사이의 거리
        *   **작물 적합도**: 관심 작물, 추천 작물, 보유 장비의 연관성
        *   **시설 일치도**: 선호 시설과 실제 시설의 일치도
        *   **면적 적합도**: 보유 장비에 따른 최적 면적과의 일치도
    *   계산된 총점이 높은 순으로 최종 추천 목록이 생성됩니다.

## 사용 방법

### 1. `ai_match` 폴더로 이동

터미널에서 이 프로젝트의 루트 폴더인 `ai_match`로 이동합니다.

```bash
# 예시: cd /path/to/your/project/ai/Re Local/ai_match
```

### 2. 필수 라이브러리 설치

아래 명령어로 필요한 모든 라이브러리를 설치합니다.

```bash
pip install -r requirements.txt
```

### 3. FastAPI 서버 실행

다음 명령어로 API 서버를 실행합니다.

```bash
python3 -m uvicorn ai_match.ai_app.main:app --reload --port 8000
```

### 4. AI 모델 학습 (매칭 전 필수)

매칭 기능을 사용하기 전에, 반드시 K-means 모델을 최신 데이터로 학습시켜야 합니다.

-   **API 문서**: 웹 브라우저에서 [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) 로 접속합니다.
-   **엔드포인트**: `POST /train`
-   API 문서에서 `POST /train`을 찾아 **Try it out** -> **Execute** 버튼을 순서대로 클릭하여 모델을 학습시킵니다.
-   성공적으로 완료되면 `ai_app/model/` 폴더에 모델 파일(`kmeans_model_v2.joblib`)과 전처리기 파일(`preprocessors_v2.joblib`)이 생성됩니다.

### 5. 매칭 기능 테스트

모델 학습 후, API를 호출하여 매칭 기능을 테스트할 수 있습니다.

-   **API 문서**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
-   **엔드포인트**: `POST /match`

#### 요청 본문 (Request Body) 예시

API 문서의 `Request body` 필드나 다른 API 테스트 도구에 아래의 JSON 데이터를 복사하여 사용하세요. 유일한 구매자 예시인 `id: 101`의 데이터입니다.

```json
{
  "id": 101,
  "home_lat": 37.4979,
  "home_lng": 127.0276,
  "tools": [
    "트랙터",
    "경운기"
  ],
  "interested_crops": [
    "쌀",
    "콩"
  ],
  "preferences": {
    "soil": [
      "사양토",
      "양토"
    ],
    "water_source": [
      "지하수",
      "저수지"
    ],
    "agri_water": true,
    "electricity": true,
    "warehouse": true,
    "greenhouse": false,
    "fence": false,
    "paved_road": true,
    "car_access": true,
    "public_transport": false,
    "machine_access": true,
    "road_adjacent": true
  },
  "transaction": {
    "trade_type": "임대",
    "budget": 6000000,
    "preferred_sell_period": "즉시"
  }
}
```

#### `curl`을 이용한 터미널 테스트 예시

아래는 터미널에서 `curl`을 사용하여 직접 테스트하는 전체 명령어입니다.

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/match' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "id": 101,
  "home_lat": 37.4979,
  "home_lng": 127.0276,
  "tools": [
    "트랙터",
    "경운기"
  ],
  "interested_crops": [
    "쌀",
    "콩"
  ],
  "preferences": {
    "soil": [
      "사양토",
      "양토"
    ],
    "water_source": [
      "지하수",
      "저수지"
    ],
    "agri_water": true,
    "electricity": true,
    "warehouse": true,
    "greenhouse": false,
    "fence": false,
    "paved_road": true,
    "car_access": true,
    "public_transport": false,
    "machine_access": true,
    "road_adjacent": true
  },
  "transaction": {
    "trade_type": "임대",
    "budget": 6000000,
    "preferred_sell_period": "즉시"
  }
}'
```

#### 응답 (Response) 예시

요청이 성공하면, 아래와 같이 조건에 맞는 여러 농지들이 점수가 높은 순으로 정렬되어 반환될 수 있습니다.

점수 계산 방식

  먼저, 저희가 만든 점수 계산 로직은 4가지 주요 항목(거리, 작물, 시설, 면적)으로 구성되어 있습니다. 각 항목의
  만점은 25점으로 설정되어 있습니다.

  그리고 이 4가지 항목의 점수를 단순히 합산하는 것이 아니라, 각각 다른 가중치(Weights)를 곱해서 최종 점수를
  계산합니다. 현재 matching_engine.py에 설정된 가중치는 다음과 같습니다.

   * 거리 (`distance`): 30% (가중치 0.3)
   * 작물 (`crop`): 30% (가중치 0.3)
   * 시설 (`facility`): 20% (가중치 0.2)
   * 면적 (`area`): 20% (가중치 0.2)