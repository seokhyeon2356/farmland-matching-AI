# Re:Local AI 매칭 서버 (최종 개선판)

이 프로젝트는 FastAPI와 Scikit-learn을 사용하여, Re:Local 커뮤니티를 위한 AI 매칭 시스템의 **최종 개선판**입니다. 농지 소유자가 제공하는 모든 유의미한 정보를 AI 모델 학습과 필터링에 사용하여, 사용자에게 최적의 매칭 결과를 제공합니다.

## 실행 순서

AI 모델을 먼저 **학습**시킨 후, **매칭** 기능을 사용해야 합니다.

### 1. `ai_match` 폴더로 이동

터미널에서 `ai_match` 폴더로 이동해야 합니다.

```bash
cd ai_match
```

### 2. 필수 라이브러리 설치

`ai_match` 폴더 안에 있는 상태에서, 아래 명령어로 필요한 모든 라이브러리를 설치합니다.

```bash
pip install -r requirements.txt
```

### 3. FastAPI 서버 실행

다음 명령어로 API 서버를 실행합니다.

```bash
uvicorn ai_app.main:app --reload --port 8000
```

### 3. AI 모델 학습시키기 (v2)

매칭 기능을 사용하기 전에, 반드시 먼저 개선된 모델을 학습시켜야 합니다.

- 웹 브라우저에서 [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) 로 접속합니다.
- `POST /train_v2` 엔드포인트를 찾아 클릭합니다.
- **Try it out** 버튼을 누른 후, **Execute** 버튼을 클릭합니다.
- 실행이 성공하면, `ai_app/model/` 폴더 안에 `kmeans_model_v2.joblib` 파일과 `preprocessors_v2.joblib` 파일이 생성됩니다.

### 4. 구매자 매칭 테스트하기 (v2)

모델 학습이 완료되었다면, 이제 특정 구매자에게 맞는 농지 그룹을 찾아볼 수 있습니다.

- 다시 API 문서([http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs))로 돌아옵니다.
- `POST /match_v2` 엔드포인트를 찾아 클릭합니다.
- **Try it out** 버튼을 누릅니다.
- **Request body**에 매칭을 원하는 구매자(신청자)의 상세 정보를 JSON 형식으로 입력합니다. 아래 예시를 복사해서 사용해 보세요.

```json
{
  "id": 101,
  "home_lat": 36.765,
  "home_lng": 126.965,
  "tools": [
    "소형트랙터",
    "관리기"
  ],
  "interested_crops": [
    "양파",
    "감자"
  ],
  "preferences": {
    "soil": [
      "사양토",
      "양토"
    ],
    "water_source": [
      "지하수"
    ],
    "agri_water": true,
    "electricity": true,
    "warehouse": true,
    "greenhouse": false,
    "fence": true,
    "paved_road": true,
    "car_access": true,
    "public_transport": false,
    "machine_access": true,
    "road_adjacent": true
  },
  "transaction": {
    "trade_type": "임대",
    "budget": 2000000,
    "preferred_sell_period": "3개월 이내"
  }
}
```

- **Execute** 버튼을 클릭합니다.

### 5. 최종 결과 확인

Response body에서 최종 매칭 결과를 확인합니다. 결과는 두 종류의 목록을 포함합니다.

- **`matched_aimatch`**: AI가 클러스터링을 통해 "유사하다"고 판단한 모든 농지 목록입니다.
- **`filtered_aimatch`**: 위 목록에서 사용자의 거래 조건(거래 형태, 예산, 희망 시기)에 맞는 것을 추가로 걸러낸 **최종 추천 목록**입니다.