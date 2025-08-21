# Re:Local AI 하이브리드 매칭 서버 (ERD 연동 버전)

이 프로젝트는 Re:Local 서비스를 위한 AI 기반 농지-청년 매칭 시스템의 **ERD 연동 리팩토링 버전**입니다.
기존의 매칭 로직을 백엔드 데이터베이스 스키마(ERD)에 맞춰 재구성하고, 코드 구조를 개선하여 관리 효율성을 높였습니다.

## 주요 변경 사항

-   **ERD 완전 동기화**: 백엔드 ERD의 모든 테이블과 필드를 Pydantic 모델에 100% 반영하여 데이터 무결성을 확보했습니다.
-   **매칭 로직 고도화**: '작물'을 기준으로 1차 클러스터링을 수행한 후, 거리, 가격, 시설, 판매자 평판 등 복합적인 요소를 점수화하여 최종 추천하는 정교한 하이브리드 로직을 구현했습니다.
-   **API 기능 확장**: 데이터 조회(`GET`) 뿐만 아니라, 새로운 농지나 판매자 정보를 동적으로 등록(`POST`)할 수 있는 데이터 관리 API를 추가하여 시스템의 확장성을 높였습니다.
-   **코드 구조 개선**: 프로젝트를 `app`, `core`, `ml`, `data` 폴더로 재구성하여 역할과 책임(SoC)을 명확히 분리했습니다.

## 프로젝트 구조

```
ai_match_integrated/
├── app/                  # FastAPI 애플리케이션 관련 파일
│   ├── main.py           # API 엔드포인트 정의
│   └── models.py         # ERD 기반 Pydantic 데이터 모델
├── core/                 # 핵심 비즈니스 로직
│   └── matching_logic.py # 매칭 점수 계산 로직
├── ml/                   # 머신러닝 모델 및 학습 관련
│   ├── trainer.py        # K-means 모델 학습 스크립트
│   └── models/           # 학습된 모델/전처리기 저장 폴더
├── data/                 # 샘플 데이터
│   └── sample_data.json
├── README.md             # 프로젝트 안내 문서
└── requirements.txt      # 필수 라이브러리 목록
```

## API 엔드포인트

API 문서는 서버 실행 후 [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) 에서 확인할 수 있습니다.

### AI 모델 및 매칭
- `POST /train`: K-Means 모델을 학습시키고 최신 데이터로 업데이트합니다.
- `POST /match`: 하이브리드 매칭 엔진을 사용하여 신청자에게 최적의 농지를 추천합니다.

### 데이터 관리
- `POST /farmlands`: 새로운 농지 정보를 `form-data` 형식으로 받아 등록합니다.
- `POST /sellers`: 새로운 판매자 정보를 `JSON` 형식으로 받아 등록합니다.

### 데이터 조회
- `GET /sellers`: 모든 판매자 정보를 조회합니다.
- `GET /licenses`: 모든 자격증 정보를 조회합니다.
- `GET /matching_statuses`: 모든 매칭 현황 정보를 조회합니다.
- `GET /profit_informations`: 모든 수익 정보를 조회합니다.
- `GET /recommenders`: 모든 추천인 정보를 조회합니다.
- `GET /favorite_farmlands`: 모든 찜한 농지 정보를 조회합니다.

## 사용 방법

### 1. `ai_match_integrated` 폴더로 이동

```bash
cd /path/to/your/project/ai/Re Local/ai_match_integrated
```

### 2. 필수 라이브러리 설치

```bash
pip install -r requirements.txt
```

### 3. FastAPI 서버 실행

```bash
python3 -m uvicorn app.main:app --reload --port 8000
```

### 4. AI 모델 학습 (매칭 또는 데이터 변경 후)

-   새로운 데이터를 추가했거나, 매칭 로직을 사용하기 전에는 반드시 모델을 최신 상태로 학습시켜야 합니다.
-   API 문서에서 `POST /train`을 찾아 **Try it out** -> **Execute** 버튼을 클릭하여 모델을 학습시킵니다.

### 5. 매칭 기능 테스트

-   API 문서에서 `POST /match`를 찾아 테스트할 수 있습니다.

#### 요청 본문 (Request Body) 예시

```json
{
  "buyerId": 101,
  "buyerName": "김청년",
  "buyerAge": 30,
  "buyerGender": "남성",
  "buyerAddress": "서울특별시 강남구",
  "buyerNumber": "010-1111-2222",
  "buyerEmail": "kim.youth@example.com",
  "profileImage": "s3://bucket/profiles/kim.jpg",
  "home_lat": 37.4979,
  "home_lng": 127.0276,
  "trustProfile": {
    "trustId": 1,
    "awards": [
      "청년농업인상 수상"
    ],
    "interestCrop": [
      "쌀",
      "콩"
    ],
    "experience": "초보",
    "wantTrade": [
      "임대"
    ],
    "oneIntroduction": "열정으로 농사짓겠습니다!",
    "introduction": "안녕하세요. 농업에 미래를 걸고 귀농을 희망하는 김청년입니다. 쌀과 콩 재배에 관심이 많으며, 성실하게 배우고 일하겠습니다.",
    "videoURL": null,
    "sns": "instagram.com/kim.youth.farm",
    "personal": null,
    "trustScore": "85",
    "buyerId": 101,
    "equipment": [
      "트랙터",
      "경운기"
    ],
    "budget": 6000000,
    "wantPeriod": "즉시"
  }
}
```
