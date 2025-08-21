from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum
from datetime import datetime

class MatchStatusEnum(str, Enum):
    MATCHING = "MATCHING"
    PENDING = "PENDING"
    CANCELED = "CANCELED"
    COMPLETED = "COMPLETED"

class MatchingStatus(BaseModel):
    matchingId: int = Field(description="매칭 현황 ID")
    matchStatus: MatchStatusEnum = Field(description="매칭 상태")
    recommendCount: int = Field(description="추천 횟수")
    preferences: Optional[str] = Field(None, description="선호 사항")
    farmlandMatch: int = Field(description="농지 ID (FK)")
    buyerMatch: int = Field(description="구매자 ID (FK)")

class ProfitInformation(BaseModel):
    profitId: int = Field(description="수익 정보 ID")
    yearlyProfit: str = Field(description="연 수익 정보")
    yield_info: str = Field(description="수확량")
    unitPrice: str = Field(description="단위 가격")
    material: str = Field(description="재료비")
    labor: str = Field(description="인건비")
    machine: str = Field(description="기계, 장비비")
    netProfit: str = Field(description="순수익")
    landId: int = Field(description="농지 ID (FK)")

class TrustProfile(BaseModel):
    trustId: int = Field(description="신뢰 정보 ID")
    awards: Optional[List[str]] = Field(None, description="수상/활동이력")
    interestCrop: Optional[List[str]] = Field(None, description="관심 작물")
    experience: Optional[str] = Field(None, description="농업 경험")
    wantTrade: Optional[List[str]] = Field(None, description="희망 거래 형태")
    oneIntroduction: Optional[str] = Field(None, description="한줄 자기소개")
    introduction: Optional[str] = Field(None, description="자기소개 입력")
    videoURL: Optional[str] = Field(None, description="자기소개 영상 URL")
    sns: Optional[str] = Field(None, description="SNS 아이디")
    personal: Optional[str] = Field(None, description="개인정보") # ERD의 '개인정보' 필드
    trustScore: Optional[str] = Field(None, description="신뢰도 점수")
    buyerId: int = Field(description="구매자 ID (FK)")
    equipment: Optional[List[str]] = Field(None, description="장비")
    budget: Optional[int] = Field(None, description="예산")
    wantPeriod: Optional[str] = Field(None, description="희망 거래 기간")

class Recommender(BaseModel):
    suggestId: int = Field(description="추천인 ID")
    suggestName: str = Field(description="추천인 이름")
    suggestRelationship: str = Field(description="추천인과의 관계")
    suggestNumber: str = Field(description="추천인 전화번호")
    suggestEmail: Optional[str] = Field(None, description="추천인 이메일")
    buyerSuggest: int = Field(description="구매자 ID (FK)")

class Farmland(BaseModel):
    landId: int = Field(description="농지 ID")
    landName: str = Field(description="농지 이름")
    landAddress: str = Field(description="농지 주소")
    landRoadAddress: Optional[str] = Field(None, description="농지 도로명 주소")
    landNumber: str = Field(description="지번")
    landCrop: str = Field(description="직전 재배 농작물")
    landArea: int = Field(description="농지 면적 (m^2)")
    soiltype: str = Field(description="토양 유형")
    waterSource: str = Field(description="용수 접근성")
    ownerName: str = Field(description="소유자 이름")
    ownerAge: int = Field(description="소유자 나이")
    ownerAddress: str = Field(description="소유자 주소")
    landRegisterDate: datetime = Field(description="농지 등록일")
    landWater: str = Field(description="농업 용수")
    landElec: str = Field(description="전기")
    landMachine: str = Field(description="농기계 접근")
    landStorage: Optional[str] = Field(None, description="창고 여부")
    landHouse: Optional[str] = Field(None, description="비닐하우스 여부")
    landFence: Optional[str] = Field(None, description="울타리 여부")
    landRoad: str = Field(description="도로 인접 여부")
    landWellRoad: str = Field(description="포장도로 여부")
    landBus: str = Field(description="대중교통 접근성")
    landCar: str = Field(description="차량 진입 가능성")
    landTrade: str = Field(description="농지 거래 형태")
    landMatch: Optional[str] = Field(None, description="희망 매칭 유형")
    landPrice: Optional[int] = Field(None, description="농지희망가격")
    landWhen: Optional[str] = Field(None, description="농지 매도 희망 시기")
    landWhy: Optional[str] = Field(None, description="농지 내놓는 이유")
    landComent: Optional[str] = Field(None, description="코멘트")
    landRegister: Optional[str] = Field(None, description="농지대장 URL")
    landCadastre: Optional[str] = Field(None, description="토지대장 URL")
    landCertification: Optional[str] = Field(None, description="친환경/GAP 인증서 URL")
    landImage: Optional[str] = Field(None, description="농지 이미지 URL")
    sellerFarmland: int = Field(description="판매자 ID (FK)")
    # ERD에 없지만 서비스 기능에 필수적인 필드
    lat: float = Field(description="위도")
    lng: float = Field(description="경도")

class FavoriteFarmland(BaseModel):
    favoriteLandId: int = Field(description="찜한 농지 ID")
    buyer_id: int = Field(description="구매자 ID (FK)")
    land_id: int = Field(description="농지 ID (FK)")

class Seller(BaseModel):
    sellerId: int = Field(description="판매자 ID")
    sellerName: str = Field(description="판매자 이름")
    sellerYear: int = Field(description="판매자 생년")
    sellerNumber: str = Field(description="판매자 연락처")
    sellerAddress: str = Field(description="판매자 주소")
    seller_land: int = Field(description="판매 농지 수")

class Buyer(BaseModel):
    buyerId: int = Field(description="구매자 ID")
    buyerName: str = Field(description="구매자 이름")
    buyerAge: int = Field(description="구매자 나이")
    buyerGender: str = Field(description="구매자 성별")
    buyerAddress: str = Field(description="구매자 주소")
    buyerNumber: str = Field(description="구매자 연락처")
    buyerEmail: str = Field(description="구매자 이메일")
    profileImage: Optional[str] = Field(None, description="구매자 프로필 이미지 URL") # ERD의 'Field'를 명확한 이름으로 변경
    trustProfile: TrustProfile # 신뢰 정보는 별도 모델로 분리
    # ERD에 없지만 서비스 기능에 필수적인 필드
    home_lat: float = Field(description="거주지 위도")
    home_lng: float = Field(description="거주지 경도")

class License(BaseModel):
    licenseId: int = Field(description="자격증 ID")
    licenseName: Optional[str] = Field(None, description="자격증 이름")
    licenseFile: Optional[str] = Field(None, description="자격증 파일 URL")
    buyerLicense: int = Field(description="구매자 ID (FK)")

# API 응답을 위한 모델 (ERD와는 별개)
class MatchResult(BaseModel):
    matches: List[dict]
    cluster_info: dict
