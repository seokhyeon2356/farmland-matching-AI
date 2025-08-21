from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum

class MatchStatusEnum(str, Enum):
    pending = "pending"
    matched = "matched"
    canceled = "canceled"

class Farmland(BaseModel):
    landId: int = Field(description="농지 ID")
    landName: str = Field(description="농지 이름")
    landAddress: str = Field(description="농지 주소")
    landArea: int = Field(description="농지 면적 (m^2)")
    landCrop: str = Field(description="직전 재배 농작물")
    soiltype: str = Field(description="토양 유형")
    waterSource: str = Field(description="용수 접근성")
    landWater: bool = Field(description="농업 용수")
    landElec: bool = Field(description="전기")
    landMachine: bool = Field(description="농기계 접근")
    landStorage: bool = Field(description="창고 여부")
    landHouse: bool = Field(description="비닐하우스 여부")
    landFence: bool = Field(description="울타리 여부")
    landRoad: bool = Field(description="도로 인접 여부")
    landWellRoad: bool = Field(description="포장도로 여부")
    landBus: bool = Field(description="대중교통 접근성")
    landCar: bool = Field(description="차량 진입 가능성")
    landTrade: str = Field(description="농지 거래 형태")
    landPrice: int = Field(description="농지희망가격")
    landWhen: str = Field(description="농지 매도 희망 시기")
    lat: float = Field(description="위도")
    lng: float = Field(description="경도")

class TrustProfile(BaseModel):
    interestCrop: List[str] = Field(description="관심 작물")
    experience: Optional[str] = Field(description="농업 경험")
    wantTrade: List[str] = Field(description="거래 형태")
    equipment: List[str] = Field(description="장비")
    budget: int = Field(description="예산")
    wantPeriod: str = Field(description="원하는 거래 기간")

class Buyer(BaseModel):
    buyerId: int = Field(description="구매자 ID")
    buyerName: str = Field(description="구매자 이름")
    buyerAge: int = Field(description="구매자 나이")
    buyerAddress: str = Field(description="구매자 주소")
    home_lat: float = Field(description="거주지 위도")
    home_lng: float = Field(description="거주지 경도")
    trustProfile: TrustProfile

class MatchResult(BaseModel):
    matches: List[dict]
    cluster_info: dict

class MatchingStatus(BaseModel):
    matchingId: int = Field(description="매칭 현황 ID")
    matchStatus: MatchStatusEnum = Field(description="매칭 상태")
    recommendCount: int = Field(description="추천 횟수")
    preferences: Optional[str] = Field(description="선호 사항")
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

class RecommenderInformation(BaseModel):
    suggestId: int = Field(description="추천인 ID")
    suggestName: str = Field(description="추천인 이름")
    suggestRelationship: str = Field(description="추천인과의 관계")
    suggestNumber: str = Field(description="추천인 전화번호")
    suggestEmail: Optional[str] = Field(description="추천인 이메일")
    buyerSuggest: int = Field(description="구매자 ID (FK)")

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

class License(BaseModel):
    licenseId: int = Field(description="자격증 ID")
    licenseName: Optional[str] = Field(description="자격증 이름")
    licenseFile: Optional[str] = Field(description="자격증 파일 URL")
    buyerLicense: int = Field(description="구매자 ID (FK)")