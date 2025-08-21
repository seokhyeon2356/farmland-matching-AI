from pydantic import BaseModel, Field
from typing import List, Optional

# --- New Data Structure Models ---

class Farmland(BaseModel):
    landId: int
    landName: str
    landLat: float
    landLng: float
    ownerAge: str
    ownerAddress: str
    ownerName: str
    landCrop: str
    landArea: int
    soiltype: str
    waterSource: str
    landWater: bool
    landElec: bool
    landMachine: bool
    landStorage: bool
    landHouse: bool
    landFence: bool
    landRoad: bool
    landWellRoad: bool
    landBus: bool
    landCar: bool
    landTrade: str
    landPrice: int
    landMatch: str
    landWhen: str
    landWhy: str
    landComent: str

class BuyerTrustProfile(BaseModel):
    interestCrop: List[str]
    equipment: List[str]
    wantTrade: List[str]
    budget: int
    wantPeriod: str

class Buyer(BaseModel):
    buyerId: int
    buyerLat: float
    buyerLng: float
    trustProfile: BuyerTrustProfile

# --- API I/O Models ---

class MatchResult(BaseModel):
    matches: List[dict]
    cluster_info: dict
