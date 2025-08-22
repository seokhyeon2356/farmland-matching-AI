# app/models.py
from __future__ import annotations
from typing import List, Any
from pydantic import BaseModel, ConfigDict, model_validator
import numpy as np

def to_native(x: Any) -> Any:
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, (list, tuple)):
        return [to_native(v) for v in x]
    if isinstance(x, dict):
        return {k: to_native(v) for k, v in x.items()}
    return x

class Farmland(BaseModel):
    # 안전망(임시 우회): 넘파이 같은 임의 타입이 들어와도 허용
    model_config = ConfigDict(arbitrary_types_allowed=True)

    landId: int
    landName: str
    landLat: float
    landLng: float
    ownerAge: int
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

    @model_validator(mode="before")
    @classmethod
    def _coerce_numpy(cls, data: Any):
        return to_native(data)

class BuyerTrustProfile(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    interestCrop: List[str]
    equipment: List[str]
    wantTrade: List[str]
    budget: int
    wantPeriod: str

    @model_validator(mode="before")
    @classmethod
    def _coerce_numpy(cls, data: Any):
        return to_native(data)

class Buyer(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    buyerId: int
    buyerLat: float
    buyerLng: float
    trustProfile: BuyerTrustProfile

    @model_validator(mode="before")
    @classmethod
    def _coerce_numpy(cls, data: Any):
        return to_native(data)

class MatchResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    matches: List[dict]
    cluster_info: dict

    @model_validator(mode="before")
    @classmethod
    def _coerce_numpy(cls, data: Any):
        return to_native(data)
