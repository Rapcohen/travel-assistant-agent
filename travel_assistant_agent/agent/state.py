from enum import Enum
from typing import List, Optional

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from pydantic import BaseModel, Field
from typing_extensions import Annotated


class UserQueryIntent(Enum):
    DESTINATION_RECOMMENDATION = 'destination_recommendation'
    ACTIVITY_RECOMMENDATION = 'activity_recommendation'
    PACKING_RECOMMENDATION = 'packing_recommendation'
    FOOD_RECOMMENDATION = 'food_recommendation'
    UNKNOWN = 'unknown'


class UserPreferences(BaseModel):
    origin: Optional[str] = Field(default=None, description='Starting location (city or airport)')
    destination: Optional[str] = Field(default=None, description='Primary destination (city / region / country)')
    trip_duration: Optional[str] = Field(default=None, description='Approximate trip length (e.g. "5 days", "2 weeks")')
    budget: Optional[str] = Field(default=None, description='Budget level or range (e.g. "low", "mid-range", "under 2000 USD")')
    time_of_year: Optional[str] = Field(default=None, description='Season or month timeframe or exact dates')
    travel_style: Optional[str] = Field(default=None, description='Travel style (e.g. "laid-back", "luxury", "backpacking", "family-friendly", "adventure")')
    trip_participants: Optional[str] = Field(default=None, description='Who is traveling (e.g. "solo", "couple", "family with kids")')
    interests: Optional[str] = Field(default=None, description='High-level interests (e.g. "history", "food", "beaches")')
    activities: Optional[str] = Field(default=None, description='Specific planned or desired activities (e.g. "hiking", "snorkeling")')
    cuisine_preferences: Optional[str] = Field(default=None, description='Cuisine types or food interests (e.g. "seafood", "street food", "vegan")')
    transportation_preferences: Optional[str] = Field(default=None, description='Preferred transport modes (e.g. "train", "driving", "public transit only")')

    @classmethod
    def merge(
            cls,
            existing: 'UserPreferences',
            new: 'UserPreferences'
    ) -> 'UserPreferences':
        merged_data = existing.model_dump()
        for field, value in new.model_dump().items():
            if value is not None:
                merged_data[field] = value
        return cls(**merged_data)


class InputState(BaseModel):
    messages: Annotated[List[AnyMessage], add_messages] = Field(default_factory=list)


class State(InputState):
    user_query_intent: UserQueryIntent = UserQueryIntent.UNKNOWN
    user_preferences: UserPreferences = Field(default_factory=UserPreferences)
