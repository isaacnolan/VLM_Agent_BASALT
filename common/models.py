"""Shared data models for client and server."""
from pydantic import BaseModel
from typing import Optional, List, Dict, Any


class ImageData(BaseModel):
    """Base64 encoded image data."""
    data: str  # base64 encoded image


class StateActionPair(BaseModel):
    """Represents a historical state-action pair."""
    image: ImageData
    action: Optional[Dict[str, Any]] = None  # The action taken from this state


class PolicyRequest(BaseModel):
    """Request model for policy server."""
    task_name: str
    image: Optional[ImageData] = None  # For backward compatibility
    history: Optional[List[StateActionPair]] = None  # History of state-action pairs
    step: Optional[int] = 0
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 512
    max_history_length: Optional[int] = 5  # Maximum number of historical frames to include


class ActionResponse(BaseModel):
    """Response model from policy server."""
    action: Dict[str, Any]
    reasoning: str
    step: int
