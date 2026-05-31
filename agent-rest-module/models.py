from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class AgentContext(BaseModel):
    run_id: Optional[str] = Field(default=None, description="Batch run identifier")
    profile_id: Optional[str] = Field(default=None, description="Dataset profile identifier")
    step_id: Optional[str] = Field(default=None, description="Optional step identifier")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AgentStartRequest(BaseModel):
    agent_id: str = Field(..., min_length=1)
    initial_prompt: str = Field(default="/start")
    parameters: Dict[str, Any] = Field(default_factory=dict)
    context: AgentContext = Field(default_factory=AgentContext)


class AgentMessageRequest(BaseModel):
    message: str = Field(..., min_length=1)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    context: AgentContext = Field(default_factory=AgentContext)


class AgentActionRequest(BaseModel):
    parameters: Dict[str, Any] = Field(default_factory=dict)
    context: AgentContext = Field(default_factory=AgentContext)


class ProfessionRequest(BaseModel):
    profession_name: str = Field(..., min_length=1)
    context: AgentContext = Field(default_factory=AgentContext)


class AgentResponse(BaseModel):
    agent_id: str
    message: str
    user_state: Optional[str] = None
    professions: Dict[str, Any] = Field(default_factory=dict)
    test_info: Dict[str, Any] = Field(default_factory=dict)
    test_version: Optional[str] = None
    top5_professions: List[str] = Field(default_factory=list)
    context: AgentContext = Field(default_factory=AgentContext)


class AgentSessionSnapshot(BaseModel):
    agent_id: str
    app_user_id: Optional[int] = None
    user_state: Optional[str] = None
    user_type: Optional[str] = None
    conversation_history_len: int = 0
    user_metadata: Dict[str, Any] = Field(default_factory=dict)

