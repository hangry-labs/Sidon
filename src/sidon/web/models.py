# SPDX-License-Identifier: Apache-2.0
"""Shared processing settings and the persisted job contract."""
from typing import Literal
import uuid

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .audio import OutputFormat

JobStatus = Literal['uploading', 'queued', 'running', 'done', 'failed', 'cancelled']
JobMode = Literal['enhance', 'separate', 'separate_enhance', 'convert']
TERMINAL = {'done', 'failed', 'cancelled'}


class Settings(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    strength: float = Field(default=1, ge=0, le=1)
    output_format: OutputFormat = 'wav'
    chunk_seconds: float = Field(default=4, ge=2, le=16)
    overlap_seconds: float = Field(default=1, ge=0.1, le=2)

    @model_validator(mode='after')
    def valid_overlap(self):
        if self.overlap_seconds >= self.chunk_seconds:
            raise ValueError('Overlap must be shorter than the separation window')
        return self


class StoredOutput(BaseModel):
    model_config = ConfigDict(extra='allow', allow_inf_nan=False)
    name: str
    file: str = Field(pattern=r'^[A-Za-z0-9][A-Za-z0-9_.-]*$')
    sample_rate: int = Field(gt=0)
    duration: float = Field(ge=0)
    channels: int = Field(default=1, ge=1, le=8)
    format: str | None = None
    lossy: bool | None = None
    samples_over_full_scale: int = Field(default=0, ge=0)


class StoredJob(BaseModel):
    model_config = ConfigDict(extra='allow', allow_inf_nan=False)
    id: str
    status: JobStatus
    mode: JobMode
    created: float = Field(ge=0)
    stage: str
    filename: str
    settings: Settings
    outputs: list[StoredOutput]
    cancel_requested: bool = False
    request_id: str | None = None
    error: str | None = None
    queue_wait_seconds: float | None = Field(default=None, ge=0)
    processing_seconds: float | None = Field(default=None, ge=0)
    total_seconds: float | None = Field(default=None, ge=0)

    @field_validator('id')
    @classmethod
    def canonical_id(cls, value):
        if str(uuid.UUID(value)) != value:
            raise ValueError('Job IDs must be canonical UUID strings')
        return value
