from fastapi import APIRouter

from backend.api.dependencies import SettingsDep
from backend.models.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(settings: SettingsDep) -> HealthResponse:
    return HealthResponse(
        version=settings.app_version,
        llm_model=settings.llm_model,
        llm_configured=bool(settings.openai_api_key),
    )
