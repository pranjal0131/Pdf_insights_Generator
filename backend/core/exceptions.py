"""Domain exceptions.

Every error the API can surface to a client derives from AppError, which carries
an HTTP status code so the FastAPI exception handler can translate it directly.
"""


class AppError(Exception):
    """Base class for all application errors."""

    status_code: int = 500

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


class InvalidFileTypeError(AppError):
    status_code = 415


class FileTooLargeError(AppError):
    status_code = 413


class PDFExtractionError(AppError):
    status_code = 422


class EmptyDocumentError(AppError):
    """PDF parsed but contained no extractable text (e.g. scanned images)."""

    status_code = 422


class DocumentNotFoundError(AppError):
    status_code = 404


class LLMConfigurationError(AppError):
    """The LLM provider is not configured (e.g. missing API key)."""

    status_code = 503


class AnalysisError(AppError):
    status_code = 502
