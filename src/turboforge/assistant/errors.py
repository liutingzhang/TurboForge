class AssistantError(Exception):
    """Public, secret-free boundary error. Never include an upstream response body."""

    def __init__(self, code: str, message: str, status: int = 400):
        super().__init__(message)
        self.code = code
        self.status = status
