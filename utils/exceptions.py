class RetrievalError(Exception):
    """Custom exception for retrieval errors."""
    def __init__(self, message):
        self.message = message
        super().__init__(message)


class APIError(Exception):
    def __init__(self, message=None):
        self.message = message
        super().__init__(message)


class FatalAPIError(APIError):
    pass


class NonFatalAPIError(APIError):
    pass


class FileWriteError(Exception):
    """Custom exception for file write errors."""
    def __init__(self, message):
        self.message = message
        super().__init__(message)