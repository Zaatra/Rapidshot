class RapidShotError(Exception):
    """Base class for exceptions in RapidShot."""
    pass

class RapidShotDXGIError(RapidShotError):
    """
    Wrapper for DXGI/COM errors.
    """
    def __init__(self, message, hresult=None):
        super().__init__(message)
        self.hresult = hresult
        self.message = message

    def __str__(self):
        if self.hresult is not None:
            try:
                # Try to format hresult as hex, fallback if it's not an int
                hresult_str = f" (HRESULT: {self.hresult:#010x})"
            except (TypeError, ValueError):
                hresult_str = f" (HRESULT: {self.hresult})"
            return f"{self.message}{hresult_str}"
        return self.message

class RapidShotReinitError(RapidShotDXGIError):
    """
    Indicates that re-initialization of capture resources is needed,
    typically due to DXGI_ERROR_ACCESS_LOST.
    """
    pass

class RapidShotDeviceError(RapidShotDXGIError):
    """
    Indicates a critical device error (e.g., DEVICE_REMOVED, DEVICE_RESET)
    that requires re-initialization.
    """
    pass

class RapidShotConfigError(RapidShotError):
    """
    Indicates a configuration error or invalid API call.
    """
    pass

class RapidShotTimeoutError(RapidShotError):
    """
    Indicates a timeout error, typically for frame acquisition.
    """
    pass
