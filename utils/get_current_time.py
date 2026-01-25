from datetime import datetime

def _format_file_timestamp(timestamp: float, include_time: bool = False):
    """
    Format file timestamp to a %Y-%m-%d string.

    Args:
        timestamp (float): timestamp in float
        include_time (bool): whether to include time in the formatted string

    Returns:
        str: formatted timestamp
    """
    try:
        if include_time:
            return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%dT%H:%M:%SZ")
        return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d")
    except Exception:
        return None