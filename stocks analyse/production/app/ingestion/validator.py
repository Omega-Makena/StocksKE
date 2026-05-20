import structlog

logger = structlog.get_logger(__name__)

def validate_price_data(data: dict) -> bool:
    """
    Validates EOD stock price data dictionary.
    Expects keys: open, high, low, close, volume
    """
    required_keys = ['open', 'high', 'low', 'close', 'volume']
    
    for key in required_keys:
        if key not in data:
            logger.warning("Validation failed: missing key", key=key)
            return False
            
        val = data[key]
        if not isinstance(val, (int, float)):
            logger.warning("Validation failed: wrong type", key=key, type=type(val))
            return False
            
        if val < 0:
            logger.warning("Validation failed: negative value", key=key, value=val)
            return False
            
    # Basic logic check
    if data['low'] > data['high']:
        logger.warning("Validation failed: low > high")
        return False
        
    return True
