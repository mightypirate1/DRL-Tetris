import logging
logger = logging.getLogger(__name__)
def mix(x,y, alpha=0.99):
    return (1-alpha) * x + alpha * y
