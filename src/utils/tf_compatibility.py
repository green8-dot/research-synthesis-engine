"""
TensorFlow/Transformers Compatibility Module
Handles version compatibility issues
"""

import logging
logger = logging.getLogger(__name__)

# Handle TFPreTrainedModel import compatibility
try:
    from transformers import TFPreTrainedModel
    TF_AVAILABLE = True
    logger.info("TensorFlow transformers available")
except ImportError:
    TFPreTrainedModel = None
    TF_AVAILABLE = False
    logger.warning("TFPreTrainedModel not available - using compatibility mode")

# Handle other potential compatibility issues
try:
    import tensorflow as tf
    TF_VERSION = tf.__version__
    logger.info(f"TensorFlow version: {TF_VERSION}")
except ImportError:
    tf = None
    TF_VERSION = "not installed"
    logger.warning("TensorFlow not available")

def get_tf_compatibility_info():
    """Get TensorFlow compatibility information"""
    return {
        "tf_available": tf is not None,
        "tf_version": TF_VERSION,
        "tfpretrained_available": TF_AVAILABLE,
        "compatibility_mode": not TF_AVAILABLE
    }
