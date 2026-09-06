from app.models.cnn import CNNBaseline
from app.models.cnn_dnn import CNNDNN
from app.models.cnn_rf import CNNFeatureExtractor, CNNWithRFWrapper
from app.models.multimodal_cnn import MultimodalCNN

MODEL_REGISTRY = {
    "cnn": {
        "class": CNNBaseline,
        "display_name": "CNN Baseline",
        "description": "ResNet-18 backbone with linear regression head",
        "description": "ResNet backbone with linear regression head",
        "requires_gender": False,
        "supports_gradcam": True,
    },
    "cnn_dnn": {
        "class": CNNDNN,
        "display_name": "CNN + DNN",
        "description": "ResNet-18 with multi-layer dense regression head",
        "description": "ResNet with multi-layer dense regression head",
        "requires_gender": False,
        "supports_gradcam": True,
    },
    "multimodal_cnn": {
        "class": MultimodalCNN,
        "display_name": "Multimodal CNN",
        "description": "Image CNN fused with gender metadata",
        "requires_gender": True,
        "supports_gradcam": True,
    },
    "cnn_rf": {
        "class": CNNFeatureExtractor,
        "display_name": "CNN + Random Forest",
        "description": "CNN feature extractor with sklearn RF regressor",
        "requires_gender": False,
        "supports_gradcam": True,
        "supports_gradcam": False,
        "wrapper": CNNWithRFWrapper,
    },
}

SUPPORTED_MODELS = list(MODEL_REGISTRY.keys())
