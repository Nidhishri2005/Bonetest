"""Verification script for bone age model architectures, target scaling, and inference."""

import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import cv2
import numpy as np
import torch
from app.models.cnn import CNNBaseline
from app.models.cnn_dnn import CNNDNN
from app.models.multimodal_cnn import MultimodalCNN
from app.models.cnn_rf import CNNFeatureExtractor
from ml.preprocessing import standardize_target, inverse_transform_target


def test_models():
    print("Test 1: Verifying Model Forward Passes...", flush=True)
    x1 = torch.randn(1, 1, 512, 512)
    x4 = torch.randn(4, 1, 512, 512)
    g1 = torch.tensor([[1.0]])
    g4 = torch.tensor([[1.0], [0.0], [1.0], [0.0]])

    m1 = CNNBaseline(pretrained=False).eval()
    m2 = CNNDNN(pretrained=False).eval()
    m3 = MultimodalCNN(pretrained=False).eval()
    m4 = CNNFeatureExtractor(pretrained=False).eval()

    out1_1 = m1(x1)
    out1_4 = m1(x4)
    assert out1_1.shape == torch.Size([1]), f"m1(x1) shape: {out1_1.shape}"
    assert out1_4.shape == torch.Size([4]), f"m1(x4) shape: {out1_4.shape}"
    print("  [PASS] CNNBaseline forward pass", flush=True)

    out2_1 = m2(x1)
    out2_4 = m2(x4)
    assert out2_1.shape == torch.Size([1]), f"m2(x1) shape: {out2_1.shape}"
    assert out2_4.shape == torch.Size([4]), f"m2(x4) shape: {out2_4.shape}"
    print("  [PASS] CNNDNN forward pass", flush=True)

    out3_1 = m3(x1, g1)
    out3_4 = m3(x4, g4)
    assert out3_1.shape == torch.Size([1]), f"m3(x1) shape: {out3_1.shape}"
    assert out3_4.shape == torch.Size([4]), f"m3(x4) shape: {out3_4.shape}"
    print("  [PASS] MultimodalCNN forward pass", flush=True)

    out4_1 = m4(x1)
    out4_4 = m4(x4)
    assert out4_1.shape == torch.Size([1, 512]), f"m4(x1) shape: {out4_1.shape}"
    assert out4_4.shape == torch.Size([4, 512]), f"m4(x4) shape: {out4_4.shape}"
    print("  [PASS] CNNFeatureExtractor forward pass", flush=True)


def test_target_scaling():
    print("\nTest 2: Verifying Target Standardization Math...", flush=True)
    mu, sigma = 127.32, 41.18
    y_orig = torch.tensor([12.0, 84.0, 127.32, 180.0, 228.0])
    z = standardize_target(y_orig, mu, sigma)
    y_rec = inverse_transform_target(z, mu, sigma)
    diff = torch.abs(y_orig - y_rec).max().item()
    assert diff < 1e-4, f"Target scaling error: {diff}"
    print(f"  [PASS] Target roundtrip error: {diff:.6f}", flush=True)


def test_inference_service():
    print("\nTest 3: Verifying Inference Service...", flush=True)
    from app.services.inference_service import get_inference_service
    service = get_inference_service()

    dummy_img = np.full((512, 512), 128, dtype=np.uint8)
    _, buf = cv2.imencode(".png", dummy_img)
    img_bytes = buf.tobytes()

    for m in ["cnn", "cnn_dnn", "multimodal_cnn"]:
        res = service.predict(img_bytes, m, gender="male")
        pred = res["bone_age_months"]
        print(f"  [PASS] {m} inference: {pred} months (confidence: {res['confidence']})", flush=True)


def test_gradcam_service():
    print("\nTest 4: Verifying Grad-CAM Service...", flush=True)
    from app.services.gradcam_service import get_gradcam_service
    service = get_gradcam_service()

    dummy_img = np.full((512, 512), 128, dtype=np.uint8)
    _, buf = cv2.imencode(".png", dummy_img)
    img_bytes = buf.tobytes()

    for m in ["cnn", "cnn_dnn", "multimodal_cnn"]:
        res = service.generate(img_bytes, m, gender="male")
        assert "heatmap_base64" in res and len(res["heatmap_base64"]) > 100
        assert "overlay_base64" in res and len(res["overlay_base64"]) > 100
        print(f"  [PASS] {m} Grad-CAM: heatmap & overlay generated successfully", flush=True)


def main():
    test_models()
    test_target_scaling()
    test_inference_service()
    test_gradcam_service()
    print("\nALL VERIFICATION TESTS COMPLETED SUCCESSFULLY!", flush=True)


if __name__ == "__main__":
    main()


