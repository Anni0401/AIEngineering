import os
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)
import numpy as np
from src.models.cnn import MNISTCNN
from src.inference import predict

def test_inference_output_range():
    model = MNISTCNN()
    x = np.random.randint(0, 255, (28, 28))
    pred = predict(x, model)

    assert 0 <= pred <= 9
