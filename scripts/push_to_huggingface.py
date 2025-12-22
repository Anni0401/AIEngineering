from huggingface_hub import HfApi
import os

api = HfApi()

repo_id = "Anni0401/mnist-cnn"
repo_type = "model"

# Files to upload: (local_path, path_in_repo)
files_to_upload = [
    ("deployment/mnist_cnn.pth", "mnist_cnn.pth"),
    ("src/models/cnn.py", "cnn.py"),
    ("src/inference.py", "inference.py"),
]

for local_path, repo_path in files_to_upload:
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=repo_path,
        repo_id=repo_id,
        repo_type=repo_type,
        commit_message="Add MNIST CNN model and inference code"
    )

print("Model and code successfully pushed to Hugging Face")
