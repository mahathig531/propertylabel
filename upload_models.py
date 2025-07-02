from huggingface_hub import HfApi, create_repo
import os

# Initialize the API
api = HfApi()

# Your username and token
username = "Mahathig"
token = "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"

# Repository name
repo_name = "property-labeling-models"
repo_id = f"{username}/{repo_name}"

try:
    # Create the repository
    print(f"Creating repository: {repo_id}")
    create_repo(repo_id, token=token, repo_type="model", exist_ok=True)
    print(f"Repository created successfully!")
    
    # Upload model files
    model_files = [
        "backend/best_room_classifier.pth",
        "backend/yolov8x.pt", 
        "backend/class_mapping.json"
    ]
    
    for file_path in model_files:
        if os.path.exists(file_path):
            print(f"Uploading {file_path}...")
            api.upload_file(
                path_or_fileobj=file_path,
                path_in_repo=os.path.basename(file_path),
                repo_id=repo_id,
                token=token
            )
            print(f"Successfully uploaded {file_path}")
        else:
            print(f"File not found: {file_path}")
    
    print(f"\nAll models uploaded successfully!")
    print(f"Repository URL: https://huggingface.co/{repo_id}")
    
except Exception as e:
    print(f"Error: {e}") 