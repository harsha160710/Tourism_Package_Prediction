from huggingface_hub import HfApi,create_repo
from huggingface_hub.utils import RepositoryNotFoundError,HfHubHTTPError
import os

repo_id='Harsha1001/Tourism-Package-Prediction'
repo_type='dataset'

api=HfApi(token=os.getenv('HF_TOKEN'))
try:
  api.repo_info(repo_id=repo_id,repo_type=repo_type)
  print(f"Space '{repo_id}' already exists, uysing it")
except RepositoryNotFoundError:
  print(f"Space '{repo_id}' does not exist, creating it")
  api.create_repo(repo_id=repo_id,repo_type=repo_type)
  print(f"Space '{repo_id}' created.")

api.upload_folder(
    folder_path="tourism_project/data",
    repo_id=repo_id,
    repo_type=repo_type,
)
