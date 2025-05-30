import kagglehub
import shutil
import os


if __name__ == '__main__':
	# Make sure directory exists
	os.makedirs("../data", exist_ok=True)
	
	# Download latest version of the file
	path = kagglehub.dataset_download("stackoverflow/so-survey-2017")
	print(f"Downloaded to: {path}")
	
	shutil.copytree(path, "../data", dirs_exist_ok=True)
