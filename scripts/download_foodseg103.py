import kagglehub

path = kagglehub.dataset_download(
    "kmader/foodseg103"
)

print(path)