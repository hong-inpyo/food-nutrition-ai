classes = {}

with open(
    "datasets/foodseg103/FoodSeg103/category_id.txt",
    encoding="utf-8"
) as f:

    for line in f:

        idx, name = line.strip().split(maxsplit=1)

        classes[int(idx)-1] = name

with open("foodseg103.yaml", "w", encoding="utf-8") as f:

    f.write("path: datasets/foodseg_yolo\n")
    f.write("\n")
    f.write("train: images/train\n")
    f.write("val: images/train\n")
    f.write("\n")
    f.write("names:\n")

    for i, name in classes.items():

        f.write(f"  {i}: {name}\n")

print("foodseg103.yaml 생성 완료")