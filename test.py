import os

dataset_path = "data/custom_dataset"

classes = os.listdir(dataset_path)

total_images = 0

for cls in classes:
    cls_path = os.path.join(dataset_path, cls)

    if os.path.isdir(cls_path):
        count = len(os.listdir(cls_path))
        print(cls, count)
        total_images += count

print("\nClasses:", len(classes))
print("Total images:", total_images)