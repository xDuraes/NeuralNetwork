import os
import torchvision
from torchvision import datasets, transforms
from collections import Counter

"""
SCRIPT TO GENERATE A DATASET IN PNG
"""

transform = transforms.ToTensor()

data_dir = os.path.join(".", "data")

mnist_train = datasets.MNIST(
    root=data_dir,
    train=True,
    download=True,
    transform=transform
)

mnist_data_dir = os.path.join(data_dir, "mnist_png")
os.makedirs(mnist_data_dir, exist_ok=True)

# number of images to save
N = 10000  

for i in range(N):
    image, label = mnist_train[i]
    # image: (1, 28, 28)
    filepath = os.path.join(mnist_data_dir, f"img_{i}_label_{label}.png")
    torchvision.utils.save_image(image, filepath)

files = os.listdir(mnist_data_dir)
numbers = list()

for file in files:
    numbers.append(file.split("_")[-1].replace(".png",""))

print(Counter(numbers))