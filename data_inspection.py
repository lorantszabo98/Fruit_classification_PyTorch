from utils import config
import os
import matplotlib.pyplot as plt


def get_data_distribution(data):
    data_path = data
    class_counts = {}

    for class_label in os.listdir(data_path):
        class_path = os.path.join(data_path, class_label)
        number_of_images = len(os.listdir(class_path))
        class_counts[class_label] = number_of_images

    for class_name, counts in class_counts.items():
        print(f"Class: {class_name}, Number of images: {counts}")

    plot_distribution(class_counts)


def plot_distribution(class_counts):
    plt.bar(class_counts.keys(), class_counts.values())
    plt.xlabel('Classes')
    plt.ylabel('Number of images')
    plt.title('Class distribution')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    get_data_distribution(config.DATA_PATH)