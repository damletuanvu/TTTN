import numpy as np
import matplotlib.pyplot as plt
import os

def load_idx_file(filepath):
    """
    Loads an IDX file (e.g., MNIST images or labels).
    """
    with open(filepath, 'rb') as f:
        # Read magic number and dimensions
        magic = int.from_bytes(f.read(4), 'big')
        num_dims = magic & 0x000000FF  # Last byte indicates number of dimensions

        dims = []
        for _ in range(num_dims):
            dims.append(int.from_bytes(f.read(4), 'big'))

        # Read the data
        data = np.frombuffer(f.read(), dtype=np.uint8)

        # Reshape if necessary
        if num_dims > 1:
            data = data.reshape(dims)
        return data


data_dir = "Data\mninst" 

# Tên file ảnh (ví dụ: file ảnh training)
train_images_path = os.path.join(data_dir, 'train-images.idx3-ubyte')
train_labels_path = os.path.join(data_dir, 'train-labels.idx1-ubyte')

# Tên file ảnh (ví dụ: file ảnh test)
test_images_path = os.path.join(data_dir, 't10k-images.idx3-ubyte')
test_labels_path = os.path.join(data_dir, 't10k-labels.idx1-ubyte')


try:
    # Load training images and labels
    print(f"Loading {train_images_path}...")
    train_images = load_idx_file(train_images_path)
    print(f"Loading {train_labels_path}...")
    train_labels = load_idx_file(train_labels_path)

    # Load test images and labels
    print(f"Loading {test_images_path}...")
    test_images = load_idx_file(test_images_path)
    print(f"Loading {test_labels_path}...")
    test_labels = load_idx_file(test_labels_path)

    print(f"Shape of training images: {train_images.shape}") # Ví dụ: (60000, 28, 28)
    print(f"Shape of training labels: {train_labels.shape}") # Ví dụ: (60000,)
    print(f"Shape of test images: {test_images.shape}")     # Ví dụ: (10000, 28, 28)
    print(f"Shape of test labels: {test_labels.shape}")     # Ví dụ: (10000,)

    # Hiển thị một vài ảnh ví dụ từ tập training
    plt.figure(figsize=(10, 5))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(train_images[i], cmap='gray')
        plt.title(f"Label: {train_labels[i]}")
        plt.axis('off')
    plt.suptitle("Ví dụ ảnh từ tập dữ liệu Training")
    plt.show()

    # Hiển thị một vài ảnh ví dụ từ tập test
    plt.figure(figsize=(10, 5))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(test_images[i], cmap='gray')
        plt.title(f"Label: {test_labels[i]}")
        plt.axis('off')
    plt.suptitle("Ví dụ ảnh từ tập dữ liệu Test")
    plt.show()

except FileNotFoundError as e:
    print(f"Lỗi: Không tìm thấy file. Hãy đảm bảo bạn đã đặt đúng đường dẫn tới thư mục chứa các file IDX.")
    print(f"Chi tiết lỗi: {e}")
except Exception as e:
    print(f"Đã xảy ra lỗi khi xử lý file: {e}")