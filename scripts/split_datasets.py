import os
import random
import shutil

class splitter:
    def __init__(self, dir_path, train_path, val_path, test_path, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        self.dirPath = str(dir_path)
        self.trainPath = str(train_path)
        self.valPath = str(val_path)
        self.testPath = str(test_path)
        self.trainRatio = train_ratio
        self.valRatio = val_ratio
        self.testRatio = test_ratio

    def splitData(self):
        image_files = os.listdir(self.dirPath)
        num_images = len(image_files)
        random.shuffle(image_files)

        num_train = int(num_images * self.trainRatio)
        num_val = int(num_images * self.valRatio)
        num_test = num_images - num_train - num_val

        train_files = image_files[:num_train]
        val_files = image_files[num_train:num_train + num_val]
        test_files = image_files[num_train + num_val:]

        os.makedirs(self.trainPath, exist_ok=True)
        os.makedirs(self.valPath, exist_ok=True)
        os.makedirs(self.testPath, exist_ok=True)

        for filename in train_files:
            src_path = os.path.join(self.dirPath, filename)
            dst_path = os.path.join(self.trainPath, filename)
            shutil.copy(src_path, dst_path)  # Use shutil.copy if you want to copy instead of move

        for filename in val_files:
            src_path = os.path.join(self.dirPath, filename)
            dst_path = os.path.join(self.valPath, filename)
            shutil.copy(src_path, dst_path)  # Use shutil.copy if you want to copy instead of move

        for filename in test_files:
            src_path = os.path.join(self.dirPath, filename)
            dst_path = os.path.join(self.testPath, filename)
            shutil.copy(src_path, dst_path)  # Use shutil.copy if you want to copy instead of move

def main():
    split = splitter(dir_path = "/home/hari/cement/datasets/images", train_path = "/home/hari/cement/datasets/images/train", val_path = "/home/hari/cement/datasets/images/val", test_path = "/home/hari/cement/datasets/images/test")
    split.splitData()

if __name__ == "__main__":
    main()