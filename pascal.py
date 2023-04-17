"""
Load in pascal dataset to data generator 


def data_augmentation(image, label):
    # use numpy for random scalar between 0 and 512 for variable called x
    NUM_BOXES = 1
    boxes = tf.random.uniform(shape=(NUM_BOXES, 4))

    print(boxes.shape)

    image_ = tf.image.crop_and_resize(image, boxes=boxes, crop_size=[image_size, image_size], method="bilinear")
    label_ = tf.image.crop_and_resize(label, boxes=boxes, crop_size=[image_size, image_size], methd="nearest")

    return image_, label_

Pretty good augmentatoin tensorflow:
    https://www.datacamp.com/tutorial/complete-guide-data-augmentation



"""
import os
from skimage import io
import random
import numpy as np
import tensorflow as tf
from skimage.transform import resize
import tensorflow as tf

# from sklearn.preprocessing import LabelEncoder
from PIL import Image


class PascalDataGenerator:
    def __init__(self, image_size=572, batch_size=8, train_test_val="train"):
        self.image_size = image_size
        self.train, self.trainval, self.val = self.load_txts()
        self.train_test_val = train_test_val
        self.batch_size = batch_size

    def get_pascal_labels(self):
        """Load the mapping that associates pascal classes with label colors
        Returns:
            np.ndarray with dimensions (21, 3)
        """
        return np.asarray(
            [
                [0, 0, 0],
                [128, 0, 0],
                [0, 128, 0],
                [128, 128, 0],
                [0, 0, 128],
                [128, 0, 128],
                [0, 128, 128],
                [128, 128, 128],
                [64, 0, 0],
                [192, 0, 0],
                [64, 128, 0],
                [192, 128, 0],
                [64, 0, 128],
                [192, 0, 128],
                [64, 128, 128],
                [192, 128, 128],
                [0, 64, 0],
                [128, 64, 0],
                [0, 192, 0],
                [128, 192, 0],
                [0, 64, 128],
            ]
        )

    def encode_segmap(self, mask):
        """Encode segmentation label images as pascal classes
        Args:
            mask (np.ndarray): raw segmentation label image of dimension
                (M, N, 3), in which the Pascal classes are encoded as colours.
        Returns:
            (np.ndarray): class map with dimensions (M,N), where the value at
            a given location is the integer denoting the class index.
        """
        mask = mask.astype(int)
        label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
        for ii, label in enumerate(self.get_pascal_labels()):
            label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
        label_mask = label_mask.astype(int)
        return label_mask

    def augmentation(self, image, label):
        if np.random.uniform(0, 1, 1) > 0.7:
            image = tf.image.flip_left_right(image)
            label = tf.image.flip_left_right(label)
        if np.random.uniform(0, 1, 1) > 0.5:
            central_frac = np.random.uniform(0.5, 0.9, 1)
            image = tf.image.central_crop(image, central_fraction=central_frac)
            label = tf.image.central_crop(label, central_fraction=central_frac)
            image = tf.image.resize(image[np.newaxis, ...], (self.image_size, self.image_size), method="bilinear")[0]
            label = tf.image.resize(label[np.newaxis, ...], (self.image_size, self.image_size), method="nearest")[0]
        return image, label

    def load_txts(self):
        with open(f"data/VOCdevkit/VOC2010/ImageSets/Segmentation/train.txt") as f:
            train = f.read().splitlines()
        with open(f"data/VOCdevkit/VOC2010/ImageSets/Segmentation/trainval.txt") as f:
            trainval = f.read().splitlines()
        with open(f"data/VOCdevkit/VOC2010/ImageSets/Segmentation/val.txt") as f:
            val = f.read().splitlines()
        return train, trainval, val

    def load_filename(self, filename):
        image = io.imread(f"data/VOCdevkit/VOC2010/JPEGImages/{filename}.jpg")
        label = io.imread(f"data/VOCdevkit/VOC2010/SegmentationClass/{filename}.png")[..., :3]
        label = self.encode_segmap(label)
        label = label[..., np.newaxis]

        image = tf.image.resize(image[np.newaxis, ...], (self.image_size, self.image_size), method="bilinear")[0]
        label = tf.image.resize(label[np.newaxis, ...], (self.image_size, self.image_size), method="nearest")[0]

        # tf.debugging.check_numerics(label, "label is nan")
        # tf.debugging.check_numerics(image, "image is nan")

        return image / 255, tf.keras.utils.to_categorical(label, num_classes=21)

    def data_generator(self):
        if self.train_test_val == "train":
            data = self.train
        if self.train_test_val == "trainval":
            data = self.trainval
        if self.train_test_val == "val":
            data = self.val
        print(f"there are {len(data)} training examples")

        while True:
            # sample
            batch_filenames = random.sample(data, self.batch_size)
            batch = [self.load_filename(b) for b in batch_filenames]
            images, labels = [], []
            for b in batch:
                image, label = b
                image, label = self.augmentation(image, label)
                images.append(image)
                labels.append(label)
            images = np.array(images)
            labels = np.array(labels)
            yield (images, labels)


if __name__ == "__main__":
    pdg = PascalDataGenerator()

    # tfds = tf.data.Dataset.from_generator(
    #     pdg.data_generator, output_types=(tf.float32, tf.float32), output_shapes=([None, None, None, 3], [None, None, None, 1])
    # )

    # pdg.load_filename("2007_000032")

    image_size = 512
    batch_size = 16

    tfds = tf.data.Dataset.from_generator(
        pdg.data_generator,
        output_types=(tf.float32, tf.float32),
        output_shapes=([batch_size, image_size, image_size, 3], [batch_size, image_size, image_size, 21]),
    )

    print(tfds.element_spec)

    gen = PascalDataGenerator().data_generator()
    print(next(iter(gen)))

# img, msk = load_filename("2007_000032")
# print(msk.shape)
# print(msk)
