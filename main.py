# Import necessary packages
import tensorflow as tf
from tensorflow.keras.applications import VGG16, VGG19, ResNet50
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers as keras_layers
from tensorflow.keras import models as keras_models

from sklearn.model_selection import StratifiedShuffleSplit

import numpy as np
import matplotlib.pyplot as plt
import argparse
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("--train", default=False, type=bool, help="Include training")
parser.add_argument("--test_size", default=0.2, type=float, help="Proportion of the test set")
parser.add_argument("--image", default=None, type=str, help="Path to the input image")
parser.add_argument("--model", default="vgg16", type=str, help="Name of the model to use")
parser.add_argument("--seed", default=42, type=int, help="Random seed")

NUM_CLASSES = 5
CLASS_SIZE = 100
MODELS = {"vgg16": VGG16,
          "vgg19": VGG19,
          "resnet": ResNet50}
LABELS = {0 : "cat",
          1: "dog",
          2: "bird",
          3: "fish",
          4: "insect"}


def main(args):
    # Training phase
    if args.train:
        # Set the random seed
        np.random.seed(args.seed)
        tf.random.set_seed(args.seed)

        # Download the pre-trained model
        # include_top=False for not including the 3 fully-connected layers at the top of the network - output layers will be created on our own
        # However, in this case we need to specify the input shape
        if args.model not in MODELS.keys():
            raise AssertionError("The --model command line argument is invalid")
        
        print(f"[INFO] loading {args.model}...")
        model = MODELS[args.model](weights='imagenet', include_top=False, input_shape=(224, 224, 3))

        # Modify the model to fit our needs

        # Freeze initial layers to keep the lower-level features learned from the original dataset - decreases the time needed for training and reduces the risk of overfitting
        for layer in model.layers:
            layer.trainable = False

        # Modify the model architecture - add three 
        new_output = keras_layers.GlobalAveragePooling2D()(model.output)
        new_output = keras_layers.Dense(1024, activation='relu')(new_output)
        new_output = keras_layers.Dense(NUM_CLASSES, activation='softmax')(new_output)
        model = keras_models.Model(inputs=model.input, outputs=new_output)
        
        # Compile the model
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Load the data
        cats = [load_img(f"Dataset/Cats/cat{i}.jpg") for i in range(1, CLASS_SIZE+1)]
        dogs = [load_img(f"Dataset/Dogs/dog{i}.jpg") for i in range(1, CLASS_SIZE+1)]
        birds = [load_img(f"Dataset/Birds/bird{i}.jpg") for i in range(1, CLASS_SIZE+1)]
        fish = [load_img(f"Dataset/Fish/fish{i}.jpg") for i in range(1, CLASS_SIZE+1)]
        insects = [load_img(f"Dataset/Insects/insect{i}.jpg") for i in range(1, CLASS_SIZE+1)]

        # Process the data and create labels
        data = cats + dogs + birds + fish + insects
        data = np.array([img_to_array(image) for image in data])
        # data = np.array([preprocess_input(image) for image in data])
        labels = np.array([i for i in range(NUM_CLASSES) for _ in range(CLASS_SIZE)])

        # Split the data using stratified shuffle split - allows to maintain the overall class distribution in the train and test sets
        stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
        for train_idx, test_idx in stratified_split.split(data, labels):
            train_data, test_data = data[train_idx], data[test_idx]
            train_labels, test_labels = labels[train_idx], labels[test_idx]

        # One-hot encode the labels
        train_labels = to_categorical(train_labels, num_classes=NUM_CLASSES)
        test_labels = to_categorical(test_labels, num_classes=NUM_CLASSES)

        # Create an ImageDataGenerator object with the desired augmentations
        generator = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            preprocessing_function=preprocess_input
        )
        generator.fit(train_data)

        # Train the model
        print("[INFO] training the model...")
        batch_size = 16
        epochs = 3
        history = model.fit(
            generator.flow(train_data, train_labels, batch_size=batch_size),
            steps_per_epoch=len(train_data) // batch_size,
            epochs=epochs
        )
        # Show training history
        accuracy, loss = np.array(history.history["accuracy"]), np.array(history.history["loss"])
        plt.figure(figsize=(5,5))
        plt.plot(accuracy * 100, label='Train accuracy')
        plt.plot(loss * 100, label='Train loss')
        plt.xticks(np.arange(0,epochs))
        plt.xlabel('Epoch')
        plt.legend()
        plt.show()

        # Evaluate the model
        train_loss, train_accuracy = model.evaluate(preprocess_input(train_data), train_labels)
        test_loss, test_accuracy = model.evaluate(preprocess_input(test_data), test_labels)
        print("Train accuracy:", train_accuracy)
        print("Test accuracy:", test_accuracy)

        # Save the model
        model.save(f"{args.model}.h5")


    # Predict the class of the image
    if args.image:
        # Preprocess the image
        image = load_img(args.image, target_size=(224,224))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)

        # Load the model
        if args.model not in MODELS.keys():
            raise AssertionError("The --model command line argument is invalid")
        model = keras_models.load_model(f"{args.model}.h5")
        
        # Make prediction
        pred = model.predict(image)
        label = np.argmax(pred)
        prob = pred[0][label]
        
        # Show the result
        orig = cv2.imread(args.image)
        cv2.putText(orig, "Label: {}, {:.2f}% ({})".format(LABELS[label], prob * 100, args.model), 
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
        cv2.imshow("Classification", orig)
        cv2.waitKey(0)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)