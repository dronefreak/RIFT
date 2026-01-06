"""
MNIST CNN Classifier

A simple convolutional neural network for MNIST digit classification.
Expected performance: ~99.45% test accuracy after 30 epochs.

This script serves as a test/benchmark for the CNN architecture used
in the junction detection project.
"""

from __future__ import print_function
import sys
import argparse
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

# Default hyperparameters
DEFAULT_BATCH_SIZE = 100
DEFAULT_NUM_CLASSES = 10
DEFAULT_EPOCHS = 30

# Image dimensions
IMG_ROWS, IMG_COLS = 28, 28


def load_and_preprocess_data():
	"""
	Load MNIST dataset and preprocess for training.

	Returns:
		Tuple of (x_train, y_train, x_test, y_test, input_shape)
	"""
	print("Loading MNIST dataset...")
	try:
		(x_train, y_train), (x_test, y_test) = mnist.load_data()
	except Exception as e:
		print(f"Error loading MNIST dataset: {e}")
		sys.exit(1)

	# Reshape data based on Keras backend configuration
	if K.image_data_format() == 'channels_first':
		x_train = x_train.reshape(x_train.shape[0], 1, IMG_ROWS, IMG_COLS)
		x_test = x_test.reshape(x_test.shape[0], 1, IMG_ROWS, IMG_COLS)
		input_shape = (1, IMG_ROWS, IMG_COLS)
	else:
		x_train = x_train.reshape(x_train.shape[0], IMG_ROWS, IMG_COLS, 1)
		x_test = x_test.reshape(x_test.shape[0], IMG_ROWS, IMG_COLS, 1)
		input_shape = (IMG_ROWS, IMG_COLS, 1)

	# Normalize pixel values to [0, 1]
	x_train = x_train.astype('float32') / 255.0
	x_test = x_test.astype('float32') / 255.0

	print(f'x_train shape: {x_train.shape}')
	print(f'{x_train.shape[0]} train samples')
	print(f'{x_test.shape[0]} test samples')

	# Convert labels to one-hot encoding
	y_train = keras.utils.to_categorical(y_train, DEFAULT_NUM_CLASSES)
	y_test = keras.utils.to_categorical(y_test, DEFAULT_NUM_CLASSES)

	return x_train, y_train, x_test, y_test, input_shape


def build_model(input_shape, num_classes):
	"""
	Build CNN model for MNIST classification.

	Args:
		input_shape: Shape of input images
		num_classes: Number of output classes

	Returns:
		Compiled Keras model
	"""
	model = Sequential()

	# First convolutional layer
	model.add(Conv2D(32, kernel_size=(5, 5),
	                 activation='relu',
	                 input_shape=input_shape))

	# Second convolutional layer
	model.add(Conv2D(64, (5, 5), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.5))

	# Flatten and fully connected layers
	model.add(Flatten())
	model.add(Dense(1024, activation='relu'))
	model.add(Dropout(0.5))

	# Output layer
	model.add(Dense(num_classes, activation='softmax'))

	# Compile model
	model.compile(loss=keras.losses.categorical_crossentropy,
	              optimizer=keras.optimizers.Adadelta(),
	              metrics=['accuracy'])

	print("\nModel architecture:")
	model.summary()

	return model


def parse_arguments():
	"""Parse command line arguments."""
	parser = argparse.ArgumentParser(
		description='Train CNN on MNIST dataset',
		formatter_class=argparse.ArgumentDefaultsHelpFormatter
	)

	parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE,
	                    help='Batch size for training')
	parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS,
	                    help='Number of training epochs')

	return parser.parse_args()


def main():
	"""Main execution function."""
	args = parse_arguments()

	print("=" * 60)
	print("MNIST CNN Classifier")
	print("=" * 60)
	print(f"Configuration:")
	print(f"  Batch size: {args.batch_size}")
	print(f"  Epochs: {args.epochs}")
	print(f"  Image size: {IMG_ROWS}x{IMG_COLS}")
	print("=" * 60)

	try:
		# Load and preprocess data
		x_train, y_train, x_test, y_test, input_shape = load_and_preprocess_data()

		# Build model
		print("\nBuilding model...")
		model = build_model(input_shape, DEFAULT_NUM_CLASSES)

		# Train model
		print("\nTraining model...")
		model.fit(x_train, y_train,
		          batch_size=args.batch_size,
		          epochs=args.epochs,
		          verbose=1,
		          validation_data=(x_test, y_test))

		# Evaluate model
		print("\nEvaluating model...")
		score = model.evaluate(x_test, y_test, verbose=0)
		print(f'Test loss: {score[0]:.4f}')
		print(f'Test accuracy: {score[1]:.4f} ({score[1]*100:.2f}%)')

	except Exception as e:
		print(f"\nError: {e}")
		import traceback
		traceback.print_exc()
		sys.exit(1)


if __name__ == '__main__':
	main()

