"""
CIFAR-10 CNN Classifier

A convolutional neural network for CIFAR-10 image classification with
data augmentation support.

This script serves as a test/benchmark for CNN architectures.
"""

from __future__ import print_function
import sys
import argparse
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

import os
import pickle
import numpy as np

# Default hyperparameters
DEFAULT_BATCH_SIZE = 32  # FIXED: Was 1, which is very inefficient
DEFAULT_NUM_CLASSES = 10
DEFAULT_EPOCHS = 50
DEFAULT_NUM_PREDICTIONS = 20


def load_and_preprocess_data():
	"""
	Load CIFAR-10 dataset and preprocess for training.

	Returns:
		Tuple of (x_train, y_train, x_test, y_test)
	"""
	print("Loading CIFAR-10 dataset...")
	try:
		(x_train, y_train), (x_test, y_test) = cifar10.load_data()
	except Exception as e:
		print(f"Error loading CIFAR-10 dataset: {e}")
		sys.exit(1)

	print(f'x_train shape: {x_train.shape}')
	print(f'{x_train.shape[0]} train samples')
	print(f'{x_test.shape[0]} test samples')

	# Normalize pixel values to [0, 1]
	x_train = x_train.astype('float32') / 255.0
	x_test = x_test.astype('float32') / 255.0

	# Convert class vectors to binary class matrices
	y_train = keras.utils.to_categorical(y_train, DEFAULT_NUM_CLASSES)
	y_test = keras.utils.to_categorical(y_test, DEFAULT_NUM_CLASSES)

	return x_train, y_train, x_test, y_test


def build_model(input_shape, num_classes):
	"""
	Build CNN model for CIFAR-10 classification.

	Args:
		input_shape: Shape of input images
		num_classes: Number of output classes

	Returns:
		Compiled Keras model
	"""
	model = Sequential()

	# First convolutional block
	model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
	model.add(Activation('relu'))

	# Second convolutional block
	model.add(Conv2D(64, (3, 3), padding='same'))
	model.add(Activation('relu'))

	# Flatten and fully connected layers
	model.add(Flatten())
	model.add(Dense(1024))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))

	# Output layer
	model.add(Dense(num_classes))
	model.add(Activation('softmax'))

	# FIXED: Use RMSprop() class instead of deprecated rmsprop() function
	# FIXED: Use learning_rate instead of deprecated lr parameter
	opt = keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-6)

	# Compile model
	model.compile(loss='categorical_crossentropy',
	              optimizer=opt,
	              metrics=['accuracy'])

	print("\nModel architecture:")
	model.summary()

	return model


def train_model(model, x_train, y_train, x_test, y_test, batch_size, epochs,
                use_data_augmentation, save_dir, model_name):
	"""
	Train the CIFAR-10 model with optional data augmentation.

	Args:
		model: Compiled Keras model
		x_train, y_train: Training data
		x_test, y_test: Test data
		batch_size: Batch size for training
		epochs: Number of epochs
		use_data_augmentation: Whether to use data augmentation
		save_dir: Directory to save model
		model_name: Name for saved model file
	"""
	if not use_data_augmentation:
		print('Training without data augmentation...')
		model.fit(x_train, y_train,
		          batch_size=batch_size,
		          epochs=epochs,
		          validation_data=(x_test, y_test),
		          shuffle=True,
		          verbose=1)
	else:
		print('Training with real-time data augmentation...')
		# Create data generator with augmentation
		datagen = ImageDataGenerator(
			featurewise_center=False,
			samplewise_center=False,
			featurewise_std_normalization=False,
			samplewise_std_normalization=False,
			zca_whitening=False,
			rotation_range=0,
			width_shift_range=0.1,
			height_shift_range=0.1,
			horizontal_flip=True,
			vertical_flip=False)

		# Compute statistics for normalization
		datagen.fit(x_train)

		# FIXED: Use fit() instead of deprecated fit_generator()
		model.fit(datagen.flow(x_train, y_train, batch_size=batch_size),
		          steps_per_epoch=x_train.shape[0] // batch_size,
		          epochs=epochs,
		          validation_data=(x_test, y_test),
		          verbose=1)

	# Save model
	if not os.path.isdir(save_dir):
		os.makedirs(save_dir)
		print(f"Created directory: {save_dir}")

	model_path = os.path.join(save_dir, model_name)
	model.save(model_path)
	print(f'Saved trained model at {model_path}')

	return model_path


def load_label_names():
	"""
	Load CIFAR-10 label names from the dataset.

	Returns:
		Dictionary with label names
	"""
	label_list_path = 'datasets/cifar-10-batches-py/batches.meta'

	keras_dir = os.path.expanduser(os.path.join('~', '.keras'))
	datadir_base = os.path.expanduser(keras_dir)
	if not os.access(datadir_base, os.W_OK):
		datadir_base = os.path.join('/tmp', '.keras')
	label_list_path = os.path.join(datadir_base, label_list_path)

	try:
		with open(label_list_path, mode='rb') as f:
			labels = pickle.load(f)
		return labels
	except FileNotFoundError:
		print(f"Warning: Label file not found at {label_list_path}")
		# Return default labels
		return {'label_names': ['airplane', 'automobile', 'bird', 'cat', 'deer',
		                        'dog', 'frog', 'horse', 'ship', 'truck']}


def evaluate_and_predict(model, x_test, y_test, num_predictions):
	"""
	Evaluate model and show sample predictions.

	Args:
		model: Trained model
		x_test, y_test: Test data
		num_predictions: Number of sample predictions to display
	"""
	# Load label names
	labels = load_label_names()

	# Evaluate model
	# FIXED: Use evaluate() instead of deprecated evaluate_generator()
	print("\nEvaluating model...")
	score = model.evaluate(x_test, y_test, verbose=0)
	print(f'Test loss: {score[0]:.4f}')
	print(f'Test accuracy: {score[1]:.4f} ({score[1]*100:.2f}%)')

	# Make predictions
	# FIXED: Use predict() instead of deprecated predict_generator()
	print(f"\nSample predictions (first {num_predictions}):")
	predictions = model.predict(x_test[:num_predictions])

	for i in range(min(num_predictions, len(predictions))):
		actual_label = labels['label_names'][np.argmax(y_test[i])]
		predicted_label = labels['label_names'][np.argmax(predictions[i])]
		match = "✓" if actual_label == predicted_label else "✗"
		print(f'{match} Actual: {actual_label:12s} | Predicted: {predicted_label:12s}')


def parse_arguments():
	"""Parse command line arguments."""
	parser = argparse.ArgumentParser(
		description='Train CNN on CIFAR-10 dataset with data augmentation',
		formatter_class=argparse.ArgumentDefaultsHelpFormatter
	)

	parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE,
	                    help='Batch size for training')
	parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS,
	                    help='Number of training epochs')
	parser.add_argument('--no-augmentation', action='store_true',
	                    help='Disable data augmentation')
	parser.add_argument('--num-predictions', type=int, default=DEFAULT_NUM_PREDICTIONS,
	                    help='Number of sample predictions to show')
	parser.add_argument('--save-dir', type=str, default='saved_models',
	                    help='Directory to save the trained model')
	parser.add_argument('--model-name', type=str, default='keras_cifar10_trained_model.h5',
	                    help='Filename for the saved model')

	return parser.parse_args()


def main():
	"""Main execution function."""
	args = parse_arguments()

	print("=" * 60)
	print("CIFAR-10 CNN Classifier")
	print("=" * 60)
	print(f"Configuration:")
	print(f"  Batch size: {args.batch_size}")
	print(f"  Epochs: {args.epochs}")
	print(f"  Data augmentation: {not args.no_augmentation}")
	print(f"  Save directory: {args.save_dir}")
	print(f"  Model name: {args.model_name}")
	print("=" * 60)

	try:
		# Load and preprocess data
		x_train, y_train, x_test, y_test = load_and_preprocess_data()

		# Build model
		print("\nBuilding model...")
		input_shape = x_train.shape[1:]
		model = build_model(input_shape, DEFAULT_NUM_CLASSES)

		# Train model
		print("\nTraining model...")
		train_model(model, x_train, y_train, x_test, y_test,
		           args.batch_size, args.epochs, not args.no_augmentation,
		           args.save_dir, args.model_name)

		# Evaluate and show predictions
		evaluate_and_predict(model, x_test, y_test, args.num_predictions)

		print("\n" + "=" * 60)
		print("Training complete!")
		print("=" * 60)

	except Exception as e:
		print(f"\nError: {e}")
		import traceback
		traceback.print_exc()
		sys.exit(1)


if __name__ == '__main__':
	main()

