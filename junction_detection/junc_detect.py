
"""
Junction Detection Training Script

This script trains a convolutional neural network to detect road junctions
from image data. It uses TFLearn (deprecated) and should be migrated to
pure TensorFlow/Keras in future versions.

Usage:
    python junc_detect.py --train-dir <path> --test-dir <path> [options]
"""

import cv2
import numpy as np
import os
import argparse
import sys
from random import shuffle
import matplotlib.pyplot as plt

# Default configuration
DEFAULT_IMG_SIZE = 80
DEFAULT_LEARN_RATE = 1e-3
DEFAULT_EPOCHS = 10
DEFAULT_BATCH_SIZE = 70
DEFAULT_VALIDATION_SPLIT = 50

font = cv2.FONT_HERSHEY_SIMPLEX


def label_image(img_filename):
	"""
	Extract label from image filename.

	Expected format: prefix.label.number.jpg
	where label is either 'junc1' or 'none'

	Args:
		img_filename: Image filename string

	Returns:
		One-hot encoded label: [1,0] for junction, [0,1] for no junction
	"""
	try:
		word_label = img_filename.split('.')[-3]
		if word_label == 'junc1':
			return [1, 0]
		elif word_label == 'none':
			return [0, 1]
		else:
			print(f"Warning: Unknown label '{word_label}' in {img_filename}, treating as 'none'")
			return [0, 1]
	except IndexError:
		print(f"Error: Could not parse label from {img_filename}")
		return [0, 1]


def create_train_data(training_directory, img_size):
	"""
	Load and prepare training data from directory.

	Args:
		training_directory: Path to training images
		img_size: Size to resize images to (square)

	Returns:
		List of [image, label] pairs
	"""
	if not os.path.exists(training_directory):
		raise FileNotFoundError(f"Training directory not found: {training_directory}")

	train_data = []
	image_list = [x for x in os.listdir(training_directory) if x.endswith('.jpg')]

	if not image_list:
		raise ValueError(f"No .jpg images found in {training_directory}")

	print(f"Loading {len(image_list)} training images from {training_directory}...")

	for img_filename in image_list:
		label = label_image(img_filename)
		path = os.path.join(training_directory, img_filename)

		img = cv2.imread(path, 0)  # Read as grayscale
		if img is None:
			print(f"Warning: Could not load image {path}")
			continue

		img = cv2.resize(img, (img_size, img_size))
		train_data.append([np.array(img), np.array(label)])

	shuffle(train_data)
	np.save('train_data.npy', train_data)
	print(f"Loaded {len(train_data)} training samples")
	return train_data


def process_test_data(testing_directory, img_size):
	"""
	Load and prepare test data from directory.

	Args:
		testing_directory: Path to test images
		img_size: Size to resize images to (square)

	Returns:
		Tuple of (test_data, image_list)
	"""
	if not os.path.exists(testing_directory):
		raise FileNotFoundError(f"Testing directory not found: {testing_directory}")

	test_data = []
	image_list = [x for x in os.listdir(testing_directory) if x.endswith('.jpg')]

	if not image_list:
		raise ValueError(f"No .jpg images found in {testing_directory}")

	print(f"Loading {len(image_list)} test images from {testing_directory}...")

	for img_filename in image_list:
		path = os.path.join(testing_directory, img_filename)
		img = cv2.imread(path, 0)  # Read as grayscale

		if img is None:
			print(f"Warning: Could not load image {path}")
			continue

		img = cv2.resize(img, (img_size, img_size))
		test_data.append(np.array(img))

	np.save('test_Data.npy', test_data)
	print(f"Loaded {len(test_data)} test samples")
	return test_data, image_list


def build_model(img_size, learn_rate):
	"""
	Build and compile the CNN model for junction detection.

	WARNING: This uses TFLearn which is deprecated. Consider migrating
	to pure TensorFlow/Keras.

	Args:
		img_size: Input image size (square)
		learn_rate: Learning rate for optimizer

	Returns:
		Compiled TFLearn model
	"""
	import tflearn
	from tflearn.layers.conv import conv_2d, max_pool_2d
	from tflearn.layers.core import input_data, dropout, fully_connected
	from tflearn.layers.estimator import regression

	# Input layer
	convnet = input_data(shape=[None, img_size, img_size, 1], name='input')

	# Convolutional layers - FIXED: Using 'relu' instead of incorrect 'softmax'
	convnet = conv_2d(convnet, 32, 5, activation='relu')
	convnet = max_pool_2d(convnet, 2)

	convnet = conv_2d(convnet, 64, 5, activation='relu')
	convnet = max_pool_2d(convnet, 2)

	# Fully connected layers
	convnet = fully_connected(convnet, 1024, activation='relu')
	convnet = dropout(convnet, 0.8)

	# Output layer - softmax is correct here for classification
	convnet = fully_connected(convnet, 2, activation='softmax')

	# Compile with optimizer and loss function
	convnet = regression(convnet,
	                     optimizer='adam',
	                     learning_rate=learn_rate,
	                     loss='categorical_crossentropy',
	                     name='targets')

	model = tflearn.DNN(convnet)
	return model


def train_model(model, train_data, img_size, epochs, batch_size, validation_split, model_name):
	"""
	Train the junction detection model.

	Args:
		model: Compiled TFLearn model
		train_data: Training data list
		img_size: Image size
		epochs: Number of training epochs
		batch_size: Batch size for training
		validation_split: Number of samples to use for validation
		model_name: Name for saving the model
	"""
	# Split data into train and validation
	train = train_data[:-validation_split]
	test = train_data[-validation_split:]

	print(f"Training samples: {len(train)}")
	print(f"Validation samples: {len(test)}")

	# Prepare training data
	X = np.array([i[0] for i in train]).reshape(-1, img_size, img_size, 1)
	Y = [i[1] for i in train]

	# Prepare validation data
	test_x = np.array([i[0] for i in test]).reshape(-1, img_size, img_size, 1)
	test_y = [i[1] for i in test]

	# Train the model
	print("Starting training...")
	model.fit({'input': X}, {'targets': Y},
	          n_epoch=epochs,
	          validation_set=({'input': test_x}, {'targets': test_y}),
	          batch_size=batch_size,
	          snapshot_step=500,
	          show_metric=True,
	          run_id=model_name)

	# Save the model
	model.save(model_name + '.tflearn')
	print(f"Model saved as {model_name}.tflearn")


def run_inference(model, testing_directory, img_size, output_dir='output'):
	"""
	Run inference on test images and save results.

	Args:
		model: Trained model
		testing_directory: Directory containing test images
		img_size: Image size for model input
		output_dir: Directory to save output images
	"""
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
		print(f"Created output directory: {output_dir}")

	test_data, image_list = process_test_data(testing_directory, img_size)

	print("\nRunning inference on test images...")
	junction_count = 0
	none_count = 0

	for idx, data in enumerate(test_data):
		data_reshaped = data.reshape(img_size, img_size, 1)
		model_out = model.predict([data_reshaped])[0]

		# Load original image in color for visualization
		path = os.path.join(testing_directory, image_list[idx])
		image = cv2.imread(path)

		if image is None:
			print(f"Warning: Could not load {path} for visualization")
			continue

		if np.argmax(model_out) == 0:
			# Junction detected
			label = 'JUNC'
			color = (0, 0, 255)  # Red
			output_filename = os.path.join(output_dir, f"junc_{junction_count}.jpg")
			junction_count += 1
			print(f"Junction detected: {image_list[idx]} (confidence: {model_out[0]:.3f})")
		else:
			# No junction
			label = 'NONE'
			color = (0, 255, 0)  # Green
			output_filename = os.path.join(output_dir, f"none_{none_count}.jpg")
			none_count += 1
			print(f"No junction: {image_list[idx]} (confidence: {model_out[1]:.3f})")

		# Add text to image
		cv2.putText(image, label, (10, 50), font, 2, color, 2)
		cv2.imwrite(output_filename, image)

	print(f"\nInference complete:")
	print(f"  Junctions detected: {junction_count}")
	print(f"  No junctions: {none_count}")
	print(f"  Results saved to: {output_dir}/")


def parse_arguments():
	"""Parse command line arguments."""
	parser = argparse.ArgumentParser(
		description='Train and test junction detection CNN',
		formatter_class=argparse.ArgumentDefaultsHelpFormatter
	)

	parser.add_argument('--train-dir', type=str, required=True,
	                    help='Path to training images directory')
	parser.add_argument('--test-dir', type=str, required=True,
	                    help='Path to test images directory')
	parser.add_argument('--img-size', type=int, default=DEFAULT_IMG_SIZE,
	                    help='Image size (square)')
	parser.add_argument('--learn-rate', type=float, default=DEFAULT_LEARN_RATE,
	                    help='Learning rate')
	parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS,
	                    help='Number of training epochs')
	parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE,
	                    help='Batch size for training')
	parser.add_argument('--validation-split', type=int, default=DEFAULT_VALIDATION_SPLIT,
	                    help='Number of samples for validation')
	parser.add_argument('--output-dir', type=str, default='output',
	                    help='Directory to save inference results')

	return parser.parse_args()


def main():
	"""Main execution function."""
	args = parse_arguments()

	# Generate model name
	model_name = f'junction_detection-lr{args.learn_rate}-epochs{args.epochs}'

	print("=" * 60)
	print("Junction Detection Training")
	print("=" * 60)
	print(f"Configuration:")
	print(f"  Training directory: {args.train_dir}")
	print(f"  Testing directory: {args.test_dir}")
	print(f"  Image size: {args.img_size}x{args.img_size}")
	print(f"  Learning rate: {args.learn_rate}")
	print(f"  Epochs: {args.epochs}")
	print(f"  Batch size: {args.batch_size}")
	print(f"  Validation split: {args.validation_split}")
	print(f"  Model name: {model_name}")
	print("=" * 60)

	try:
		# Load training data
		train_data = create_train_data(args.train_dir, args.img_size)

		# Build model
		print("\nBuilding model...")
		model = build_model(args.img_size, args.learn_rate)

		# Train model
		train_model(model, train_data, args.img_size, args.epochs,
		           args.batch_size, args.validation_split, model_name)

		# Run inference on test data
		print("\n" + "=" * 60)
		print("Running inference on test images...")
		print("=" * 60)
		run_inference(model, args.test_dir, args.img_size, args.output_dir)

		print("\n" + "=" * 60)
		print("Training and testing complete!")
		print("=" * 60)

	except Exception as e:
		print(f"\nError: {e}")
		import traceback
		traceback.print_exc()
		sys.exit(1)


if __name__ == '__main__':
	main()
