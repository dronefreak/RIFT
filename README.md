# RIFT - Road and Junction Detection

[![Watch the video](https://github.com/dronefreak/RIFT/blob/master/rift.png)](https://www.youtube.com/watch?v=XviSLDWHbTw&feature=youtu.be)

## Overview

RIFT is a computer vision project for autonomous navigation, combining deep learning-based junction detection with real-time road following algorithms. The project includes:

- **Junction Detection**: Deep learning models (CNN) for detecting road junctions
- **Road Following**: ROS-based real-time road detection and autonomous navigation
- **Multi-platform**: Python-based training and C++ ROS nodes for deployment

## Features

- Junction detection using convolutional neural networks
- Real-time road segmentation using color and edge-based filtering
- ROS integration for drone/robot control
- Support for both pre-recorded video and live camera feeds

## Project Structure

```
RIFT/
├── junction_detection/          # Python deep learning models
│   ├── junc_detect.py          # Junction detection training script
│   ├── mnist_test_conv.py      # MNIST CNN test
│   └── cifar10_test_conv.py    # CIFAR-10 CNN test
├── road_not_taken/             # C++ ROS package
│   ├── src/                    # Source files
│   │   ├── main.cpp           # ROS node main
│   │   ├── process_image.cpp  # Image processing algorithms
│   │   └── pub_img.cpp        # Image publisher
│   ├── include/               # Header files
│   └── CMakeLists.txt         # Build configuration
└── requirements.txt           # Python dependencies
```

## Dependencies

### Python Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- TensorFlow >= 2.8.0
- Keras >= 2.8.0
- OpenCV >= 4.5.0
- NumPy >= 1.21.0
- Matplotlib >= 3.3.0

### ROS Dependencies (C++)

- ROS Kinetic/Melodic/Noetic
- OpenCV 3.x or 4.x
- cv_bridge
- image_transport
- sensor_msgs
- geometry_msgs

## Installation

### Python Setup

```bash
# Clone the repository
git clone https://github.com/dronefreak/RIFT.git
cd RIFT

# Install Python dependencies
pip install -r requirements.txt
```

### ROS Package Setup

```bash
# Copy to your catkin workspace
cp -r road_not_taken ~/catkin_ws/src/

# Build the package
cd ~/catkin_ws
catkin_make

# Source the workspace
source devel/setup.bash
```

## Usage

### Junction Detection Training

```bash
cd junction_detection

# Configure paths in junc_detect.py first, then run:
python junc_detect.py --train-dir /path/to/training/data \
                      --test-dir /path/to/test/data \
                      --img-size 80 \
                      --epochs 10
```

### Road Following (ROS)

```bash
# Launch the road following node
rosrun road_not_taken road_not_taken_node

# For camera input (modify system_parameters.h and rebuild)
# Uncomment PROCESS_USB_WEB_CAM_FRAMES in include/system_parameters.h
```

## Configuration

### Image Processing Parameters

Edit `road_not_taken/include/system_parameters.h`:

```cpp
#define PROCESSED_IMAGE_HEIGHT  100
#define PROCESSED_IMAGE_WIDTH   100
#define PROCESS_USB_WEB_CAM_FRAMES  // Uncomment for live camera
```

### Neural Network Parameters

Edit the Python training scripts to adjust:
- Image size
- Learning rate
- Network architecture
- Batch size
- Number of epochs

## Algorithm Details

### Road Detection

The road detection algorithm uses:
1. **Spectral Filtering**: RGB color analysis to identify road-like pixels
2. **Edge Detection**: Canny edge detection for road boundaries
3. **Flood Fill**: Region growing from bottom center to identify road area
4. **Centroid Calculation**: Computes steering direction based on road centroid

### Junction Detection

CNN-based binary classifier:
- Input: 80x80 grayscale images
- Architecture: 2 Conv layers + Fully connected layers
- Output: Junction vs. No-junction classification

## Known Issues & TODOs

- [ ] Migrate from deprecated TFLearn to pure TensorFlow/Keras
- [ ] Add unit tests for image processing functions
- [ ] Improve error handling in Python scripts
- [ ] Add launch files for ROS nodes
- [ ] Create ROS package.xml

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## License

See LICENSE file for details.

## Demo Video

Watch the system in action: [YouTube Demo](https://www.youtube.com/watch?v=XviSLDWHbTw&feature=youtu.be)
