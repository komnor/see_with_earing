# Basic Auditory Vision Simulator - Documentation

## Overview

The Basic Auditory Vision Simulator is a software application that demonstrates how visual information can be converted to audio in real-time, based on the principles described in the scientific paper "Through the Ear, We See: A Neuroadaptive Blueprint for Non-Invasive Vision Restoration via Auditory Interfaces." This simulator implements the auditory frequency formula f = f0 + α⋅D + β⋅θ to convert 2D image data to spatialized audio, allowing users to experience the concept of "seeing through sound."

## Scientific Background

The simulator is based on the concept of cross-modal neuroplasticity, which allows the brain to repurpose neural pathways from one sensory modality to another. In this case, the visual cortex can be activated by auditory stimuli, enabling blind individuals to interpret visual information through sound.

The core of this approach is the auditory frequency formula:

```
f = f0 + α⋅D + β⋅θ
```

Where:
- f is the output frequency
- f0 is the base frequency
- α is the depth factor
- D is the depth value at a given point
- β is the angle factor
- θ is the horizontal angle from center

This formula maps spatial information (depth and horizontal position) to audio properties (frequency and stereo panning), creating a structured soundscape that can be interpreted by the brain.

## System Architecture

The simulator consists of three main components:

1. **Image Processing Module**: Processes frames from webcam or image files to extract visual features and generate depth maps.
2. **Audio Conversion Module**: Converts processed visual data to audio using the auditory frequency formula.
3. **Interactive User Interface**: Provides controls for adjusting parameters and visualizing the conversion process.

### Image Processing Module

The image processing module handles the following tasks:
- Capturing frames from webcam or loading image files
- Preprocessing frames (resizing, grayscale conversion)
- Generating depth maps using edge detection
- Extracting pixel data including intensity, depth, and position

### Audio Conversion Module

The audio conversion module handles the following tasks:
- Implementing the auditory frequency formula
- Generating audio samples in real-time
- Applying spatial audio effects (stereo panning)
- Managing audio playback

### Interactive User Interface

The user interface provides:
- Input source selection (webcam or image file)
- Visualization of original image, edges, and depth map
- Parameter controls for adjusting audio and image processing
- Real-time feedback on processing performance

## Implementation Details

### Technologies Used

- **Python**: Primary programming language
- **OpenCV**: Image processing and webcam capture
- **NumPy**: Numerical operations
- **PyAudio**: Real-time audio output
- **Tkinter**: Graphical user interface
- **Threading**: Multi-threaded processing for real-time performance

### Key Classes

#### RealTimeImageProcessor

This class handles frame processing in a separate thread, with features for:
- ROI selection
- Parameter adjustment
- Visualization
- Depth map generation

#### WebcamCapture

This class manages webcam input with:
- Frame capture in a separate thread
- Camera selection
- FPS monitoring

#### RealTimeAudioConverter

This class converts visual data to audio in real-time with:
- Implementation of the auditory frequency formula
- Stereo panning based on horizontal position
- Audio effects (reverb, compression)
- Continuous audio streaming

#### AuditoryVisionSimulator

This is the main application class that:
- Integrates all components
- Manages the user interface
- Handles user interactions
- Coordinates data flow between modules

## Features

- **Real-time Processing**: Processes webcam input or image files in real-time
- **Parameter Customization**: Adjustable parameters for audio and image processing
- **Visualization**: Visual feedback showing original image, edges, and depth map
- **Stereo Audio**: Spatial audio representation of visual data
- **Cross-platform**: Works on Windows, macOS, and Linux

## Usage Instructions

### Installation

1. Ensure Python 3.x is installed on your system
2. Install required dependencies:
   ```
   pip install numpy matplotlib scipy pillow librosa soundfile opencv-python pyaudio
   ```
3. On Linux, you may need to install additional system packages:
   ```
   sudo apt-get install python3-tk portaudio19-dev
   ```

### Running the Simulator

1. Navigate to the simulator directory
2. Run the main application:
   ```
   python src/simulator_app.py
   ```

### Using the Interface

1. **Input Selection**:
   - Choose between webcam and image file input
   - Select camera or image file

2. **Start/Stop**:
   - Click "Start" to begin processing
   - Click "Stop" to end processing

3. **Parameter Adjustment**:
   - Adjust image processing parameters (blur radius, depth scale)
   - Adjust audio parameters (base frequency, depth factor, angle factor, volume, reverb)
   - Adjust processing parameters (row step, column step)

4. **Visualization**:
   - The top section shows the original image
   - The middle section shows the edge detection
   - The bottom section shows the depth map

## Testing

The simulator includes comprehensive test scripts to verify functionality:

- `test_simulator.py`: Tests all components including integration
- `test_image_processing.py`: Tests only image processing components (for environments without audio hardware)

## Limitations and Future Improvements

### Current Limitations

- Requires audio hardware for full functionality
- Simplified depth estimation using edge detection
- Limited to 2D images or video frames
- Processing performance depends on hardware capabilities

### Future Improvements

- Support for depth cameras (e.g., Intel RealSense, Microsoft Kinect)
- Machine learning-based object detection and recognition
- More advanced audio synthesis techniques
- Training mode to help users learn to interpret the audio signals
- Mobile device support

## Conclusion

The Basic Auditory Vision Simulator demonstrates the potential of using auditory interfaces for non-invasive vision restoration. By converting visual information into structured soundscapes, it provides a foundation for further research and development in this field. The simulator's modular architecture and customizable parameters allow for experimentation with different approaches to audio-visual mapping.

This implementation serves as a proof of concept for the principles described in the paper "Through the Ear, We See," showing how cross-modal neuroplasticity can be leveraged to enable blind individuals to "see" through sound.
