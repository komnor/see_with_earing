# README - Basic Auditory Vision Simulator

## Introduction

The Basic Auditory Vision Simulator is a software application that demonstrates how visual information can be converted to audio in real-time, based on the principles described in the scientific paper "Through the Ear, We See: A Neuroadaptive Blueprint for Non-Invasive Vision Restoration via Auditory Interfaces."

This simulator implements the auditory frequency formula `f = f0 + α⋅D + β⋅θ` to convert 2D image data to spatialized audio, allowing users to experience the concept of "seeing through sound" through cross-modal neuroplasticity.

## Quick Start

1. Extract the archive:
   ```
   tar -xzvf basic_auditory_vision_simulator.tar.gz
   ```

2. Navigate to the simulator directory:
   ```
   cd simulator
   ```

3. Run the simulator:
   ```
   ./run_simulator.sh
   ```
   
   Or manually:
   ```
   python3 src/simulator_app.py
   ```

## System Requirements

- Python 3.x
- Required Python packages:
  - numpy
  - matplotlib
  - scipy
  - pillow
  - librosa
  - soundfile
  - opencv-python
  - pyaudio
  - tkinter (python3-tk)
- Audio output device
- Webcam (optional, for live video input)

## Installation

1. Install required Python packages:
   ```
   pip install numpy matplotlib scipy pillow librosa soundfile opencv-python pyaudio
   ```

2. On Linux, you may need to install additional system packages:
   ```
   sudo apt-get install python3-tk portaudio19-dev
   ```

## Documentation

For detailed information about the simulator, please refer to the following documentation files:

- `documentation.md`: Comprehensive documentation on implementation, features, and usage
- `design_document.md`: Detailed design architecture and technical specifications

## Testing

The simulator includes test scripts to verify functionality:

- `test/test_simulator.py`: Tests all components including integration
- `test/test_image_processing.py`: Tests only image processing components (for environments without audio hardware)

To run the image processing tests:
```
python3 test/test_image_processing.py
```

## Features

- Real-time processing of webcam input or image files
- Conversion of visual information to spatialized audio
- Interactive user interface with parameter controls
- Visualization of original image, edges, and depth map
- Customizable audio and image processing parameters

## License

This software is provided for educational and research purposes only.

## Acknowledgments

This simulator is based on the concepts presented in the paper "Through the Ear, We See: A Neuroadaptive Blueprint for Non-Invasive Vision Restoration via Auditory Interfaces" by Ferenc Lengyel.
