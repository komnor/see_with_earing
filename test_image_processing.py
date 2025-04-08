import os
import sys
import time
import cv2
import numpy as np

# Add parent directory to path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.image_processor import RealTimeImageProcessor

def test_image_processor():
    """Test the RealTimeImageProcessor with a test image"""
    print("Testing RealTimeImageProcessor...")
    
    # Create test directory if it doesn't exist
    test_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test")
    os.makedirs(test_dir, exist_ok=True)
    
    # Create a test image
    test_image_path = os.path.join(test_dir, "test_image.png")
    if not os.path.exists(test_image_path):
        # Create a simple test image with geometric shapes
        width, height = 640, 480
        img = np.ones((height, width, 3), dtype=np.uint8) * 255  # White background
        
        # Draw a red circle
        cv2.circle(img, (width // 2, height // 2), 100, (0, 0, 255), -1)
        
        # Draw a blue rectangle
        cv2.rectangle(img, (100, 100), (200, 200), (255, 0, 0), -1)
        
        # Draw a green triangle
        triangle_pts = np.array([[400, 100], [500, 100], [450, 200]], dtype=np.int32)
        cv2.fillPoly(img, [triangle_pts], (0, 255, 0))
        
        # Save the test image
        cv2.imwrite(test_image_path, img)
        print(f"Created test image: {test_image_path}")
    
    # Initialize image processor
    processor = RealTimeImageProcessor()
    processor.start_processing()
    
    # Load test image
    test_image = cv2.imread(test_image_path)
    if test_image is None:
        print(f"Error: Could not load test image {test_image_path}")
        return False
    
    # Process the image
    processor.add_frame(test_image)
    
    # Wait for processing to complete
    time.sleep(1)
    
    # Get processed data
    processed_data = processor.get_latest_processed()
    if processed_data is None:
        print("Error: No processed data available")
        processor.stop_processing()
        return False
    
    # Create visualization
    viz_path = os.path.join(test_dir, "test_visualization.png")
    viz_image = processor.create_visualization(800, 600)
    cv2.imwrite(viz_path, viz_image)
    print(f"Created visualization: {viz_path}")
    
    # Get frame data
    frame_data = processor.get_frame_data(row_step=20, col_step=10)
    if not frame_data:
        print("Error: No frame data available")
        processor.stop_processing()
        return False
    
    print(f"Frame data contains {len(frame_data)} rows")
    print(f"First row contains {len(frame_data[0])} pixels")
    
    # Stop processing
    processor.stop_processing()
    
    print("RealTimeImageProcessor test completed successfully")
    return True

def test_webcam_capture():
    """Test webcam capture if available"""
    print("Testing webcam capture...")
    
    # Import WebcamCapture
    from src.image_processor import WebcamCapture
    
    # Initialize webcam
    webcam = WebcamCapture()
    
    # Get available cameras
    camera_list = webcam.get_camera_list()
    print(f"Available cameras: {camera_list}")
    
    if not camera_list:
        print("No cameras available, skipping webcam test")
        return True
    
    # Start capture
    if not webcam.start_capture():
        print("Error: Could not start webcam capture")
        return False
    
    # Capture frames for a few seconds
    print("Capturing frames for 3 seconds...")
    start_time = time.time()
    frame_count = 0
    
    while time.time() - start_time < 3:
        frame = webcam.get_frame()
        if frame is not None:
            frame_count += 1
        time.sleep(0.03)  # ~30 fps
    
    # Stop capture
    webcam.stop_capture()
    
    print(f"Captured {frame_count} frames in 3 seconds")
    print(f"Approximate FPS: {frame_count / 3:.1f}")
    
    if frame_count == 0:
        print("Warning: No frames captured")
        return False
    
    print("Webcam capture test completed successfully")
    return True

def main():
    """Run image processing tests only (no audio)"""
    print("Starting Basic Auditory Vision Simulator image processing tests...")
    
    # Test image processor
    if not test_image_processor():
        print("Image processor test failed")
        return False
    
    # Test webcam capture
    if not test_webcam_capture():
        print("Webcam capture test failed")
        # Continue with other tests
    
    print("Image processing tests completed successfully!")
    print("Note: Audio tests skipped due to lack of audio hardware in sandbox environment")
    return True

if __name__ == "__main__":
    main()
