import cv2
import numpy as np
import scipy.ndimage as ndimage
import time
from PIL import Image, ImageTk
import threading
import queue

class RealTimeImageProcessor:
    """
    Class for real-time processing of images and video frames
    with features for audio conversion
    """
    def __init__(self, frame_rate=30, resolution=(640, 480)):
        """
        Initialize the image processor with customizable parameters
        
        Parameters:
        - frame_rate (int): Target frame rate for processing
        - resolution (tuple): Target resolution (width, height) for processing
        """
        self.frame_rate = frame_rate
        self.target_width, self.target_height = resolution
        
        # Processing parameters
        self.blur_radius = 3
        self.edge_threshold = 100
        self.depth_scale = 1.0
        
        # Current frame data
        self.current_frame = None
        self.grayscale = None
        self.edges = None
        self.depth_map = None
        self.processed_timestamp = 0
        
        # Processing flags
        self.processing_active = False
        self.processing_thread = None
        self.frame_queue = queue.Queue(maxsize=2)  # Limit queue size to avoid memory issues
        self.processed_frame_queue = queue.Queue(maxsize=2)
        
        # Region of interest (ROI) - default to full frame
        self.roi_enabled = False
        self.roi = (0, 0, self.target_width, self.target_height)  # (x, y, width, height)
        
        # Statistics
        self.fps = 0
        self.processing_time = 0
    
    def start_processing(self):
        """Start the background processing thread"""
        if not self.processing_active:
            self.processing_active = True
            self.processing_thread = threading.Thread(target=self._processing_loop)
            self.processing_thread.daemon = True
            self.processing_thread.start()
            return True
        return False
    
    def stop_processing(self):
        """Stop the background processing thread"""
        self.processing_active = False
        if self.processing_thread:
            self.processing_thread.join(timeout=1.0)
            self.processing_thread = None
        return True
    
    def _processing_loop(self):
        """Background thread for continuous frame processing"""
        while self.processing_active:
            try:
                # Get the latest frame from the queue, non-blocking
                frame = self.frame_queue.get(block=False)
                
                # Process the frame
                start_time = time.time()
                self._process_frame(frame)
                self.processing_time = time.time() - start_time
                
                # Calculate FPS
                current_time = time.time()
                time_diff = current_time - self.processed_timestamp
                if time_diff > 0:
                    self.fps = 1.0 / time_diff
                self.processed_timestamp = current_time
                
                # Put processed frame in the output queue
                try:
                    # Package processed data
                    processed_data = {
                        'original': self.current_frame,
                        'grayscale': self.grayscale,
                        'edges': self.edges,
                        'depth_map': self.depth_map,
                        'timestamp': self.processed_timestamp,
                        'fps': self.fps,
                        'processing_time': self.processing_time
                    }
                    
                    # Try to put in queue without blocking
                    self.processed_frame_queue.put(processed_data, block=False)
                except queue.Full:
                    # Queue is full, skip this frame
                    pass
                
                # Mark task as done
                self.frame_queue.task_done()
                
                # Sleep to maintain target frame rate
                target_time = 1.0 / self.frame_rate
                if self.processing_time < target_time:
                    time.sleep(target_time - self.processing_time)
            
            except queue.Empty:
                # No frames in queue, sleep briefly
                time.sleep(0.01)
            except Exception as e:
                print(f"Error in processing loop: {str(e)}")
                time.sleep(0.1)  # Sleep to avoid tight loop on error
    
    def add_frame(self, frame):
        """
        Add a new frame to the processing queue
        
        Parameters:
        - frame (numpy.ndarray): BGR image frame from camera or file
        
        Returns:
        - success (bool): Whether the frame was added to the queue
        """
        if not self.processing_active:
            return False
        
        try:
            # Resize frame to target resolution
            if frame.shape[1] != self.target_width or frame.shape[0] != self.target_height:
                frame = cv2.resize(frame, (self.target_width, self.target_height))
            
            # Try to add to queue without blocking
            self.frame_queue.put(frame, block=False)
            return True
        except queue.Full:
            # Queue is full, skip this frame
            return False
    
    def get_latest_processed(self):
        """
        Get the latest processed frame data
        
        Returns:
        - processed_data (dict): Dictionary containing processed frame data
        """
        try:
            # Get the latest processed frame without blocking
            return self.processed_frame_queue.get(block=False)
        except queue.Empty:
            # No processed frames available
            return None
    
    def _process_frame(self, frame):
        """
        Process a single frame
        
        Parameters:
        - frame (numpy.ndarray): BGR image frame
        """
        # Store the current frame
        self.current_frame = frame
        
        # Apply ROI if enabled
        if self.roi_enabled:
            x, y, w, h = self.roi
            frame = frame[y:y+h, x:x+w]
        
        # Convert to grayscale
        self.grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect edges using Sobel filters
        edges_x = ndimage.sobel(self.grayscale, axis=0)
        edges_y = ndimage.sobel(self.grayscale, axis=1)
        self.edges = np.hypot(edges_x, edges_y)
        self.edges = self.edges / (self.edges.max() + 1e-10) * 255  # Normalize with epsilon to avoid division by zero
        
        # Create a simple depth map (inverse of edge intensity with blur)
        self.depth_map = 255 - self.edges
        
        # Convert to float32 to avoid float16 issues
        self.depth_map = self.depth_map.astype(np.float32)
        
        # Apply Gaussian blur for smoothing
        self.depth_map = ndimage.gaussian_filter(self.depth_map, sigma=self.blur_radius)
        
        # Normalize depth map
        self.depth_map = self.depth_map / (self.depth_map.max() + 1e-10) * 255 * self.depth_scale
    
    def set_roi(self, x, y, width, height, enable=True):
        """
        Set region of interest for processing
        
        Parameters:
        - x, y (int): Top-left corner coordinates
        - width, height (int): ROI dimensions
        - enable (bool): Whether to enable ROI
        """
        self.roi = (x, y, width, height)
        self.roi_enabled = enable
    
    def disable_roi(self):
        """Disable region of interest processing"""
        self.roi_enabled = False
    
    def set_parameters(self, blur_radius=None, edge_threshold=None, depth_scale=None):
        """
        Update processing parameters
        
        Parameters:
        - blur_radius (float): Radius for Gaussian blur
        - edge_threshold (int): Threshold for edge detection
        - depth_scale (float): Scaling factor for depth map
        """
        if blur_radius is not None:
            self.blur_radius = blur_radius
        if edge_threshold is not None:
            self.edge_threshold = edge_threshold
        if depth_scale is not None:
            self.depth_scale = depth_scale
    
    def get_pixel_data(self, row, col):
        """
        Get pixel data at a specific position for audio conversion
        
        Parameters:
        - row, col (int): Pixel coordinates
        
        Returns:
        - pixel_data (dict): Dictionary with pixel data or None if invalid
        """
        if self.grayscale is None or self.depth_map is None:
            return None
        
        # Check if coordinates are within bounds
        if row < 0 or row >= self.grayscale.shape[0] or col < 0 or col >= self.grayscale.shape[1]:
            return None
        
        # Get grayscale intensity (brightness)
        intensity = self.grayscale[row, col]
        
        # Get depth value
        depth = self.depth_map[row, col]
        
        # Calculate horizontal angle from center (normalized to -1 to 1)
        center_col = self.grayscale.shape[1] / 2
        angle = (col - center_col) / center_col  # -1 (left) to 1 (right)
        
        # Calculate vertical position (normalized to 0 to 1)
        vert_pos = row / self.grayscale.shape[0]  # 0 (top) to 1 (bottom)
        
        return {
            'intensity': intensity / 255.0,  # Normalize to 0-1
            'depth': depth / 255.0,  # Normalize to 0-1
            'angle': angle,  # -1 to 1
            'vert_pos': vert_pos  # 0 to 1
        }
    
    def get_row_data(self, row, step=10):
        """
        Get data for an entire row with specified step size
        
        Parameters:
        - row (int): Row index
        - step (int): Step size for sampling columns
        
        Returns:
        - row_data (list): List of pixel data dictionaries
        """
        if self.grayscale is None or self.depth_map is None:
            return []
        
        row_data = []
        for col in range(0, self.grayscale.shape[1], step):
            pixel_data = self.get_pixel_data(row, col)
            if pixel_data:
                row_data.append(pixel_data)
        
        return row_data
    
    def get_frame_data(self, row_step=20, col_step=10):
        """
        Get data for the entire frame with specified step sizes
        
        Parameters:
        - row_step (int): Step size for sampling rows
        - col_step (int): Step size for sampling columns
        
        Returns:
        - frame_data (list): List of lists containing pixel data
        """
        if self.grayscale is None or self.depth_map is None:
            return []
        
        frame_data = []
        for row in range(0, self.grayscale.shape[0], row_step):
            row_data = self.get_row_data(row, col_step)
            if row_data:
                frame_data.append(row_data)
        
        return frame_data
    
    def create_visualization(self, width=800, height=600):
        """
        Create a visualization of the processed frame
        
        Parameters:
        - width, height (int): Dimensions of the visualization
        
        Returns:
        - visualization (numpy.ndarray): BGR visualization image
        """
        if self.current_frame is None or self.grayscale is None or self.edges is None or self.depth_map is None:
            # Create a blank image if no data is available
            return np.zeros((height, width, 3), dtype=np.uint8)
        
        # Create a visualization with original, edges, and depth map
        # Calculate the height of each section
        section_height = height // 3
        
        # Create a blank image
        visualization = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Resize images to fit in the visualization
        orig_resized = cv2.resize(self.current_frame, (width, section_height))
        
        # Convert grayscale edges to BGR for visualization
        edges_colored = cv2.cvtColor(self.edges.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        edges_resized = cv2.resize(edges_colored, (width, section_height))
        
        # Create a colored depth map using a colormap
        depth_colored = cv2.applyColorMap(self.depth_map.astype(np.uint8), cv2.COLORMAP_JET)
        depth_resized = cv2.resize(depth_colored, (width, section_height))
        
        # Combine the images vertically
        visualization[0:section_height, :, :] = orig_resized
        visualization[section_height:2*section_height, :, :] = edges_resized
        visualization[2*section_height:3*section_height, :, :] = depth_resized
        
        # Add text with processing information
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(visualization, f"FPS: {self.fps:.1f}", (10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(visualization, f"Processing Time: {self.processing_time*1000:.1f} ms", (10, 70), font, 1, (255, 255, 255), 2)
        
        return visualization
    
    def create_tk_image(self, frame):
        """
        Convert a BGR frame to a PhotoImage for Tkinter
        
        Parameters:
        - frame (numpy.ndarray): BGR image
        
        Returns:
        - tk_image (ImageTk.PhotoImage): Image for Tkinter
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(rgb_frame)
        
        # Convert to PhotoImage
        return ImageTk.PhotoImage(image=pil_image)


class WebcamCapture:
    """
    Class for capturing frames from webcam
    """
    def __init__(self, camera_index=0, resolution=(640, 480), fps=30):
        """
        Initialize webcam capture
        
        Parameters:
        - camera_index (int): Index of the camera to use
        - resolution (tuple): Target resolution (width, height)
        - fps (int): Target frame rate
        """
        self.camera_index = camera_index
        self.width, self.height = resolution
        self.fps = fps
        
        self.cap = None
        self.is_capturing = False
        self.capture_thread = None
        
        # Frame queue
        self.frame_queue = queue.Queue(maxsize=2)
        
        # Statistics
        self.frame_count = 0
        self.start_time = 0
        self.current_fps = 0
    
    def start_capture(self):
        """Start webcam capture"""
        if self.is_capturing:
            return False
        
        # Initialize camera
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            print(f"Error: Could not open camera {self.camera_index}")
            return False
        
        # Set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        
        # Set FPS
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        # Start capture thread
        self.is_capturing = True
        self.frame_count = 0
        self.start_time = time.time()
        self.capture_thread = threading.Thread(target=self._capture_loop)
        self.capture_thread.daemon = True
        self.capture_thread.start()
        
        return True
    
    def stop_capture(self):
        """Stop webcam capture"""
        self.is_capturing = False
        
        if self.capture_thread:
            self.capture_thread.join(timeout=1.0)
            self.capture_thread = None
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        return True
    
    def _capture_loop(self):
        """Background thread for continuous frame capture"""
        while self.is_capturing and self.cap and self.cap.isOpened():
            try:
                # Capture frame
                ret, frame = self.cap.read()
                
                if not ret:
                    print("Error: Failed to capture frame")
                    time.sleep(0.1)
                    continue
                
                # Update statistics
                self.frame_count += 1
                elapsed_time = time.time() - self.start_time
                if elapsed_time > 0:
                    self.current_fps = self.frame_count / elapsed_time
                
                # Reset statistics every second
                if elapsed_time > 1.0:
                    self.start_time = time.time()
                    self.frame_count = 0
                
                # Try to add to queue without blocking
                try:
                    self.frame_queue.put(frame, block=False)
                except queue.Full:
                    # Queue is full, skip this frame
                    pass
                
                # Sleep to maintain target frame rate
                time.sleep(1.0 / self.fps)
            
            except Exception as e:
                print(f"Error in capture loop: {str(e)}")
                time.sleep(0.1)
    
    def get_frame(self):
        """
        Get the latest frame from the webcam
        
        Returns:
        - frame (numpy.ndarray): BGR image frame or None if no frame is available
        """
        if not self.is_capturing:
            return None
        
        try:
            # Get the latest frame without blocking
            return self.frame_queue.get(block=False)
        except queue.Empty:
            # No frames available
            return None
    
    def get_camera_list(self):
        """
        Get a list of available cameras
        
        Returns:
        - camera_list (list): List of available camera indices
        """
        camera_list = []
        
        # Try the first 10 camera indices
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                camera_list.append(i)
                cap.release()
        
        return camera_list
