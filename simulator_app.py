import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
import threading
import time
import os
import sys

# Add parent directory to path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.image_processor import RealTimeImageProcessor, WebcamCapture
from src.audio_converter import RealTimeAudioConverter

class AuditoryVisionSimulator:
    """
    Main application class for the Basic Auditory Vision Simulator
    """
    def __init__(self, root):
        """
        Initialize the simulator application
        
        Parameters:
        - root: Tkinter root window
        """
        self.root = root
        self.root.title("Basic Auditory Vision Simulator")
        self.root.geometry("1200x800")
        self.root.minsize(800, 600)
        
        # Set up components
        self.webcam = WebcamCapture()
        self.image_processor = RealTimeImageProcessor()
        self.audio_converter = RealTimeAudioConverter()
        
        # Application state
        self.running = False
        self.using_webcam = True
        self.current_image_path = None
        self.update_interval = 50  # ms
        
        # Create UI
        self.create_ui()
        
        # Bind window close event
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
    
    def create_ui(self):
        """Create the user interface"""
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create top control panel
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Input source controls
        source_frame = ttk.LabelFrame(control_frame, text="Input Source")
        source_frame.pack(side=tk.LEFT, padx=5, fill=tk.X)
        
        self.source_var = tk.StringVar(value="webcam")
        ttk.Radiobutton(source_frame, text="Webcam", variable=self.source_var, 
                        value="webcam", command=self.on_source_change).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(source_frame, text="Image File", variable=self.source_var, 
                        value="file", command=self.on_source_change).pack(side=tk.LEFT, padx=5)
        
        self.file_button = ttk.Button(source_frame, text="Select File", command=self.on_select_file)
        self.file_button.pack(side=tk.LEFT, padx=5)
        self.file_button.config(state=tk.DISABLED)
        
        # Camera selection
        self.camera_var = tk.StringVar(value="0")
        self.camera_combo = ttk.Combobox(source_frame, textvariable=self.camera_var, width=5)
        self.camera_combo.pack(side=tk.LEFT, padx=5)
        self.camera_combo['values'] = [str(i) for i in range(5)]  # Default to 5 possible cameras
        self.camera_combo.bind("<<ComboboxSelected>>", self.on_camera_change)
        
        # Start/Stop button
        self.start_button = ttk.Button(control_frame, text="Start", command=self.on_start_stop)
        self.start_button.pack(side=tk.RIGHT, padx=5)
        
        # Create main content area with visualization and controls
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Visualization area (left side)
        viz_frame = ttk.LabelFrame(content_frame, text="Visualization")
        viz_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        self.canvas = tk.Canvas(viz_frame, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Parameter controls (right side)
        param_frame = ttk.LabelFrame(content_frame, text="Parameters")
        param_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0), ipadx=10, ipady=10, width=300)
        
        # Image processing parameters
        img_param_frame = ttk.LabelFrame(param_frame, text="Image Processing")
        img_param_frame.pack(fill=tk.X, padx=5, pady=5, ipady=5)
        
        # Blur radius
        ttk.Label(img_param_frame, text="Blur Radius:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.blur_var = tk.DoubleVar(value=3.0)
        blur_scale = ttk.Scale(img_param_frame, from_=0.5, to=10.0, variable=self.blur_var, 
                              orient=tk.HORIZONTAL, command=self.on_image_param_change)
        blur_scale.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=2)
        ttk.Label(img_param_frame, textvariable=self.blur_var, width=5).grid(row=0, column=2, padx=5, pady=2)
        
        # Depth scale
        ttk.Label(img_param_frame, text="Depth Scale:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.depth_scale_var = tk.DoubleVar(value=1.0)
        depth_scale = ttk.Scale(img_param_frame, from_=0.1, to=3.0, variable=self.depth_scale_var, 
                               orient=tk.HORIZONTAL, command=self.on_image_param_change)
        depth_scale.grid(row=1, column=1, sticky=tk.EW, padx=5, pady=2)
        ttk.Label(img_param_frame, textvariable=self.depth_scale_var, width=5).grid(row=1, column=2, padx=5, pady=2)
        
        # Audio parameters
        audio_param_frame = ttk.LabelFrame(param_frame, text="Audio Conversion")
        audio_param_frame.pack(fill=tk.X, padx=5, pady=5, ipady=5)
        
        # Base frequency
        ttk.Label(audio_param_frame, text="Base Frequency:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.base_freq_var = tk.DoubleVar(value=440.0)
        base_freq_scale = ttk.Scale(audio_param_frame, from_=220.0, to=880.0, variable=self.base_freq_var, 
                                   orient=tk.HORIZONTAL, command=self.on_audio_param_change)
        base_freq_scale.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=2)
        ttk.Label(audio_param_frame, textvariable=self.base_freq_var, width=5).grid(row=0, column=2, padx=5, pady=2)
        
        # Depth factor (α)
        ttk.Label(audio_param_frame, text="Depth Factor (α):").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.depth_factor_var = tk.DoubleVar(value=500.0)
        depth_factor_scale = ttk.Scale(audio_param_frame, from_=100.0, to=1000.0, variable=self.depth_factor_var, 
                                      orient=tk.HORIZONTAL, command=self.on_audio_param_change)
        depth_factor_scale.grid(row=1, column=1, sticky=tk.EW, padx=5, pady=2)
        ttk.Label(audio_param_frame, textvariable=self.depth_factor_var, width=5).grid(row=1, column=2, padx=5, pady=2)
        
        # Angle factor (β)
        ttk.Label(audio_param_frame, text="Angle Factor (β):").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.angle_factor_var = tk.DoubleVar(value=300.0)
        angle_factor_scale = ttk.Scale(audio_param_frame, from_=100.0, to=800.0, variable=self.angle_factor_var, 
                                      orient=tk.HORIZONTAL, command=self.on_audio_param_change)
        angle_factor_scale.grid(row=2, column=1, sticky=tk.EW, padx=5, pady=2)
        ttk.Label(audio_param_frame, textvariable=self.angle_factor_var, width=5).grid(row=2, column=2, padx=5, pady=2)
        
        # Volume
        ttk.Label(audio_param_frame, text="Volume:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)
        self.volume_var = tk.DoubleVar(value=0.8)
        volume_scale = ttk.Scale(audio_param_frame, from_=0.0, to=1.0, variable=self.volume_var, 
                                orient=tk.HORIZONTAL, command=self.on_audio_param_change)
        volume_scale.grid(row=3, column=1, sticky=tk.EW, padx=5, pady=2)
        ttk.Label(audio_param_frame, textvariable=self.volume_var, width=5).grid(row=3, column=2, padx=5, pady=2)
        
        # Reverb
        ttk.Label(audio_param_frame, text="Reverb:").grid(row=4, column=0, sticky=tk.W, padx=5, pady=2)
        self.reverb_var = tk.DoubleVar(value=0.3)
        reverb_scale = ttk.Scale(audio_param_frame, from_=0.0, to=0.9, variable=self.reverb_var, 
                                orient=tk.HORIZONTAL, command=self.on_audio_param_change)
        reverb_scale.grid(row=4, column=1, sticky=tk.EW, padx=5, pady=2)
        ttk.Label(audio_param_frame, textvariable=self.reverb_var, width=5).grid(row=4, column=2, padx=5, pady=2)
        
        # Compression
        self.compression_var = tk.BooleanVar(value=True)
        compression_check = ttk.Checkbutton(audio_param_frame, text="Apply Compression", 
                                           variable=self.compression_var, command=self.on_audio_param_change)
        compression_check.grid(row=5, column=0, columnspan=3, sticky=tk.W, padx=5, pady=2)
        
        # Processing parameters
        proc_param_frame = ttk.LabelFrame(param_frame, text="Processing")
        proc_param_frame.pack(fill=tk.X, padx=5, pady=5, ipady=5)
        
        # Row step
        ttk.Label(proc_param_frame, text="Row Step:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.row_step_var = tk.IntVar(value=20)
        row_step_scale = ttk.Scale(proc_param_frame, from_=5, to=50, variable=self.row_step_var, 
                                  orient=tk.HORIZONTAL, command=lambda x: self.row_step_var.set(int(float(x))))
        row_step_scale.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=2)
        ttk.Label(proc_param_frame, textvariable=self.row_step_var, width=5).grid(row=0, column=2, padx=5, pady=2)
        
        # Column step
        ttk.Label(proc_param_frame, text="Column Step:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.col_step_var = tk.IntVar(value=10)
        col_step_scale = ttk.Scale(proc_param_frame, from_=5, to=30, variable=self.col_step_var, 
                                  orient=tk.HORIZONTAL, command=lambda x: self.col_step_var.set(int(float(x))))
        col_step_scale.grid(row=1, column=1, sticky=tk.EW, padx=5, pady=2)
        ttk.Label(proc_param_frame, textvariable=self.col_step_var, width=5).grid(row=1, column=2, padx=5, pady=2)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X, pady=(5, 0))
        
        # Configure grid weights
        for frame in [img_param_frame, audio_param_frame, proc_param_frame]:
            frame.columnconfigure(1, weight=1)
    
    def on_start_stop(self):
        """Handle start/stop button click"""
        if not self.running:
            self.start_simulator()
        else:
            self.stop_simulator()
    
    def start_simulator(self):
        """Start the simulator"""
        try:
            # Start image processor
            self.image_processor.start_processing()
            
            # Start audio converter
            self.audio_converter.start_audio()
            
            # Start webcam if using webcam
            if self.using_webcam:
                camera_index = int(self.camera_var.get())
                self.webcam = WebcamCapture(camera_index=camera_index)
                if not self.webcam.start_capture():
                    messagebox.showerror("Error", f"Could not open camera {camera_index}")
                    self.stop_simulator()
                    return
            elif self.current_image_path is None or not os.path.exists(self.current_image_path):
                messagebox.showerror("Error", "Please select an image file")
                self.stop_simulator()
                return
            
            # Update UI
            self.running = True
            self.start_button.config(text="Stop")
            self.status_var.set("Running")
            
            # Start update loop
            self.update_ui()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start simulator: {str(e)}")
            self.stop_simulator()
    
    def stop_simulator(self):
        """Stop the simulator"""
        # Stop components
        if hasattr(self, 'webcam') and self.webcam:
            self.webcam.stop_capture()
        
        if hasattr(self, 'image_processor'):
            self.image_processor.stop_processing()
        
        if hasattr(self, 'audio_converter'):
            self.audio_converter.stop_audio()
        
        # Update UI
        self.running = False
        self.start_button.config(text="Start")
        self.status_var.set("Stopped")
    
    def update_ui(self):
        """Update the UI with the latest data"""
        if not self.running:
            return
        
        try:
            # Get frame from webcam or image file
            if self.using_webcam:
                frame = self.webcam.get_frame()
                if frame is None:
                    # No frame available, try again later
                    self.root.after(self.update_interval, self.update_ui)
                    return
            else:
                # Load image file
                if self.current_image_path and os.path.exists(self.current_image_path):
                    frame = cv2.imread(self.current_image_path)
                    if frame is None:
                        messagebox.showerror("Error", f"Could not load image: {self.current_image_path}")
                        self.stop_simulator()
                        return
                else:
                    self.root.after(self.update_interval, self.update_ui)
                    return
            
            # Add frame to image processor
            self.image_processor.add_frame(frame)
            
            # Get processed data
            processed_data = self.image_processor.get_latest_processed()
            if processed_data:
                # Create visualization
                viz_width = self.canvas.winfo_width()
                viz_height = self.canvas.winfo_height()
                if viz_width > 1 and viz_height > 1:  # Ensure canvas has been drawn
                    viz_image = self.image_processor.create_visualization(viz_width, viz_height)
                    
                    # Convert to PhotoImage for Tkinter
                    self.photo_image = self.image_processor.create_tk_image(viz_image)
                    
                    # Display on canvas
                    self.canvas.create_image(0, 0, image=self.photo_image, anchor=tk.NW)
                    
                    # Get frame data for audio
                    frame_data = self.image_processor.get_frame_data(
                        row_step=self.row_step_var.get(),
                        col_step=self.col_step_var.get()
                    )
                    
                    # Send to audio converter
                    self.audio_converter.add_frame_data(frame_data)
                    
                    # Update status
                    fps = processed_data.get('fps', 0)
                    proc_time = processed_data.get('processing_time', 0) * 1000
                    self.status_var.set(f"Running | FPS: {fps:.1f} | Processing: {proc_time:.1f} ms")
            
            # Schedule next update
            self.root.after(self.update_interval, self.update_ui)
        except Exception as e:
            messagebox.showerror("Error", f"Error updating UI: {str(e)}")
            self.stop_simulator()
    
    def on_source_change(self):
        """Handle input source change"""
        source = self.source_var.get()
        if source == "webcam":
            self.using_webcam = True
            self.file_button.config(state=tk.DISABLED)
            self.camera_combo.config(state=tk.NORMAL)
        else:
            self.using_webcam = False
            self.file_button.config(state=tk.NORMAL)
            self.camera_combo.config(state=tk.DISABLED)
    
    def on_select_file(self):
        """Handle file selection"""
        file_path = filedialog.askopenfilename(
            title="Select Image File",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif"),
                ("All files", "*.*")
            ]
        )
        if file_path:
            self.current_image_path = file_path
            self.status_var.set(f"Selected file: {os.path.basename(file_path)}")
    
    def on_camera_change(self, event):
        """Handle camera selection change"""
        if self.running and self.using_webcam:
            # Restart with new camera
            self.stop_simulator()
            self.start_simulator()
    
    def on_image_param_change(self, *args):
        """Handle image processing parameter changes"""
        if hasattr(self, 'image_processor'):
            self.image_processor.set_parameters(
                blur_radius=self.blur_var.get(),
                depth_scale=self.depth_scale_var.get()
            )
    
    def on_audio_param_change(self, *args):
        """Handle audio parameter changes"""
        if hasattr(self, 'audio_converter'):
            self.audio_converter.set_parameters(
                base_freq=self.base_freq_var.get(),
                depth_factor=self.depth_factor_var.get(),
                angle_factor=self.angle_factor_var.get(),
                volume_factor=self.volume_var.get(),
                reverb=self.reverb_var.get(),
                compression=self.compression_var.get()
            )
    
    def on_close(self):
        """Handle window close event"""
        self.stop_simulator()
        self.root.destroy()


def main():
    """Main entry point"""
    root = tk.Tk()
    app = AuditoryVisionSimulator(root)
    root.mainloop()


if __name__ == "__main__":
    main()
