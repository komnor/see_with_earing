import numpy as np
import pyaudio
import threading
import queue
import time
import scipy.signal as signal

class RealTimeAudioConverter:
    """
    Class for converting visual data to audio in real-time
    using the formula f = f0 + α⋅D + β⋅θ
    """
    def __init__(self, sample_rate=44100, buffer_size=1024, channels=2):
        """
        Initialize the audio converter with customizable parameters
        
        Parameters:
        - sample_rate (int): Audio sample rate in Hz
        - buffer_size (int): Size of audio buffer in samples
        - channels (int): Number of audio channels (1=mono, 2=stereo)
        """
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.channels = channels
        
        # Audio formula parameters
        self.base_freq = 440  # Base frequency f0 in Hz (A4 note)
        self.depth_factor = 500  # α parameter, controls how depth affects frequency
        self.angle_factor = 300  # β parameter, controls how horizontal angle affects frequency
        self.volume_factor = 0.8  # Maximum volume scaling factor (0-1)
        
        # Audio effects parameters
        self.reverb = 0.3  # Reverb amount (0-1)
        self.compression = True  # Whether to apply compression
        
        # Audio processing flags
        self.is_playing = False
        self.audio_thread = None
        self.frame_data_queue = queue.Queue(maxsize=2)
        
        # PyAudio objects
        self.p = None
        self.stream = None
        
        # Audio buffer
        self.audio_buffer = np.zeros((self.buffer_size, self.channels))
        self.buffer_position = 0
        
        # Statistics
        self.processing_time = 0
        self.buffer_underruns = 0
        self.last_frame_time = 0
    
    def start_audio(self):
        """Start audio playback"""
        if self.is_playing:
            return False
        
        try:
            # Initialize PyAudio
            self.p = pyaudio.PyAudio()
            
            # Open audio stream
            self.stream = self.p.open(
                format=pyaudio.paFloat32,
                channels=self.channels,
                rate=self.sample_rate,
                output=True,
                frames_per_buffer=self.buffer_size,
                stream_callback=self._audio_callback
            )
            
            # Start audio thread
            self.is_playing = True
            self.audio_thread = threading.Thread(target=self._audio_processing_loop)
            self.audio_thread.daemon = True
            self.audio_thread.start()
            
            return True
        except Exception as e:
            print(f"Error starting audio: {str(e)}")
            self.stop_audio()
            return False
    
    def stop_audio(self):
        """Stop audio playback"""
        self.is_playing = False
        
        if self.audio_thread:
            self.audio_thread.join(timeout=1.0)
            self.audio_thread = None
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        
        if self.p:
            self.p.terminate()
            self.p = None
        
        return True
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """
        Callback function for PyAudio stream
        
        Returns:
        - audio_data (bytes): Audio data to play
        - flag (int): PyAudio status flag
        """
        # Convert audio buffer to bytes
        audio_data = self.audio_buffer.astype(np.float32).tobytes()
        
        # Check for buffer underruns
        if status == pyaudio.paOutputUnderflow:
            self.buffer_underruns += 1
        
        return (audio_data, pyaudio.paContinue)
    
    def _audio_processing_loop(self):
        """Background thread for continuous audio processing"""
        while self.is_playing:
            try:
                # Get the latest frame data from the queue, non-blocking
                frame_data = self.frame_data_queue.get(block=False)
                
                # Process the frame data to audio
                start_time = time.time()
                self._process_frame_data(frame_data)
                self.processing_time = time.time() - start_time
                
                # Update last frame time
                self.last_frame_time = time.time()
                
                # Mark task as done
                self.frame_data_queue.task_done()
            
            except queue.Empty:
                # No frame data in queue, generate silence
                self._generate_silence()
                time.sleep(0.01)
            except Exception as e:
                print(f"Error in audio processing loop: {str(e)}")
                time.sleep(0.1)  # Sleep to avoid tight loop on error
    
    def _generate_silence(self):
        """Generate silence in the audio buffer"""
        # Gradually fade out current buffer to avoid clicks
        fade_factor = 0.95
        self.audio_buffer *= fade_factor
    
    def _process_frame_data(self, frame_data):
        """
        Process frame data to generate audio
        
        Parameters:
        - frame_data (list): List of lists containing pixel data
        """
        # Reset audio buffer
        self.audio_buffer = np.zeros((self.buffer_size, self.channels))
        
        # Process each row of pixel data
        buffer_position = 0
        samples_per_pixel = max(1, self.buffer_size // (len(frame_data) * len(frame_data[0])))
        
        for row_data in frame_data:
            for pixel_data in row_data:
                # Generate tone for this pixel
                tone = self._generate_tone(pixel_data, samples_per_pixel)
                
                # Add to buffer at current position
                end_pos = min(buffer_position + samples_per_pixel, self.buffer_size)
                samples_to_add = end_pos - buffer_position
                
                if samples_to_add > 0 and buffer_position < self.buffer_size:
                    self.audio_buffer[buffer_position:end_pos] += tone[:samples_to_add]
                
                # Move buffer position
                buffer_position += samples_per_pixel
                
                # Stop if buffer is full
                if buffer_position >= self.buffer_size:
                    break
            
            # Stop if buffer is full
            if buffer_position >= self.buffer_size:
                break
        
        # Apply audio effects
        self._apply_audio_effects()
        
        # Normalize to prevent clipping
        max_val = np.max(np.abs(self.audio_buffer))
        if max_val > 1.0:
            self.audio_buffer /= max_val
    
    def _generate_tone(self, pixel_data, num_samples):
        """
        Generate a tone based on pixel data
        
        Parameters:
        - pixel_data (dict): Dictionary containing 'intensity', 'depth', 'angle', and 'vert_pos'
        - num_samples (int): Number of samples to generate
        
        Returns:
        - tone (numpy.ndarray): Generated tone samples
        """
        # Extract pixel data
        intensity = pixel_data['intensity']  # 0-1
        depth = pixel_data['depth']  # 0-1
        angle = pixel_data['angle']  # -1 to 1
        vert_pos = pixel_data['vert_pos']  # 0-1
        
        # Calculate frequency using the formula: f = f0 + α⋅D + β⋅θ
        freq = self.base_freq + (self.depth_factor * depth) + (self.angle_factor * angle)
        
        # Ensure frequency is in audible range (20Hz - 20kHz)
        freq = np.clip(freq, 20, 20000)
        
        # Calculate volume based on intensity
        volume = intensity * self.volume_factor
        
        # Generate time array for the tone
        t = np.linspace(0, num_samples / self.sample_rate, num_samples, False)
        
        # Generate sine wave
        tone = np.sin(freq * 2 * np.pi * t) * volume
        
        # Apply stereo panning based on angle
        # -1 (left) to 1 (right)
        if angle < 0:
            # More on the left
            left_vol = 1.0
            right_vol = 1.0 + angle  # Reduces right volume
        else:
            # More on the left
            left_vol = 1.0 - angle  # Reduces left volume
            right_vol = 1.0
        
        # Create stereo tone
        stereo_tone = np.zeros((num_samples, 2))
        stereo_tone[:, 0] = tone * left_vol  # Left channel
        stereo_tone[:, 1] = tone * right_vol  # Right channel
        
        return stereo_tone
    
    def _apply_audio_effects(self):
        """Apply audio effects to the audio buffer"""
        # Apply simple reverb effect (simplified)
        if self.reverb > 0:
            reverb_buffer = np.copy(self.audio_buffer)
            delay_samples = int(0.05 * self.sample_rate)  # 50ms delay
            
            # Create a delayed version of the buffer
            if delay_samples < self.buffer_size:
                delayed_buffer = np.zeros_like(self.audio_buffer)
                delayed_buffer[delay_samples:] = reverb_buffer[:-delay_samples] * 0.6
                
                # Mix original with reverb
                self.audio_buffer = self.audio_buffer * (1 - self.reverb) + delayed_buffer * self.reverb
        
        # Apply simple compression (simplified)
        if self.compression:
            # Calculate RMS
            rms = np.sqrt(np.mean(self.audio_buffer**2))
            
            # Apply soft knee compression
            threshold = 0.5
            ratio = 4.0
            makeup_gain = 1.2
            
            # Simple compression algorithm
            mask = np.abs(self.audio_buffer) > threshold
            self.audio_buffer[mask] = threshold + (np.abs(self.audio_buffer[mask]) - threshold) / ratio * np.sign(self.audio_buffer[mask])
            
            # Apply makeup gain
            self.audio_buffer = self.audio_buffer * makeup_gain
            
            # Clip to prevent distortion
            self.audio_buffer = np.clip(self.audio_buffer, -1.0, 1.0)
    
    def add_frame_data(self, frame_data):
        """
        Add frame data to the processing queue
        
        Parameters:
        - frame_data (list): List of lists containing pixel data
        
        Returns:
        - success (bool): Whether the frame data was added to the queue
        """
        if not self.is_playing:
            return False
        
        try:
            # Try to add to queue without blocking
            self.frame_data_queue.put(frame_data, block=False)
            return True
        except queue.Full:
            # Queue is full, skip this frame
            return False
    
    def set_parameters(self, base_freq=None, depth_factor=None, angle_factor=None, 
                      volume_factor=None, reverb=None, compression=None):
        """
        Update audio parameters
        
        Parameters:
        - base_freq (float): Base frequency f0 in Hz
        - depth_factor (float): α parameter, controls how depth affects frequency
        - angle_factor (float): β parameter, controls how horizontal angle affects frequency
        - volume_factor (float): Maximum volume scaling factor (0-1)
        - reverb (float): Reverb amount (0-1)
        - compression (bool): Whether to apply compression
        """
        if base_freq is not None:
            self.base_freq = base_freq
        if depth_factor is not None:
            self.depth_factor = depth_factor
        if angle_factor is not None:
            self.angle_factor = angle_factor
        if volume_factor is not None:
            self.volume_factor = volume_factor
        if reverb is not None:
            self.reverb = reverb
        if compression is not None:
            self.compression = compression
    
    def get_status(self):
        """
        Get current status of the audio converter
        
        Returns:
        - status (dict): Dictionary with status information
        """
        return {
            'is_playing': self.is_playing,
            'processing_time': self.processing_time,
            'buffer_underruns': self.buffer_underruns,
            'last_frame_time': self.last_frame_time,
            'parameters': {
                'base_freq': self.base_freq,
                'depth_factor': self.depth_factor,
                'angle_factor': self.angle_factor,
                'volume_factor': self.volume_factor,
                'reverb': self.reverb,
                'compression': self.compression
            }
        }
