"""Episode recording utilities for video and data."""
import cv2
import os
import logging

logger = logging.getLogger(__name__)


class EpisodeRecorder:
    """Handles recording of episodes to video and data files."""
    
    def __init__(self, record_dir, episode_num, fps=20.0):
        """
        Initialize episode recorder.
        
        Args:
            record_dir: Directory to save recordings
            episode_num: Episode number for filename
            fps: Frames per second for video
        """
        self.record_dir = record_dir
        self.episode_num = episode_num
        self.fps = fps
        self.writer = None
        self.video_path = None
        
    def start_recording(self, obs):
        """
        Start recording video for this episode.
        
        Args:
            obs: Initial observation with 'pov' key
            
        Returns:
            bool: True if recording started successfully
        """
        if obs is None or "pov" not in obs:
            return False
        
        try:
            h, w, _ = obs["pov"].shape
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.video_path = os.path.join(
                self.record_dir, 
                f"episode_{self.episode_num:03d}.mp4"
            )
            # OpenCV expects (width, height)
            self.writer = cv2.VideoWriter(self.video_path, fourcc, self.fps, (w, h))
            logger.info(f"Recording video to: {self.video_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to start recording: {e}")
            return False
    
    def write_frame(self, obs, step):
        """
        Write a frame to the video.
        
        Args:
            obs: Observation with 'pov' key
            step: Current step number (for logging)
        """
        if self.writer is not None and obs is not None and "pov" in obs:
            try:
                frame = obs["pov"]
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                self.writer.write(frame_bgr)
            except Exception as e:
                logger.warning(f"Could not write frame {step}: {e}")
    
    def stop_recording(self):
        """Stop recording and save the video."""
        if self.writer is not None:
            self.writer.release()
            logger.info(f"Video saved: episode_{self.episode_num:03d}.mp4")
            self.writer = None
