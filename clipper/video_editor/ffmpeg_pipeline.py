"""FFmpeg video processing pipeline."""

import subprocess
import shutil
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional

from ..caption_generator import Caption
from ..clip_selector import ClipCandidate
from ..config import Config
from .ass_generator import ASSGenerator
from ..utils import get_logger

logger = get_logger(__name__)


@dataclass
class RenderResult:
    """Result of video rendering."""
    output_path: Path
    duration: float
    success: bool
    error: Optional[str] = None


class FFmpegPipeline:
    """FFmpeg-based video processing pipeline."""
    
    def __init__(self, config: Config):
        self.config = config
        self.temp_dir = config.temp_dir
        self.ass_generator = ASSGenerator(
            width=config.target_width,
            height=config.target_height,
            style=config.caption_style
        )
        
        # Verify FFmpeg is available
        if not self._check_ffmpeg():
            raise RuntimeError("FFmpeg not found. Please install FFmpeg.")
    
    def _check_ffmpeg(self) -> bool:
        """Check if FFmpeg is available."""
        try:
            result = subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                timeout=10
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def render(
        self,
        video_path: Path,
        clip: ClipCandidate,
        captions: List[Caption],
        output_path: Path
    ) -> RenderResult:
        """
        Render a clip with burned-in captions.
        
        Pipeline:
        1. Extract clip segment
        2. Convert to 9:16 with blur background
        3. Burn subtitles
        
        Args:
            video_path: Source video path
            clip: Clip with start/end times
            captions: List of captions (times relative to clip start)
            output_path: Output video path
            
        Returns:
            RenderResult with output path and status
        """
        video_path = Path(video_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Rendering clip: {clip.start:.1f}s - {clip.end:.1f}s")
        
        try:
            # Generate ASS subtitle file
            ass_path = self.temp_dir / f"captions_{output_path.stem}.ass"
            self.ass_generator.generate(captions, ass_path)
            
            # Run combined FFmpeg command
            duration = clip.end - clip.start
            self._run_ffmpeg_render(
                video_path, clip.start, clip.end, ass_path, output_path
            )
            
            # Cleanup temp files
            if ass_path.exists():
                ass_path.unlink()
            
            logger.info(f"Rendered: {output_path}")
            
            return RenderResult(
                output_path=output_path,
                duration=duration,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Render failed: {e}")
            return RenderResult(
                output_path=output_path,
                duration=0,
                success=False,
                error=str(e)
            )
    
    def _run_ffmpeg_render(
        self,
        input_path: Path,
        start: float,
        end: float,
        ass_path: Path,
        output_path: Path
    ):
        """Run FFmpeg to render final video."""
        
        w = self.config.target_width
        h = self.config.target_height
        
        # Complex filter for:
        # 1. Create blurred background scaled to fill 9:16
        # 2. Scale foreground to fit within 9:16
        # 3. Overlay foreground on background
        # 4. Burn in ASS subtitles
        
        # Escape special characters in ASS path for FFmpeg
        ass_path_escaped = str(ass_path).replace("\\", "/").replace(":", "\\:")
        
        filter_complex = (
            # Create blurred background
            f"[0:v]scale={w}:{h}:force_original_aspect_ratio=increase,"
            f"crop={w}:{h},boxblur=20:5[bg];"
            # Scale foreground to fit
            f"[0:v]scale={w}:-2:force_original_aspect_ratio=decrease[fg];"
            # Overlay foreground centered on background
            f"[bg][fg]overlay=(W-w)/2:(H-h)/2,"
            # Burn subtitles
            f"ass='{ass_path_escaped}'[outv]"
        )
        
        cmd = [
            "ffmpeg",
            "-y",  # Overwrite output
            "-ss", str(start),  # Seek before input (faster)
            "-to", str(end),
            "-i", str(input_path),
            "-filter_complex", filter_complex,
            "-map", "[outv]",
            "-map", "0:a?",  # Map audio if exists
            "-c:v", "libx264",
            "-preset", self.config.ffmpeg_preset,
            "-crf", str(self.config.ffmpeg_crf),
            "-c:a", "aac",
            "-b:a", self.config.audio_bitrate,
            "-movflags", "+faststart",  # Enable fast start for web
            "-pix_fmt", "yuv420p",  # Compatibility
            str(output_path)
        ]
        
        logger.debug(f"FFmpeg command: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )
        
        if result.returncode != 0:
            logger.error(f"FFmpeg stderr: {result.stderr}")
            raise RuntimeError(f"FFmpeg failed: {result.stderr[-500:]}")
    
    def extract_clip_only(
        self,
        video_path: Path,
        start: float,
        end: float,
        output_path: Path
    ) -> Path:
        """Extract clip without processing (for debugging)."""
        cmd = [
            "ffmpeg",
            "-y",
            "-ss", str(start),
            "-to", str(end),
            "-i", str(video_path),
            "-c", "copy",
            str(output_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        if result.returncode != 0:
            raise RuntimeError(f"Clip extraction failed: {result.stderr}")
        
        return output_path
    
    def get_video_info(self, video_path: Path) -> dict:
        """Get video information using ffprobe."""
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            str(video_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode != 0:
            raise RuntimeError(f"ffprobe failed: {result.stderr}")
        
        import json
        return json.loads(result.stdout)
