"""yt-dlp wrapper for downloading YouTube videos."""

import subprocess
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Callable

from ..utils import get_logger, validate_youtube_url, extract_video_id

logger = get_logger(__name__)


@dataclass
class DownloadResult:
    """Result of a video download."""
    video_path: Path
    subtitle_path: Optional[Path]
    video_id: str
    title: str
    duration: float
    description: str
    thumbnail_url: Optional[str] = None


class DownloadError(Exception):
    """Custom exception for download errors."""
    pass


class YouTubeDownloader:
    """Downloads YouTube videos using yt-dlp."""
    
    def __init__(self, temp_dir: Path):
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
    
    def download(
        self,
        url: str,
        quality: str = "best",
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> DownloadResult:
        """
        Download video with specified quality and optional subtitles.
        
        Args:
            url: YouTube URL
            quality: Quality format ID or 'best' (default)
            progress_callback: Optional callback for progress updates
            
        Returns:
            DownloadResult with paths and metadata
            
        Raises:
            DownloadError: If download fails
        """
        if not validate_youtube_url(url):
            raise DownloadError(f"Invalid YouTube URL: {url}")
        
        video_id = extract_video_id(url)
        if not video_id:
            raise DownloadError(f"Could not extract video ID from URL: {url}")
        
        logger.info(f"Downloading video: {video_id} (quality: {quality})")
        
        # First, get video metadata
        metadata = self._get_metadata(url)
        
        # Check for issues
        self._check_downloadability(metadata)
        
        # Download video
        video_path = self._download_video(url, video_id, quality)
        
        # Try to download subtitles
        subtitle_path = self._download_subtitles(url, video_id)
        
        return DownloadResult(
            video_path=video_path,
            subtitle_path=subtitle_path,
            video_id=video_id,
            title=metadata.get("title", ""),
            duration=metadata.get("duration", 0),
            description=metadata.get("description", ""),
            thumbnail_url=metadata.get("thumbnail")
        )
    
    def _get_metadata(self, url: str) -> dict:
        """Extract video metadata without downloading."""
        cmd = [
            "yt-dlp",
            "--dump-json",
            "--no-download",
            url
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode != 0:
                raise DownloadError(f"Failed to get metadata: {result.stderr}")
            
            return json.loads(result.stdout)
            
        except subprocess.TimeoutExpired:
            raise DownloadError("Timeout while fetching metadata")
        except json.JSONDecodeError:
            raise DownloadError("Failed to parse metadata JSON")
    
    def _check_downloadability(self, metadata: dict):
        """Check if video can be downloaded."""
        # Check if live
        if metadata.get("is_live"):
            raise DownloadError("Cannot download live streams")
        
        # Check duration (max 2 hours for processing)
        duration = metadata.get("duration", 0)
        if duration > 7200:
            logger.warning(f"Video is {duration/60:.1f} minutes long. Processing may take a while.")
        
        # Check availability
        availability = metadata.get("availability")
        if availability in ["private", "premium_only"]:
            raise DownloadError(f"Video is {availability}")
        
        # Check age-gating
        if metadata.get("age_limit", 0) > 0:
            logger.warning("Video is age-restricted. May require authentication.")
    
    def _download_video(self, url: str, video_id: str, quality: str = "best") -> Path:
        """Download video file with specified quality."""
        output_path = self.temp_dir / f"{video_id}.mp4"
        
        # Build format string based on quality
        if quality == "best" or not quality:
            format_str = "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best"
        else:
            # Use specific format ID with fallback
            format_str = f"{quality}+bestaudio[ext=m4a]/{quality}/bestvideo[height<={quality}]+bestaudio/best"
            # If quality looks like a height (e.g., "720", "1080")
            if quality.isdigit():
                format_str = f"bestvideo[height<={quality}][ext=mp4]+bestaudio[ext=m4a]/bestvideo[height<={quality}]+bestaudio/best[height<={quality}]/best"
        
        cmd = [
            "yt-dlp",
            "-f", format_str,
            "--merge-output-format", "mp4",
            "-o", str(output_path),
            "--no-playlist",
            url
        ]
        
        logger.info(f"Downloading video with format: {format_str}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1800  # 30 minutes max
            )
            
            if result.returncode != 0:
                raise DownloadError(f"Download failed: {result.stderr}")
            
            if not output_path.exists():
                raise DownloadError("Video file not created")
            
            logger.info(f"Video downloaded: {output_path}")
            return output_path
            
        except subprocess.TimeoutExpired:
            raise DownloadError("Download timed out")
    
    def _download_subtitles(self, url: str, video_id: str) -> Optional[Path]:
        """Download subtitles if available."""
        output_template = str(self.temp_dir / video_id)
        
        cmd = [
            "yt-dlp",
            "--write-auto-sub",
            "--write-sub",
            "--sub-lang", "en",
            "--sub-format", "vtt",
            "--skip-download",
            "-o", output_template,
            url
        ]
        
        logger.info("Checking for subtitles...")
        
        try:
            subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120
            )
            
            # Look for subtitle file
            subtitle_patterns = [
                self.temp_dir / f"{video_id}.en.vtt",
                self.temp_dir / f"{video_id}.en-orig.vtt",
            ]
            
            for pattern in subtitle_patterns:
                if pattern.exists():
                    logger.info(f"Subtitles found: {pattern}")
                    return pattern
            
            # Check for any .vtt file
            vtt_files = list(self.temp_dir.glob(f"{video_id}*.vtt"))
            if vtt_files:
                logger.info(f"Subtitles found: {vtt_files[0]}")
                return vtt_files[0]
            
            logger.warning("No subtitles available, will use Whisper")
            return None
            
        except subprocess.TimeoutExpired:
            logger.warning("Subtitle download timed out")
            return None
