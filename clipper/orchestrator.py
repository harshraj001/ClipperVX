"""Main orchestrator for the ClipperVX pipeline."""

import json
import shutil
from pathlib import Path
from typing import List, Optional, Callable
from dataclasses import dataclass, field

from .config import Config
from .downloader import YouTubeDownloader, DownloadResult
from .transcriber import VTTParser, WhisperTranscriber, Transcript
from .clip_selector import HeuristicSelector, LLMClipSelector, ClipCandidate
from .caption_generator import CaptionGenerator
from .video_editor import FFmpegPipeline, RenderResult
from .utils import get_logger, setup_logging

logger = get_logger(__name__)


@dataclass
class ClipResult:
    """Result for a single processed clip."""
    output_path: Path
    start: float
    end: float
    duration: float
    hook: str
    virality_score: float
    title: str = ""
    description: str = ""
    hashtags: List[str] = field(default_factory=list)


@dataclass
class ProcessResult:
    """Result of processing a YouTube URL."""
    source_url: str
    source_title: str
    video_id: str
    clips: List[ClipResult] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "source_url": self.source_url,
            "source_title": self.source_title,
            "video_id": self.video_id,
            "clips": [
                {
                    "filename": c.output_path.name,
                    "start": c.start,
                    "end": c.end,
                    "duration": c.duration,
                    "hook": c.hook,
                    "virality_score": c.virality_score,
                    "title": c.title,
                    "description": c.description,
                    "hashtags": c.hashtags,
                    "relative_path": f"{self.video_id}/{c.output_path.name}"
                }
                for c in self.clips
            ],
            "errors": self.errors
        }


class ClipperOrchestrator:
    """Main orchestrator for the video processing pipeline."""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config.load_default()
        
        # Initialize components
        self.downloader = YouTubeDownloader(self.config.temp_dir)
        self.vtt_parser = VTTParser()
        self.whisper = WhisperTranscriber(model_size=self.config.whisper_model)
        self.heuristic_selector = HeuristicSelector(
            min_length=self.config.min_clip_length,
            max_length=self.config.max_clip_length
        )
        self.llm_selector = LLMClipSelector(self.config)
        self.caption_generator = CaptionGenerator(self.config)
        self.video_editor = FFmpegPipeline(self.config)
    
    def run(
        self,
        url: str,
        max_clips: int = 3,
        min_length: int = None,
        max_length: int = None,
        quality: str = "best",
        use_llm: bool = True,
        progress_callback: Optional[Callable[[str, float], None]] = None,
        stop_event: Optional[any] = None
    ) -> ProcessResult:
        """
        Process a YouTube URL and generate clips.
        
        Args:
            url: YouTube video URL
            max_clips: Maximum number of clips to generate
            min_length: Minimum clip length (seconds)
            max_length: Maximum clip length (seconds)
            quality: Video quality (format ID or 'best')
            use_llm: Whether to use LLM for clip selection
            progress_callback: Optional callback(stage, progress)
            stop_event: Optional threading.Event to check for cancellation
            
        Returns:
            ProcessResult with generated clips
        """
        min_length = min_length or self.config.min_clip_length
        max_length = max_length or self.config.max_clip_length
        
        result = ProcessResult(
            source_url=url,
            source_title="",
            video_id=""
        )
        
        def update_progress(stage: str, percent: int):
            """Update progress with direct percentage."""
            if stop_event and stop_event.is_set():
                raise InterruptedError("Job cancelled by user")
                
            if progress_callback:
                progress_callback(stage, percent / 100.0)
            logger.info(f"[{percent}%] {stage}")
        
        try:
            # Check stop before starting
            if stop_event and stop_event.is_set():
                raise InterruptedError("Job cancelled by user")

            # Step 1: Download video (0-15%)
            update_progress("Downloading video...", 5)
            download = self.downloader.download(url, quality=quality)
            result.source_title = download.title
            result.video_id = download.video_id
            update_progress("Download complete", 15)
            
            # Step 2: Transcribe (15-35%)
            update_progress("Transcribing audio...", 20)
            transcript = self._get_transcript(download)
            update_progress("Transcription complete", 35)
            
            # Step 3: Select clips (35-45%)
            update_progress("Analyzing for best clips...", 38)
            clips = self._select_clips(
                transcript, download.title, max_clips,
                min_length, max_length, use_llm
            )
            update_progress("Clips selected", 45)
            
            if not clips:
                result.errors.append("No suitable clips found")
                return result
            
            # Step 4: Process each clip (45-95%)
            output_dir = self.config.output_dir / download.video_id
            output_dir.mkdir(parents=True, exist_ok=True)
            
            clip_start = 45
            clip_range = 50  # 45% to 95%
            
            for i, clip in enumerate(clips):
                # Check stop before each clip
                if stop_event and stop_event.is_set():
                    raise InterruptedError("Job cancelled by user")
                    
                clip_pct = clip_start + int(clip_range * i / len(clips))
                update_progress(f"Rendering clip {i+1}/{len(clips)}...", clip_pct)
                
                try:
                    clip_result = self._process_clip(
                        download.video_path,
                        transcript,
                        clip,
                        output_dir,
                        i + 1
                    )
                    result.clips.append(clip_result)
                    done_pct = clip_start + int(clip_range * (i + 1) / len(clips))
                    update_progress(f"Clip {i+1}/{len(clips)} done", done_pct)
                except Exception as e:
                    error_msg = f"Failed to process clip {i+1}: {e}"
                    logger.error(error_msg)
                    result.errors.append(error_msg)
            
            # Complete (100%)
            update_progress("Processing complete!", 100)
            
            # Save metadata
            self._save_metadata(result, output_dir)
            
            # Cleanup
            self._cleanup(download)
            
        except InterruptedError as e:
            logger.info(f"Cancellation caught in orchestrator: {e}")
            raise e
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            result.errors.append(str(e))
        
        return result
    
    def _get_transcript(self, download: DownloadResult) -> Transcript:
        """Get transcript from subtitles or Whisper."""
        if download.subtitle_path and download.subtitle_path.exists():
            logger.info("Using YouTube captions")
            return self.vtt_parser.parse(download.subtitle_path)
        else:
            logger.info("Using Whisper transcription")
            return self.whisper.transcribe(download.video_path)
    
    def _select_clips(
        self,
        transcript: Transcript,
        title: str,
        max_clips: int,
        min_length: int,
        max_length: int,
        use_llm: bool
    ) -> List[ClipCandidate]:
        """Select best clips from transcript."""
        clips = []
        if use_llm:
            try:
                clips = self.llm_selector.select(
                    transcript, title, max_clips, min_length, max_length
                )
            except Exception as e:
                logger.warning(f"LLM selection failed: {e}, using heuristics")
        
        if not clips:
            # Fallback to heuristic selection
            self.heuristic_selector.min_length = min_length
            self.heuristic_selector.max_length = max_length
            clips = self.heuristic_selector.select(transcript, max_clips)
            
        # Log metadata check
        if clips:
            c = clips[0]
            logger.info(f"Clip metadata check - Title: '{c.title}', Desc: '{c.description[:20]}...', Tags: {c.hashtags}")
            
        return clips
    
    def _process_clip(
        self,
        video_path: Path,
        transcript: Transcript,
        clip: ClipCandidate,
        output_dir: Path,
        clip_number: int
    ) -> ClipResult:
        """Process a single clip."""
        # Generate captions
        captions = self.caption_generator.generate(transcript, clip)
        
        # Render video
        output_path = output_dir / f"clip_{clip_number:03d}.mp4"
        render_result = self.video_editor.render(
            video_path, clip, captions, output_path
        )
        
        if not render_result.success:
            raise RuntimeError(render_result.error)
        
        return ClipResult(
            output_path=render_result.output_path,
            start=clip.start,
            end=clip.end,
            duration=clip.end - clip.start,
            hook=clip.hook,
            virality_score=clip.virality_score,
            title=getattr(clip, 'title', ''),
            description=getattr(clip, 'description', ''),
            hashtags=getattr(clip, 'hashtags', [])
        )
    
    def _save_metadata(self, result: ProcessResult, output_dir: Path):
        """Save metadata JSON file."""
        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        logger.info(f"Metadata saved: {metadata_path}")
    
    def _cleanup(self, download: DownloadResult):
        """Clean up temporary files."""
        try:
            if download.video_path.exists():
                download.video_path.unlink()
            if download.subtitle_path and download.subtitle_path.exists():
                download.subtitle_path.unlink()
        except Exception as e:
            logger.warning(f"Cleanup failed: {e}")
    
    def clean_temp(self):
        """Clean all temporary files."""
        if self.config.temp_dir.exists():
            shutil.rmtree(self.config.temp_dir)
            self.config.temp_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Temp directory cleaned")
    
    def run_from_file(
        self,
        video_path: Path,
        max_clips: int = 3,
        min_length: int = None,
        max_length: int = None,
        use_llm: bool = True,
        progress_callback: Optional[Callable[[str, float], None]] = None,
        stop_event: Optional[any] = None
    ) -> ProcessResult:
        """
        Process a local video file and generate clips.
        
        Args:
            video_path: Path to local video file
            max_clips: Maximum number of clips to generate
            min_length: Minimum clip length (seconds)
            max_length: Maximum clip length (seconds)
            use_llm: Whether to use LLM for clip selection
            progress_callback: Optional callback(stage, progress)
            stop_event: Optional threading.Event
            
        Returns:
            ProcessResult with generated clips
        """
        video_path = Path(video_path)
        min_length = min_length or self.config.min_clip_length
        max_length = max_length or self.config.max_clip_length
        
        video_id = video_path.stem.replace(' ', '_')[:20]
        
        result = ProcessResult(
            source_url=str(video_path),
            source_title=video_path.name,
            video_id=video_id
        )
        
        def update_progress(stage: str, progress: float = 0):
            if progress_callback:
                progress_callback(stage, progress)
            logger.info(f"[{stage}] {progress:.0%}" if progress else f"[{stage}]")
        
        try:
            # Step 1: Transcribe using Whisper (no subtitles for local files)
            update_progress("Transcribing audio", 0.1)
            transcript = self.whisper.transcribe(video_path)
            update_progress("Transcribing audio", 0.3)
            
            # Step 2: Select clips
            update_progress("Selecting clips", 0.35)
            clips = self._select_clips(
                transcript, video_path.name, max_clips,
                min_length, max_length, use_llm
            )
            update_progress("Selecting clips", 0.5)
            
            if not clips:
                result.errors.append("No suitable clips found")
                return result
            
            # Step 3: Process each clip
            output_dir = self.config.output_dir / video_id
            output_dir.mkdir(parents=True, exist_ok=True)
            
            for i, clip in enumerate(clips):
                # Progress during rendering: 50-95%
                progress = 0.5 + (0.45 * (i / len(clips)))
                update_progress(f"Rendering clip {i+1}/{len(clips)}", progress)
                
                try:
                    clip_result = self._process_clip(
                        video_path,
                        transcript,
                        clip,
                        output_dir,
                        i + 1
                    )
                    result.clips.append(clip_result)
                    # Update after clip completes
                    progress = 0.5 + (0.45 * ((i + 1) / len(clips)))
                    update_progress(f"Rendered clip {i+1}/{len(clips)}", progress)
                except Exception as e:
                    error_msg = f"Failed to process clip {i+1}: {e}"
                    logger.error(error_msg)
                    result.errors.append(error_msg)
            
            update_progress("Complete!", 1.0)
            
            # Save metadata
            self._save_metadata(result, output_dir)
            
        except InterruptedError as e:
            logger.info(f"Cancellation caught in run_from_file: {e}")
            raise e
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            result.errors.append(str(e))
        
        return result
