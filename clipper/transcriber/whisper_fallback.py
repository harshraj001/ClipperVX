"""Whisper transcription fallback."""

from pathlib import Path
from typing import Optional

from ..utils import get_logger
from .vtt_parser import Transcript, Segment, Word

logger = get_logger(__name__)


class WhisperTranscriber:
    """Transcribe audio using Whisper when subtitles are unavailable."""
    
    def __init__(self, model_size: str = "base", device: str = "auto"):
        """
        Initialize Whisper transcriber.
        
        Args:
            model_size: Whisper model size (tiny, base, small, medium, large)
            device: Device to use (auto, cpu, cuda)
        """
        self.model_size = model_size
        self.device = device
        self._model = None
    
    def _load_model(self):
        """Lazy load the Whisper model."""
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
                
                logger.info(f"Loading Whisper model: {self.model_size}")
                
                # Determine compute type based on device
                if self.device == "cuda":
                    compute_type = "float16"
                else:
                    compute_type = "int8"
                
                self._model = WhisperModel(
                    self.model_size,
                    device=self.device if self.device != "auto" else "auto",
                    compute_type=compute_type
                )
                
                logger.info("Whisper model loaded")
                
            except ImportError:
                raise ImportError(
                    "faster-whisper not installed. Install with: pip install faster-whisper"
                )
    
    def transcribe(
        self,
        audio_path: Path,
        language: Optional[str] = "en"
    ) -> Transcript:
        """
        Transcribe audio file.
        
        Args:
            audio_path: Path to audio/video file
            language: Language code (en, es, etc.)
            
        Returns:
            Transcript object with word-level timestamps
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        self._load_model()
        
        logger.info(f"Transcribing: {audio_path}")
        
        # Transcribe with word timestamps
        segments_generator, info = self._model.transcribe(
            str(audio_path),
            language=language,
            word_timestamps=True,
            vad_filter=True,  # Filter out non-speech
            vad_parameters=dict(
                min_silence_duration_ms=500
            )
        )
        
        logger.info(f"Detected language: {info.language} (probability: {info.language_probability:.2f})")
        
        segments = []
        max_end = 0.0
        
        for segment in segments_generator:
            words = []
            
            if segment.words:
                for word_info in segment.words:
                    words.append(Word(
                        word=word_info.word.strip(),
                        start=word_info.start,
                        end=word_info.end
                    ))
            
            segments.append(Segment(
                start=segment.start,
                end=segment.end,
                text=segment.text.strip(),
                words=words
            ))
            
            max_end = max(max_end, segment.end)
        
        logger.info(f"Transcribed {len(segments)} segments")
        
        return Transcript(
            segments=segments,
            duration=max_end
        )
