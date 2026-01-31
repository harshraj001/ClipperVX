"""VTT subtitle parser with word-level timestamps."""

import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional

import webvtt

from ..utils import get_logger

logger = get_logger(__name__)


@dataclass
class Word:
    """A single word with timestamp."""
    word: str
    start: float
    end: float


@dataclass
class Segment:
    """A transcript segment with word-level timestamps."""
    start: float
    end: float
    text: str
    words: List[Word] = field(default_factory=list)


@dataclass
class Transcript:
    """Complete transcript with segments."""
    segments: List[Segment] = field(default_factory=list)
    duration: float = 0.0
    
    def get_text_in_range(self, start: float, end: float) -> str:
        """Get all text within a time range."""
        texts = []
        for segment in self.segments:
            if segment.end > start and segment.start < end:
                texts.append(segment.text)
        return " ".join(texts)
    
    def get_segments_in_range(self, start: float, end: float) -> List[Segment]:
        """Get all segments within a time range."""
        return [s for s in self.segments if s.end > start and s.start < end]
    
    def get_all_timestamps(self) -> List[float]:
        """Get all unique timestamps from segments."""
        timestamps = set()
        for segment in self.segments:
            timestamps.add(segment.start)
            timestamps.add(segment.end)
            for word in segment.words:
                timestamps.add(word.start)
                timestamps.add(word.end)
        return sorted(timestamps)
    
    def to_dict(self) -> dict:
        """Convert to dictionary format."""
        return {
            "segments": [
                {
                    "start": s.start,
                    "end": s.end,
                    "text": s.text,
                    "words": [
                        {"word": w.word, "start": w.start, "end": w.end}
                        for w in s.words
                    ]
                }
                for s in self.segments
            ],
            "duration": self.duration
        }


class VTTParser:
    """Parser for YouTube VTT subtitle files."""
    
    def __init__(self):
        # Pattern to match VTT timestamp format
        self.time_pattern = re.compile(
            r'<(\d{2}):(\d{2}):(\d{2})\.(\d{3})>'
        )
    
    def parse(self, vtt_path: Path) -> Transcript:
        """
        Parse VTT file into Transcript with word-level timestamps.
        
        Args:
            vtt_path: Path to .vtt file
            
        Returns:
            Transcript object with segments
        """
        vtt_path = Path(vtt_path)
        if not vtt_path.exists():
            raise FileNotFoundError(f"VTT file not found: {vtt_path}")
        
        logger.info(f"Parsing VTT file: {vtt_path}")
        
        segments = []
        max_end = 0.0
        
        try:
            for caption in webvtt.read(str(vtt_path)):
                start = self._timestamp_to_seconds(caption.start)
                end = self._timestamp_to_seconds(caption.end)
                text = caption.text.strip()
                
                if not text:
                    continue
                
                # Clean up text (remove duplicate lines from YouTube auto-subs)
                text = self._clean_text(text)
                
                # Try to extract word-level timestamps from text
                words = self._extract_words(text, start, end)
                
                # Clean text by removing timing tags
                clean_text = self._remove_timing_tags(text)
                
                if clean_text:
                    segment = Segment(
                        start=start,
                        end=end,
                        text=clean_text,
                        words=words
                    )
                    segments.append(segment)
                    max_end = max(max_end, end)
        
        except Exception as e:
            logger.error(f"Error parsing VTT: {e}")
            raise
        
        # Merge overlapping/duplicate segments
        segments = self._merge_segments(segments)
        
        logger.info(f"Parsed {len(segments)} segments")
        
        return Transcript(segments=segments, duration=max_end)
    
    def _timestamp_to_seconds(self, timestamp: str) -> float:
        """Convert VTT timestamp to seconds."""
        parts = timestamp.replace(',', '.').split(':')
        if len(parts) == 3:
            hours, minutes, seconds = parts
            return float(hours) * 3600 + float(minutes) * 60 + float(seconds)
        elif len(parts) == 2:
            minutes, seconds = parts
            return float(minutes) * 60 + float(seconds)
        else:
            return float(parts[0])
    
    def _clean_text(self, text: str) -> str:
        """Clean text from common VTT artifacts."""
        # Remove newlines
        text = text.replace('\n', ' ')
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def _remove_timing_tags(self, text: str) -> str:
        """Remove inline timing tags from text."""
        # Remove <HH:MM:SS.mmm> style tags
        text = re.sub(r'<\d{2}:\d{2}:\d{2}\.\d{3}>', '', text)
        # Remove <c> style tags
        text = re.sub(r'</?c[^>]*>', '', text)
        return text.strip()
    
    def _extract_words(self, text: str, segment_start: float, segment_end: float) -> List[Word]:
        """Extract words with timestamps from text containing timing tags."""
        words = []
        
        # Check if text has inline timestamps
        matches = list(self.time_pattern.finditer(text))
        
        if matches:
            # Text has inline timestamps
            last_pos = 0
            last_time = segment_start
            
            for match in matches:
                # Get text before this timestamp
                before_text = text[last_pos:match.start()]
                before_text = self._remove_timing_tags(before_text).strip()
                
                if before_text:
                    # Parse timestamp
                    h, m, s, ms = match.groups()
                    current_time = int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000
                    
                    # Create word entries for text before timestamp
                    word_list = before_text.split()
                    if word_list:
                        time_per_word = (current_time - last_time) / len(word_list)
                        for i, word in enumerate(word_list):
                            word_start = last_time + i * time_per_word
                            word_end = word_start + time_per_word
                            words.append(Word(word=word, start=word_start, end=word_end))
                    
                    last_time = current_time
                
                last_pos = match.end()
            
            # Handle remaining text after last timestamp
            remaining = text[last_pos:]
            remaining = self._remove_timing_tags(remaining).strip()
            if remaining:
                word_list = remaining.split()
                if word_list:
                    time_per_word = (segment_end - last_time) / len(word_list)
                    for i, word in enumerate(word_list):
                        word_start = last_time + i * time_per_word
                        word_end = word_start + time_per_word
                        words.append(Word(word=word, start=word_start, end=word_end))
        
        else:
            # No inline timestamps, distribute evenly
            clean_text = self._remove_timing_tags(text)
            word_list = clean_text.split()
            if word_list:
                duration = segment_end - segment_start
                time_per_word = duration / len(word_list)
                for i, word in enumerate(word_list):
                    word_start = segment_start + i * time_per_word
                    word_end = word_start + time_per_word
                    words.append(Word(word=word, start=word_start, end=word_end))
        
        return words
    
    def _merge_segments(self, segments: List[Segment]) -> List[Segment]:
        """Merge overlapping segments (common in YouTube auto-captions)."""
        if not segments:
            return []
        
        # Sort by start time
        segments = sorted(segments, key=lambda s: s.start)
        
        merged = [segments[0]]
        
        for segment in segments[1:]:
            last = merged[-1]
            
            # If this segment overlaps significantly with the last one
            if segment.start < last.end and segment.text == last.text:
                # Skip duplicate
                continue
            elif segment.start < last.end - 0.1:
                # Overlapping but different text - adjust timing
                segment.start = last.end
                if segment.start < segment.end:
                    merged.append(segment)
            else:
                merged.append(segment)
        
        return merged
