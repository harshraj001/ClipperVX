"""Input validation utilities."""

import re
from typing import List, Tuple


YOUTUBE_URL_PATTERN = re.compile(
    r'^(https?://)?(www\.)?(youtube\.com/watch\?v=|youtu\.be/|youtube\.com/shorts/)[\w-]{11}'
)


def validate_youtube_url(url: str) -> bool:
    """Validate YouTube URL format."""
    return bool(YOUTUBE_URL_PATTERN.match(url))


def extract_video_id(url: str) -> str | None:
    """Extract video ID from YouTube URL."""
    patterns = [
        r'youtube\.com/watch\?v=([\w-]{11})',
        r'youtu\.be/([\w-]{11})',
        r'youtube\.com/shorts/([\w-]{11})',
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def validate_timestamps(
    clips: List[dict],
    video_duration: float,
    transcript_timestamps: List[float]
) -> Tuple[List[dict], List[str]]:
    """
    Validate clip timestamps against video duration and transcript.
    
    Returns:
        Tuple of (valid_clips, error_messages)
    """
    valid_clips = []
    errors = []
    
    for i, clip in enumerate(clips):
        start = clip.get("start", 0)
        end = clip.get("end", 0)
        
        # Check basic validity
        if start < 0:
            errors.append(f"Clip {i+1}: start time {start} is negative")
            continue
        
        if end > video_duration:
            errors.append(f"Clip {i+1}: end time {end} exceeds video duration {video_duration}")
            continue
        
        if start >= end:
            errors.append(f"Clip {i+1}: start time {start} >= end time {end}")
            continue
        
        # Check if timestamps exist in transcript (within 1 second tolerance)
        start_valid = any(abs(t - start) < 1.0 for t in transcript_timestamps)
        end_valid = any(abs(t - end) < 1.0 for t in transcript_timestamps)
        
        if not start_valid:
            errors.append(f"Clip {i+1}: start time {start} not found in transcript")
            continue
        
        if not end_valid:
            errors.append(f"Clip {i+1}: end time {end} not found in transcript")
            continue
        
        valid_clips.append(clip)
    
    return valid_clips, errors


def validate_no_overlap(clips: List[dict]) -> List[dict]:
    """Remove overlapping clips, keeping higher scored ones."""
    if not clips:
        return []
    
    # Sort by virality score descending
    sorted_clips = sorted(clips, key=lambda x: x.get("virality_score", 0), reverse=True)
    
    non_overlapping = []
    for clip in sorted_clips:
        start, end = clip["start"], clip["end"]
        
        # Check if overlaps with any existing clip
        overlaps = False
        for existing in non_overlapping:
            ex_start, ex_end = existing["start"], existing["end"]
            if not (end <= ex_start or start >= ex_end):
                overlaps = True
                break
        
        if not overlaps:
            non_overlapping.append(clip)
    
    # Re-sort by start time
    return sorted(non_overlapping, key=lambda x: x["start"])
