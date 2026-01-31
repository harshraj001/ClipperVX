"""Heuristic-based clip selection."""

import re
from dataclasses import dataclass
from typing import List, Tuple

from ..transcriber import Transcript, Segment
from ..utils import get_logger

logger = get_logger(__name__)


@dataclass
class ClipCandidate:
    """A candidate clip identified by the selector."""
    start: float
    end: float
    hook: str
    virality_score: float
    reason: str


# Keywords that indicate engaging content
HOOK_KEYWORDS = [
    # Attention grabbers
    r"\b(here's|here is)\b.*\b(secret|trick|thing|how)\b",
    r"\b(watch this|look at this|check this out)\b",
    r"\b(you won't believe|unbelievable|incredible|amazing)\b",
    r"\b(the truth|truth is|actually|really)\b",
    r"\b(biggest mistake|common mistake|don't do this)\b",
    r"\b(game changer|life changing|changed my life)\b",
    r"\b(number one|top|best|worst)\b",
    r"\b(pro tip|tip|hack|secret)\b",
    r"\b(wait|hold on|stop|listen)\b",
    r"\b(this is why|that's why|the reason)\b",
    r"\b(never|always|must|have to)\b",
]

# Question patterns
QUESTION_PATTERNS = [
    r"\b(why|how|what|when|where|who)\b.*\?",
    r"\b(did you know|have you ever|can you believe)\b",
]

# Story/narrative indicators
STORY_PATTERNS = [
    r"\b(so|well|okay so|alright so)\b.*\b(happened|story|told|said)\b",
    r"\b(let me tell you|i'll tell you|i'm going to show)\b",
]


class HeuristicSelector:
    """Rule-based clip selection using heuristics."""
    
    def __init__(
        self,
        min_length: int = 15,
        max_length: int = 60,
        keyword_weight: float = 0.4,
        density_weight: float = 0.3,
        question_weight: float = 0.2,
        story_weight: float = 0.1
    ):
        self.min_length = min_length
        self.max_length = max_length
        self.keyword_weight = keyword_weight
        self.density_weight = density_weight
        self.question_weight = question_weight
        self.story_weight = story_weight
        
        # Compile patterns
        self.hook_patterns = [re.compile(p, re.IGNORECASE) for p in HOOK_KEYWORDS]
        self.question_patterns = [re.compile(p, re.IGNORECASE) for p in QUESTION_PATTERNS]
        self.story_patterns = [re.compile(p, re.IGNORECASE) for p in STORY_PATTERNS]
    
    def select(
        self,
        transcript: Transcript,
        max_clips: int = 3
    ) -> List[ClipCandidate]:
        """
        Select best clips using heuristics.
        
        Args:
            transcript: Transcript with segments
            max_clips: Maximum number of clips to return
            
        Returns:
            List of ClipCandidate objects
        """
        logger.info("Running heuristic clip selection")
        
        # Generate candidate windows
        candidates = self._generate_candidates(transcript)
        
        # Score each candidate
        scored = [(c, self._score_candidate(c, transcript)) for c in candidates]
        
        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)
        
        # Remove overlapping clips
        selected = self._remove_overlaps([s[0] for s in scored], max_clips)
        
        logger.info(f"Selected {len(selected)} clips")
        
        return selected
    
    def _generate_candidates(self, transcript: Transcript) -> List[Tuple[float, float, str]]:
        """Generate candidate time windows."""
        candidates = []
        
        if not transcript.segments:
            return candidates
        
        # Use sliding window approach
        window_sizes = [15, 30, 45, 60]
        step_size = 5  # seconds
        
        total_duration = transcript.duration
        
        for window_size in window_sizes:
            if window_size > total_duration:
                continue
            
            current = 0.0
            while current + window_size <= total_duration:
                end = current + window_size
                
                # Snap to segment boundaries if possible
                start_seg = self._find_nearest_segment_start(transcript, current)
                end_seg = self._find_nearest_segment_end(transcript, end)
                
                if end_seg - start_seg >= self.min_length:
                    text = transcript.get_text_in_range(start_seg, end_seg)
                    candidates.append((start_seg, end_seg, text))
                
                current += step_size
        
        return candidates
    
    def _find_nearest_segment_start(self, transcript: Transcript, time: float) -> float:
        """Find nearest segment start time."""
        nearest = time
        min_diff = float('inf')
        
        for segment in transcript.segments:
            diff = abs(segment.start - time)
            if diff < min_diff:
                min_diff = diff
                nearest = segment.start
        
        return nearest
    
    def _find_nearest_segment_end(self, transcript: Transcript, time: float) -> float:
        """Find nearest segment end time."""
        nearest = time
        min_diff = float('inf')
        
        for segment in transcript.segments:
            diff = abs(segment.end - time)
            if diff < min_diff:
                min_diff = diff
                nearest = segment.end
        
        return nearest
    
    def _score_candidate(
        self,
        candidate: Tuple[float, float, str],
        transcript: Transcript
    ) -> float:
        """Score a candidate clip."""
        start, end, text = candidate
        text_lower = text.lower()
        
        # Keyword score
        keyword_score = 0
        matched_keywords = []
        for pattern in self.hook_patterns:
            if pattern.search(text_lower):
                keyword_score += 1
                matched_keywords.append(pattern.pattern)
        keyword_score = min(1.0, keyword_score / 3)  # Normalize
        
        # Question score
        question_score = 0
        for pattern in self.question_patterns:
            if pattern.search(text_lower):
                question_score = 1.0
                break
        
        # Story score
        story_score = 0
        for pattern in self.story_patterns:
            if pattern.search(text_lower):
                story_score = 1.0
                break
        
        # Density score (words per second)
        duration = end - start
        word_count = len(text.split())
        words_per_second = word_count / duration if duration > 0 else 0
        # Optimal is ~2-3 words per second (not too fast, not too slow)
        density_score = 1.0 - abs(2.5 - words_per_second) / 2.5
        density_score = max(0, min(1, density_score))
        
        # Combined score
        total_score = (
            keyword_score * self.keyword_weight +
            density_score * self.density_weight +
            question_score * self.question_weight +
            story_score * self.story_weight
        )
        
        # Bonus for clips starting with strong hooks
        first_words = text[:100].lower()
        for pattern in self.hook_patterns[:5]:  # Top hook patterns
            if pattern.search(first_words):
                total_score *= 1.2
                break
        
        return total_score
    
    def _remove_overlaps(
        self,
        candidates: List[Tuple[float, float, str]],
        max_clips: int
    ) -> List[ClipCandidate]:
        """Remove overlapping candidates and convert to ClipCandidate."""
        selected = []
        
        for start, end, text in candidates:
            if len(selected) >= max_clips:
                break
            
            # Check for overlap with existing selections
            overlaps = False
            for existing in selected:
                if not (end <= existing.start or start >= existing.end):
                    overlaps = True
                    break
            
            if not overlaps:
                # Generate hook from first sentence
                hook = self._generate_hook(text)
                reason = self._generate_reason(text)
                score = self._score_candidate((start, end, text), None) if candidates else 5
                
                selected.append(ClipCandidate(
                    start=start,
                    end=end,
                    hook=hook,
                    virality_score=min(10, score * 10),
                    reason=reason
                ))
        
        return selected
    
    def _generate_hook(self, text: str) -> str:
        """Generate a hook from the text."""
        # Get first sentence or first 10 words
        sentences = re.split(r'[.!?]', text)
        first_sentence = sentences[0].strip() if sentences else text
        
        words = first_sentence.split()[:10]
        hook = " ".join(words)
        
        if len(hook) > 50:
            hook = hook[:47] + "..."
        
        return hook
    
    def _generate_reason(self, text: str) -> str:
        """Generate a reason for why this clip was selected."""
        text_lower = text.lower()
        reasons = []
        
        for pattern in self.hook_patterns[:3]:
            if pattern.search(text_lower):
                reasons.append("Contains attention-grabbing hook")
                break
        
        for pattern in self.question_patterns:
            if pattern.search(text_lower):
                reasons.append("Opens with engaging question")
                break
        
        for pattern in self.story_patterns:
            if pattern.search(text_lower):
                reasons.append("Contains narrative element")
                break
        
        return "; ".join(reasons) if reasons else "High speech density and flow"
