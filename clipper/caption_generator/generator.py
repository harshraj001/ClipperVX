"""Caption generation with LLM enhancement."""

import json
import re
from dataclasses import dataclass
from typing import List, Optional

from ..transcriber import Transcript, Segment
from ..clip_selector import ClipCandidate
from ..config import Config
from ..utils import get_logger

logger = get_logger(__name__)


@dataclass
class Caption:
    """A single caption entry."""
    start: float  # Relative to clip start
    end: float    # Relative to clip start
    text: str
    emphasis_words: List[str] = None
    
    def __post_init__(self):
        if self.emphasis_words is None:
            self.emphasis_words = []


CAPTION_PROMPT = '''You are a short-form video caption specialist. Convert the following transcript segment into punchy, hook-style captions optimized for sound-off viewing.

TRANSCRIPT SEGMENT (timestamps relative to clip start):
{segment_text}

RULES:
1. Maximum 6 words per line
2. Break at natural speech pauses
3. CAPITALIZE key emotional/action words (2-3 per caption max)
4. Each caption: 1.5-3 seconds display time
5. Maintain timing alignment with original transcript
6. Total captions should cover the entire segment

OUTPUT FORMAT (JSON only, no markdown):
{{
  "captions": [
    {{"start": 0.0, "end": 2.5, "text": "Here's the SECRET", "emphasis_words": ["SECRET"]}},
    {{"start": 2.5, "end": 5.0, "text": "nobody tells you", "emphasis_words": []}}
  ]
}}

Return ONLY the JSON, no other text.'''


class CaptionGenerator:
    """Generates captions for video clips."""
    
    def __init__(self, config: Config, use_llm: bool = True):
        self.config = config
        self.use_llm = use_llm
        self._client = None
    
    def generate(
        self,
        transcript: Transcript,
        clip: ClipCandidate
    ) -> List[Caption]:
        """
        Generate captions for a clip.
        
        Args:
            transcript: Full video transcript
            clip: Clip candidate with start/end times
            
        Returns:
            List of Caption objects with times relative to clip start
        """
        # Get segments within clip range
        segments = transcript.get_segments_in_range(clip.start, clip.end)
        
        if not segments:
            logger.warning(f"No segments found for clip {clip.start}-{clip.end}")
            return []
        
        if self.use_llm:
            try:
                return self._generate_llm(segments, clip)
            except Exception as e:
                logger.warning(f"LLM caption generation failed: {e}, falling back to basic")
                return self._generate_basic(segments, clip)
        else:
            return self._generate_basic(segments, clip)
    
    def _generate_basic(
        self,
        segments: List[Segment],
        clip: ClipCandidate
    ) -> List[Caption]:
        """Generate basic captions from transcript segments."""
        captions = []
        
        for segment in segments:
            # Adjust times to be relative to clip start
            rel_start = max(0, segment.start - clip.start)
            rel_end = segment.end - clip.start
            
            # Split long text into caption-sized chunks
            text = segment.text
            words = text.split()
            
            if len(words) <= 6:
                # Short enough for single caption
                emphasis = self._find_emphasis_words(text)
                captions.append(Caption(
                    start=rel_start,
                    end=rel_end,
                    text=self._apply_emphasis(text, emphasis),
                    emphasis_words=emphasis
                ))
            else:
                # Split into multiple captions
                chunk_size = 5
                duration = rel_end - rel_start
                time_per_word = duration / len(words) if words else 0
                
                for i in range(0, len(words), chunk_size):
                    chunk_words = words[i:i + chunk_size]
                    chunk_text = " ".join(chunk_words)
                    
                    chunk_start = rel_start + i * time_per_word
                    chunk_end = min(rel_end, chunk_start + len(chunk_words) * time_per_word)
                    
                    emphasis = self._find_emphasis_words(chunk_text)
                    captions.append(Caption(
                        start=chunk_start,
                        end=chunk_end,
                        text=self._apply_emphasis(chunk_text, emphasis),
                        emphasis_words=emphasis
                    ))
        
        return captions
    
    def _generate_llm(
        self,
        segments: List[Segment],
        clip: ClipCandidate
    ) -> List[Caption]:
        """Generate captions using LLM."""
        # Format segment text with relative timestamps
        segment_text = ""
        for segment in segments:
            rel_start = max(0, segment.start - clip.start)
            rel_end = segment.end - clip.start
            segment_text += f"[{rel_start:.1f}s - {rel_end:.1f}s] {segment.text}\n"
        
        prompt = CAPTION_PROMPT.format(segment_text=segment_text.strip())
        
        # Call LLM
        response = self._call_llm(prompt)
        
        # Parse response
        return self._parse_response(response)
    
    def _call_llm(self, prompt: str) -> str:
        """Call LLM."""
        if self._client is None:
            if self.config.llm_provider == "gemini":
                import google.generativeai as genai
                genai.configure(api_key=self.config.get_api_key())
                self._client = genai.GenerativeModel(self.config.llm_model)
            else:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.config.get_api_key())
        
        if self.config.llm_provider == "gemini":
            response = self._client.generate_content(prompt)
            return response.text
        else:
            response = self._client.chat.completions.create(
                model=self.config.openai_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5
            )
            return response.choices[0].message.content
    
    def _parse_response(self, response: str) -> List[Caption]:
        """Parse LLM response."""
        # Clean response
        response = response.strip()
        response = re.sub(r'^```json\s*', '', response)
        response = re.sub(r'^```\s*', '', response)
        response = re.sub(r'\s*```$', '', response)
        
        try:
            data = json.loads(response)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse caption response: {e}")
            return []
        
        captions = []
        for cap in data.get("captions", []):
            try:
                captions.append(Caption(
                    start=float(cap["start"]),
                    end=float(cap["end"]),
                    text=str(cap["text"]),
                    emphasis_words=cap.get("emphasis_words", [])
                ))
            except (KeyError, ValueError) as e:
                logger.warning(f"Invalid caption data: {e}")
                continue
        
        return captions
    
    def _find_emphasis_words(self, text: str) -> List[str]:
        """Find words that should be emphasized."""
        emphasis_triggers = [
            "secret", "amazing", "incredible", "never", "always",
            "best", "worst", "only", "must", "need", "stop",
            "wait", "look", "watch", "listen", "important",
            "truth", "real", "actually", "literally", "seriously",
            "game", "changer", "changed", "everything", "nothing"
        ]
        
        words = text.lower().split()
        emphasis = []
        
        for word in words:
            clean_word = re.sub(r'[^\w]', '', word)
            if clean_word in emphasis_triggers:
                # Find original casing
                for orig_word in text.split():
                    if re.sub(r'[^\w]', '', orig_word.lower()) == clean_word:
                        emphasis.append(orig_word)
                        break
        
        return emphasis[:3]  # Max 3 emphasis words
    
    def _apply_emphasis(self, text: str, emphasis_words: List[str]) -> str:
        """Apply uppercase emphasis to words."""
        for word in emphasis_words:
            # Replace with uppercase version
            pattern = re.compile(re.escape(word), re.IGNORECASE)
            text = pattern.sub(word.upper(), text, count=1)
        
        return text
