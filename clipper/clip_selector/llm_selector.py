"""LLM-powered clip selection."""

import json
import re
from dataclasses import dataclass
from typing import List, Optional

from ..transcriber import Transcript
from ..config import Config
from ..utils import get_logger
from ..utils.validators import validate_timestamps, validate_no_overlap

logger = get_logger(__name__)


def _repair_json(json_str: str) -> str:
    """Attempt to repair truncated JSON by closing open brackets/braces."""
    # Count open brackets
    open_braces = json_str.count('{') - json_str.count('}')
    open_brackets = json_str.count('[') - json_str.count(']')
    
    # Try to find the last complete object in an array
    if open_braces > 0 or open_brackets > 0:
        # Find the last complete entry (ends with })
        last_complete = json_str.rfind('}')
        if last_complete > 0:
            # Check if there's an incomplete entry after it
            after_last = json_str[last_complete+1:].strip()
            if after_last.startswith(','):
                # Truncate at the last complete entry
                json_str = json_str[:last_complete+1]
                # Recalculate
                open_braces = json_str.count('{') - json_str.count('}')
                open_brackets = json_str.count('[') - json_str.count(']')
    
    # Close any remaining open structures
    json_str = json_str.rstrip()
    if json_str.endswith(','):
        json_str = json_str[:-1]
    
    json_str += ']' * open_brackets
    json_str += '}' * open_braces
    
    return json_str


@dataclass
class ClipCandidate:
    """A candidate clip identified by the selector."""
    start: float
    end: float
    hook: str
    virality_score: float
    reason: str
    title: str = ""
    description: str = ""
    hashtags: List[str] = None


CLIP_SELECTION_PROMPT = '''You are a viral content analyst. Given a video transcript with timestamps, identify the most engaging segments for short-form content (15-60 seconds).

TRANSCRIPT:
{transcript}

VIDEO METADATA:
- Title: {title}
- Duration: {duration:.1f}s

OUTPUT REQUIREMENTS:
1. Return ONLY valid JSON, no markdown code blocks
2. Each clip must have exact start/end timestamps that exist in the transcript
3. Timestamps must be floating point numbers in seconds
4. No overlapping clips
5. Each clip should be {min_length}-{max_length} seconds long

SELECTION CRITERIA:
- Strong hooks that grab attention in first 3 seconds
- Complete thoughts or stories (don't cut mid-sentence)
- Emotional peaks, revelations, or surprising moments
- Actionable tips or insights
- Controversial or debate-worthy statements

VIRAL METADATA GENERATION:
For each selected clip, generate:
1. A catchy, clickbait-style TITLE (max 50 chars)
2. A compelling DESCRIPTION (max 200 chars) that encourages watching
3. 3-5 relevant, high-traffic HASHTAGS

OUTPUT FORMAT:
{{
  "clips": [
    {{
      "start": <float>,
      "end": <float>,
      "hook": "<one-line hook for this clip>",
      "virality_score": <1-10>,
      "reason": "<why this segment is engaging>",
      "title": "<viral title>",
      "description": "<viral description>",
      "hashtags": ["#tag1", "#tag2", "#tag3"]
    }}
  ]
}}

Select up to {max_clips} clips. Return ONLY the JSON, no other text.'''


class LLMClipSelector:
    """LLM-powered clip selection."""
    
    def __init__(self, config: Config):
        self.config = config
        self.provider = config.llm_provider
        self._client = None
    
    def _get_client(self):
        """Initialize LLM client lazily."""
        if self._client is not None:
            return self._client
        
        if self.provider == "antigravity":
            from ..llm.antigravity_client import AntigravityClient
            self._client = AntigravityClient()
            if not self._client.is_authenticated():
                self._client.authenticate()
        elif self.provider == "gemini":
            from google import genai
            self._client = genai.Client(api_key=self.config.get_api_key())
        else:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.config.get_api_key())
        
        return self._client
    
    def select(
        self,
        transcript: Transcript,
        title: str,
        max_clips: int = 3,
        min_length: int = 15,
        max_length: int = 60
    ) -> List[ClipCandidate]:
        """
        Select clips using LLM.
        
        Args:
            transcript: Video transcript
            title: Video title
            max_clips: Maximum clips to select
            min_length: Minimum clip length in seconds
            max_length: Maximum clip length in seconds
            
        Returns:
            List of validated ClipCandidate objects
        """
        logger.info("Running LLM clip selection")
        
        # Format transcript for prompt
        transcript_text = self._format_transcript(transcript)
        
        # Build prompt
        prompt = CLIP_SELECTION_PROMPT.format(
            transcript=transcript_text,
            title=title,
            duration=transcript.duration,
            min_length=min_length,
            max_length=max_length,
            max_clips=max_clips
        )
        
        # Call LLM
        response_text = self._call_llm(prompt)
        
        # Parse response
        clips = self._parse_response(response_text)
        
        # Validate timestamps
        all_timestamps = transcript.get_all_timestamps()
        valid_clips, errors = validate_timestamps(
            [c.__dict__ for c in clips],
            transcript.duration,
            all_timestamps
        )
        
        for error in errors:
            logger.warning(f"Validation error: {error}")
        
        # Convert back to ClipCandidate and remove overlaps
        candidates = [ClipCandidate(**c) for c in valid_clips]
        candidates = validate_no_overlap([c.__dict__ for c in candidates])
        
        result = [ClipCandidate(**c) for c in candidates[:max_clips]]
        
        logger.info(f"LLM selected {len(result)} valid clips")
        
        return result
    
    def _format_transcript(self, transcript: Transcript, max_chars: int = 15000) -> str:
        """Format transcript for LLM prompt."""
        lines = []
        
        for segment in transcript.segments:
            line = f"[{segment.start:.1f}s - {segment.end:.1f}s] {segment.text}"
            lines.append(line)
        
        text = "\n".join(lines)
        
        # Truncate if too long
        if len(text) > max_chars:
            text = text[:max_chars] + "\n... [transcript truncated]"
        
        return text
    
    def _call_llm(self, prompt: str) -> str:
        """Call LLM and return response text."""
        from rich.console import Console
        from rich.panel import Panel
        from rich.syntax import Syntax
        
        console = Console()
        client = self._get_client()
        
        # Log the request
        console.print()
        console.print(Panel(
            f"[bold cyan]Model:[/] {self.config.llm_model}\n"
            f"[bold cyan]Provider:[/] {self.provider}",
            title="LLM Request - Clip Selection",
            border_style="cyan"
        ))
        console.print(Panel(prompt[:500] + "..." if len(prompt) > 500 else prompt, 
                           title="Prompt (truncated)", border_style="dim"))
        
        try:
            if self.provider == "antigravity":
                result = client.generate(
                    prompt=prompt,
                    model=self.config.llm_model,
                    temperature=0.7
                )
            elif self.provider == "gemini":
                from google import genai
                # Ensure JSON mode for gemini
                response = client.models.generate_content(
                    model=self.config.llm_model,
                    contents=prompt,
                    config=genai.types.GenerateContentConfig(
                        response_mime_type="application/json"
                    )
                )
                result = response.text
            else:
                response = client.chat.completions.create(
                    model=self.config.openai_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7
                )
                result = response.choices[0].message.content
            
            # Log the response
            console.print(Panel(
                Syntax(result[:1000] + "..." if len(result) > 1000 else result, 
                       "json", theme="monokai", word_wrap=True),
                title="LLM Response",
                border_style="green"
            ))
            console.print()
            
            # Log to file
            try:
                from datetime import datetime
                log_dir = self.config.output_dir / ".debug_logs"
                log_dir.mkdir(parents=True, exist_ok=True)
                log_file = log_dir / "llm_selector.log"
                with open(log_file, "a") as f:
                    f.write(f"\n{'='*50}\nTIMESTAMP: {datetime.now()}\nTYPE: CLIP_SELECTION\nPROMPT:\n{prompt}\n{'-'*20}\nRESPONSE:\n{result}\n{'='*50}\n")
            except Exception:
                pass
            
            return result
                
        except Exception as e:
            console.print(Panel(f"[red]{e}[/]", title="LLM Error", border_style="red"))
            logger.error(f"LLM call failed: {e}")
            raise
    
    def _parse_response(self, response: str) -> List[ClipCandidate]:
        """Parse LLM response into ClipCandidate objects."""
        # Clean response - remove markdown code blocks if present
        response = response.strip()
        response = re.sub(r'^```json\s*', '', response)
        response = re.sub(r'^```\s*', '', response)
        response = re.sub(r'\s*```$', '', response)
        
        # Try to find JSON object in response (for thinking models that include reasoning)
        json_match = re.search(r'\{[\s\S]*"clips"[\s\S]*\}', response)
        if json_match:
            response = json_match.group(0)
        
        try:
            data = json.loads(response)
        except json.JSONDecodeError as e:
            # Try to repair truncated JSON
            logger.warning(f"Initial JSON parse failed: {e}, attempting repair...")
            try:
                repaired = _repair_json(response)
                data = json.loads(repaired)
                logger.info("JSON repair successful")
            except json.JSONDecodeError as e2:
                logger.error(f"Failed to parse LLM response even after repair: {e2}")
                logger.debug(f"Response was: {response[:500]}")
                return []
        
        clips = []
        
        for clip_data in data.get("clips", []):
            try:
                # Handle hashtags list properly
                hashtags = clip_data.get("hashtags", [])
                if isinstance(hashtags, str):
                    hashtags = [t.strip() for t in hashtags.split(',')]
                
                clip = ClipCandidate(
                    start=float(clip_data["start"]),
                    end=float(clip_data["end"]),
                    hook=str(clip_data.get("hook", "")),
                    virality_score=float(clip_data.get("virality_score", 5)),
                    reason=str(clip_data.get("reason", "")),
                    title=str(clip_data.get("title", "")),
                    description=str(clip_data.get("description", "")),
                    hashtags=hashtags or []
                )
                clips.append(clip)
            except (KeyError, ValueError) as e:
                logger.warning(f"Invalid clip data: {e}")
                continue
        
        return clips
