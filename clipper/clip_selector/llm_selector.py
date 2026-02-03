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


def _extract_json_object(text: str) -> str:
    """Extract the outermost balanced JSON object from text."""
    # Find the first opening brace
    start_idx = text.find('{')
    if start_idx == -1:
        return text
    
    # Track brace balance to find the matching closing brace
    brace_count = 0
    in_string = False
    escape_next = False
    
    for i, char in enumerate(text[start_idx:], start=start_idx):
        if escape_next:
            escape_next = False
            continue
        
        if char == '\\':
            escape_next = True
            continue
        
        if char == '"' and not escape_next:
            in_string = not in_string
            continue
        
        if in_string:
            continue
        
        if char == '{':
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0:
                return text[start_idx:i+1]
    
    # If we get here, the JSON is incomplete - return what we have
    return text[start_idx:]


def _repair_json(json_str: str) -> str:
    """Attempt to repair truncated or malformed JSON."""
    # First, extract the JSON object
    json_str = _extract_json_object(json_str)
    
    # Remove trailing content after what looks like a complete clips array
    # Find the clips array and try to close it properly
    clips_match = re.search(r'"clips"\s*:\s*\[', json_str)
    if clips_match:
        # Find all complete objects within the clips array
        clip_start = clips_match.end()
        
        # Track the last position of a complete clip object
        last_complete_pos = clip_start
        brace_depth = 0
        bracket_depth = 1  # We're inside the clips array
        in_string = False
        escape_next = False
        
        for i, char in enumerate(json_str[clip_start:], start=clip_start):
            if escape_next:
                escape_next = False
                continue
            if char == '\\':
                escape_next = True
                continue
            if char == '"' and not escape_next:
                in_string = not in_string
                continue
            if in_string:
                continue
            
            if char == '{':
                brace_depth += 1
            elif char == '}':
                brace_depth -= 1
                if brace_depth == 0 and bracket_depth == 1:
                    # We just closed a clip object
                    last_complete_pos = i + 1
            elif char == '[':
                bracket_depth += 1
            elif char == ']':
                bracket_depth -= 1
                if bracket_depth == 0:
                    # Clips array properly closed
                    last_complete_pos = i + 1
                    break
        
        # Truncate at the last complete position and close properly
        json_str = json_str[:last_complete_pos].rstrip()
        
        # Remove trailing comma if present
        if json_str.endswith(','):
            json_str = json_str[:-1]
    
    # Count and close remaining brackets/braces
    open_braces = json_str.count('{') - json_str.count('}')
    open_brackets = json_str.count('[') - json_str.count(']')
    
    json_str = json_str.rstrip()
    if json_str.endswith(','):
        json_str = json_str[:-1]
    
    json_str += ']' * max(0, open_brackets)
    json_str += '}' * max(0, open_braces)
    
    return json_str


def _extract_clips_fallback(text: str) -> list:
    """Last-resort fallback: extract individual clip objects using regex."""
    clips = []
    
    # Pattern to match individual clip objects
    clip_pattern = re.compile(
        r'\{\s*"start"\s*:\s*([0-9.]+)\s*,\s*"end"\s*:\s*([0-9.]+)[^}]*?"hook"\s*:\s*"([^"]*)"[^}]*?"virality_score"\s*:\s*([0-9]+)',
        re.DOTALL
    )
    
    for match in clip_pattern.finditer(text):
        try:
            clips.append({
                "start": float(match.group(1)),
                "end": float(match.group(2)),
                "hook": match.group(3),
                "virality_score": int(match.group(4)),
                "reason": "",
                "title": "",
                "description": "",
                "hashtags": []
            })
        except (ValueError, IndexError):
            continue
    
    return clips


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
3. 20 trending, high-traffic HASHTAGS related to the content/niche

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
                    temperature=0.7,
                    max_tokens=16384  # Large enough for 10 clips with full metadata
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
            response_len = len(result)
            console.print(Panel(
                Syntax(result[:1000] + "..." if response_len > 1000 else result, 
                       "json", theme="monokai", word_wrap=True),
                title=f"LLM Response ({response_len} chars)",
                border_style="green"
            ))
            console.print()
            
            # Log COMPLETE response to file for debugging
            try:
                from datetime import datetime
                log_dir = self.config.output_dir / ".debug_logs"
                log_dir.mkdir(parents=True, exist_ok=True)
                log_file = log_dir / "llm_selector.log"
                with open(log_file, "a") as f:
                    f.write(f"\n{'='*80}\n")
                    f.write(f"TIMESTAMP: {datetime.now()}\n")
                    f.write(f"TYPE: CLIP_SELECTION\n")
                    f.write(f"MODEL: {self.config.llm_model}\n")
                    f.write(f"RESPONSE_LENGTH: {response_len} chars\n")
                    f.write(f"{'-'*40}\nPROMPT:\n{prompt}\n")
                    f.write(f"{'-'*40}\nCOMPLETE RESPONSE:\n{result}\n")
                    f.write(f"{'='*80}\n")
            except Exception:
                pass
            
            return result
                
        except Exception as e:
            console.print(Panel(f"[red]{e}[/]", title="LLM Error", border_style="red"))
            logger.error(f"LLM call failed: {e}")
            raise
    
    def _parse_response(self, response: str) -> List[ClipCandidate]:
        """Parse LLM response into ClipCandidate objects."""
        original_response = response
        
        # Clean response - remove markdown code blocks if present
        response = response.strip()
        response = re.sub(r'^```json\s*', '', response)
        response = re.sub(r'^```\s*', '', response)
        response = re.sub(r'\s*```$', '', response)
        
        # Use the robust JSON extraction instead of greedy regex
        response = _extract_json_object(response)
        
        data = None
        
        # Try direct parse first
        try:
            data = json.loads(response)
        except json.JSONDecodeError as e:
            logger.warning(f"Initial JSON parse failed: {e}, attempting repair...")
            
            # Try repair
            try:
                repaired = _repair_json(response)
                data = json.loads(repaired)
                logger.info("JSON repair successful")
            except json.JSONDecodeError as e2:
                logger.warning(f"JSON repair failed: {e2}, trying fallback extraction...")
                
                # Last resort: regex-based extraction
                fallback_clips = _extract_clips_fallback(original_response)
                if fallback_clips:
                    logger.info(f"Fallback extraction found {len(fallback_clips)} clips")
                    data = {"clips": fallback_clips}
                else:
                    logger.error("All JSON parsing attempts failed")
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
