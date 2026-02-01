"""ASS subtitle file generator."""

from pathlib import Path
from typing import List
from dataclasses import dataclass

from ..caption_generator import Caption
from ..config import CaptionStyle


@dataclass
class ASSStyle:
    """ASS subtitle style configuration."""
    name: str = "Default"
    fontname: str = "Komika Axis"
    fontsize: int = 72
    primary_color: str = "&H00FFFFFF"  # White (AABBGGRR format)
    secondary_color: str = "&H000000FF"  # Red
    outline_color: str = "&H00000000"  # Black
    back_color: str = "&H80000000"  # Semi-transparent black
    bold: int = 1
    italic: int = 0
    underline: int = 0
    strikeout: int = 0
    scale_x: int = 100
    scale_y: int = 100
    spacing: int = 0
    angle: int = 0
    border_style: int = 1  # 1 = outline + shadow
    outline: int = 4
    shadow: int = 2
    alignment: int = 2  # 2 = bottom center
    margin_l: int = 50
    margin_r: int = 50
    margin_v: int = 400
    encoding: int = 1


class ASSGenerator:
    """Generates ASS subtitle files for FFmpeg burning."""
    
    def __init__(
        self,
        width: int = 1080,
        height: int = 1920,
        style: CaptionStyle = None
    ):
        self.width = width
        self.height = height
        
        # Build ASS style from config
        self.style = ASSStyle()
        if style:
            self.style.fontname = style.font.replace("-", " ")
            self.style.fontsize = style.fontsize
            self.style.primary_color = style.color
            self.style.outline_color = style.outline_color
            self.style.outline = style.outline_width
            self.style.shadow = style.shadow_depth
            self.style.margin_v = style.margin_v
    
    def generate(self, captions: List[Caption], output_path: Path) -> Path:
        """
        Generate ASS file from captions.
        
        Args:
            captions: List of Caption objects
            output_path: Path to save ASS file
            
        Returns:
            Path to generated ASS file
        """
        output_path = Path(output_path)
        
        content = self._build_header()
        content += self._build_styles()
        content += self._build_events(captions)
        
        output_path.write_text(content, encoding='utf-8-sig')  # BOM for compatibility
        
        return output_path
    
    def _build_header(self) -> str:
        """Build ASS script info header."""
        return f"""[Script Info]
Title: ClipperVX Subtitles
ScriptType: v4.00+
WrapStyle: 0
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.709
PlayResX: {self.width}
PlayResY: {self.height}

"""
    
    def _build_styles(self) -> str:
        """Build ASS styles section."""
        s = self.style
        
        style_line = (
            f"Style: {s.name},{s.fontname},{s.fontsize},"
            f"{s.primary_color},{s.secondary_color},{s.outline_color},{s.back_color},"
            f"{s.bold},{s.italic},{s.underline},{s.strikeout},"
            f"{s.scale_x},{s.scale_y},{s.spacing},{s.angle},"
            f"{s.border_style},{s.outline},{s.shadow},{s.alignment},"
            f"{s.margin_l},{s.margin_r},{s.margin_v},{s.encoding}"
        )
        
        # Create emphasis style (for highlighted words)
        emphasis_style = ASSStyle(
            name="Emphasis",
            fontname=s.fontname,
            fontsize=int(s.fontsize * 1.1),  # Slightly larger
            primary_color="&H0000FFFF",  # Yellow
            outline_color=s.outline_color,
            outline=s.outline + 1,
            shadow=s.shadow,
            margin_v=s.margin_v,
            bold=1
        )
        e = emphasis_style
        
        emphasis_line = (
            f"Style: {e.name},{e.fontname},{e.fontsize},"
            f"{e.primary_color},{e.secondary_color},{e.outline_color},{e.back_color},"
            f"{e.bold},{e.italic},{e.underline},{e.strikeout},"
            f"{e.scale_x},{e.scale_y},{e.spacing},{e.angle},"
            f"{e.border_style},{e.outline},{e.shadow},{e.alignment},"
            f"{e.margin_l},{e.margin_r},{e.margin_v},{e.encoding}"
        )
        
        return f"""[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
{style_line}
{emphasis_line}

"""
    
    def _build_events(self, captions: List[Caption]) -> str:
        """Build ASS events/dialogue section."""
        lines = ["[Events]"]
        lines.append("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text")
        
        for caption in captions:
            start_time = self._seconds_to_ass_time(caption.start)
            end_time = self._seconds_to_ass_time(caption.end)
            
            # Apply emphasis formatting
            text = self._format_text_with_emphasis(caption.text, caption.emphasis_words)
            
            # Create dialogue line
            dialogue = f"Dialogue: 0,{start_time},{end_time},Default,,0,0,0,,{text}"
            lines.append(dialogue)
        
        return "\n".join(lines) + "\n"
    
    def _seconds_to_ass_time(self, seconds: float) -> str:
        """Convert seconds to ASS timestamp format (H:MM:SS.CC)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        centiseconds = int((secs % 1) * 100)
        whole_secs = int(secs)
        
        return f"{hours}:{minutes:02d}:{whole_secs:02d}.{centiseconds:02d}"
    
    def _format_text_with_emphasis(self, text: str, emphasis_words: List[str]) -> str:
        """Format text with ASS emphasis tags."""
        if not emphasis_words:
            return text
        
        for word in emphasis_words:
            # Use ASS override tags for emphasis
            # {\fs80} = larger font, {\c&H0000FFFF&} = yellow color
            emphasis_tag = r"{\fs80\c&H0000FFFF&}"
            reset_tag = r"{\r}"
            
            # Replace word with emphasized version
            # Handle both uppercase and original case
            for variant in [word, word.upper()]:
                if variant in text:
                    text = text.replace(
                        variant,
                        f"{emphasis_tag}{variant}{reset_tag}",
                        1  # Only first occurrence
                    )
                    break
        
        return text
