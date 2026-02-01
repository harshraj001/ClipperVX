# ClipperVX

AI-powered tool to automatically extract and transform long-form videos into viral short-form content (YouTube Shorts, TikTok, Reels).

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## ✨ Features

- 🎬 **YouTube Download** - Fetch videos directly from YouTube URLs with quality selection
- 📁 **Local File Support** - Process your own video files
- 🤖 **AI Clip Selection** - Uses Gemini/OpenAI to identify the most engaging segments
- 📝 **Auto Captions** - Generates word-by-word animated captions
- 🎨 **Vertical Format** - Automatically crops to 9:16 aspect ratio
- 🌐 **Web GUI** - Beautiful web interface for easy use
- ⚡ **CLI Support** - Command-line interface for automation

## 📸 Screenshots

The web GUI provides an intuitive interface for processing videos:
- Paste YouTube URL or upload local files
- Configure clip count, length, and quality
- Real-time progress tracking
- Preview and download generated clips

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- FFmpeg
- yt-dlp

### Installation

```bash
# Clone the repository
git clone https://github.com/harshraj001/ClipperVX.git
cd ClipperVX

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .
```

### Configuration

Set your API keys (optional, enables AI clip selection):

```bash
export GEMINI_API_KEY="your-gemini-api-key"
# OR
export OPENAI_API_KEY="your-openai-api-key"
```

### Usage

#### Web GUI (Recommended)

```bash
python -m clipper.cli gui
```

Then open http://localhost:5000 in your browser.

#### Command Line

```bash
# Process a YouTube video
python -m clipper.cli process "https://youtube.com/watch?v=VIDEO_ID"

# Process with options
python -m clipper.cli process "URL" --clips 5 --min-length 15 --max-length 45

# Process a local file
python -m clipper.cli local video.mp4 --clips 3
```

## 🏗️ Architecture

```
ClipperVX/
├── clipper/
│   ├── cli.py              # Command-line interface
│   ├── web_server.py       # Flask web server
│   ├── orchestrator.py     # Main processing pipeline
│   ├── config.py           # Configuration management
│   ├── downloader/         # YouTube download module
│   ├── transcriber/        # Whisper & VTT parsing
│   ├── clip_selector/      # LLM & heuristic selection
│   ├── caption_generator/  # ASS subtitle generation
│   ├── video_editor/       # FFmpeg video processing
│   └── web/                # Frontend templates & assets
├── configs/                # Configuration files
└── output/                 # Generated clips
```

## ⚙️ Configuration

Edit `configs/defaults.yaml` to customize:

```yaml
# Clip settings
min_clip_length: 15
max_clip_length: 60

# LLM settings
llm_provider: "gemini"
llm_model: "gemini-2.5-flash"

# Caption styling
caption:
  font: "Komika Axis"
  size: 20
  primary_color: "&H00FFFFFF"
```

## 🔧 Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black clipper/
```

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- [yt-dlp](https://github.com/yt-dlp/yt-dlp) for video downloading
- [Whisper](https://github.com/openai/whisper) for transcription
- [FFmpeg](https://ffmpeg.org/) for video processing
- [Google Gemini](https://ai.google.dev/) / [OpenAI](https://openai.com/) for AI clip selection
