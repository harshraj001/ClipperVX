# ClipperVX

AI-powered tool to automatically extract and transform long-form videos into viral short-form content (YouTube Shorts, TikTok, Reels).

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## ✨ Features

- 🎬 **YouTube Download** - Fetch videos directly from YouTube URLs with quality selection
- 📁 **Local File Support** - Process your own video files
- 🤖 **AI Clip Selection** - Uses Claude, Gemini, or OpenAI to identify the most engaging segments
- 📝 **Auto Captions** - Generates word-by-word animated captions
- 🎨 **Vertical Format** - Automatically crops to 9:16 aspect ratio
- 🌐 **Web GUI** - Beautiful web interface with model selection
- ⚡ **CLI Support** - Command-line interface for automation
- 🚀 **Antigravity Mode** - Free access to Claude & Gemini models via Google OAuth

## 🤖 LLM Providers

ClipperVX supports three LLM providers for AI-powered clip selection:

### Antigravity (Recommended - Free)

Access Claude Sonnet 4.5, Claude Opus 4.5, and Gemini 3 models for free via Google OAuth authentication. No API keys required!

| Model | Description |
|-------|-------------|
| `claude-sonnet-4-5-thinking` | Claude Sonnet 4.5 with extended thinking |
| `claude-opus-4-5-thinking` | Claude Opus 4.5 with extended thinking |
| `gemini-3-flash` | Gemini 3 Flash with thinking |
| `gemini-3-pro-low` | Gemini 3 Pro Low |
| `gemini-3-pro-high` | Gemini 3 Pro High |

### Google Gemini

Requires a Gemini API key. Get one at [Google AI Studio](https://aistudio.google.com/).

| Model | Description |
|-------|-------------|
| `gemini-2.5-flash` | Fast Gemini model |
| `gemini-2.5-pro` | Pro Gemini model |
| `gemma-3-12b-it` | Gemma 3 12B Instruct |

### OpenAI

Requires an OpenAI API key. Get one at [OpenAI Platform](https://platform.openai.com/).

| Model | Description |
|-------|-------------|
| `gpt-4o-mini` | Fast GPT-4o |
| `gpt-4o` | Full GPT-4o |

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

For Antigravity (free, no API keys needed):
1. Open the web GUI
2. Go to Settings
3. Select "Antigravity (Claude/Gemini)" as provider
4. Click "Authenticate with Google"
5. Complete OAuth flow in browser

For API-based providers:
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
│   ├── llm/                # LLM providers (Antigravity, etc.)
│   └── web/                # Frontend templates & assets
├── configs/                # Configuration files
└── output/                 # Generated clips
```

## ⚙️ Configuration

Edit `configs/defaults.yaml` to customize:

```yaml
# LLM settings
llm_provider: "antigravity"  # antigravity, gemini, or openai
llm_model: "claude-sonnet-4-5-thinking"

# Clip settings
min_clip_length: 15
max_clip_length: 60

# Caption styling
caption_font: "Komika Axis"
caption_fontsize: 72
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
- [Claude](https://anthropic.com/) / [Google Gemini](https://ai.google.dev/) / [OpenAI](https://openai.com/) for AI clip selection
