# ClipperVX

![ClipperVX Banner](banner.png)

AI-powered tool to automatically extract and transform long-form videos into viral short-form content (YouTube Shorts, TikTok, Reels).

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## ✨ Features

- 🎬 **YouTube Download** - Fetch videos directly from YouTube URLs with quality selection
- 📁 **Local File Support** - Process your own video files
- 🤖 **AI Clip Selection** - Uses Claude, Gemini, or OpenAI to identify the most engaging segments
- 📅 **Smart Scheduling** - Auto-upload clips to YouTube with randomized intervals and viral tags
- 📢 **Viral Metadata** - Automatically generates clickbait titles, descriptions, and hashtags
- 📝 **Auto Captions** - Generates word-by-word animated captions with custom fonts
- 🎨 **Vertical Format** - Automatically crops to 9:16 aspect ratio
- 🌐 **Web GUI** - Beautiful web interface with preview and history
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
| `gemini-3-flash` | **(Recommended)** Gemini 3 Flash with thinking |
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

## 📺 YouTube Automation

ClipperVX includes powerful YouTube integration features:

### 1. Google Cloud Setup (Required for YouTube)
Since ClipperVX is open-source and runs locally on your machine, you must create your own Google Cloud Project to interact with the YouTube API.

1. Go to [Google Cloud Console](https://console.cloud.google.com/).
2. Create a new project.
3. Enable **YouTube Data API v3**.
4. Go to **APIs & Services > Credentials** and create **OAuth 2.0 Client IDs**.
   - Application Type: *Web application*
   - Authorized Redirect URIs: `http://localhost:5000/api/auth/youtube/callback`
5. Download the JSON file, rename it to `client_secrets.json`, and place it in the `configs/` folder.
6. **Important**: Go to **OAuth consent screen** and add your Google email as a **Test User**. This allows you to use the app without verification.

### 2. Connect Account
1. Go to **Settings** in the Web GUI.
2. Click **Connect YouTube Account**.
3. Grant permissions to allow ClipperVX to upload videos to your channel.
   - You may see a "Google hasn't verified this app" warning. Click **Advanced > Go to ClipperVX (unsafe)** to proceed (this is safe because you created the app yourself).

### 3. Post Clips
- **Single Upload**: Click the "Post to YouTube" button on any generated clip card.
- **Batch Schedule**: Click **"Post All (Scheduled)"** to automatically upload ALL generated clips.
  - Clips are scheduled **1 hour apart**.
  - Includes randomized timing (+/- 15 mins) to appear natural.
  - Automatically includes:
    - AI-generated viral title
    - Engaging description
    - **20+ Trending Hashtags** for maximum reach

> **Note**: Scheduled videos are uploaded as `private` initially (YouTube requirement) and will go public at the scheduled time.

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
caption_font: "Luckiest Guy"
caption_fontsize: 96
caption_margin_v: 500
```

### Custom Fonts

Place `.ttf` or `.otf` font files in the `fonts/` directory. Each font should be in its own subdirectory:

```
fonts/
  ├── Komika_Axis/
  │   └── komika.ttf
  └── Luckiest_Guy/
      └── luckiest.ttf
```

These will automatically appear in the font dropdown in the Web GUI settings.

### Viral Metadata

After processing, click "View Details" on any generated clip to:
- Copy the AI-generated **Viral Title**
- Copy the engaging **Description**
- Copy optimized **Hashtags**

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
