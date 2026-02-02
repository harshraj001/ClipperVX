"""Flask web server for ClipperVX GUI."""

import os
import json
import time
import uuid
import shutil
import threading
import signal
import subprocess
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory, Response, session, redirect, url_for
from werkzeug.utils import secure_filename
import datetime
import random

from .config import Config, AVAILABLE_MODELS, AVAILABLE_FONTS
from .orchestrator import ClipperOrchestrator
from .downloader import YouTubeDownloader
from .uploader import YouTubeUploader
from .utils import get_logger, validate_youtube_url, extract_video_id

logger = get_logger(__name__)

# Initialize Flask app
app = Flask(__name__, 
    template_folder='web/templates',
    static_folder='web/static'
)
app.config['SECRET_KEY'] = os.urandom(24)
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024 * 1024  # 2GB max upload

# Paths
JOBS_FILE = Path.home() / ".clippervx" / "jobs.json"
FONTS_DIR = Path(__file__).parent.parent / "fonts"

# Allow OAuth over HTTP for local testing
os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'

# Global state
processing_jobs = {}
stop_flags = {}  # job_id -> threading.Event for cancellation


def get_video_formats(url: str) -> list:
    """Get available video formats for quality selection."""
    import subprocess
    
    try:
        result = subprocess.run(
            ["yt-dlp", "-j", "--no-download", url],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode != 0:
            return []
        
        data = json.loads(result.stdout)
        formats = []
        seen = set()
        
        for fmt in data.get("formats", []):
            height = fmt.get("height")
            if height and height not in seen and fmt.get("vcodec") != "none":
                seen.add(height)
                formats.append({
                    "format_id": fmt.get("format_id"),
                    "resolution": f"{height}p",
                    "height": height,
                    "ext": fmt.get("ext", "mp4"),
                    "filesize": fmt.get("filesize_approx", 0)
                })
        
        formats.sort(key=lambda x: x["height"], reverse=True)
        return formats
        
    except Exception as e:
        logger.error(f"Failed to get formats: {e}")
        return []


def load_jobs() -> dict:
    """Load jobs from persistent storage."""
    if JOBS_FILE.exists():
        try:
            return json.loads(JOBS_FILE.read_text())
        except Exception as e:
            logger.error(f"Failed to load jobs: {e}")
    return {}


def save_jobs(jobs: dict):
    """Save jobs to persistent storage."""
    try:
        JOBS_FILE.parent.mkdir(parents=True, exist_ok=True)
        JOBS_FILE.write_text(json.dumps(jobs, indent=2))
    except Exception as e:
        logger.error(f"Failed to save jobs: {e}")


# Load existing jobs on startup
processing_jobs.update(load_jobs())


@app.route('/')
def index():
    """Main page."""
    return render_template('index.html')


@app.route('/api/models')
def get_models():
    """Get available models by provider."""
    provider = request.args.get('provider', 'gemini')
    models = AVAILABLE_MODELS.get(provider, {})
    return jsonify({
        "provider": provider,
        "models": [{"id": k, "name": v} for k, v in models.items()]
    })


@app.route('/api/providers')
def get_providers():
    """Get available LLM providers."""
    return jsonify({
        "providers": [
            {"id": "antigravity", "name": "Antigravity (Claude/Gemini)", "requires_key": False},
            {"id": "gemini", "name": "Google Gemini", "requires_key": True},
            {"id": "openai", "name": "OpenAI", "requires_key": True}
        ]
    })


@app.route('/api/fonts')
def get_fonts():
    """Get available fonts with file paths."""
    fonts = []
    for name, rel_path in AVAILABLE_FONTS.items():
        font_path = FONTS_DIR / rel_path
        fonts.append({
            "name": name,
            "path": f"/fonts/{rel_path}",  # Web accessible path
            "file": rel_path.split("/")[-1],
            "available": font_path.exists()
        })
    return jsonify({"fonts": fonts})


@app.route('/fonts/<path:filename>')
def serve_font(filename):
    """Serve font files."""
    return send_from_directory(FONTS_DIR, filename)


@app.route('/api/jobs')
def get_all_jobs():
    """Get all jobs for history."""
    return jsonify({"jobs": list(processing_jobs.values())})


@app.route('/api/jobs', methods=['DELETE'])
def clear_jobs():
    """Clear all job history."""
    processing_jobs.clear()
    save_jobs({})
    return jsonify({"status": "cleared"})


@app.route('/api/job/<job_id>/stop', methods=['POST'])
def stop_job(job_id):
    """Stop a running job."""
    if job_id not in processing_jobs:
        return jsonify({"error": "Job not found"}), 404
    
    job = processing_jobs[job_id]
    if job.get("status") != "processing":
        return jsonify({"error": "Job is not running"}), 400
    
    # Set stop flag
    if job_id in stop_flags:
        stop_flags[job_id].set()
    
    # Update job status
    job["status"] = "stopped"
    job["error"] = "Cancelled by user"
    save_jobs(processing_jobs)
    
    return jsonify({"status": "stopped"})


@app.route('/api/auth/antigravity', methods=['POST'])
def auth_antigravity():
    """Authenticate with Antigravity (Google OAuth)."""
    try:
        from .llm.antigravity_client import AntigravityClient
        client = AntigravityClient()
        
        if client.is_authenticated():
            return jsonify({"status": "already_authenticated"})
        
        success = client.authenticate(timeout=120)
        if success:
            return jsonify({"status": "authenticated"})
        else:
            return jsonify({"error": "Authentication failed"}), 401
    except Exception as e:
        logger.error(f"Antigravity auth error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/auth/antigravity/status')
def antigravity_status():
    """Check Antigravity authentication status."""
    try:
        from .llm.antigravity_client import AntigravityClient
        client = AntigravityClient()
        return jsonify({"authenticated": client.is_authenticated()})
    except Exception as e:
        return jsonify({"authenticated": False, "error": str(e)})


@app.route('/api/auth/youtube/url')
def youtube_auth_url():
    """Get YouTube OAuth URL."""
    try:
        config = Config.load_default()
        uploader = YouTubeUploader(config.config_dir)
        
        callback_url = url_for('youtube_callback', _external=True)
        # Fix: Google doesn't accept 0.0.0.0, force localhost
        callback_url = callback_url.replace('0.0.0.0', 'localhost')
        
        # Avoid https mismatch if behind proxy
        if request.headers.get('X-Forwarded-Proto') == 'https':
            callback_url = callback_url.replace('http:', 'https:')
            
        auth_url, state = uploader.get_auth_url(callback_url)
        
        # Save state flow to session to validate in callback
        session['youtube_oauth_state'] = state
        
        return jsonify({"url": auth_url})
    except Exception as e:
        logger.error(f"YouTube auth URL error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/auth/youtube/callback')
def youtube_callback():
    """Handle YouTube OAuth callback."""
    code = request.args.get('code')
    if not code:
        return "Error: No code provided", 400
        
    try:
        config = Config.load_default()
        uploader = YouTubeUploader(config.config_dir)
        
        callback_url = url_for('youtube_callback', _external=True)
        # Fix: Google doesn't accept 0.0.0.0, force localhost
        callback_url = callback_url.replace('0.0.0.0', 'localhost')

        if request.headers.get('X-Forwarded-Proto') == 'https':
            callback_url = callback_url.replace('http:', 'https:')
            
        # Retrieve state from session
        state = session.get('youtube_oauth_state')
        if not state:
            return "Error: Missing OAuth state in session. Please try again.", 400
            
        # Re-create flow with the SAME state for validation
        flow = uploader.create_flow(callback_url, state=state)
        uploader.fetch_token(flow, request.url)
        
        return redirect('/')  # Back to home
        
    except Exception as e:
        logger.error(f"YouTube callback error: {e}")
        return f"Authentication failed: {e}", 500


@app.route('/api/auth/youtube/status')
def youtube_status():
    """Check YouTube auth status."""
    try:
        config = Config.load_default()
        uploader = YouTubeUploader(config.config_dir)
        
        if uploader.is_authenticated():
            info = uploader.get_channel_info()
            return jsonify({"authenticated": True, "channel": info})
        else:
            return jsonify({"authenticated": False})
            
    except Exception as e:
        return jsonify({"authenticated": False, "error": str(e)})


@app.route('/api/upload', methods=['POST'])
def upload_clip():
    """Upload a single clip to YouTube."""
    data = request.json
    job_id = data.get('job_id')
    clip_index = data.get('clip_index')
    
    if job_id not in processing_jobs:
        return jsonify({"error": "Job not found"}), 404
        
    job = processing_jobs[job_id]
    if clip_index >= len(job.get('clips', [])):
        return jsonify({"error": "Clip index out of range"}), 400
        
    clip = job['clips'][clip_index]
    
    try:
        config = Config.load_default()
        uploader = YouTubeUploader(config.config_dir)
        
        path = Path(clip['path'])
        
        # Ensure hashtags are list
        tags = clip.get('hashtags', [])
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(',')]
            
        video_id = uploader.upload_video(
            file_path=path,
            title=clip.get('title', 'ClipperVX Clip'),
            description=clip.get('description', '') + '\n\n' + ' '.join(tags),
            tags=tags + ['shorts', 'ClipperVX'],
            privacy_status='unlisted'  # Default to unlisted for manual check
        )
        
        return jsonify({"status": "success", "video_id": video_id})
        
    except Exception as e:
        logger.error(f"Upload error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/upload_all', methods=['POST'])
def upload_all_clips():
    """Schedule all clips to YouTube."""
    data = request.json
    job_id = data.get('job_id')
    
    if job_id not in processing_jobs:
        return jsonify({"error": "Job not found"}), 404
        
    job = processing_jobs[job_id]
    clips = job.get('clips', [])
    
    if not clips:
        return jsonify({"error": "No clips to upload"}), 400
        
    results = []
    
    try:
        config = Config.load_default()
        uploader = YouTubeUploader(config.config_dir)
        
        # Start scheduling from 1 hour in future
        schedule_time = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=1)
        
        for clip in clips:
            path = Path(clip['path'])
            tags = clip.get('hashtags', [])
            if isinstance(tags, str):
                tags = [t.strip() for t in tags.split(',')]
                
            # Random offset +/- 15 mins for natural timing
            offset_minutes = random.randint(-15, 15)
            publish_at = schedule_time + datetime.timedelta(minutes=offset_minutes)
            
            video_id = uploader.upload_video(
                file_path=path,
                title=clip.get('title', 'ClipperVX Clip'),
                description=clip.get('description', '') + '\n\n' + ' '.join(tags),
                tags=tags + ['shorts'],
                privacy_status='private',  # Required for scheduled
                publish_at=publish_at
            )
            
            results.append({
                "filename": clip['filename'],
                "video_id": video_id,
                "scheduled_for": publish_at.isoformat()
            })
            
            # Next clip 1 hour later
            schedule_time += datetime.timedelta(hours=1)
            
        return jsonify({"status": "success", "uploads": results})
        
    except Exception as e:
        logger.error(f"Bulk upload error: {e}")
        return jsonify({"error": str(e)}), 500
@app.route('/api/video-info', methods=['POST'])
def get_video_info():
    data = request.json
    url = data.get('url', '')
    
    if not validate_youtube_url(url):
        return jsonify({"error": "Invalid YouTube URL"}), 400
    
    video_id = extract_video_id(url)
    
    import subprocess
    try:
        result = subprocess.run(
            ["yt-dlp", "--dump-json", "--no-download", url],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode != 0:
            error_msg = result.stderr or "Failed to fetch video info"
            logger.error(f"yt-dlp error: {error_msg}")
            return jsonify({"error": f"yt-dlp error: {error_msg[:200]}"}), 500
        
        if not result.stdout.strip():
            return jsonify({"error": "No data returned from YouTube"}), 500
        
        metadata = json.loads(result.stdout)
        formats = get_video_formats(url)
        
        return jsonify({
            "video_id": video_id,
            "title": metadata.get("title", ""),
            "duration": metadata.get("duration", 0),
            "thumbnail": metadata.get("thumbnail", ""),
            "formats": formats
        })
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}")
        return jsonify({"error": "Failed to parse video data"}), 500
    except subprocess.TimeoutExpired:
        return jsonify({"error": "Request timed out. Try again."}), 500
    except Exception as e:
        logger.error(f"Video info error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/upload', methods=['POST'])
def upload_video():
    """Handle local video file upload."""
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    # Generate unique ID for this upload
    video_id = f"local_{uuid.uuid4().hex[:8]}"
    
    # Save to temp directory
    config = Config.load_default()
    upload_dir = config.temp_dir / video_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    filename = secure_filename(file.filename)
    video_path = upload_dir / filename
    file.save(str(video_path))
    
    # Get video duration using ffprobe
    import subprocess
    thumbnail_url = ""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "json", str(video_path)],
            capture_output=True,
            text=True,
            timeout=30
        )
        probe_data = json.loads(result.stdout)
        duration = float(probe_data.get("format", {}).get("duration", 0))
        
        # Generate thumbnail at 10% into the video
        thumbnail_path = upload_dir / "thumbnail.jpg"
        seek_time = max(1, duration * 0.1)  # 10% in or 1 second
        subprocess.run(
            ["ffmpeg", "-y", "-ss", str(seek_time), "-i", str(video_path),
             "-vframes", "1", "-q:v", "2", str(thumbnail_path)],
            capture_output=True,
            timeout=30
        )
        if thumbnail_path.exists():
            thumbnail_url = f"/thumbnails/{video_id}/thumbnail.jpg"
    except:
        duration = 0
    
    return jsonify({
        "video_id": video_id,
        "title": filename,
        "duration": duration,
        "path": str(video_path),
        "thumbnail": thumbnail_url,
        "formats": []
    })


@app.route('/api/process', methods=['POST'])
def process_video():
    """Start video processing."""
    data = request.json
    url = data.get('url')
    local_path = data.get('local_path')
    video_id = data.get('video_id')
    quality = data.get('quality', 'best')
    max_clips = data.get('clips', 3)
    min_length = data.get('min_length', 15)
    max_length = data.get('max_length', 60)
    gemini_key = data.get('gemini_key', '')
    openai_key = data.get('openai_key', '')
    llm_provider = data.get('llm_provider', 'gemini')
    llm_model = data.get('llm_model', '')
    font = data.get('font', 'Komika Axis')
    
    # Validate - need either URL or local path
    if not url and not local_path:
        return jsonify({"error": "No video source provided"}), 400
    
    if url and not validate_youtube_url(url):
        return jsonify({"error": "Invalid URL"}), 400
    
    if not video_id:
        video_id = extract_video_id(url) if url else f"local_{uuid.uuid4().hex[:8]}"
    
    job_id = f"{video_id}_{int(time.time())}"
    
    # Create stop flag for this job
    stop_flags[job_id] = threading.Event()
    
    # Initialize job
    processing_jobs[job_id] = {
        "id": job_id,
        "status": "started", 
        "progress": 0, 
        "stage": "Initializing...",
        "video_id": video_id,
        "title": data.get('title', 'Video'),
        "created": time.time(),
        "font": font
    }
    save_jobs(processing_jobs)
    
    # Start processing in background thread
    thread = threading.Thread(
        target=run_processing,
        args=(job_id, url, local_path, quality, max_clips, min_length, max_length,
              gemini_key, openai_key, llm_provider, llm_model, font)
    )
    thread.daemon = True
    thread.start()
    
    return jsonify({"job_id": job_id})


def run_processing(job_id: str, url: str, local_path: str, quality: str, 
                   max_clips: int, min_length: int, max_length: int,
                   gemini_key: str, openai_key: str, llm_provider: str, llm_model: str, font: str):
    """Run processing and update job status."""
    
    stop_flag = stop_flags.get(job_id)
    
    def progress_callback(stage: str, progress: float):
        # Check if job was stopped
        if stop_flag and stop_flag.is_set():
            raise InterruptedError("Job cancelled by user")
        
        # Update job status
        if job_id in processing_jobs:
            processing_jobs[job_id].update({
                "status": "processing",
                "stage": stage,
                "progress": int(progress * 100)
            })
            save_jobs(processing_jobs)
    
    try:
        progress_callback("Initializing...", 0.05)
        
        config = Config.load_default()
        
        # Override with user-provided keys
        if gemini_key:
            config.gemini_api_key = gemini_key
        if openai_key:
            config.openai_api_key = openai_key
        if llm_provider and llm_provider != 'none':
            config.llm_provider = llm_provider
        if llm_model:
            config.llm_model = llm_model
        
        # Apply font selection
        if font:
            config.caption_style.font = font
        
        # Antigravity doesn't need API keys, other providers do
        use_llm = llm_provider != 'none' and (
            llm_provider == 'antigravity' or 
            gemini_key or openai_key or 
            config.gemini_api_key or config.openai_api_key
        )
        
        orchestrator = ClipperOrchestrator(config)
        
        # Handle local path vs URL
        if local_path:
            result = orchestrator.run_from_file(
                video_path=Path(local_path),
                max_clips=max_clips,
                min_length=min_length,
                max_length=max_length,
                use_llm=use_llm,
                progress_callback=progress_callback,
                stop_event=stop_flag
            )
        else:
            result = orchestrator.run(
                url=url,
                max_clips=max_clips,
                min_length=min_length,
                max_length=max_length,
                quality=quality,
                use_llm=use_llm,
                progress_callback=progress_callback,
                stop_event=stop_flag
            )
        
        # Build clips data with proper paths
        clips_data = []
        for c in result.clips:
            clips_data.append({
                'filename': c.output_path.name,
                'path': str(c.output_path.resolve()),
                'relative_path': f"{result.video_id}/{c.output_path.name}",
                'duration': c.duration,
                'hook': c.hook,
                'title': c.title,
                'description': c.description,
                'hashtags': c.hashtags
            })
        
        if job_id in processing_jobs:
            processing_jobs[job_id].update({
                "status": "complete",
                "clips": clips_data,
                "errors": result.errors,
                "video_id": result.video_id,
                "progress": 100
            })
            save_jobs(processing_jobs)
        
    except InterruptedError:
        logger.info(f"Job {job_id} was cancelled by user")
        if job_id in processing_jobs:
            processing_jobs[job_id].update({
                "status": "stopped",
                "error": "Cancelled by user"
            })
            save_jobs(processing_jobs)
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        import traceback
        traceback.print_exc()
        
        if job_id in processing_jobs:
            processing_jobs[job_id].update({
                "status": "error",
                "error": str(e),
                "progress": 0
            })
            save_jobs(processing_jobs)
    
    finally:
        # Cleanup stop flag
        if job_id in stop_flags:
            del stop_flags[job_id]


@app.route('/api/job/<job_id>')
def job_status(job_id):
    """Get job status (used for polling)."""
    if job_id not in processing_jobs:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(processing_jobs[job_id])


@app.route('/output/<path:filename>')
def serve_output(filename):
    """Serve generated video files."""
    config = Config.load_default()
    # Resolve to absolute path
    output_dir = Path(config.output_dir).resolve()
    file_path = output_dir / filename
    
    logger.info(f"Serving file: {file_path}")
    
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return jsonify({"error": f"File not found: {filename}"}), 404
    
    # Read and serve file directly for better compatibility
    def generate():
        with open(file_path, 'rb') as f:
            while True:
                chunk = f.read(1024 * 1024)  # 1MB chunks
                if not chunk:
                    break
                yield chunk
    
    return Response(generate(), mimetype='video/mp4')


@app.route('/thumbnails/<path:filename>')
def serve_thumbnail(filename):
    """Serve thumbnail images for local uploads."""
    config = Config.load_default()
    temp_dir = Path(config.temp_dir).resolve()
    file_path = temp_dir / filename
    
    if not file_path.exists():
        return jsonify({"error": "Thumbnail not found"}), 404
    
    return send_from_directory(str(file_path.parent), file_path.name, mimetype='image/jpeg')


def run_server(host='0.0.0.0', port=5000, debug=False):
    """Run the web server."""
    app.run(host=host, port=port, debug=debug, threaded=True)


if __name__ == '__main__':
    run_server(debug=True)
