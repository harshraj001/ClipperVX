"""CLI interface for ClipperVX."""

import typer
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table
from rich import print as rprint

from . import __version__
from .config import Config
from .orchestrator import ClipperOrchestrator
from .utils import setup_logging, validate_youtube_url

app = typer.Typer(
    name="clipper",
    help="ClipperVX - YouTube to Shorts/Reels automation tool",
    add_completion=False
)

console = Console()


def version_callback(value: bool):
    if value:
        rprint(f"[bold cyan]ClipperVX[/] version [green]{__version__}[/]")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        None, "--version", "-v",
        callback=version_callback,
        is_eager=True,
        help="Show version and exit"
    )
):
    """ClipperVX - Convert YouTube videos to viral Shorts/Reels."""
    pass


@app.command()
def run(
    url: str = typer.Option(..., "--url", "-u", help="YouTube video URL"),
    clips: int = typer.Option(3, "--clips", "-c", help="Number of clips to generate"),
    min_length: int = typer.Option(15, "--min-length", help="Minimum clip length (seconds)"),
    max_length: int = typer.Option(60, "--max-length", help="Maximum clip length (seconds)"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory"),
    llm: str = typer.Option("gemini", "--llm", help="LLM provider (gemini/openai)"),
    no_llm: bool = typer.Option(False, "--no-llm", help="Disable LLM, use heuristics only"),
    verbose: bool = typer.Option(False, "--verbose", "-V", help="Verbose output")
):
    """
    Process a YouTube video and generate clips.
    
    Example:
        clipper run --url "https://youtube.com/watch?v=xxx" --clips 3
    """
    # Validate URL
    if not validate_youtube_url(url):
        console.print("[red]Error:[/] Invalid YouTube URL")
        raise typer.Exit(1)
    
    # Setup logging
    import logging
    setup_logging(logging.DEBUG if verbose else logging.INFO)
    
    # Load config
    config = Config.load_default()
    
    # Override config with CLI options
    if output:
        config.output_dir = Path(output)
    config.llm_provider = llm
    
    console.print(f"\n[bold cyan]ClipperVX[/] v{__version__}")
    console.print(f"[dim]Processing:[/] {url}\n")
    
    # Create orchestrator
    orchestrator = ClipperOrchestrator(config)
    
    # Progress tracking
    current_stage = {"name": "", "progress": 0}
    
    def progress_callback(stage: str, progress: float):
        current_stage["name"] = stage
        current_stage["progress"] = progress
    
    # Run pipeline with progress display
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console
    ) as progress:
        
        task = progress.add_task("Processing...", total=100)
        
        def update_progress(stage: str, pct: float):
            progress.update(task, description=stage, completed=pct * 100)
        
        result = orchestrator.run(
            url=url,
            max_clips=clips,
            min_length=min_length,
            max_length=max_length,
            use_llm=not no_llm,
            progress_callback=update_progress
        )
    
    # Display results
    console.print()
    
    if result.clips:
        table = Table(title="Generated Clips")
        table.add_column("Clip", style="cyan")
        table.add_column("Duration", style="green")
        table.add_column("Hook", style="yellow")
        table.add_column("Score", style="magenta")
        
        for i, clip in enumerate(result.clips, 1):
            table.add_row(
                clip.output_path.name,
                f"{clip.duration:.1f}s",
                clip.hook[:40] + "..." if len(clip.hook) > 40 else clip.hook,
                f"{clip.virality_score:.1f}"
            )
        
        console.print(table)
        console.print(f"\n[green]✓[/] Output directory: [cyan]{config.output_dir / result.video_id}[/]")
    
    if result.errors:
        console.print("\n[yellow]Warnings:[/]")
        for error in result.errors:
            console.print(f"  [dim]•[/] {error}")
    
    if not result.clips:
        console.print("[red]✗[/] No clips generated")
        raise typer.Exit(1)


@app.command()
def config(
    show: bool = typer.Option(True, "--show", "-s", help="Show current configuration")
):
    """Show or edit configuration."""
    cfg = Config.load_default()
    
    table = Table(title="Current Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Output Directory", str(cfg.output_dir))
    table.add_row("Temp Directory", str(cfg.temp_dir))
    table.add_row("Target Resolution", f"{cfg.target_width}x{cfg.target_height}")
    table.add_row("Default Clips", str(cfg.default_clips))
    table.add_row("Clip Length Range", f"{cfg.min_clip_length}-{cfg.max_clip_length}s")
    table.add_row("LLM Provider", cfg.llm_provider)
    table.add_row("LLM Model", cfg.llm_model if cfg.llm_provider == "gemini" else cfg.openai_model)
    table.add_row("Whisper Model", cfg.whisper_model)
    table.add_row("FFmpeg Preset", cfg.ffmpeg_preset)
    table.add_row("FFmpeg CRF", str(cfg.ffmpeg_crf))
    
    # API key status
    gemini_status = "[green]Set[/]" if cfg.gemini_api_key else "[red]Not set[/]"
    openai_status = "[green]Set[/]" if cfg.openai_api_key else "[red]Not set[/]"
    table.add_row("GEMINI_API_KEY", gemini_status)
    table.add_row("OPENAI_API_KEY", openai_status)
    
    console.print(table)


@app.command()
def clean():
    """Clean temporary files."""
    cfg = Config.load_default()
    orchestrator = ClipperOrchestrator(cfg)
    orchestrator.clean_temp()
    console.print("[green]✓[/] Temporary files cleaned")


@app.command()
def gui(
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Host to bind to"),
    port: int = typer.Option(5000, "--port", "-p", help="Port to bind to"),
    debug: bool = typer.Option(False, "--debug", help="Enable debug mode")
):
    """Launch the web GUI interface."""
    from .web_server import run_server
    
    console.print(f"\n[bold cyan]ClipperVX[/] Web Interface")
    console.print(f"[dim]Starting server at[/] http://{host}:{port}\n")
    
    run_server(host=host, port=port, debug=debug)


if __name__ == "__main__":
    app()
