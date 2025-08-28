#!/usr/bin/env python3
"""
Quick start script for audio analysis pipeline.
Run this to test the pipeline with your audio file.
"""

import os
from pathlib import Path
import click

@click.command()
def quick_start():
    """Interactive quick start guide for audio analysis."""
    
    click.echo(click.style("🎵 Audio Analysis Pipeline - Quick Start", fg='cyan', bold=True))
    click.echo(click.style("=" * 50, fg='cyan'))
    click.echo()
    
    # Check if test audio exists
    test_audio = Path("test_audio.wav")
    if not test_audio.exists():
        click.echo("📝 Creating test audio file...")
        os.system("python create_test_audio.py")
        click.echo(click.style("✅ Test audio created!", fg='green'))
        click.echo()
    
    # Show options
    click.echo("Choose what you want to analyze:")
    click.echo()
    click.echo("1. Test audio file (test_audio.wav)")
    click.echo("2. Your own audio/video file")
    click.echo("3. YouTube URL")
    click.echo("4. Show me an example")
    click.echo()
    
    choice = click.prompt("Enter your choice (1-4)", type=int, default=4)
    
    if choice == 1:
        # Test audio
        click.echo()
        click.echo(click.style("🚀 Analyzing test audio...", fg='yellow'))
        cmd = "python main.py test_audio.wav --output ./quick_test --format both --verbose"
        click.echo(f"Running: {cmd}")
        click.echo()
        os.system(cmd)
        
        click.echo()
        click.echo(click.style("✨ Analysis complete!", fg='green', bold=True))
        click.echo("Check the results in: ./quick_test/")
        click.echo()
        click.echo("Files created:")
        click.echo("  • quick_test/report_*.txt - Summary report")
        click.echo("  • quick_test/timeline_*.json - Detailed results")
        click.echo("  • quick_test/timeline_*.csv - CSV format")
        
    elif choice == 2:
        # User's file
        click.echo()
        click.echo("Enter the path to your audio/video file:")
        click.echo("Examples:")
        click.echo("  • audio.wav")
        click.echo("  • ./input/speech.mp3")
        click.echo("  • /Users/you/Downloads/video.mp4")
        click.echo()
        
        file_path = click.prompt("File path")
        
        if not Path(file_path).exists():
            click.echo(click.style(f"❌ File not found: {file_path}", fg='red'))
            click.echo()
            click.echo("Tips:")
            click.echo("  • Make sure the file exists")
            click.echo("  • Use relative path from current directory")
            click.echo("  • Or use full absolute path")
            return
        
        # Ask for options
        visualize = click.confirm("Generate visualization plots?", default=False)
        verbose = click.confirm("Show detailed progress?", default=True)
        
        # Build command
        cmd = f"python main.py \"{file_path}\" --output ./analysis_output --format both"
        if visualize:
            cmd += " --visualize"
        if verbose:
            cmd += " --verbose"
        
        click.echo()
        click.echo(click.style("🚀 Starting analysis...", fg='yellow'))
        click.echo(f"Running: {cmd}")
        click.echo()
        os.system(cmd)
        
        click.echo()
        click.echo(click.style("✨ Analysis complete!", fg='green', bold=True))
        click.echo("Check the results in: ./analysis_output/")
        
    elif choice == 3:
        # YouTube URL
        click.echo()
        click.echo("Enter YouTube URL:")
        click.echo("Example: https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        click.echo()
        
        url = click.prompt("YouTube URL")
        
        if not url.startswith("http"):
            click.echo(click.style("❌ Invalid URL", fg='red'))
            return
        
        click.echo()
        click.echo(click.style("🚀 Downloading and analyzing...", fg='yellow'))
        cmd = f'python main.py "{url}" --output ./youtube_output --format json --verbose'
        click.echo(f"Running: {cmd}")
        click.echo()
        os.system(cmd)
        
        click.echo()
        click.echo(click.style("✨ Analysis complete!", fg='green', bold=True))
        click.echo("Check the results in: ./youtube_output/")
        
    elif choice == 4:
        # Show example
        click.echo()
        click.echo(click.style("📚 Example Commands:", fg='cyan', bold=True))
        click.echo()
        
        examples = [
            ("Basic analysis", "python main.py audio.wav"),
            ("With output directory", "python main.py audio.wav --output ./results"),
            ("With visualizations", "python main.py audio.wav --visualize"),
            ("Export as CSV", "python main.py audio.wav --format csv"),
            ("From video file", "python main.py video.mp4"),
            ("From YouTube", 'python main.py "https://youtube.com/watch?v=..." --output ./yt'),
            ("Verbose mode", "python main.py audio.wav --verbose"),
            ("Without GPU", "python main.py audio.wav --no-gpu"),
        ]
        
        for desc, cmd in examples:
            click.echo(f"{desc:25} {click.style(cmd, fg='yellow')}")
        
        click.echo()
        click.echo(click.style("📁 File Structure:", fg='cyan', bold=True))
        click.echo()
        click.echo("Put your audio files anywhere, for example:")
        click.echo()
        click.echo("ai_prac/")
        click.echo("├── input/")
        click.echo("│   ├── interview.wav     <- Your audio files here")
        click.echo("│   ├── podcast.mp3")
        click.echo("│   └── meeting.mp4")
        click.echo("├── output/              <- Results will go here")
        click.echo("└── main.py")
        
        click.echo()
        click.echo("Then run: python main.py input/interview.wav")
    
    click.echo()
    click.echo(click.style("-" * 50, fg='cyan'))
    click.echo("For more options, run: python main.py --help")
    click.echo("Full guide: cat GETTING_STARTED.md")
    click.echo()

if __name__ == "__main__":
    quick_start()