# Input Folder

Place your audio and video files here for analysis.

## Supported Formats

### Audio Files
- WAV (.wav)
- MP3 (.mp3)  
- FLAC (.flac)
- M4A (.m4a)
- OGG (.ogg)
- And more...

### Video Files  
- MP4 (.mp4)
- AVI (.avi)
- MOV (.mov)
- MKV (.mkv)
- And more...

## How to Use

1. Copy your audio/video files to this folder
2. Run the analysis:
   ```bash
   python main.py input/yourfile.wav
   ```

3. Results will be saved in the `output/` folder

## Examples

```bash
# Analyze a WAV file
python main.py input/speech.wav

# Analyze an MP3 with visualizations
python main.py input/music.mp3 --visualize

# Analyze a video file
python main.py input/interview.mp4 --format both
```