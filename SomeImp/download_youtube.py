import subprocess
import os
import re

def download_youtube_video(url, output_path=".", filename=None, audio_only=False, resolution=None):
    """
    Downloads a YouTube video using yt-dlp.

    Args:
        url (str): The URL of the YouTube video to download.
        output_path (str, optional): The path where the video should be saved. Defaults to the current directory.
        filename (str, optional): The filename to use for the downloaded video. If None, yt-dlp will generate one.
        audio_only (bool, optional): If True, downloads only the audio. Defaults to False.
        resolution (str, optional):  The desired video resolution (e.g., "1080p", "720p", "best").
            If None, downloads the best available resolution.
    """
    # Construct the yt-dlp command.
    command = ["yt-dlp"]

    if filename:
        command.extend(["-o", os.path.join(output_path, filename)])
    else:
        command.extend(["-o", output_path + '/%(title)s.%(ext)s']) # Ensure output_path is included

    if audio_only:
        command.extend(["-x", "--audio-format", "mp3"])  # Extract audio and save as mp3
    elif resolution:
        command.extend(["-f", f"bestvideo[height<={resolution}]+bestaudio/best[height<={resolution}]/best"]) # Select best video and audio below specified resolution
    else:
        command.append("-f best") # download the best quality video and audio available

    command.append(url)

    try:
        # Run the command and capture the output.
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()

        # Print the output from yt-dlp (optional, for debugging or user feedback)
        print(stdout.decode("utf-8"))
        if stderr:
            print(stderr.decode("utf-8"))

        if process.returncode != 0:
            print(f"Error: yt-dlp returned code {process.returncode}")
            return False  # Indicate failure

        print(f"Download complete!")
        return True #Indicate Success

    except FileNotFoundError:
        print("Error: yt-dlp is not installed. Please install it using 'pip install yt-dlp'")
        return False
    except Exception as e:
        print(f"An error occurred: {e}")
        return False

def main():
    """
    Main function to take user input and call the download function.
    """
    url = input("Enter the YouTube video URL: ")
    output_path = input("Enter the output path (default: current directory): ") or "."
    filename = input("Enter the filename (optional, default: video title): ")
    audio_only = input("Download audio only? (y/n, default: n): ").lower() == "y"
    resolution = input("Enter maximum resolution (e.g., 1080p, 720p, or press Enter for best): ")

    # Validate the URL
    if not re.match(r"https?://(?:www\.)?youtube\.com/watch\?v=[a-zA-Z0-9_-]+", url):
        print("Error: Invalid YouTube URL.")
        return

    # Validate the output path.
    if not os.path.exists(output_path):
        print(f"Error: Output path '{output_path}' does not exist.  Creating it.")
        try:
            os.makedirs(output_path)
        except OSError as e:
            print(f"Error creating output directory: {e}")
            return

    success = download_youtube_video(url, output_path, filename, audio_only, resolution)
    if success:
        print("Video downloaded successfully!")
    else:
        print("Video download failed.")

if __name__ == "__main__":
    main()

