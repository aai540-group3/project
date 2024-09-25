#!/usr/bin/env python3
"""
.. module:: src.upload-video
   :synopsis: Replace a video on Vimeo using the Vimeo API.

This script replaces an existing video on Vimeo with a new video file using the Vimeo API.
It requires environment variables to be set for authentication and specifying the video details.

Usage:
    Set the required environment variables and run the script.

Requirements:
    - Python 3.x
    - vimeo-python library
    - Vimeo API credentials (access token, client ID, client secret)
    - Existing video ID on Vimeo
    - Path to the new video file

Environment Variables:
    VIMEO_ACCESS_TOKEN: Your Vimeo API access token
    VIMEO_CLIENT_ID: Your Vimeo API client ID
    VIMEO_CLIENT_SECRET: Your Vimeo API client secret
    VIMEO_VIDEO_ID: The ID of the existing video to be replaced
    VIDEO_PATH: The file path of the new video

Note:
    Ensure that you have the necessary permissions to modify videos on your Vimeo account.
"""

import os
import vimeo

# Initialize the Vimeo client with credentials from environment variables
client = vimeo.VimeoClient(
    token=os.environ["VIMEO_ACCESS_TOKEN"],
    key=os.environ["VIMEO_CLIENT_ID"],
    secret=os.environ["VIMEO_CLIENT_SECRET"],
)

# Get the Vimeo video ID and new video file path from environment variables
VIMEO_VIDEO_ID = os.environ["VIMEO_VIDEO_ID"]
VIDEO_PATH = os.environ["VIDEO_PATH"]

# Construct the video URI for the Vimeo API
VIDEO_URI = f"https://api.vimeo.com/videos/{VIMEO_VIDEO_ID}"

try:
    # Attempt to replace the video
    response = client.replace(VIDEO_URI, filename=VIDEO_PATH)
    print(f"Video successfully replaced. New video URI: {response}")
except Exception as e:
    # Handle any errors that occur during the replacement process
    print(f"Failed to replace video: {e}")
