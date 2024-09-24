#!/usr/bin/env python3
import os

import vimeo

client = vimeo.VimeoClient(
    token=os.environ["VIMEO_ACCESS_TOKEN"],
    key=os.environ["VIMEO_CLIENT_ID"],
    secret=os.environ["VIMEO_CLIENT_SECRET"],
)

VIMEO_VIDEO_ID = os.environ["VIMEO_VIDEO_ID"]
VIDEO_PATH = os.environ["VIDEO_PATH"]
VIDEO_URI = f"https://api.vimeo.com/videos/{VIMEO_VIDEO_ID}"

try:
    response = client.replace(VIDEO_URI, filename=VIDEO_PATH)
    print(f"Video replaced at {response}")
except Exception as e:
    print(f"Failed to replace video: {e}")
