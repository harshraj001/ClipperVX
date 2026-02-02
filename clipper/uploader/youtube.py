import os
import json
import logging
import datetime
import random
from pathlib import Path
from typing import Optional, List, Dict, Tuple

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

from ..utils import get_logger

logger = get_logger(__name__)

SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload", 
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/youtube.readonly"
]

class YouTubeUploader:
    """Handles YouTube authentication and video uploading."""
    
    def __init__(self, config_dir: Path):
        self.config_dir = config_dir
        self.client_secrets_file = config_dir / "client_secrets.json"
        self.token_file = config_dir / "youtube_token.json"
        self.creds = None
        self.service = None
        
        # Load credentials if they exist
        self._load_credentials()

    def _load_credentials(self):
        """Load saved credentials from token file."""
        if self.token_file.exists():
            try:
                self.creds = Credentials.from_authorized_user_file(str(self.token_file), SCOPES)
                # Refresh if expired
                if self.creds and self.creds.expired and self.creds.refresh_token:
                    self.creds.refresh(Request())
                    # Save refreshed token
                    with open(self.token_file, "w") as token:
                        token.write(self.creds.to_json())
            except Exception as e:
                logger.error(f"Failed to load credentials: {e}")
                self.creds = None

    def is_authenticated(self) -> bool:
        """Check if we have valid credentials."""
        return self.creds is not None and self.creds.valid

    def create_flow(self, redirect_uri: str, state: str = None):
        """Create OAuth flow instance."""
        if not self.client_secrets_file.exists():
            raise FileNotFoundError(f"Client secrets file not found at {self.client_secrets_file}")
            
        return Flow.from_client_secrets_file(
            str(self.client_secrets_file),
            scopes=SCOPES,
            redirect_uri=redirect_uri,
            state=state
        )

    def get_auth_url(self, redirect_uri: str) -> Tuple[str, str]:
        """
        Generate the authorization URL for the user.
        Returns (auth_url, state)
        """
        flow = self.create_flow(redirect_uri)
        auth_url, state = flow.authorization_url(prompt='consent')
        return auth_url, state

    def fetch_token(self, flow, authorization_response: str):
        """Exchange auth code for token."""
        flow.fetch_token(authorization_response=authorization_response)
        self.creds = flow.credentials
        
        # Save credentials
        with open(self.token_file, "w") as token:
            token.write(self.creds.to_json())
            
        return self.creds

    def get_channel_info(self) -> Dict:
        """Get connected channel info."""
        if not self.is_authenticated():
            return {}
            
        try:
            service = build('youtube', 'v3', credentials=self.creds)
            request = service.channels().list(
                part="snippet",
                mine=True
            )
            response = request.execute()
            
            if response.get("items"):
                snippet = response["items"][0]["snippet"]
                return {
                    "title": snippet["title"],
                    "thumbnail": snippet["thumbnails"]["default"]["url"]
                }
        except Exception as e:
            logger.error(f"Failed to fetch channel info: {e}")
            
        return {}

    def upload_video(
        self, 
        file_path: Path, 
        title: str, 
        description: str, 
        tags: List[str], 
        privacy_status: str = "private",
        publish_at: datetime.datetime = None
    ) -> str:
        """
        Uploads a video to YouTube.
        Returns the video ID.
        """
        if not self.is_authenticated():
            raise Exception("Not authenticated with YouTube")
            
        if not file_path.exists():
            raise FileNotFoundError(f"Video file not found: {file_path}")

        service = build('youtube', 'v3', credentials=self.creds)
        
        body = {
            "snippet": {
                "title": title[:100],  # YouTube max title length
                "description": description[:5000],
                "tags": tags,
                "categoryId": "22" # People & Blogs default
            },
            "status": {
                "privacyStatus": privacy_status,
                "selfDeclaredMadeForKids": False
            }
        }
        
        # Scheduling
        if publish_at:
            # YouTube requires "private" status for scheduled uploads
            body["status"]["privacyStatus"] = "private"
            body["status"]["publishAt"] = publish_at.isoformat()
            
        media = MediaFileUpload(
            str(file_path),
            chunksize=-1, 
            resumable=True,
            mimetype="video/mp4"
        )
        
        request = service.videos().insert(
            part="snippet,status",
            body=body,
            media_body=media
        )
        
        response = None
        while response is None:
            status, response = request.next_chunk()
            if status:
                logger.info(f"Uploaded {int(status.progress() * 100)}%")
        
        logger.info(f"Upload complete! Video ID: {response['id']}")
        return response['id']
