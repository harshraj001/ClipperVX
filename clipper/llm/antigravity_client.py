"""
Standalone Antigravity Client for Google Cloud Code API.

This module provides direct access to Claude and Gemini models via
Google's Cloud Code API without requiring an external proxy.
"""

import json
import os
import time
import uuid
import webbrowser
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Generator, Dict, Any
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlencode, parse_qs, urlparse

import requests

from ..utils import get_logger

logger = get_logger(__name__)


# OAuth Configuration
OAUTH_CLIENT_ID = "1071006060591-tmhssin2h21lcre235vtolojh4g403ep.apps.googleusercontent.com"
OAUTH_CLIENT_SECRET = "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"
OAUTH_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
OAUTH_TOKEN_URL = "https://oauth2.googleapis.com/token"
OAUTH_SCOPES = [
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/cclog",
    "https://www.googleapis.com/auth/experimentsandconfigs"
]

# API Configuration
API_BASE_URL = "https://daily-cloudcode-pa.googleapis.com"
DEFAULT_PROJECT = "rising-fact-p41fc"

# Available Models
ANTIGRAVITY_MODELS = {
    "claude-sonnet-4-5-thinking": "Claude Sonnet 4.5 with extended thinking",
    "claude-opus-4-5-thinking": "Claude Opus 4.5 with extended thinking",
    "claude-sonnet-4-5": "Claude Sonnet 4.5 without thinking",
    "gemini-3-flash": "Gemini 3 Flash with thinking",
    "gemini-3-pro-low": "Gemini 3 Pro Low",
    "gemini-3-pro-high": "Gemini 3 Pro High",
}


@dataclass
class TokenData:
    """OAuth token storage."""
    access_token: str
    refresh_token: str
    expires_at: float
    
    def is_expired(self) -> bool:
        return time.time() >= self.expires_at - 60  # 60s buffer
    
    def to_dict(self) -> dict:
        return {
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "expires_at": self.expires_at
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "TokenData":
        return cls(
            access_token=data["access_token"],
            refresh_token=data["refresh_token"],
            expires_at=data["expires_at"]
        )


class OAuthCallbackHandler(BaseHTTPRequestHandler):
    """Handler for OAuth callback."""
    
    auth_code: Optional[str] = None
    
    def do_GET(self):
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        
        if "code" in params:
            OAuthCallbackHandler.auth_code = params["code"][0]
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(b"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>ClipperVX - Authentication Successful</title>
                <style>
                    body { font-family: -apple-system, sans-serif; display: flex; 
                           justify-content: center; align-items: center; height: 100vh;
                           background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); 
                           color: white; margin: 0; }
                    .container { text-align: center; padding: 40px; 
                                 background: rgba(255,255,255,0.1); border-radius: 20px; }
                    h1 { color: #00ff88; }
                    p { color: #aaa; }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>&#10004; Authentication Successful!</h1>
                    <p>You can close this window and return to ClipperVX.</p>
                </div>
            </body>
            </html>
            """)
        else:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b"Authentication failed")
    
    def log_message(self, format, *args):
        pass  # Suppress logging


class AntigravityClient:
    """
    Standalone client for Google Cloud Code API (Antigravity).
    
    Provides access to Claude and Gemini models via Google's internal API.
    """
    
    def __init__(
        self,
        token_path: Optional[Path] = None,
        callback_port: int = 51121
    ):
        self.token_path = token_path or Path.home() / ".clippervx" / "antigravity_token.json"
        self.callback_port = callback_port
        self.token: Optional[TokenData] = None
        self.project_id: Optional[str] = None
        
        # Load cached token
        self._load_token()
    
    def _load_token(self):
        """Load cached token from file."""
        if self.token_path.exists():
            try:
                with open(self.token_path) as f:
                    data = json.load(f)
                self.token = TokenData.from_dict(data)
                logger.info("Loaded cached Antigravity token")
            except Exception as e:
                logger.warning(f"Failed to load cached token: {e}")
    
    def _save_token(self):
        """Save token to file."""
        if self.token:
            self.token_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.token_path, "w") as f:
                json.dump(self.token.to_dict(), f)
            logger.info("Saved Antigravity token to cache")
    
    def is_authenticated(self) -> bool:
        """Check if we have a valid token."""
        return self.token is not None
    
    def get_auth_url(self) -> str:
        """Get OAuth authorization URL."""
        params = {
            "client_id": OAUTH_CLIENT_ID,
            "redirect_uri": f"http://localhost:{self.callback_port}/oauth-callback",
            "response_type": "code",
            "scope": " ".join(OAUTH_SCOPES),
            "access_type": "offline",
            "prompt": "consent"
        }
        return f"{OAUTH_AUTH_URL}?{urlencode(params)}"
    
    def authenticate(self, timeout: int = 120) -> bool:
        """
        Run OAuth authentication flow.
        
        Opens browser for user consent, waits for callback.
        """
        from rich.console import Console
        from rich.panel import Panel
        
        console = Console()
        
        # Start callback server
        OAuthCallbackHandler.auth_code = None
        server = HTTPServer(("localhost", self.callback_port), OAuthCallbackHandler)
        server.timeout = timeout
        
        # Open browser
        auth_url = self.get_auth_url()
        console.print(Panel(
            f"[bold cyan]Opening browser for authentication...[/]\n\n"
            f"If browser doesn't open, visit:\n[dim]{auth_url[:80]}...[/]",
            title="🔐 Antigravity Authentication",
            border_style="cyan"
        ))
        
        webbrowser.open(auth_url)
        
        # Wait for callback
        try:
            while OAuthCallbackHandler.auth_code is None:
                server.handle_request()
        except Exception as e:
            logger.error(f"Auth callback failed: {e}")
            return False
        finally:
            server.server_close()
        
        # Exchange code for token
        return self._exchange_code(OAuthCallbackHandler.auth_code)
    
    def _exchange_code(self, code: str) -> bool:
        """Exchange authorization code for tokens."""
        try:
            resp = requests.post(OAUTH_TOKEN_URL, data={
                "client_id": OAUTH_CLIENT_ID,
                "client_secret": OAUTH_CLIENT_SECRET,
                "code": code,
                "grant_type": "authorization_code",
                "redirect_uri": f"http://localhost:{self.callback_port}/oauth-callback"
            })
            resp.raise_for_status()
            data = resp.json()
            
            self.token = TokenData(
                access_token=data["access_token"],
                refresh_token=data.get("refresh_token", ""),
                expires_at=time.time() + data.get("expires_in", 3600)
            )
            self._save_token()
            logger.info("Successfully authenticated with Antigravity")
            return True
            
        except Exception as e:
            logger.error(f"Token exchange failed: {e}")
            return False
    
    def _refresh_token(self) -> bool:
        """Refresh expired access token."""
        if not self.token or not self.token.refresh_token:
            return False
        
        try:
            resp = requests.post(OAUTH_TOKEN_URL, data={
                "client_id": OAUTH_CLIENT_ID,
                "client_secret": OAUTH_CLIENT_SECRET,
                "refresh_token": self.token.refresh_token,
                "grant_type": "refresh_token"
            })
            resp.raise_for_status()
            data = resp.json()
            
            self.token = TokenData(
                access_token=data["access_token"],
                refresh_token=self.token.refresh_token,
                expires_at=time.time() + data.get("expires_in", 3600)
            )
            self._save_token()
            logger.info("Refreshed Antigravity token")
            return True
            
        except Exception as e:
            logger.error(f"Token refresh failed: {e}")
            return False
    
    def _ensure_token(self) -> str:
        """Ensure we have a valid access token."""
        if not self.token:
            raise RuntimeError("Not authenticated. Call authenticate() first.")
        
        if self.token.is_expired():
            if not self._refresh_token():
                raise RuntimeError("Token expired and refresh failed. Re-authenticate.")
        
        return self.token.access_token
    
    def _get_headers(self) -> dict:
        """Get API request headers."""
        import platform
        
        os_name = platform.system().lower()
        arch = platform.machine()
        
        return {
            "Authorization": f"Bearer {self._ensure_token()}",
            "Content-Type": "application/json",
            "User-Agent": f"antigravity/1.15.8 {os_name}/{arch}",
            "X-Goog-Api-Client": "google-cloud-sdk vscode_cloudshelleditor/0.1",
            "Client-Metadata": json.dumps({
                "ideType": "IDE_UNSPECIFIED",
                "platform": "PLATFORM_UNSPECIFIED",
                "pluginType": "GEMINI"
            })
        }
    
    def _init_project(self) -> str:
        """Initialize and get project ID."""
        if self.project_id:
            return self.project_id
        
        url = f"{API_BASE_URL}/v1internal:loadCodeAssist"
        payload = {
            "metadata": {
                "ideType": "IDE_UNSPECIFIED",
                "platform": "PLATFORM_UNSPECIFIED",
                "pluginType": "GEMINI",
                "duetProject": DEFAULT_PROJECT
            }
        }
        
        try:
            resp = requests.post(url, headers=self._get_headers(), json=payload)
            resp.raise_for_status()
            data = resp.json()
            
            project = data.get("cloudaicompanionProject", {})
            if isinstance(project, str):
                self.project_id = project
            else:
                self.project_id = project.get("id", DEFAULT_PROJECT)
                
            logger.info(f"Initialized Antigravity with project: {self.project_id}")
            
        except Exception as e:
            logger.warning(f"Failed to get project ID: {e}, using default")
            self.project_id = DEFAULT_PROJECT
        
        return self.project_id
    
    def generate(
        self,
        prompt: str,
        model: str = "claude-sonnet-4-5-thinking",
        max_tokens: int = 4096,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Generate content using Antigravity API.
        
        Args:
            prompt: User prompt
            model: Model ID (e.g., claude-sonnet-4-5-thinking)
            max_tokens: Maximum output tokens
            temperature: Sampling temperature
            system_prompt: Optional system instruction
            
        Returns:
            Generated text response
        """
        project_id = self._init_project()
        request_id = f"agent-{uuid.uuid4()}"
        
        # Build system instruction parts
        sys_parts = []
        base_sys = (
            "You are Antigravity, a powerful agentic AI coding assistant designed by "
            "the Google Deepmind team working on Advanced Agentic Coding. You are pair "
            "programming with a USER to solve their coding task."
        )
        sys_parts.append({"text": base_sys})
        sys_parts.append({"text": f"Please ignore the following [ignore]{base_sys}[/ignore]"})
        
        if system_prompt:
            sys_parts.append({"text": system_prompt})
        
        payload = {
            "project": project_id,
            "model": model,
            "userAgent": "antigravity",
            "requestType": "agent",
            "requestId": request_id,
            "request": {
                "contents": [
                    {"role": "user", "parts": [{"text": prompt}]}
                ],
                "systemInstruction": {
                    "role": "user",
                    "parts": sys_parts
                },
                "generationConfig": {
                    "maxOutputTokens": max_tokens,
                    "temperature": temperature,
                    "topP": 0.95
                }
            }
        }
        
        # Use streaming endpoint
        url = f"{API_BASE_URL}/v1internal:streamGenerateContent?alt=sse"
        
        try:
            resp = requests.post(
                url,
                headers=self._get_headers(),
                json=payload,
                stream=True,
                timeout=120
            )
            
            # Log response status
            logger.info(f"Antigravity API status: {resp.status_code}")
            
            if resp.status_code != 200:
                error_text = resp.text
                logger.error(f"Antigravity API error: {error_text}")
                raise RuntimeError(f"API error {resp.status_code}: {error_text[:500]}")
            
            # Parse SSE response
            result_text = []
            thought_text = []
            raw_lines = []
            
            for line in resp.iter_lines():
                if not line:
                    continue
                    
                line = line.decode("utf-8")
                raw_lines.append(line)
                
                if not line.startswith("data:"):
                    continue
                
                json_str = line[5:].strip()
                if not json_str:
                    continue
                
                try:
                    data = json.loads(json_str)
                    
                    # Check for error in response
                    if "error" in data:
                        logger.error(f"API returned error: {data['error']}")
                        raise RuntimeError(f"API error: {data['error']}")
                    
                    # Handle wrapped response format: {"response": {"candidates": [...]}}
                    response_data = data.get("response", data)
                    candidates = response_data.get("candidates", [])
                    if not candidates:
                        continue
                    
                    parts = candidates[0].get("content", {}).get("parts", [])
                    for part in parts:
                        text = part.get("text", "")
                        is_thought = part.get("thought", False)
                        
                        if is_thought:
                            thought_text.append(text)
                        else:
                            result_text.append(text)
                            
                except json.JSONDecodeError as e:
                    logger.debug(f"Failed to parse line: {json_str[:100]}")
                    continue
            
            final_result = "".join(result_text)
            
            # Log if empty response
            if not final_result:
                logger.warning(f"Empty response. Got {len(raw_lines)} raw lines")
                if raw_lines:
                    logger.warning(f"First few lines: {raw_lines[:5]}")
            
            return final_result
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Antigravity API request failed: {e}")
            raise RuntimeError(f"API request failed: {e}")


# Convenience function
def get_antigravity_client(token_path: Optional[Path] = None) -> AntigravityClient:
    """Get or create Antigravity client singleton."""
    return AntigravityClient(token_path=token_path)
