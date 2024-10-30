from datetime import datetime, timedelta
from typing import Dict, Optional

import jwt
from fastapi import Depends, HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from passlib.context import CryptContext

class SecurityHandler:
    """Handle API security."""

    def __init__(self):
        """Initialize security handler."""
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        self.secret_key = os.getenv("JWT_SECRET_KEY")
        self.algorithm = "HS256"
        self.security = HTTPBearer()

    def create_access_token(
        self,
        data: Dict,
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """Create JWT access token.

        Args:
            data: Token data
            expires_delta: Token expiration time

        Returns:
            JWT token
        """
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=15)

        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(
            to_encode,
            self.secret_key,
            algorithm=self.algorithm
        )
        return encoded_jwt

    def verify_token(
        self,
        credentials: HTTPAuthorizationCredentials = Security(HTTPBearer())
    ) -> Dict:
        """Verify JWT token.

        Args:
            credentials: HTTP credentials

        Returns:
            Token payload

        Raises:
            HTTPException: If token is invalid
        """
        try:
            payload = jwt.decode(
                credentials.credentials,
                self.secret_key,
                algorithms=[self.algorithm]
            )
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=401,
                detail="Token has expired"
            )
        except jwt.JWTError:
            raise HTTPException(
                status_code=401,
                detail="Could not validate credentials"
            )

    def verify_api_key(self, api_key: str) -> bool:
        """Verify API key.

        Args:
            api_key: API key to verify

        Returns:
            Whether API key is valid
        """
        valid_api_key = os.getenv("API_KEY")
        return self.pwd_context.verify(api_key, valid_api_key)

security_handler = SecurityHandler()

async def verify_token_dependency(
    credentials: HTTPAuthorizationCredentials = Security(HTTPBearer())
) -> Dict:
    """Dependency for token verification."""
    return security_handler.verify_token(credentials)

async def verify_api_key_dependency(
    api_key: str = Depends(HTTPBearer())
) -> bool:
    """Dependency for API key verification."""
    if not security_handler.verify_api_key(api_key):
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )
    return True
