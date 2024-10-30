from typing import Dict, Optional

import redis
from fastapi import HTTPException, Request

class RateLimiter:
    """Rate limiting middleware."""

    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        rate_limit: int = 100,
        time_window: int = 60
    ):
        """Initialize rate limiter.

        Args:
            redis_host: Redis host
            redis_port: Redis port
            rate_limit: Maximum requests per time window
            time_window: Time window in seconds
        """
        self.redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            decode_responses=True
        )
        self.rate_limit = rate_limit
        self.time_window = time_window

    async def check_rate_limit(
        self,
        request: Request,
        key: Optional[str] = None
    ) -> None:
        """Check rate limit for request.

        Args:
            request: FastAPI request
            key: Rate limit key (defaults to IP address)

        Raises:
            HTTPException: If rate limit exceeded
        """
        if not key:
            key = request.client.host

        # Get current count
        current = self.redis_client.get(f"rate_limit:{key}")

        if current is None:
            # First request, set initial count
            self.redis_client.setex(
                f"rate_limit:{key}",
                self.time_window,
                1
            )
        elif int(current) >= self.rate_limit:
            # Rate limit exceeded
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded"
            )
        else:
            # Increment count
            self.redis_client.incr(f"rate_limit:{key}")

rate_limiter = RateLimiter()
