"""
core/auth.py — JWT Authentication implementation.

WHAT IS JWT (JSON Web Token)?
  A JWT is a compact, URL-safe token that encodes claims (user info) and is
  cryptographically signed. It looks like:

    eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1c2VyMSJ9.signature
    ─────── header ──────── ─── payload ────── ─ signature ─

  The server creates the token at login. The client sends it with every request.
  The server verifies the signature — no database lookup needed.

WHY JWT OVER SESSION COOKIES?
  - Stateless: no session storage needed on the server
  - Works with microservices (any service can verify the token with the same secret)
  - Works with mobile apps (cookies are browser-specific)
  - The token contains user info directly (no DB lookup on each request)

HOW IT WORKS:
  1. User sends email/password to POST /api/auth/login
  2. Server verifies credentials → creates JWT with user_id as "sub" claim
  3. Server returns {"access_token": "eyJ...", "token_type": "bearer"}
  4. Client stores token (localStorage or memory)
  5. Client sends: Authorization: Bearer eyJ...
  6. Server calls get_current_user() → decodes token → returns user dict

SECURITY:
  - Token is signed with SECRET_KEY (from .env) using HS256 algorithm
  - Changing SECRET_KEY invalidates all existing tokens (useful for security incidents)
  - Tokens expire after JWT_EXPIRE_MINUTES (default: 60)
  - We never store tokens in the database (stateless design)

DEPENDENCIES:
  pip install python-jose[cryptography] passlib[bcrypt]
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

from app.core.config import settings

logger = logging.getLogger(__name__)

# ── OAuth2 scheme ──────────────────────────────────────────────────────────────
# This tells FastAPI where to find the token in requests.
# "tokenUrl" is the endpoint clients call to GET a token.
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="/api/auth/login",
    auto_error=False,  # Return None instead of 401 for optional auth
)

# ── Optional: password hashing ────────────────────────────────────────────────
# In production, passwords are stored as bcrypt hashes, never plain text.
# bcrypt is deliberately slow (work factor 12) to make brute force impractical.
try:
    from passlib.context import CryptContext
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    PASSLIB_AVAILABLE = True
except ImportError:
    pwd_context = None
    PASSLIB_AVAILABLE = False
    logger.warning("passlib not installed — password hashing disabled. pip install passlib[bcrypt]")

# ── JWT library ───────────────────────────────────────────────────────────────
try:
    from jose import JWTError, jwt
    JOSE_AVAILABLE = True
except ImportError:
    JWTError = Exception
    jwt = None
    JOSE_AVAILABLE = False
    logger.warning("python-jose not installed — JWT disabled. pip install python-jose[cryptography]")


# ─────────────────────────────────────────────────────────────────────────────
# Token creation
# ─────────────────────────────────────────────────────────────────────────────

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Create a signed JWT access token.

    Args:
        data: Payload dict. Must include "sub" (subject = user identifier).
              Can include any additional claims: email, roles, etc.
        expires_delta: How long until the token expires. Defaults to settings value.

    Returns:
        JWT string (3 base64url segments joined by dots).

    Example:
        token = create_access_token({"sub": "user123", "email": "a@b.com"})
        # Returns: "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1c2VyMTIzIn0.SIGNATURE"
    """
    if not JOSE_AVAILABLE:
        # Fallback: return a simple placeholder token for development
        logger.warning("JWT library not available — returning placeholder token")
        return f"dev-token-{data.get('sub', 'unknown')}"

    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=settings.jwt_expire_minutes)
    )
    to_encode["exp"] = expire
    to_encode["iat"] = datetime.now(timezone.utc)   # issued at

    return jwt.encode(to_encode, settings.secret_key, algorithm=settings.jwt_algorithm)


def verify_token(token: str) -> Optional[dict]:
    """
    Decode and verify a JWT token.

    Returns:
        The decoded payload dict if valid, None if invalid/expired.

    Failure modes:
        - Expired token → JWTError with "Signature has expired"
        - Wrong signature → JWTError (tampering detected)
        - Malformed token → JWTError
    """
    if not JOSE_AVAILABLE:
        if token.startswith("dev-token-"):
            return {"sub": token.replace("dev-token-", ""), "email": "dev@localhost"}
        return None

    try:
        payload = jwt.decode(
            token,
            settings.secret_key,
            algorithms=[settings.jwt_algorithm],
        )
        return payload
    except JWTError as e:
        logger.debug(f"Token verification failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Password utilities
# ─────────────────────────────────────────────────────────────────────────────

def hash_password(plain_password: str) -> str:
    """
    Hash a plain-text password using bcrypt.

    WHY BCRYPT?
      MD5/SHA-256 are too fast — an attacker with a GPU can test billions of
      hashes/second. bcrypt deliberately takes ~100ms, making brute force
      attacks 10,000× slower. The "12" work factor means 2^12 = 4096 iterations.

    NEVER store plain passwords. Even if your DB is stolen, bcrypt hashes
    cannot be reversed in practical time.
    """
    if not PASSLIB_AVAILABLE or pwd_context is None:
        # Fallback for development (NOT secure — install passlib in production)
        import hashlib
        return hashlib.sha256(plain_password.encode()).hexdigest()
    return pwd_context.hash(plain_password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a plain password against a bcrypt hash.

    Returns True if the password matches the hash.
    Uses constant-time comparison to prevent timing attacks.
    """
    if not PASSLIB_AVAILABLE or pwd_context is None:
        import hashlib
        return hashlib.sha256(plain_password.encode()).hexdigest() == hashed_password
    return pwd_context.verify(plain_password, hashed_password)


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI dependency: get the current authenticated user
# ─────────────────────────────────────────────────────────────────────────────

async def get_current_user(token: Optional[str] = Depends(oauth2_scheme)) -> dict:
    """
    FastAPI dependency: extracts and validates the JWT token from the request.

    USAGE:
        @router.get("/protected")
        async def protected_route(current_user: dict = Depends(get_current_user)):
            return {"user_id": current_user["sub"]}

    HOW IT WORKS:
        1. OAuth2PasswordBearer extracts the token from "Authorization: Bearer <token>"
        2. We decode the token with verify_token()
        3. If valid, return the user payload dict
        4. If invalid/missing, raise HTTP 401

    KUBERNETES NOTE:
        In K8s, multiple pod replicas all share the same SECRET_KEY (from K8s Secret).
        A token issued by pod-1 can be verified by pod-2 — no shared session storage needed.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials. Please log in again.",
        headers={"WWW-Authenticate": "Bearer"},
    )

    if token is None:
        raise credentials_exception

    payload = verify_token(token)
    if payload is None:
        raise credentials_exception

    user_id = payload.get("sub")
    if not user_id:
        raise credentials_exception

    return {
        "sub": user_id,
        "email": payload.get("email", ""),
        "roles": payload.get("roles", []),
    }


async def get_optional_user(token: Optional[str] = Depends(oauth2_scheme)) -> Optional[dict]:
    """
    Like get_current_user but returns None instead of raising 401.
    Use for endpoints that work for both authenticated and anonymous users.

    Example: chat endpoint can use a demo user for unauthenticated requests.
    """
    if token is None:
        return None
    payload = verify_token(token)
    if payload is None:
        return None
    return {
        "sub": payload.get("sub", "anonymous"),
        "email": payload.get("email", ""),
        "roles": payload.get("roles", []),
    }


async def require_admin(current_user: dict = Depends(get_current_user)) -> dict:
    """
    Dependency that requires the user to have the 'admin' role.

    USAGE:
        @router.delete("/api/admin/reset")
        async def reset(admin: dict = Depends(require_admin)):
            ...
    """
    if "admin" not in current_user.get("roles", []):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin role required for this operation.",
        )
    return current_user
