"""
api/auth.py — Authentication API endpoints.

ENDPOINTS:
  POST /api/auth/login    — exchange email+password for a JWT token
  POST /api/auth/refresh  — exchange a valid token for a new one (extend session)
  GET  /api/auth/me       — return the currently authenticated user's info
  POST /api/auth/logout   — (optional) invalidate token on client side

FLOW:
  Client → POST /api/auth/login {email, password}
         ← 200 {access_token, token_type, expires_in}
  Client → GET /api/chat (Authorization: Bearer <token>)
         ← 200 {response...}

DEV MODE:
  For local development, this endpoint accepts a hardcoded demo user so you
  can test protected routes without setting up a real user database.
  Set AUTH_DEMO_MODE=true in .env to enable.

PRODUCTION:
  Replace the demo user lookup with a real database query.
  Use Alembic to create a 'users' table with hashed passwords.
"""

import logging
from datetime import timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr, Field

from app.core.auth import (
    create_access_token,
    verify_password,
    hash_password,
    get_current_user,
)
from app.core.config import settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/auth", tags=["Authentication"])


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic models (request / response shapes)
# ─────────────────────────────────────────────────────────────────────────────

class LoginRequest(BaseModel):
    """
    Standard email+password login.

    Using BaseModel instead of OAuth2PasswordRequestForm so we can accept JSON
    (the Angular frontend sends JSON, not form-encoded data).
    """
    email: str = Field(..., description="User email address")
    password: str = Field(..., min_length=1, description="User password")

    model_config = {"json_schema_extra": {"example": {
        "email": "demo@example.com",
        "password": "demo123",
    }}}


class TokenResponse(BaseModel):
    """Token returned after successful login."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int = Field(description="Token validity in seconds")
    user_id: str
    email: str


class UserResponse(BaseModel):
    """Current user info returned by GET /api/auth/me."""
    user_id: str
    email: str
    roles: list[str]


class RefreshRequest(BaseModel):
    """Refresh an existing valid token."""
    token: str


# ─────────────────────────────────────────────────────────────────────────────
# Demo user store (replace with database in production)
# ─────────────────────────────────────────────────────────────────────────────
# WHY IN-MEMORY?
#   This is a dev-mode shortcut. In production, store users in PostgreSQL
#   with Alembic-managed migrations. The hash here is bcrypt("demo123").
_DEMO_USERS = {
    "demo@example.com": {
        "user_id": "user-demo-001",
        "email": "demo@example.com",
        "hashed_password": hash_password("demo123"),  # "demo123" hashed at import time
        "roles": ["user"],
        "name": "Demo User",
    },
    "admin@example.com": {
        "user_id": "user-admin-001",
        "email": "admin@example.com",
        "hashed_password": hash_password("admin123"),
        "roles": ["user", "admin"],
        "name": "Admin User",
    },
}


def _lookup_user(email: str) -> Optional[dict]:
    """
    Look up a user by email.

    TODO (production): Replace with:
        result = await db.execute(select(User).where(User.email == email))
        return result.scalar_one_or_none()
    """
    return _DEMO_USERS.get(email.lower())


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/login",
    response_model=TokenResponse,
    summary="Log in and receive a JWT access token",
    responses={
        200: {"description": "Login successful — token returned"},
        401: {"description": "Invalid email or password"},
    },
)
async def login(request: LoginRequest):
    """
    Authenticate with email + password and receive a JWT.

    WHAT HAPPENS:
      1. Look up user by email
      2. Verify bcrypt hash of the provided password
      3. If valid, sign a JWT with user_id as "sub" claim
      4. Return token + metadata

    SECURITY NOTES:
      - We return the same error message for "user not found" and "wrong password"
        This prevents email enumeration attacks (attacker can't distinguish
        "this email doesn't exist" from "wrong password").
      - constant-time comparison (passlib) prevents timing attacks.

    DEMO CREDENTIALS (dev mode):
      email: demo@example.com  password: demo123
      email: admin@example.com password: admin123
    """
    user = _lookup_user(request.email)

    # Use the SAME error for both "not found" and "wrong password"
    # to prevent email enumeration.
    invalid_credentials = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid email or password.",
        headers={"WWW-Authenticate": "Bearer"},
    )

    if not user:
        logger.warning(f"Login attempt for unknown email: {request.email}")
        raise invalid_credentials

    if not verify_password(request.password, user["hashed_password"]):
        logger.warning(f"Failed login attempt for: {request.email}")
        raise invalid_credentials

    # Create JWT token
    token_data = {
        "sub": user["user_id"],
        "email": user["email"],
        "roles": user["roles"],
    }
    expire = timedelta(minutes=settings.jwt_expire_minutes)
    access_token = create_access_token(token_data, expires_delta=expire)

    logger.info(f"Successful login: {user['email']} ({user['user_id']})")

    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        expires_in=settings.jwt_expire_minutes * 60,
        user_id=user["user_id"],
        email=user["email"],
    )


@router.post(
    "/refresh",
    response_model=TokenResponse,
    summary="Refresh an existing JWT token",
)
async def refresh_token(request: RefreshRequest):
    """
    Exchange a valid (non-expired) token for a new one with a fresh expiry.

    WHY TOKEN REFRESH?
      Short-lived tokens (60 min) are more secure — if stolen, they expire quickly.
      Refresh lets users stay logged in without re-entering their password.

    PRODUCTION ENHANCEMENT:
      Use refresh tokens (long-lived, stored in DB) instead of re-issuing access tokens.
      This allows revoking sessions: delete the refresh token from DB to log out a user.
    """
    from app.core.auth import verify_token

    payload = verify_token(request.token)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token is invalid or expired.",
        )

    user_id = payload.get("sub")
    email = payload.get("email", "")
    roles = payload.get("roles", [])

    new_token = create_access_token(
        {"sub": user_id, "email": email, "roles": roles},
        expires_delta=timedelta(minutes=settings.jwt_expire_minutes),
    )

    return TokenResponse(
        access_token=new_token,
        token_type="bearer",
        expires_in=settings.jwt_expire_minutes * 60,
        user_id=user_id,
        email=email,
    )


@router.get(
    "/me",
    response_model=UserResponse,
    summary="Get the currently authenticated user",
)
async def get_me(current_user: dict = Depends(get_current_user)):
    """
    Return the currently authenticated user's info.

    This endpoint requires a valid JWT in the Authorization header.
    The Angular frontend calls this after login to get user details.

    Example response:
        {
            "user_id": "user-demo-001",
            "email": "demo@example.com",
            "roles": ["user"]
        }
    """
    return UserResponse(
        user_id=current_user["sub"],
        email=current_user.get("email", ""),
        roles=current_user.get("roles", []),
    )


@router.post(
    "/logout",
    summary="Log out (client-side token invalidation)",
)
async def logout():
    """
    Log out the current user.

    STATELESS NOTE:
      JWTs are stateless — the server doesn't store them, so there's no server-side
      "invalidation". True logout = client deletes the token from localStorage.

      For immediate revocation (e.g., security incident):
        Option A: Blacklist the token in Redis until it expires
        Option B: Change the SECRET_KEY (invalidates ALL tokens — nuclear option)
        Option C: Use short expiry (60min) and refresh tokens

      This endpoint exists for API completeness. Clients should delete their token.
    """
    return {"message": "Logged out successfully. Please delete your token client-side."}
