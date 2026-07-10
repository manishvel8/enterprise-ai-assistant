#!/usr/bin/env bash
# =============================================================================
# setup-local.sh — One-shot local development environment setup
# Run this once before starting Docker Compose for the first time.
# =============================================================================

set -euo pipefail

echo "============================================="
echo " Enterprise AI Chat Assistant — Local Setup"
echo "============================================="

# --- Check prerequisites ---

check_command() {
  if ! command -v "$1" &>/dev/null; then
    echo "ERROR: '$1' is not installed. Please install it first."
    echo "  See: $2"
    exit 1
  fi
}

check_command "docker"  "https://docs.docker.com/get-docker/"
check_command "node"    "https://nodejs.org/"
check_command "python3" "https://www.python.org/downloads/"
check_command "ng"      "npm install -g @angular/cli@17"

echo ""
echo "All prerequisites found."
echo "  Docker:  $(docker --version)"
echo "  Node:    $(node --version)"
echo "  Python:  $(python3 --version)"
echo "  Angular: $(ng version 2>/dev/null | head -1 || echo 'ng found')"
echo ""

# --- Create .env from .env.example if not present ---
if [ ! -f ".env" ]; then
  echo "Creating .env from .env.example ..."
  cp .env.example .env
  echo ""
  echo "IMPORTANT: Edit .env and add your OpenAI API key before continuing."
  echo "  nano .env"
  echo ""
  read -p "Press Enter once you have added your OpenAI API key to .env ..."
else
  echo ".env already exists, skipping copy."
fi

# --- Validate that OPENAI_API_KEY is set ---
source .env 2>/dev/null || true
if [ -z "${OPENAI_API_KEY:-}" ] || [ "${OPENAI_API_KEY}" = "sk-your-openai-key-here" ]; then
  echo ""
  echo "ERROR: OPENAI_API_KEY is not set in .env"
  echo "Please edit .env and add your real OpenAI API key."
  exit 1
fi

echo "OPENAI_API_KEY is set."
echo ""

# --- Create Python virtual environment for local development (outside Docker) ---
if [ ! -d "backend/venv" ]; then
  echo "Creating Python virtual environment for local development ..."
  python3 -m venv backend/venv
  echo "Activating and installing dependencies ..."
  source backend/venv/bin/activate
  pip install --upgrade pip --quiet
  echo "Virtual environment created at backend/venv"
  echo "To activate: source backend/venv/bin/activate"
else
  echo "Python venv already exists at backend/venv"
fi

echo ""
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Start all services:  docker compose up --build"
echo "  2. Open frontend:       http://localhost:4200"
echo "  3. Open API docs:       http://localhost:8000/docs"
echo "  4. Open Neo4j browser:  http://localhost:7474"
echo "  5. Open Qdrant UI:      http://localhost:6333/dashboard"
echo "  6. Open MinIO console:  http://localhost:9001"
echo ""
