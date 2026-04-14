#!/bin/bash
# Stratos AI Coaching Companion — Start Script
set -e

cd "$(dirname "$0")"

echo "🚀 Starting Stratos AI Coaching Companion..."
echo ""

# Check for API key
if [ -z "$ANTHROPIC_API_KEY" ]; then
  if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
  fi
fi

if [ -z "$ANTHROPIC_API_KEY" ] || [ "$ANTHROPIC_API_KEY" = "your_api_key_here" ]; then
  echo "⚠️  ANTHROPIC_API_KEY not set."
  echo ""
  echo "   Add your key to the .env file:"
  echo "   ANTHROPIC_API_KEY=sk-ant-..."
  echo ""
  echo "   Get your key at: https://console.anthropic.com"
  echo ""
  exit 1
fi

echo "✅ API key configured"
echo "✅ Starting server on http://localhost:8000"
echo ""
echo "   Open your browser to: http://localhost:8000"
echo "   Press Ctrl+C to stop"
echo ""

PATH="$HOME/Library/Python/3.9/bin:$PATH" uvicorn main:app --reload --port 8000 --host 0.0.0.0
