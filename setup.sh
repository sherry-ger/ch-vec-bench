#!/usr/bin/env bash
set -e

echo ""
echo "=================================================="
echo "  ClickHouse Vector Benchmark — Setup"
echo "=================================================="
echo ""

# Check Python 3.11+
if ! command -v python3 &>/dev/null; then
    echo "Error: python3 not found. Install Python 3.11+ from https://python.org"
    exit 1
fi

python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
required="3.11"
if [ "$(printf '%s\n' "$required" "$python_version" | sort -V | head -n1)" != "$required" ]; then
    echo "Error: Python 3.11+ required (found $python_version)"
    echo "Download from: https://python.org/downloads"
    exit 1
fi
echo "✓ Python $python_version"

# Create virtual environment
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Install dependencies
echo "  Installing dependencies (this takes ~30s on first run)..."
.venv/bin/pip install -e . -q
echo "✓ Dependencies installed"

# Set up .env
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo ""
    echo "=================================================="
    echo "  Next step: fill in your ClickHouse credentials"
    echo "=================================================="
    echo ""
    echo "  Open .env in your editor and fill in:"
    echo "    CLICKHOUSE_HOST     — from ClickHouse Cloud console"
    echo "    CLICKHOUSE_PASSWORD — from ClickHouse Cloud console"
    echo ""
    echo "  Find these under: Settings → Connection details → HTTPS"
    echo ""
else
    echo "✓ .env already configured"
fi

echo ""
echo "=================================================="
echo "  Setup complete!"
echo "=================================================="
echo ""
echo "  1. Fill in .env with your ClickHouse credentials (if not done)"
echo "  2. Activate the venv:  source .venv/bin/activate"
echo "  3. Run a quick test:   python -m src.main run --dims 64 --n-vectors 1000"
echo "  4. Full help:          python -m src.main --help"
echo ""
