<div align="center">

# 🎬 Epiclips

**Open-source AI-powered video clipping tool**

Transform long-form videos into viral short-form clips with AI-driven analysis, intelligent face tracking, and automated captions.

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![Next.js](https://img.shields.io/badge/Next.js-14-black.svg)](https://nextjs.org)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://docker.com)

[Documentation](https://github.com/akshaynstack/epiclips/tree/main/docs) • [Demo](#demo) • [Quick Start](#quick-start) • [API Reference](#api-reference)

</div>

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🤖 **AI-Powered Clipping** | Uses Gemini AI to identify the most viral-worthy segments |
| 👤 **Smart Face Tracking** | Multi-tier face detection (MediaPipe + Haar) with smooth tracking |
| 📱 **9:16 Portrait Output** | Perfect for TikTok, Reels, and YouTube Shorts |
| 💬 **Auto Captions** | Word-by-word highlighting with customizable styles |
| 🎨 **Caption Presets** | 5+ built-in styles including viral gold, neon pop, opus bold |
| 🖥️ **Split-Screen Layout** | Auto-detects screen share + webcam for split rendering |
| 🎙️ **Podcast Mode** | 2-person podcast detection with top/bottom stacking |
| ⚡ **Fast Transcription** | Groq Whisper at 216x realtime speed |
| 🌐 **Modern Frontend** | Next.js 14 with real-time progress updates |

---

## 🎥 Demo

<div align="center">
  <img src="docs/assets/demo.gif" alt="Epiclips Demo" width="600">
</div>

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- FFmpeg
- Docker (optional)

### Option 1: Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/akshaynstack/epiclips.git
cd epiclips

# Copy environment file
cp .env.example .env

# Edit .env with your API keys (see Configuration section)

# Run with Docker Compose
docker-compose up -d
```

The API will be available at `http://localhost:8000` and frontend at `http://localhost:3000`.

### Option 2: Manual Setup (using uv - Recommended)

[uv](https://github.com/astral-sh/uv) is a lightning-fast Python package manager that ensures environment isolation.

```bash
# Clone and enter directory
git clone https://github.com/akshaynstack/epiclips.git
cd epiclips

# Initialize and install dependencies (Python 3.11 required)
# uv will automatically create a .venv on its first run
uv sync --python 3.11

# Copy and configure environment
cp .env.example .env
# Edit .env with your API keys

# Run the server
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Option 3: Legacy Manual Setup (venv)

> [!WARNING]
> Python 3.12+ is NOT compatible with MediaPipe 0.10.9. You MUST use Python 3.11.

```bash
# Create virtual environment (Python 3.11)
python3.11 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

#### Frontend (Next.js)

```bash
# In a new terminal
cd frontend

# Install dependencies
npm install

# Run development server
npm run dev
```

Visit `http://localhost:3000` to access the app.

---

## ⚙️ Configuration

Create a `.env` file with the following variables:

```env
# Required - AI Services
OPENROUTER_API_KEY=your_openrouter_api_key    # For AI clip planning
OPENROUTER_MODEL=google/gemini-flash-1.5-8b   # Model to use (free options available)
GROQ_API_KEY=your_groq_api_key                # For fast transcription

# Optional - Cloud Storage
AWS_ACCESS_KEY_ID=your_aws_key
AWS_SECRET_ACCESS_KEY=your_aws_secret
AWS_S3_BUCKET=your-bucket-name
AWS_REGION=us-east-1

# Optional - Webhooks
WEBHOOK_URL=https://your-api.com/webhook    # Receive job completion events
WEBHOOK_SECRET=your_webhook_secret
```

### Getting API Keys

| Service | Purpose | Get Key |
|---------|---------|---------|
| **OpenRouter** | AI clip planning | [OpenRouter](https://openrouter.ai/keys) |
| **Groq** | Fast transcription | [Groq Console](https://console.groq.com/keys) |

---

## 📡 API Reference

### Submit a Clipping Job

```bash
curl -X POST http://localhost:8000/ai-clipping/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "video_url": "https://www.youtube.com/watch?v=VIDEO_ID",
    "max_clips": 5,
    "duration_ranges": ["short", "medium"],
    "include_captions": true,
    "caption_preset": "viral_gold"
  }'
```

**Response:**
```json
{
  "job_id": "abc123-...",
  "status": "accepted",
  "estimated_processing_minutes": 8
}
```

### Check Job Status

```bash
curl http://localhost:8000/ai-clipping/jobs/{job_id}
```

### Available Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ai-clipping/jobs` | POST | Submit new clipping job |
| `/ai-clipping/jobs/{id}` | GET | Get job status and results |
| `/ai-clipping/caption-presets` | GET | List available caption styles |
| `/health` | GET | Health check |
| `/health/ready` | GET | Readiness check |

### Caption Presets

| Preset ID | Style | Colors |
|-----------|-------|--------|
| `viral_gold` | Classic viral | White + Gold |
| `clean_white` | Professional | White + Blue |
| `neon_pop` | Bold neon | Cyan + Magenta |
| `bold_boxed` | High contrast | White + Red |
| `opus_bold` | OpusClip-style | White + Green |

---

## 🏗️ Project Structure

```
epiclips/
├── app/                    # FastAPI backend
│   ├── main.py            # Application entry
│   ├── config.py          # Configuration
│   ├── routers/           # API endpoints
│   └── services/          # Core services
│       ├── ai_clipping_pipeline.py
│       ├── face_detector.py
│       ├── rendering_service.py
│       ├── caption_generator.py
│       └── ...
├── frontend/              # Next.js frontend
│   ├── app/              # App router pages
│   ├── lib/              # Utilities & store
│   └── public/           # Static assets
├── docs/                  # Documentation
├── examples/              # Example scripts
├── tests/                 # Test suite
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## 🎨 Rendering Modes

### Talking Head Mode
Full-frame face tracking with the subject centered in the upper third.

```
┌─────────────────┐
│   ┌─────────┐   │
│   │  Face   │   │  ← Dynamic tracking
│   └─────────┘   │
│                 │
│                 │
│   [Captions]    │
└─────────────────┘
```

### Split-Screen Mode
Screen content on top, speaker on bottom - perfect for tutorials.

```
┌─────────────────┐
│   ┌─────────┐   │
│   │ Screen  │   │  ← 50%
│   └─────────┘   │
│   [Captions]    │
│   ┌─────────┐   │
│   │  Face   │   │  ← 50%
│   └─────────┘   │
└─────────────────┘
```

### Podcast Mode
Two speakers stacked vertically for interviews and podcasts.

```
┌─────────────────┐
│   ┌─────────┐   │
│   │Speaker 1│   │  ← 50%
│   └─────────┘   │
│   ┌─────────┐   │
│   │Speaker 2│   │  ← 50%
│   └─────────┘   │
│   [Captions]    │
└─────────────────┘
```

---

## 🛠️ Development

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=app
```

### Code Style

```bash
# Format code
black app/
isort app/

# Type checking
mypy app/
```

---

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [MediaPipe](https://mediapipe.dev/) - Face detection
- [FFmpeg](https://ffmpeg.org/) - Video processing
- [Groq](https://groq.com/) - Fast transcription
- [Google Gemini](https://deepmind.google/technologies/gemini/) - AI planning

---

<div align="center">

**Made with ❤️ by [Akshay](https://github.com/akshaynstack)**

⭐ Star this repo if you find it useful!

</div>
