# Epiclips Frontend

Next.js frontend for Epiclips - AI-powered video clipping.

## Development

```bash
# Install dependencies
npm install

# Run development server
npm run dev
```

Visit [http://localhost:3000](http://localhost:3000)

## Building

```bash
# Production build
npm run build

# Start production server
npm start
```

## Environment Variables

Create `.env.local`:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## Features

- 📹 YouTube URL input
- ⚙️ Clip configuration (count, duration, style)
- 📊 Real-time job progress
- 🎬 Video preview and download
- 🎨 Caption style selector

## Tech Stack

- Next.js 14 (App Router)
- TypeScript
- Tailwind CSS
- Zustand (State Management)
