# CausalForge: Setup Guide

## Prerequisites

- **Node.js** 18+ (recommended: 20+)
- **Python** 3.10+
- **npm** or **pnpm**
- **NVIDIA NIM API key** (from build.nvidia.com)
- **MongoDB Atlas account** (free tier works)

## Installation

### 1. Clone the Repository

```bash
git clone <repo-url>
cd CausalForge
```

### 2. Frontend Setup

```bash
cd apps/web
npm install
```

Create `.env.local`:
```
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### 3. Backend Setup

```bash
cd services/causal-engine
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

Create `.env`:
```
NIM_API_KEY=nvapi-your-key-here
MONGODB_URI=mongodb+srv://user:pass@cluster.mongodb.net/causalforge?retryWrites=true&w=majority
MONGODB_DB=causalforge
```

### 4. MongoDB Atlas Setup

1. Create a free cluster at cloud.mongodb.com
2. Create a database user with read/write access
3. Add your IP to the allowlist (or use 0.0.0.0/0 for development)
4. Copy the connection string to your `.env` file
5. The database `causalforge` will be created automatically on first write

### 5. NVIDIA NIM Setup

1. Visit build.nvidia.com
2. Sign up or log in
3. Generate an API key
4. Add it to your `.env` as `NIM_API_KEY`
5. Note: The demo works without live NIM calls (preloaded data)

## Running Locally

### Terminal 1 — Backend
```bash
cd services/causal-engine
source .venv/bin/activate
python main.py
# Runs on http://localhost:8000
```

### Terminal 2 — Frontend
```bash
cd apps/web
npm run dev
# Runs on http://localhost:3000
```

## Deployment

### Frontend (Vercel)
```bash
cd apps/web
npx vercel deploy
```

### Backend (any Python host)
The FastAPI backend can be deployed to:
- Railway
- Render
- Fly.io
- Any VPS with Python 3.10+

Set environment variables in your hosting platform's dashboard.

## Environment Variables Reference

| Variable | Required | Description |
|---|---|---|
| `NIM_API_KEY` | Yes | NVIDIA NIM API key |
| `MONGODB_URI` | No* | MongoDB Atlas connection string |
| `MONGODB_DB` | No | Database name (default: causalforge) |
| `NEXT_PUBLIC_API_URL` | Yes | Backend API URL for frontend |

*MongoDB is optional — the system runs without persistence in demo mode.

## Troubleshooting

**"NIM call failed"**: Check your API key and ensure you have credits at build.nvidia.com.

**"MongoDB unavailable"**: The system will run without persistence. Demo mode uses preloaded data.

**Port conflicts**: Backend runs on 8000, frontend on 3000. Change in `main.py` or `next.config.ts` if needed.

**Type errors**: Run `npm run build` in `apps/web` to check for TypeScript issues.
