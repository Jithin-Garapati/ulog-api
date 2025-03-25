# Railway Deployment for ULog Reasoning Agent API

This document provides instructions for deploying the ULog Reasoning Agent API to Railway with multi-user support.

## Features

- **Multi-user Support**: Handles multiple concurrent users with isolated session management
- **Automatic Session Cleanup**: Automatically cleans up inactive sessions and temporary files
- **Railway-optimized Configuration**: Properly handles Railway's environment variables and scaling

## Prerequisites

1. A [Railway](https://railway.app/) account
2. Railway CLI tool (optional, for local testing)
3. Git installed on your machine

## Deployment Steps

### 1. Prepare Your Repository

Ensure your repository has the following files:
- `Dockerfile` - Contains the containerization configuration
- `requirements.txt` - Lists all dependencies
- `.env.example` - Example environment variables (do not commit real API keys)

### 2. Set Up Railway Project

1. Log in to [Railway Dashboard](https://railway.app/dashboard)
2. Click "New Project" and select "Deploy from GitHub repo"
3. Select your repository
4. Railway will automatically detect the Dockerfile

### 3. Configure Environment Variables

Add the following environment variables in the Railway dashboard:
- `GROQ_API_KEY` - Your Groq API key
- `GEMINI_API_KEY` - Your Gemini API key
- `SESSION_TIMEOUT` - Session timeout in seconds (default: 1800)
- `MAX_UPLOAD_FILE_SIZE` - Maximum upload file size in MB (default: 10)

### 4. Deploy the Application

1. Railway will automatically build and deploy your application
2. Click on the deployment to see the build logs
3. Once deployed, you'll see a generated domain URL for your API

### 5. Test the Deployment

Test your deployment by:
1. Creating a new session (`POST /api/new_session`)
2. Uploading a ULog file (`POST /api/upload`)
3. Sending a query (`POST /api/query`)
4. Checking multiple parallel sessions work correctly

## API Endpoints

Your API provides the following endpoints:

- `POST /api/new_session` - Create a new user session
- `POST /api/upload` - Upload a ULog file (requires session_id)
- `POST /api/query` - Send a query to the agent (requires session_id)
- `POST /api/end_session` - Explicitly end a session
- `GET /health` - Health check endpoint for monitoring

## Session Management

Sessions are managed as follows:
- Each user gets a unique session ID
- Files are stored in isolated temporary directories
- Sessions expire after 30 minutes of inactivity by default
- A background thread cleans up expired sessions every 5 minutes
- When a user starts a new chat, they should request a new session

## Scaling Considerations

Railway automatically handles scaling, but consider the following:
- Each instance can handle multiple users
- For heavy usage, consider implementing Redis for distributed session management
- Monitor resource usage in Railway dashboard
- Consider setting up auto-scaling based on your usage patterns

## Troubleshooting

If you encounter issues:
1. Check the application logs in Railway dashboard
2. Ensure all environment variables are properly set
3. Verify your API keys are valid
4. Check if the health endpoint is responding correctly
