# Deploying the ULog Reasoning Agent API to Cloudflare

This guide will walk you through deploying the ULog Reasoning Agent API to Cloudflare Workers and R2 for multi-user support.

## Prerequisites

1. A Cloudflare account (you already have this)
2. Cloudflare R2 storage enabled (you already have the credentials)
3. Cloudflare Workers enabled
4. Wrangler CLI installed locally (optional, for local development)

## Step 1: Set up Cloudflare R2

You've already created a Cloudflare R2 bucket with the following credentials:

- Access Key ID: `fcf3fa119ae916da7b908e2a9933d32f`
- Secret Access Key: `a141bac5736337be31657905017944cedddc914cdec6d4b49269cf3dcbd79171`
- Endpoint URL: `https://67ab80061f04d953828dc062a95b1525.r2.cloudflarestorage.com`
- Bucket Name: `ulog-analyzer`

These are already configured in your `.env` file.

## Step 2: Prepare Your Application for Deployment

1. Make sure your application requirements are up to date:

```bash
pip freeze > requirements.txt
```

2. Create a `wrangler.toml` file in your project root:

```toml
name = "ulog-agent-api"
main = "cloudflare_worker.js"
compatibility_date = "2023-10-30"

[vars]
API_HOSTNAME = "your-api-hostname.example.com"

[[r2_buckets]]
binding = "QWEN_AGENT_DATA"
bucket_name = "qwen-agent-data"
```

3. Create a `runtime.txt` file to specify the Python version:

```
python-3.10
```

## Step 3: Deploy to Cloudflare Pages

Cloudflare Pages is the simplest way to deploy your Flask application:

1. Log in to your Cloudflare dashboard
2. Go to "Pages" and click "Create a project"
3. Connect your GitHub repository or upload your files directly
4. Choose the following build settings:
   - Build command: `pip install -r requirements.txt`
   - Build output directory: `/`
   - Root directory: `Agent api`

5. Add the following environment variables:
   - `R2_ENDPOINT_URL`: `https://67ab80061f04d953828dc062a95b1525.r2.cloudflarestorage.com`
   - `R2_ACCESS_KEY_ID`: `fcf3fa119ae916da7b908e2a9933d32f`
   - `R2_SECRET_ACCESS_KEY`: `a141bac5736337be31657905017944cedddc914cdec6d4b49269cf3dcbd79171`
   - `R2_BUCKET_NAME`: `qwen-agent-data`
   - `DEFAULT_MODEL`: `gemini-2.0-flash`
   - `PORT`: `8080`
   - `HOST`: `0.0.0.0`
   - `DEBUG`: `False`

6. Under "Functions" tab, enable Functions and set:
   - Usage model: Unbound
   - Placement: Included
   - Compatibility date: Latest

## Step 4: Deploy as a Worker (Alternative)

If Cloudflare Pages doesn't meet your needs, you can deploy as a Worker:

1. Install Wrangler CLI:

```bash
npm install -g wrangler
```

2. Log in to Cloudflare:

```bash
wrangler login
```

3. Deploy your worker:

```bash
wrangler deploy
```

## Step 5: Configure Your Domain

1. In your Cloudflare dashboard, go to your domain
2. Add a new DNS record:
   - Type: CNAME
   - Name: api (or your preferred subdomain)
   - Target: your Cloudflare Pages/Worker URL
   - Proxy status: Proxied

## Step 6: Update Client Configuration

1. In your client code, update the API URL to point to your Cloudflare domain:

```javascript
const API_URL = 'https://api.yourdomain.com';
```

## Multi-User Support

The API now supports multiple parallel users with proper session management:

1. Each user gets a unique `user_id` and `session_token`
2. Sessions are maintained in both memory and R2 storage
3. Files are stored in R2 with user-specific paths
4. Expired sessions and files are automatically cleaned up

## Testing Your Deployment

1. Open your API in a browser: `https://api.yourdomain.com`
2. Use the client example to interact with your API: `https://api.yourdomain.com/client`
3. Test with multiple browser sessions to verify multi-user support

## API Endpoints

Your API includes the following endpoints:

- `POST /api/auth`: Authenticate a user or create a new session
- `GET /api/ulog_files`: Get list of ULog files for the current user
- `POST /api/upload_ulog`: Upload a ULog file
- `POST /api/query`: Create a new query session
- `GET /api/sessions/<session_id>`: Get session state
- `GET /api/sessions/<session_id>/stream`: Stream session updates
- `GET /api/sessions`: Get all sessions for the current user
- `DELETE /api/sessions/<session_id>`: Delete a session
- `POST /api/sessions/clear`: Clear all sessions for the current user
- `GET /api/examples`: Get example queries
- `GET /api/config`: Get agent configuration
- `GET /api/docs`: Get API documentation

## Monitoring

To monitor your application:

1. In your Cloudflare dashboard, go to "Analytics"
2. Check CPU usage, memory usage, and request count
3. Set up alerts for high resource usage

## Troubleshooting

- If you encounter errors, check the Cloudflare logs in your dashboard
- Verify your R2 credentials are correct
- Make sure your worker has permission to access R2

For more help, contact Cloudflare support or refer to the Cloudflare documentation.
