/**
 * Cloudflare Worker for the ULog Reasoning Agent API
 * This worker proxies requests to the API and handles CORS.
 */

addEventListener('fetch', event => {
  event.respondWith(handleRequest(event.request))
})

/**
 * Handle requests to the worker
 * @param {Request} request
 * @returns {Promise<Response>}
 */
async function handleRequest(request) {
  const url = new URL(request.url)
  
  // Add CORS headers to all responses
  const corsHeaders = {
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type, X-User-Id, X-Session-Token'
  }
  
  // Handle OPTIONS requests
  if (request.method === 'OPTIONS') {
    return new Response(null, {
      status: 204,
      headers: corsHeaders
    })
  }
  
  try {
    // Forward the request to the API
    const apiUrl = new URL(request.url)
    apiUrl.hostname = API_HOSTNAME // This is set in your Cloudflare Worker configuration
    
    // Clone the request
    const newRequest = new Request(apiUrl.toString(), {
      method: request.method,
      headers: request.headers,
      body: request.body,
      redirect: 'follow'
    })
    
    // Forward the request
    const response = await fetch(newRequest)
    
    // Clone the response and add CORS headers
    const newResponse = new Response(response.body, response)
    Object.keys(corsHeaders).forEach(key => {
      newResponse.headers.set(key, corsHeaders[key])
    })
    
    return newResponse
  } catch (error) {
    // Return error response
    return new Response(JSON.stringify({
      error: 'Error processing request',
      message: error.message
    }), {
      status: 500,
      headers: {
        'Content-Type': 'application/json',
        ...corsHeaders
      }
    })
  }
}
