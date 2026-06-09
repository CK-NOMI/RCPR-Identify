const BACKEND_ORIGIN = 'http://120.46.136.60:8765';

function corsHeaders(request) {
  const origin = request.headers.get('Origin') || '*';
  return {
    'Access-Control-Allow-Origin': origin,
    'Access-Control-Allow-Methods': 'POST, OPTIONS',
    'Access-Control-Allow-Headers': request.headers.get('Access-Control-Request-Headers') || 'Content-Type',
    'Access-Control-Max-Age': '86400',
    Vary: 'Origin'
  };
}

function jsonResponse(body, status = 200, headers = {}) {
  return new Response(JSON.stringify(body), {
    status,
    headers: {
      'Content-Type': 'application/json; charset=utf-8',
      ...headers
    }
  });
}

async function proxyCosod(request) {
  if (request.method === 'OPTIONS') {
    return new Response(null, { status: 204, headers: corsHeaders(request) });
  }

  if (request.method !== 'POST') {
    return jsonResponse(
      { success: false, message: '只支持 POST 请求。' },
      405,
      { Allow: 'POST, OPTIONS', ...corsHeaders(request) }
    );
  }

  const upstreamUrl = `${BACKEND_ORIGIN}/api/cosod`;
  const upstreamHeaders = new Headers(request.headers);
  upstreamHeaders.delete('host');
  upstreamHeaders.delete('cf-connecting-ip');
  upstreamHeaders.delete('cf-ipcountry');
  upstreamHeaders.delete('cf-ray');
  upstreamHeaders.delete('cf-visitor');
  upstreamHeaders.delete('x-forwarded-proto');
  upstreamHeaders.delete('x-real-ip');

  try {
    const upstreamResponse = await fetch(upstreamUrl, {
      method: 'POST',
      headers: upstreamHeaders,
      body: request.body
    });

    const responseHeaders = new Headers(upstreamResponse.headers);
    for (const [key, value] of Object.entries(corsHeaders(request))) {
      responseHeaders.set(key, value);
    }

    return new Response(upstreamResponse.body, {
      status: upstreamResponse.status,
      statusText: upstreamResponse.statusText,
      headers: responseHeaders
    });
  } catch (error) {
    return jsonResponse(
      {
        success: false,
        message: '无法连接华为云后端，请确认服务器服务已启动并放行 8765 端口。',
        detail: error instanceof Error ? error.message : String(error)
      },
      502,
      corsHeaders(request)
    );
  }
}

async function proxyBackendAsset(request) {
  const url = new URL(request.url);
  const upstreamUrl = `${BACKEND_ORIGIN}${url.pathname}${url.search}`;

  try {
    const upstreamResponse = await fetch(upstreamUrl, { method: 'GET' });
    const responseHeaders = new Headers(upstreamResponse.headers);
    responseHeaders.set('Cache-Control', 'public, max-age=3600');
    return new Response(upstreamResponse.body, {
      status: upstreamResponse.status,
      statusText: upstreamResponse.statusText,
      headers: responseHeaders
    });
  } catch (error) {
    return jsonResponse(
      {
        success: false,
        message: '无法读取华为云后端输出图片。',
        detail: error instanceof Error ? error.message : String(error)
      },
      502
    );
  }
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    if (url.pathname === '/api/cosod') {
      return proxyCosod(request);
    }

    if (url.pathname.startsWith('/outputs/')) {
      return proxyBackendAsset(request);
    }

    if (url.pathname === '/api/health') {
      return jsonResponse({ ok: true, backend: BACKEND_ORIGIN });
    }

    return env.ASSETS.fetch(request);
  }
};
