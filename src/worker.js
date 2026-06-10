const BACKEND_ORIGIN = 'http://120.46.136.60';

function corsHeaders(request) {
  const origin = request.headers.get('Origin') || '*';
  return {
    'Access-Control-Allow-Origin': origin,
    'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
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

function filteredRequestHeaders(request) {
  const headers = new Headers(request.headers);
  for (const name of [
    'host',
    'cf-connecting-ip',
    'cf-ipcountry',
    'cf-ray',
    'cf-visitor',
    'connection',
    'content-length',
    'x-forwarded-proto',
    'x-real-ip'
  ]) {
    headers.delete(name);
  }
  return headers;
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

  try {
    const upstreamResponse = await fetch(`${BACKEND_ORIGIN}/api/cosod`, {
      method: 'POST',
      headers: filteredRequestHeaders(request),
      body: request.body
    });

    const responseHeaders = new Headers(upstreamResponse.headers);
    for (const [key, value] of Object.entries(corsHeaders(request))) {
      responseHeaders.set(key, value);
    }

    const contentType = upstreamResponse.headers.get('Content-Type') || '';
    if (!contentType.toLowerCase().includes('application/json')) {
      const text = await upstreamResponse.text();
      return jsonResponse(
        {
          success: false,
          message: '后端代理返回的不是 JSON，请确认华为云后端运行在 80 端口。',
          status: upstreamResponse.status,
          detail: text.slice(0, 300)
        },
        upstreamResponse.status || 502,
        corsHeaders(request)
      );
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
        message: '无法连接华为云后端，请确认服务器服务已启动并放行 80 端口。',
        detail: error instanceof Error ? error.message : String(error)
      },
      502,
      corsHeaders(request)
    );
  }
}

async function proxyBackendAsset(request) {
  const url = new URL(request.url);
  try {
    const upstreamResponse = await fetch(`${BACKEND_ORIGIN}${url.pathname}${url.search}`, {
      method: 'GET'
    });
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
