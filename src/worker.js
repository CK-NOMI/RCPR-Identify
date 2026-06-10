import { connect } from 'cloudflare:sockets';

const BACKEND_HOST = '120.46.136.60';
const BACKEND_PORT = 80;
const BACKEND_ORIGIN = `tcp://${BACKEND_HOST}:${BACKEND_PORT}`;
const MAX_UPLOAD_BODY_SIZE = 30 * 1024 * 1024;
const encoder = new TextEncoder();
const decoder = new TextDecoder();

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

function appendBytes(chunks, totalLength) {
  const bytes = new Uint8Array(totalLength);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.length;
  }
  return bytes;
}

function findHeaderEnd(bytes) {
  for (let i = 0; i <= bytes.length - 4; i += 1) {
    if (bytes[i] === 13 && bytes[i + 1] === 10 && bytes[i + 2] === 13 && bytes[i + 3] === 10) {
      return i;
    }
  }
  return -1;
}

function parseHeaders(headerText) {
  const lines = headerText.split('\r\n');
  const statusMatch = lines[0].match(/^HTTP\/\d(?:\.\d)?\s+(\d{3})(?:\s+(.*))?$/);
  if (!statusMatch) {
    throw new Error(`后端响应状态行异常：${lines[0] || '<empty>'}`);
  }

  const headers = new Headers();
  for (const line of lines.slice(1)) {
    const index = line.indexOf(':');
    if (index <= 0) {
      continue;
    }
    const key = line.slice(0, index).trim();
    const value = line.slice(index + 1).trim();
    if (/^(connection|keep-alive|proxy-authenticate|proxy-authorization|te|trailer|transfer-encoding|upgrade)$/i.test(key)) {
      continue;
    }
    headers.append(key, value);
  }

  return {
    status: Number(statusMatch[1]),
    statusText: statusMatch[2] || '',
    headers
  };
}

function buildResponseFromBytes(bytes) {
  const headerEnd = findHeaderEnd(bytes);
  if (headerEnd < 0) {
    throw new Error('后端响应没有 HTTP 头。');
  }
  const meta = parseHeaders(decoder.decode(bytes.slice(0, headerEnd)));
  return {
    ...meta,
    body: bytes.slice(headerEnd + 4)
  };
}

async function socketRequest({ method, path, headers = {}, body = new Uint8Array() }) {
  const socket = connect({ hostname: BACKEND_HOST, port: BACKEND_PORT });
  const writer = socket.writable.getWriter();
  const headerLines = [
    `${method} ${path} HTTP/1.1`,
    `Host: ${BACKEND_HOST}`,
    'Connection: close',
    `Content-Length: ${body.byteLength}`,
    ...Object.entries(headers)
      .filter(([, value]) => value !== undefined && value !== null && value !== '')
      .map(([key, value]) => `${key}: ${value}`)
  ];

  await writer.write(encoder.encode(`${headerLines.join('\r\n')}\r\n\r\n`));
  if (body.byteLength > 0) {
    await writer.write(body);
  }
  writer.releaseLock();

  const reader = socket.readable.getReader();
  const chunks = [];
  let totalLength = 0;
  let expectedTotalLength = null;

  while (true) {
    let result;
    try {
      result = await reader.read();
    } catch (error) {
      if (totalLength > 0) {
        break;
      }
      throw error;
    }

    if (result.done) {
      break;
    }

    chunks.push(result.value);
    totalLength += result.value.length;
    const bytes = appendBytes(chunks, totalLength);
    const headerEnd = findHeaderEnd(bytes);

    if (headerEnd >= 0) {
      const headers = parseHeaders(decoder.decode(bytes.slice(0, headerEnd))).headers;
      const contentLength = Number(headers.get('Content-Length') || '');
      if (Number.isFinite(contentLength)) {
        expectedTotalLength = headerEnd + 4 + contentLength;
        if (bytes.length >= expectedTotalLength) {
          return buildResponseFromBytes(bytes.slice(0, expectedTotalLength));
        }
      }
    }
  }

  return buildResponseFromBytes(appendBytes(chunks, totalLength));
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
    const contentLength = Number(request.headers.get('Content-Length') || '0');
    if (contentLength > MAX_UPLOAD_BODY_SIZE) {
      return jsonResponse(
        { success: false, message: '上传图片总体积过大，请压缩图片或减少单次上传数量后重试。' },
        413,
        corsHeaders(request)
      );
    }

    const upstreamResponse = await socketRequest({
      method: 'POST',
      path: '/api/cosod',
      headers: {
        'Content-Type': request.headers.get('Content-Type') || 'application/octet-stream'
      },
      body: new Uint8Array(await request.arrayBuffer())
    });

    for (const [key, value] of Object.entries(corsHeaders(request))) {
      upstreamResponse.headers.set(key, value);
    }

    return new Response(upstreamResponse.body, {
      status: upstreamResponse.status,
      statusText: upstreamResponse.statusText,
      headers: upstreamResponse.headers
    });
  } catch (error) {
    return jsonResponse(
      {
        success: false,
        message: '无法连接华为云后端，请确认服务器服务已启动并监听 80 端口。',
        detail: error instanceof Error ? error.message : String(error)
      },
      502,
      corsHeaders(request)
    );
  }
}

async function proxyCosodStatus(request) {
  if (request.method === 'OPTIONS') {
    return new Response(null, { status: 204, headers: corsHeaders(request) });
  }

  if (request.method !== 'GET') {
    return jsonResponse(
      { success: false, message: '只支持 GET 请求。' },
      405,
      { Allow: 'GET, OPTIONS', ...corsHeaders(request) }
    );
  }

  const url = new URL(request.url);
  try {
    const upstreamResponse = await socketRequest({
      method: 'GET',
      path: `/api/cosod/status${url.search}`
    });
    for (const [key, value] of Object.entries(corsHeaders(request))) {
      upstreamResponse.headers.set(key, value);
    }
    return new Response(upstreamResponse.body, {
      status: upstreamResponse.status,
      statusText: upstreamResponse.statusText,
      headers: upstreamResponse.headers
    });
  } catch (error) {
    return jsonResponse(
      {
        success: false,
        message: '无法读取识别任务状态。',
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
    const upstreamResponse = await socketRequest({
      method: 'GET',
      path: `${url.pathname}${url.search}`
    });
    upstreamResponse.headers.set('Cache-Control', 'public, max-age=3600');
    return new Response(upstreamResponse.body, {
      status: upstreamResponse.status,
      statusText: upstreamResponse.statusText,
      headers: upstreamResponse.headers
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

async function backendHealth(request) {
  try {
    const upstreamResponse = await socketRequest({
      method: 'GET',
      path: '/api/health'
    });
    for (const [key, value] of Object.entries(corsHeaders(request))) {
      upstreamResponse.headers.set(key, value);
    }
    return new Response(upstreamResponse.body, {
      status: upstreamResponse.status,
      statusText: upstreamResponse.statusText,
      headers: upstreamResponse.headers
    });
  } catch (error) {
    return jsonResponse(
      {
        ok: false,
        backend: BACKEND_ORIGIN,
        detail: error instanceof Error ? error.message : String(error)
      },
      502,
      corsHeaders(request)
    );
  }
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    if (url.pathname === '/api/cosod') {
      return proxyCosod(request);
    }

    if (url.pathname === '/api/cosod/status') {
      return proxyCosodStatus(request);
    }

    if (url.pathname.startsWith('/outputs/')) {
      return proxyBackendAsset(request);
    }

    if (url.pathname === '/api/backend-health') {
      return backendHealth(request);
    }

    if (url.pathname === '/api/health') {
      return jsonResponse({ ok: true, backend: BACKEND_ORIGIN });
    }

    return env.ASSETS.fetch(request);
  }
};
