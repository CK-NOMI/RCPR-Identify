import { connect } from 'cloudflare:sockets';

const BACKEND_HOST = '120.46.136.60.sslip.io';
const BACKEND_PORT = 8765;
const BACKEND_ORIGIN = `tcp://${BACKEND_HOST}:${BACKEND_PORT}`;
const textEncoder = new TextEncoder();
const textDecoder = new TextDecoder();

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

function mergeChunks(chunks, totalLength) {
  const output = new Uint8Array(totalLength);
  let offset = 0;
  for (const chunk of chunks) {
    output.set(chunk, offset);
    offset += chunk.length;
  }
  return output;
}

function findHeaderEnd(bytes) {
  for (let i = 0; i <= bytes.length - 4; i += 1) {
    if (bytes[i] === 13 && bytes[i + 1] === 10 && bytes[i + 2] === 13 && bytes[i + 3] === 10) {
      return i;
    }
  }
  return -1;
}

function parseHttpResponse(bytes) {
  const headerEnd = findHeaderEnd(bytes);
  if (headerEnd < 0) {
    throw new Error('Upstream response did not contain HTTP headers.');
  }

  const headerText = textDecoder.decode(bytes.slice(0, headerEnd));
  const body = bytes.slice(headerEnd + 4);
  const lines = headerText.split('\r\n');
  const statusMatch = lines[0].match(/^HTTP\/\d(?:\.\d)?\s+(\d{3})(?:\s+(.*))?$/);
  if (!statusMatch) {
    throw new Error(`Invalid upstream status line: ${lines[0]}`);
  }

  const headers = new Headers();
  for (const line of lines.slice(1)) {
    const separator = line.indexOf(':');
    if (separator <= 0) {
      continue;
    }
    const key = line.slice(0, separator).trim();
    const value = line.slice(separator + 1).trim();
    if (/^(connection|keep-alive|proxy-authenticate|proxy-authorization|te|trailer|transfer-encoding|upgrade)$/i.test(key)) {
      continue;
    }
    headers.append(key, value);
  }

  return {
    status: Number(statusMatch[1]),
    statusText: statusMatch[2] || '',
    headers,
    body
  };
}

async function socketHttpRequest({ method, path, headers = {}, body }) {
  const socket = connect({ hostname: BACKEND_HOST, port: BACKEND_PORT });
  const writer = socket.writable.getWriter();
  const requestBody = body || new Uint8Array();
  const headerLines = [
    `${method} ${path} HTTP/1.1`,
    `Host: ${BACKEND_HOST}:${BACKEND_PORT}`,
    'Connection: close',
    `Content-Length: ${requestBody.byteLength}`,
    ...Object.entries(headers)
      .filter(([, value]) => value)
      .map(([key, value]) => `${key}: ${value}`)
  ];

  await writer.write(textEncoder.encode(`${headerLines.join('\r\n')}\r\n\r\n`));
  if (requestBody.byteLength > 0) {
    await writer.write(requestBody);
  }
  await writer.close();

  const reader = socket.readable.getReader();
  const chunks = [];
  let totalLength = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) {
      break;
    }
    chunks.push(value);
    totalLength += value.length;
  }

  return parseHttpResponse(mergeChunks(chunks, totalLength));
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
    const body = new Uint8Array(await request.arrayBuffer());
    const upstreamResponse = await socketHttpRequest({
      method: 'POST',
      path: '/api/cosod',
      headers: {
        'Content-Type': request.headers.get('Content-Type') || 'application/octet-stream'
      },
      body
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
  try {
    const upstreamResponse = await socketHttpRequest({
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
