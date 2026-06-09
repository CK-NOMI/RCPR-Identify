import argparse
import cgi
import json
import mimetypes
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import BytesIO
from pathlib import Path
from socketserver import ThreadingMixIn
from urllib.parse import quote, unquote, urlparse

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent
FRONTEND_DIR = ROOT / '前端'
WEB_RUNS_DIR = ROOT / 'web_runs'
WEB_OUTPUTS_DIR = ROOT / 'web_outputs'
LOG_DIR = ROOT / 'web_logs'
CHECKPOINT_NAME = r'baseline运行出的checkpoints\model_combo_base8-136_0.7291838924090067.pt'
MAX_FILE_SIZE = 20 * 1024 * 1024
INFER_LOCK = threading.Lock()


def _json_bytes(payload):
    return json.dumps(payload, ensure_ascii=False).encode('utf-8')


def _url_for(path):
    rel = path.resolve().relative_to(WEB_OUTPUTS_DIR.resolve())
    return '/outputs/' + '/'.join(quote(part) for part in rel.parts)


def _send_overlay(original_path, mask_path, overlay_path):
    original = Image.open(original_path).convert('RGB')
    mask = Image.open(mask_path).convert('L').resize(original.size, Image.BILINEAR)
    original_arr = np.asarray(original, dtype=np.float32)
    mask_arr = np.asarray(mask, dtype=np.float32) / 255.0
    color = np.zeros_like(original_arr)
    color[..., 0] = 255.0
    alpha = 0.45 * mask_arr[..., None]
    overlay = original_arr * (1.0 - alpha) + color * alpha
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8)).save(overlay_path)


def _save_uploaded_images(file_items, image_dir, public_original_dir):
    image_dir.mkdir(parents=True, exist_ok=True)
    public_original_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    for index, item in enumerate(file_items, start=1):
        raw = item.file.read(MAX_FILE_SIZE + 1)
        if len(raw) > MAX_FILE_SIZE:
            raise ValueError(f'文件 {item.filename} 超过 20MB')
        try:
            image = Image.open(BytesIO(raw)).convert('RGB')
        except Exception:
            raise ValueError(f'文件 {item.filename} 不是可读取的图片')

        name = f'{index:04d}.jpg'
        image_path = image_dir / name
        public_path = public_original_dir / name
        image.save(image_path, quality=95)
        image.save(public_path, quality=95)
        saved.append({
            'display_name': item.filename or name,
            'dataset_name': name,
            'image_path': image_path,
            'public_path': public_path,
        })

    return saved


def _run_inference(job_id, image_root, gt_root, model_folder):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f'{job_id}.log'
    cmd = [
        sys.executable,
        'test.py',
        '--model_folder', model_folder,
        '--checkpoint_name', CHECKPOINT_NAME,
        '--datasets', 'CoCA',
        '--size', '224',
        '--test_num_workers', '0',
        '--max_group_images', '20',
        '--stage2_proto', 'acre',
        '--topk_mode', 'rtg',
        '--topk_ratio', '0.04',
        '--topk_res_alpha', '0.10',
        '--topk_conf_gate', '0.58',
        '--topk_mass_min', '0.45',
        '--topk_delta_th', '0.045',
        '--rpf_rounds', '2',
        '--rpf_soft_lambda', '0',
        '--tau2_mode', 'fixed',
        '--tau2_delta', '0.005',
        '--baseline_legacy', '0',
        '--tau1_sim', '0.76',
        '--custom_img_path', str(image_root),
        '--custom_gt_path', str(gt_root),
    ]
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'

    with log_path.open('w', encoding='utf-8') as log_file:
        log_file.write('CMD=' + ' '.join(cmd) + '\n')
        log_file.flush()
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            timeout=1800,
        )

    if proc.returncode != 0:
        tail = log_path.read_text(encoding='utf-8', errors='replace').splitlines()[-40:]
        raise RuntimeError('推理进程失败，日志尾部：\n' + '\n'.join(tail))

    return log_path


def _build_results(saved_images, job_id, model_folder):
    pred_dir = ROOT / 'predictions' / model_folder / 'CoCA' / 'upload'
    public_root = WEB_OUTPUTS_DIR / job_id
    mask_dir = public_root / 'mask'
    overlay_dir = public_root / 'overlay'
    mask_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for item in saved_images:
        stem = Path(item['dataset_name']).stem
        pred_path = pred_dir / f'{stem}.png'
        if not pred_path.is_file():
            raise RuntimeError(f'未找到模型输出：{pred_path}')

        public_mask = mask_dir / f'{stem}.png'
        shutil.copy2(pred_path, public_mask)
        public_overlay = overlay_dir / f'{stem}.png'
        _send_overlay(item['public_path'], public_mask, public_overlay)

        results.append({
            'name': item['display_name'],
            'originalUrl': _url_for(item['public_path']),
            'maskUrl': _url_for(public_mask),
            'overlayUrl': _url_for(public_overlay),
        })

    return results


class CosodHandler(BaseHTTPRequestHandler):
    server_version = 'CosodBackend/1.0'

    def _send_json(self, status, payload):
        body = _json_bytes(payload)
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path):
        if not path.is_file():
            self.send_error(404)
            return
        content_type = mimetypes.guess_type(str(path))[0] or 'application/octet-stream'
        body = path.read_bytes()
        self.send_response(200)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        if path.suffix.lower() in ('.html', '.js', '.css'):
            self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        if path in ('/', '/code.html'):
            self._send_file(FRONTEND_DIR / 'code.html')
            return
        if path == '/api/health':
            self._send_json(200, {'ok': True})
            return
        if path.startswith('/outputs/'):
            rel = path[len('/outputs/'):].lstrip('/').replace('/', os.sep)
            target = (WEB_OUTPUTS_DIR / rel).resolve()
            if not str(target).startswith(str(WEB_OUTPUTS_DIR.resolve())):
                self.send_error(403)
                return
            self._send_file(target)
            return
        self.send_error(404)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != '/api/cosod':
            self.send_error(404)
            return

        try:
            content_type = self.headers.get('Content-Type', '')
            if not content_type.startswith('multipart/form-data'):
                raise ValueError('请求必须使用 multipart/form-data')

            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={
                    'REQUEST_METHOD': 'POST',
                    'CONTENT_TYPE': content_type,
                    'CONTENT_LENGTH': self.headers.get('Content-Length', '0'),
                },
            )
            file_items = form['files[]'] if 'files[]' in form else []
            if not isinstance(file_items, list):
                file_items = [file_items]
            file_items = [item for item in file_items if getattr(item, 'filename', '')]
            if len(file_items) < 2:
                raise ValueError('请至少上传 2 张图片')

            job_id = time.strftime('%Y%m%d_%H%M%S_') + uuid.uuid4().hex[:8]
            model_folder = f'web_cosod_{job_id}'
            job_dir = WEB_RUNS_DIR / job_id
            image_root = job_dir / 'dataset' / 'image'
            image_dir = image_root / 'upload'
            gt_root = job_dir / 'dataset' / 'groundtruth'
            public_original_dir = WEB_OUTPUTS_DIR / job_id / 'original'

            saved_images = _save_uploaded_images(file_items, image_dir, public_original_dir)
            with INFER_LOCK:
                log_path = _run_inference(job_id, image_root, gt_root, model_folder)
            results = _build_results(saved_images, job_id, model_folder)

            self._send_json(200, {
                'jobId': job_id,
                'count': len(results),
                'logUrl': '',
                'logPath': str(log_path),
                'results': results,
            })
        except Exception as exc:
            self._send_json(500, {'error': str(exc), 'message': str(exc)})


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


def main():
    parser = argparse.ArgumentParser(description='SCoSPARC CoSOD web backend')
    parser.add_argument('--host', default='127.0.0.1', type=str)
    parser.add_argument('--port', default=8765, type=int)
    args = parser.parse_args()

    WEB_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    WEB_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((args.host, args.port), CosodHandler)
    print(f'CoSOD backend started: http://{args.host}:{args.port}/code.html', flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
