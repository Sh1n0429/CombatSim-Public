"""Loopback-only HTTP UI for simulator live evaluation and batch results."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
import pickle
from pathlib import Path
import secrets
from urllib.parse import parse_qs, unquote, urlsplit
import webbrowser

from .config_defaults import APPLICATION_DEFAULTS
from .replay import render_page


def make_server(session, port=APPLICATION_DEFAULTS['live']['port']):
    token = secrets.token_urlsafe(24)
    assets = Path(__file__).parent/'web'

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def reply(self, status, payload, content_type='application/json; charset=utf-8', attachment=None):
            body = (payload if isinstance(payload, bytes) else payload.encode('utf-8')
                    if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False, allow_nan=False).encode('utf-8'))
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('X-Frame-Options', 'SAMEORIGIN')
            if attachment:
                self.send_header('Content-Disposition', f'attachment; filename="{attachment}"')
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def local_request(self):
            allowed = {f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}'}
            if self.headers.get('Host') not in allowed:
                self.reply(403, {'error': 'localhostから開いてください。'})
                return False
            origin = self.headers.get('Origin')
            if origin and origin not in {'http://'+host for host in allowed}:
                self.reply(403, {'error': '別のページからの操作は受け付けていません。'})
                return False
            return True

        def do_GET(self):
            if not self.local_request():
                return
            url = urlsplit(self.path)
            try:
                if url.path == '/':
                    bootstrap = {'token': token, 'state': session.state()}
                    data = json.dumps(bootstrap, ensure_ascii=False).replace('<', r'\u003c')
                    page = (assets/'evaluation.html').read_text(encoding='utf-8').replace('__BOOTSTRAP__', data)
                    self.reply(200, page, 'text/html; charset=utf-8')
                elif url.path == '/evaluation.js':
                    self.reply(200, (assets/'evaluation.js').read_bytes(), 'text/javascript; charset=utf-8')
                elif url.path == '/arena':
                    data = json.dumps({**session.watch(), "token": token}, ensure_ascii=False).replace('<', r'\u003c')
                    page = (assets/'arena.html').read_text(encoding='utf-8').replace('__BOOTSTRAP__', data)
                    self.reply(200, page, 'text/html; charset=utf-8')
                elif url.path in ('/arena.js', '/arena-scene.js', '/arena-webgl.js', '/arena-audio.js'):
                    self.reply(200, (assets/url.path[1:]).read_bytes(), 'text/javascript; charset=utf-8')
                elif url.path.startswith('/arena-assets/'):
                    root = assets/'arena-assets'
                    path = (root/unquote(url.path[len('/arena-assets/'):])).resolve()
                    if not path.is_relative_to(root.resolve()) or not path.is_file():
                        self.reply(404, {'error': 'Not found'})
                        return
                    kind = 'text/javascript' if path.suffix == '.js' else mimetypes.guess_type(path.name)[0] or 'application/octet-stream'
                    self.reply(200, path.read_bytes(), kind)
                elif url.path == '/viewer':
                    state = session.watch()
                    config = {**state['config'], 'seed': state['seed'], 'done': state['done'], 'winner': state['winner']}
                    page = render_page(state['new_frames'], config, live=True)
                    self.reply(200, page, 'text/html; charset=utf-8')
                elif url.path == '/api/state':
                    self.reply(200, session.state())
                elif url.path == '/api/watch':
                    query = parse_qs(url.query)
                    after = int(query.get('after', ['-1'])[0])
                    self.reply(200, session.watch(query.get('match', [''])[0], after))
                elif url.path == '/api/history':
                    rows = []
                    for folder in sorted(session.output.glob('20*-*'), reverse=True):
                        summary = folder/'summary.json'
                        if summary.is_file():
                            with session.condition:
                                rows.append(json.loads(summary.read_text(encoding='utf-8')))
                        if len(rows) == 20:
                            break
                    self.reply(200, rows)
                elif url.path.startswith('/files/'):
                    path = (session.output/unquote(url.path[len('/files/'):])).resolve()
                    if not path.is_relative_to(session.output) or not path.is_file() or path.suffix not in ('.html', '.json', '.jsonl', '.csv'):
                        self.reply(404, {'error': 'ファイルが見つかりません。'})
                        return
                    with session.condition:
                        data = path.read_bytes()
                    kind = mimetypes.guess_type(path.name)[0] or 'application/octet-stream'
                    self.reply(200, data, kind+'; charset=utf-8', path.name if path.suffix != '.html' else None)
                elif url.path == '/favicon.ico':
                    self.reply(204, b'', 'image/x-icon')
                else:
                    self.reply(404, {'error': 'Not found'})
            except (ValueError, TypeError, OSError) as error:
                self.reply(400, {'error': str(error)})

        def do_POST(self):
            if not self.local_request():
                return
            if not secrets.compare_digest(self.headers.get('X-CombatSim-Token', ''), token):
                self.reply(403, {'error': 'この画面を再読み込みしてください。'})
                return
            try:
                length = int(self.headers.get('Content-Length', 0))
                if not 0 < length <= 32768:
                    raise ValueError('リクエストサイズが不正です。')
                data = json.loads(self.rfile.read(length))
                if not isinstance(data, dict):
                    raise ValueError('JSONオブジェクトを指定してください。')
                route = urlsplit(self.path).path
                if route == '/api/models':
                    result = session.add_model(data)
                elif route == '/api/preview':
                    result = session.preview(data)
                elif route == '/api/start':
                    result = session.start(data)
                elif route == '/api/pace':
                    result = session.set_pace(data.get('pace'))
                elif route == '/api/pause':
                    result = session.pause(data.get('paused'))
                elif route == '/api/stop':
                    result = session.stop()
                else:
                    self.reply(404, {'error': 'Not found'})
                    return
                self.reply(200, result)
            except (ValueError, TypeError, RuntimeError, OSError, KeyError, EOFError, pickle.UnpicklingError) as error:
                self.reply(400, {'error': str(error)})

    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    server.daemon_threads = True
    server.session_token = token
    return server


def serve_live(session, port=APPLICATION_DEFAULTS['live']['port'], open_browser=True):
    server = make_server(session, port)
    url = f'http://127.0.0.1:{server.server_port}/'
    print(f'CombatSim Evaluation: {url}\nCtrl+Cで終了します。ログ保存先: {session.output}', flush=True)
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever(poll_interval=.1)
    except KeyboardInterrupt:
        pass
    finally:
        session.stop()
        if session.worker:
            session.worker.join(timeout=10)
        server.server_close()
