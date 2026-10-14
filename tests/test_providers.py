import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import patch
from agentrx.providers import HTTPProvider, ProviderError


class Handler(BaseHTTPRequestHandler):
    records = []
    fail_once = False
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        self.records.append({'body': body, 'path': self.path, 'auth': self.headers.get('Authorization')})
        if self.fail_once and len(self.records) == 1:
            self.send_response(429); self.end_headers(); return
        data = json.dumps({'id':'response-id', 'model':'actual-model', 'usage': {'total_tokens': 22},
                'choices':[{'finish_reason':'stop', 'message': {'content': '{"constraints": []}'}}]}).encode()
        self.send_response(200); self.end_headers(); self.wfile.write(data)
    def log_message(self, *args): pass


class ProviderTests(unittest.TestCase):
    def setUp(self):
        Handler.records = []; Handler.fail_once = False
        self.server = HTTPServer(('127.0.0.1',0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True); self.thread.start()
        self.url = f'http://127.0.0.1:{self.server.server_port}/v1'
    def tearDown(self): self.server.shutdown(); self.server.server_close(); self.thread.join()

    def test_http_client_sends_json_and_retains_usage_and_actual_model(self):
        p = HTTPProvider('gpt-5', self.url, 'test-key', retries=0)
        self.assertEqual(p.complete('global', {'policy': '', 'tools': {}}), {'constraints': []})
        self.assertEqual(Handler.records[0]['path'], '/v1/chat/completions')
        self.assertEqual(Handler.records[0]['auth'], 'Bearer test-key')
        self.assertNotIn('temperature', Handler.records[0]['body'])
        self.assertEqual(p.last_telemetry['model'], 'actual-model')
        self.assertEqual(p.last_telemetry['usage']['total_tokens'], 22)

    def test_transient_http_error_is_retried(self):
        Handler.fail_once = True
        p = HTTPProvider('gpt-5', self.url, 'key', retries=1, retry_delay=0)
        self.assertEqual(p.complete('global', {'policy': '', 'tools': {}}), {'constraints': []})
        self.assertEqual(len(Handler.records), 2)

    def test_missing_credentials_fail_preflight(self):
        with patch.dict('os.environ', {}, clear=True):
            with self.assertRaisesRegex(ProviderError, 'OPENAI_API_KEY'):
                HTTPProvider.from_config({'provider': 'openai', 'model': 'gpt-5'})
