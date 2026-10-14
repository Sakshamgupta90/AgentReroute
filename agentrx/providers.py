"""JSON model clients and persistent request-level provenance without API secrets."""
import copy
import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib import error, parse, request
from .io import digest, read_json, write_json
from .prompts import messages, prompt_identity


class ProviderError(RuntimeError):
    pass


class NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ProviderError('Model endpoint redirect rejected; configure its final URL')


class HTTPProvider:
    measurement_kind = 'model_measurement'

    def __init__(self, model, base_url, api_key, provider='openai', deployment=None,
                 api_version='2024-10-21', settings=None, timeout=180, retries=3,
                 retry_delay=1, max_input_chars=None):
        url = parse.urlparse(base_url)
        if url.scheme not in ('http', 'https') or not url.hostname or url.username or url.password or url.query or url.fragment:
            raise ProviderError('Invalid model base URL (no credentials/query/fragment allowed)')
        if url.scheme == 'http' and url.hostname not in ('127.0.0.1', 'localhost', '::1'):
            raise ProviderError('Remote model endpoints must use HTTPS')
        if not api_key:
            raise ProviderError('Missing model API credential')
        self.model, self.api_key = model, api_key
        self.provider, self.settings = provider, dict(settings or {})
        allowed = {'temperature', 'top_p', 'seed', 'max_completion_tokens', 'max_tokens', 'reasoning_effort'}
        if set(self.settings) - allowed:
            raise ProviderError('Unsupported sampling setting')
        self.timeout, self.retries, self.retry_delay = timeout, retries, retry_delay
        self.max_input_chars = max_input_chars
        if provider == 'azure':
            if not deployment:
                raise ProviderError('Azure needs AZURE_OPENAI_DEPLOYMENT')
            self.url = (base_url.rstrip('/') + '/openai/deployments/' + parse.quote(deployment, safe='') +
                        '/chat/completions?api-version=' + parse.quote(api_version, safe=''))
        else:
            self.url = base_url.rstrip('/') + '/chat/completions'
        self.identity = {'provider': provider, 'model': model, 'base_url': base_url,
                         'deployment': deployment, 'api_version': api_version if provider == 'azure' else 'v1',
                         'settings': self.settings, 'client': 'python-urllib', 'python': platform.python_version()}
        self.last_telemetry = {}
        self.last_raw = None

    @classmethod
    def from_config(cls, config):
        kind = config.get('provider', 'openai')
        if kind not in ('openai', 'openai-compatible', 'azure', 'replay'):
            raise ProviderError('Unsupported model provider')
        if kind == 'replay':
            return ReplayProvider(config['replay_file'])
        key_env = config.get('api_key_env', 'AZURE_OPENAI_API_KEY' if kind == 'azure' else 'OPENAI_API_KEY')
        key = os.environ.get(key_env)
        if not key:
            raise ProviderError(f'Set {key_env} in your local environment before model-backed runs')
        if config.get('base_url_env') and not os.environ.get(config['base_url_env']):
            raise ProviderError(f"Set {config['base_url_env']} to your versioned model endpoint")
        base = os.environ.get(config.get('base_url_env', '')) or config.get('base_url') or os.environ.get(
            'AZURE_OPENAI_ENDPOINT' if kind == 'azure' else 'OPENAI_BASE_URL',
            '' if kind == 'azure' else 'https://api.openai.com/v1')
        model = os.environ.get(config.get('model_env', '')) or config.get('model', 'gpt-5')
        return cls(model, base, key, provider=kind,
                   deployment=config.get('deployment') or os.environ.get('AZURE_OPENAI_DEPLOYMENT'),
                   api_version=config.get('api_version', os.environ.get('AZURE_OPENAI_API_VERSION', '2024-10-21')),
                   settings=config.get('settings'), timeout=config.get('timeout', 180),
                   retries=config.get('retries', 3), max_input_chars=config.get('max_input_chars'))

    def complete(self, stage, payload):
        body = {'model': self.model, 'messages': messages(stage, payload),
                'response_format': {'type': 'json_object'}, **self.settings}
        encoded = json.dumps(body, ensure_ascii=False).encode()
        if self.max_input_chars and sum(len(m['content']) for m in body['messages']) > self.max_input_chars:
            raise ProviderError('Configured input limit exceeded; no content was truncated')
        headers = {'Content-Type': 'application/json'}
        headers['api-key' if self.provider == 'azure' else 'Authorization'] = (
            self.api_key if self.provider == 'azure' else 'Bearer ' + self.api_key)
        opener = request.build_opener(NoRedirect())
        start = time.monotonic()
        self.last_raw = None
        for attempt in range(self.retries + 1):
            try:
                req = request.Request(self.url, data=encoded, headers=headers, method='POST')
                with opener.open(req, timeout=self.timeout) as response:
                    content = response.read(20_000_001)
                    if len(content) > 20_000_000:
                        raise ProviderError('Model response exceeds size limit')
                    raw = json.loads(content)
                self.last_raw = raw
                self.last_telemetry = {'model': raw.get('model'), 'response_id': raw.get('id'),
                    'usage': raw.get('usage', {}), 'seconds': time.monotonic() - start,
                    'attempts': attempt + 1, 'timestamp': datetime.now(timezone.utc).isoformat()}
                choice = raw['choices'][0]
                if choice.get('finish_reason') not in ('stop', None):
                    raise ProviderError(f"Incomplete model response: {choice.get('finish_reason')}")
                message = choice['message']
                if message.get('refusal'):
                    raise ProviderError('Model refused diagnostic request')
                answer = json.loads(message['content'])
                if not isinstance(answer, dict):
                    raise ProviderError('Model must return a JSON object')
                return answer
            except error.HTTPError as exc:
                exc.close()
                if exc.code not in (429, 500, 502, 503, 504) or attempt == self.retries:
                    raise ProviderError(f'Model HTTP {exc.code}; response body omitted to protect credentials') from None
            except (error.URLError, TimeoutError, OSError):
                if attempt == self.retries:
                    raise ProviderError('Model transport failed after retries') from None
            except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
                raise ProviderError(f'Malformed model JSON response ({type(exc).__name__})') from None
            time.sleep(min(10, self.retry_delay * 2**attempt))
        raise ProviderError('Model retries exhausted')


class ReplayProvider:
    """Explicit fixture replay for offline smoke tests, never benchmark evidence."""
    measurement_kind = 'offline_fixture'
    def __init__(self, path):
        self.responses = read_json(path)
        self.identity = {'provider': 'replay', 'fixture_hash': digest(self.responses)}
        self.last_telemetry, self.last_raw = {}, None

    def complete(self, stage, payload):
        if stage not in self.responses:
            raise ProviderError(f'No fixture response for stage {stage}')
        answer = copy.deepcopy(self.responses[stage])
        self.last_raw = answer
        self.last_telemetry = {'fixture': True, 'usage': {}, 'seconds': 0}
        return answer


class CachedProvider:
    def __init__(self, provider, directory, namespace):
        self.provider, self.directory, self.namespace = provider, Path(directory), namespace
        self.identity = provider.identity
        self.measurement_kind = provider.measurement_kind

    def complete(self, stage, payload):
        prompt = messages(stage, payload)
        identity = {'namespace': self.namespace, 'provider': self.identity,
                    'prompt_identity': prompt_identity(), 'stage': stage, 'messages': prompt}
        key = digest(identity)
        path = self.directory/f'{key}.json'
        if path.exists():
            record = read_json(path)
            if record['request_hash'] != key:
                raise ProviderError('Cached request hash mismatch')
            if record['status'] == 'ok':
                return record['response']
        try:
            answer = self.provider.complete(stage, payload)
            if not isinstance(answer, dict):
                raise ProviderError('Provider must return an object')
            write_json(path, {'request_hash': key, 'request': identity, 'status': 'ok',
                             'response': answer, 'raw_response': getattr(self.provider, 'last_raw', answer),
                             'telemetry': getattr(self.provider, 'last_telemetry', {}),
                             'measurement_kind': self.measurement_kind})
            return answer
        except Exception as exc:
            write_json(path, {'request_hash': key, 'request': identity, 'status': 'error',
                             'error': str(exc), 'raw_response': getattr(self.provider, 'last_raw', None),
                             'measurement_kind': self.measurement_kind})
            raise
