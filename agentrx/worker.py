"""Disposable Python-check worker. Not a general-purpose security sandbox."""
import ast
import builtins
import contextlib
import importlib
import io
import json
import math
import sys

ALLOWED_MODULES = {'json', 're', 'math', 'statistics', 'collections', 'datetime',
                   'itertools', 'functools', 'decimal', 'fractions', 'operator'}
SAFE_BUILTINS = ('abs all any bool dict enumerate filter float int isinstance issubclass '
                 'len list map max min next range reversed round set slice sorted str sum '
                 'tuple zip print Exception ValueError TypeError KeyError IndexError '
                 'RuntimeError AssertionError StopIteration').split()


def check_source(source):
    if len(source) > 100_000:
        raise ValueError('Generated check exceeds source limit')
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(n.name not in ALLOWED_MODULES for n in node.names):
                raise ValueError('Unsupported import')
        elif isinstance(node, ast.ImportFrom):
            if node.level or node.module not in ALLOWED_MODULES or any(n.name.startswith('_') for n in node.names):
                raise ValueError('Unsupported import')
        elif isinstance(node, ast.Attribute) and node.attr.startswith('_'):
            raise ValueError('Private attributes are forbidden')
        elif isinstance(node, ast.Name) and node.id.startswith('__'):
            raise ValueError('Private names are forbidden')
        elif isinstance(node, (ast.ClassDef, ast.AsyncFunctionDef, ast.Global, ast.Nonlocal)):
            raise ValueError('Unsupported generated construct')
    return tree


def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level or name not in ALLOWED_MODULES:
        raise ImportError('Unsupported import')
    return importlib.import_module(name)


class BoundedLog(io.StringIO):
    def write(self, value):
        if self.tell() + len(value) > 16_384:
            raise ValueError('Generated check output limit exceeded')
        return super().write(value)


def main():
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
        resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
        if sys.platform.startswith('linux'):
            resource.setrlimit(resource.RLIMIT_AS, (512*1024*1024, 512*1024*1024))
    except (ImportError, OSError, ValueError):
        pass
    result = None
    try:
        payload = json.load(sys.stdin)
        tree = check_source(payload['code'])
        safe = {name: getattr(builtins, name) for name in SAFE_BUILTINS}
        safe['__import__'] = safe_import
        env = {'__builtins__': safe}
        with contextlib.redirect_stdout(BoundedLog()), contextlib.redirect_stderr(BoundedLog()):
            exec(compile(tree, '<generated-check>', 'exec'), env)
            fn = env[payload['function_name']]
            value = fn(payload['trajectory'], payload['offset'])
        if type(value) is not bool:
            raise ValueError('Python check must return a boolean')
        result = {'value': value}
    except BaseException as exc:
        result = {'error': f'{type(exc).__name__}: {str(exc)[:1000]}'}
    print(json.dumps(result))


if __name__ == '__main__':
    main()
