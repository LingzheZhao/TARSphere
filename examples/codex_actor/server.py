"""Loopback-only actor adapter using the public Codex app-server protocol."""
import argparse
import json
import os
import signal
import subprocess
import tempfile
import threading
import time
import pathlib
import tomllib
from http.server import BaseHTTPRequestHandler, HTTPServer
from payload import map_actor_payload
from rpc import RPC, ProtocolError

MODEL = 'gpt-6.1-terra'
FEATURES = ('shell_tool', 'unified_exec', 'apps', 'view_image', 'sleep_tool',
            'multi_agent', 'hooks', 'memories', 'shell_snapshot')


def command(binary, mcp_names):
    result = [binary, 'app-server', '--listen', 'stdio://']
    for setting in ('web_search="disabled"', 'project_doc_max_bytes=0',
                    'skills.include_instructions=false', 'skills.bundled.enabled=false',
                    'notify=[]', 'features.daemon_auto_start=false'):
        result += ['-c', setting]
    config_path = pathlib.Path(os.environ.get('CODEX_HOME', str(pathlib.Path.home() / '.codex'))) / 'config.toml'
    config = tomllib.loads(config_path.read_text()) if config_path.exists() else {}
    for name in config.get('plugins', {}):
        if '.' in name or any(c.isspace() for c in name):
            raise ValueError('Unsupported dotted or whitespace plugin config name')
        result += ['-c', 'plugins.' + name + '.enabled=false']
    mcp_names = set(mcp_names) | set(config.get('mcp_servers', {}))
    for name in FEATURES:
        result += ['-c', 'features.' + name + '=false']
    for name in sorted(mcp_names):
        if '.' in name or any(c.isspace() for c in name):
            raise ValueError('Unsupported dotted or whitespace MCP config name')
        result += ['-c', 'mcp_servers.' + name + '.enabled=false']
    return result


def infer(payload, args):
    allowed = {'model', 'messages', 'temperature', 'top_p', 'max_tokens',
               'max_completion_tokens', 'seed', 'response_format', 'timeout'}
    unknown = set(payload) - allowed
    if unknown:
        raise ValueError('Unsupported request fields: ' + ', '.join(sorted(unknown)))
    system, items = map_actor_payload(payload)
    if payload.get('model') != args.model:
        raise ValueError('Request model must equal the configured actor model')
    controls = {key: payload[key] for key in ('temperature', 'top_p', 'max_tokens',
                 'max_completion_tokens', 'seed', 'timeout') if key in payload}
    if 'response_format' in payload:
        if payload['response_format'] != {'type': 'json_object'}:
            raise ValueError('Only acknowledged, unenforced json_object response_format is supported')
        controls['response_format'] = payload['response_format']
    if controls and not args.acknowledge_unapplied_controls:
        raise ValueError('Codex does not implement Chat Completions sampling/token controls; explicit adapter opt-in required')
    with tempfile.TemporaryDirectory(prefix='tarsphere-codex-actor-') as cwd:
        process = subprocess.Popen(command(args.codex, args.disable_mcp), cwd=cwd,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, start_new_session=True)
        def terminate():
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        timer = threading.Timer(args.timeout, terminate)
        timer.start()
        try:
            rpc = RPC(process.stdout, process.stdin)
            rpc.request('initialize', {'clientInfo': {'name': 'tarsphere_actor', 'version': '0.1'},
                                      'capabilities': {'experimentalApi': True}})
            rpc.notify('initialized')
            status = rpc.request('mcpServerStatus/list', {})
            if status.get('nextCursor'):
                raise ProtocolError('Uninspected MCP server inventory')
            for entry in status.get('data', []):
                if (entry.get('tools') or entry.get('resources') or
                        entry.get('resourceTemplates') or entry.get('runtimeStatus') is not None):
                    raise ProtocolError('MCP server capabilities remain available')
            started = rpc.request('thread/start', {
                'model': args.model, 'allowProviderModelFallback': False,
                'cwd': cwd, 'ephemeral': True, 'baseInstructions': system,
                'developerInstructions': '', 'environments': [],
                'selectedCapabilityRoots': [], 'dynamicTools': [],
                'runtimeWorkspaceRoots': [], 'sandbox': 'read-only', 'approvalPolicy': 'never'})
            if started['model'] != args.model or started.get('instructionSources'):
                raise ProtocolError('Unexpected resolved model or loaded instruction sources')
            thread_id = started['thread']['id']
            turn = rpc.request('turn/start', {'threadId': thread_id,
                'input': items, 'model': args.model, 'effort': args.effort,
                'environments': [], 'runtimeWorkspaceRoots': []})
            text = None
            usage = None
            pending = rpc.notifications
            rpc.notifications = []
            while True:
                event = pending.pop(0) if pending else rpc.receive()
                method, params = event.get('method'), event.get('params', {})
                if method == 'thread/tokenUsage/updated':
                    usage = params['tokenUsage']['last']
                if method in ('item/started', 'item/completed'):
                    item = params['item']
                    if (method == 'item/completed' and item['type'] == 'agentMessage'
                            and item.get('phase') == 'final_answer'):
                        text = item['text']
                    elif item['type'] not in ('reasoning', 'userMessage', 'agentMessage'):
                        raise ProtocolError('Unexpected actor item: ' + item['type'])
                if method == 'turn/completed':
                    if params['turn']['status'] != 'completed':
                        raise ProtocolError('Actor turn did not complete')
                    break
            if text is None or usage is None:
                raise ProtocolError('Missing final message or actual token usage')
            return {'id': turn['turn']['id'], 'object': 'chat.completion',
                'created': int(time.time()), 'model': started['model'],
                'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': text},
                             'finish_reason': 'stop'}],
                'usage': {'prompt_tokens': usage['inputTokens'], 'completion_tokens': usage['outputTokens'],
                          'total_tokens': usage['totalTokens']},
                'codex_adapter': {'transport': 'app-server-v2', 'requested_model': args.model,
                    'resolved_model': started['model'], 'model_provider': started['modelProvider'],
                    'effort': args.effort,
                    'unapplied_controls': controls, 'raw_token_usage': usage}}
        finally:
            timer.cancel()
            terminate()
            process.wait()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_POST(self):
        if self.path != '/v1/chat/completions':
            self.send_error(404)
            return
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 64 * 1024 * 1024:
                raise ValueError('Invalid actor request size')
            response = infer(json.loads(self.rfile.read(length)), self.server.args)
            status = 200
        except Exception as exc:
            response = {'error': {'type': type(exc).__name__, 'message': str(exc)}}
            status = 502
        data = json.dumps(response).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--port', type=int, default=18093)
    p.add_argument('--codex', default='codex')
    p.add_argument('--effort', default='medium')
    p.add_argument('--model', default=MODEL)
    p.add_argument('--timeout', type=float, default=180)
    p.add_argument('--disable-mcp', action='append', default=[])
    p.add_argument('--acknowledge-unapplied-controls', action='store_true')
    args = p.parse_args()
    server = HTTPServer(('127.0.0.1', args.port), Handler)
    server.args = args
    server.serve_forever()

if __name__ == '__main__':
    main()
