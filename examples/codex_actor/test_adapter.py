import io
import json
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from payload import map_actor_payload
from rpc import RPC, ProtocolError
from server import command, infer

class AdapterTests(unittest.TestCase):
    def test_actor_order_and_exact_system(self):
        content = [{'type': 'text', 'text': 'before'},
                   {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,AA==', 'detail': 'high'}},
                   {'type': 'text', 'text': 'after'}]
        system, items = map_actor_payload({'messages': [
            {'role': 'system', 'content': 'public system\n'}, {'role': 'user', 'content': content}]})
        self.assertEqual(system, 'public system\n')
        self.assertEqual([i['type'] for i in items], ['text', 'image', 'text'])
        self.assertEqual(items[1]['url'], content[1]['image_url']['url'])

    def test_rejects_role_flattening_and_remote_images(self):
        for messages in ([{'role': 'assistant', 'content': 'hidden'}],
            [{'role': 'system', 'content': 's'}, {'role': 'user', 'content': [
                {'type': 'image_url', 'image_url': {'url': 'file:///secret'}}]}]):
            with self.assertRaises(ValueError):
                map_actor_payload({'messages': messages})

    def test_notification_before_response(self):
        reader = io.StringIO('{"method":"thread/started","params":{}}\n{"id":1,"result":{"ok":true}}\n')
        rpc = RPC(reader, io.StringIO())
        self.assertEqual(rpc.request('thread/start', {}), {'ok': True})
        self.assertEqual(len(rpc.notifications), 1)

    def test_approval_never_executes(self):
        writer = io.StringIO()
        rpc = RPC(io.StringIO('{"id":8,"method":"item/commandExecution/requestApproval"}\n'), writer)
        with self.assertRaises(ProtocolError):
            rpc.receive()
        self.assertEqual(json.loads(writer.getvalue())['error']['code'], -32601)

    def test_real_event_shape_preserves_final_message_and_usage(self):
        final = '{"decision": "unchanged"}'
        events = [
            {'id': 1, 'result': {}},
            {'id': 2, 'result': {'data': [{'tools': {}, 'resources': [], 'resourceTemplates': [], 'runtimeStatus': None}]}},
            {'id': 3, 'result': {'model': 'gpt-6.1-terra', 'modelProvider': 'openai_https', 'instructionSources': [], 'thread': {'id': 'thread-test'}}},
            {'id': 4, 'result': {'turn': {'id': 'turn-test'}}},
            {'method': 'item/started', 'params': {'item': {'type': 'agentMessage', 'phase': 'final_answer', 'text': ''}}},
            {'method': 'item/completed', 'params': {'item': {'type': 'agentMessage', 'phase': 'final_answer', 'text': final}}},
            {'method': 'thread/tokenUsage/updated', 'params': {'tokenUsage': {'last': {'inputTokens': 11116, 'outputTokens': 139, 'totalTokens': 11255, 'cachedInputTokens': 0, 'reasoningOutputTokens': 0}}}},
            {'method': 'turn/completed', 'params': {'turn': {'status': 'completed'}}},
        ]
        process = SimpleNamespace(pid=999999999, stdin=io.StringIO(),
            stdout=io.StringIO(''.join(json.dumps(e) + '\n' for e in events)), wait=lambda: None)
        args = SimpleNamespace(codex='codex', disable_mcp=[], timeout=10, effort='medium', acknowledge_unapplied_controls=True)
        payload = {'model': 'gpt-6.1-terra', 'messages': [{'role': 'system', 'content': 'public'}, {'role': 'user', 'content': 'observation'}], 'temperature': 0}
        with patch('server.subprocess.Popen', return_value=process), patch('server.os.killpg'):
            response = infer(payload, args)
        self.assertEqual(response['choices'][0]['message']['content'], final)
        self.assertEqual(response['usage']['total_tokens'], 11255)
        self.assertEqual(response['codex_adapter']['model_provider'], 'openai_https')
        self.assertEqual(response['codex_adapter']['unapplied_controls'], {'temperature': 0})
        payload['unknown_control'] = 1
        with self.assertRaises(ValueError):
            infer(payload, args)

    def test_dotted_mcp_name_rejected(self):
        with self.assertRaises(ValueError):
            command('codex', ['unsafe.name'])

    def test_process_config_disables_capabilities(self):
        cmd = command('codex', ['determined-compute'])
        self.assertIn('features.shell_tool=false', cmd)
        self.assertIn('mcp_servers.determined-compute.enabled=false', cmd)
        self.assertIn('web_search="disabled"', cmd)

if __name__ == '__main__':
    unittest.main()
