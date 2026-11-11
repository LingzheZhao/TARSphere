"""Minimal line-delimited JSON-RPC lifecycle; never execute server tools."""
import json

class ProtocolError(RuntimeError):
    pass

class RPC:
    def __init__(self, reader, writer):
        self.reader, self.writer, self.ident = reader, writer, 0
        self.notifications = []

    def send(self, message):
        self.writer.write(json.dumps(message, separators=(',', ':')) + '\n')
        self.writer.flush()

    def receive(self):
        line = self.reader.readline()
        if not line:
            raise ProtocolError('App-server closed its stream')
        event = json.loads(line)
        if 'method' in event and 'id' in event:
            # Permission requests are never approval; no dynamic tool executes.
            self.send({'id': event['id'], 'error': {'code': -32601, 'message': 'Actor tools are disabled'}})
            raise ProtocolError('Unexpected server request: ' + event['method'])
        return event

    def request(self, method, params):
        self.ident += 1
        ident = self.ident
        self.send({'id': ident, 'method': method, 'params': params})
        while True:
            event = self.receive()
            if event.get('id') == ident:
                if 'error' in event:
                    raise ProtocolError(str(event['error']))
                return event['result']
            if 'method' in event:
                self.notifications.append(event)
            else:
                raise ProtocolError('Unexpected RPC response identity')

    def notify(self, method, params=None):
        event = {'method': method}
        if params is not None:
            event['params'] = params
        self.send(event)
