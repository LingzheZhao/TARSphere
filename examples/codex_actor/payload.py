"""Lossless actor payload mapping for Codex app-server v2.

Only the harness's system + one user shape is accepted. No implicit history,
role flattening, sampling controls, or JSON instructions are added.
"""
def map_actor_payload(payload):
    messages = payload['messages']
    if len(messages) != 2 or [m['role'] for m in messages] != ['system', 'user']:
        raise ValueError('Expected exactly system + one user; history belongs to harness')
    system = messages[0]['content']
    if not isinstance(system, str):
        raise ValueError('System content must be text')
    content = messages[1]['content']
    if isinstance(content, str):
        content = [{'type': 'text', 'text': content}]
    items = []
    for part in content:
        if part['type'] == 'text':
            items.append({'type': 'text', 'text': part['text']})
        elif part['type'] == 'image_url':
            source = part['image_url']
            if isinstance(source, str):
                source = {'url': source}
            # Inline observations only: never fetch paths or remote resources.
            if not source['url'].startswith('data:image/'):
                raise ValueError('Actor observations must be inline image data')
            item = {'type': 'image', 'url': source['url']}
            if 'detail' in source:
                item['detail'] = source['detail']
            items.append(item)
        else:
            raise ValueError('Unsupported actor content: ' + part['type'])
    if payload.get('tools') or payload.get('stream'):
        raise ValueError('Tool and streaming requests are unsupported')
    return system, items
