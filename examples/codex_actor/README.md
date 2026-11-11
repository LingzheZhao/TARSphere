# Codex actor transport

This optional loopback adapter maps the harness's exact system message to Codex
`baseInstructions` and its one user message to ordered text/image inputs.
The evaluator readiness check may omit the system message; that maps to empty
base instructions with the original user request unchanged. The
harness owns history; each decision starts a fresh ephemeral thread and child
app-server. Only inline image observations are accepted. The final assistant
message is returned without extracting or rewriting its decision envelope.

The adapter uses the public Codex app-server v2 protocol and the existing Codex
login. It does not read credentials. It must be launched from a machine with
Codex 0.167.0 and access to the configured model (default `gpt-6.1-terra`). It neither launches simulation nor
submits compute work.

```sh
python examples/codex_actor/server.py --port 18093 \
  --acknowledge-unapplied-controls
```

The adapter reads configured MCP and plugin names from the current Codex home
configuration and disables each name for its child process. Extra server names
can be supplied with `--disable-mcp`. It verifies an MCP inventory with only disabled, idle entries and empty capabilities
before starting a thread. The process
turns off shell execution, apps, web search, image-view tools, sleep tools,
subagents, hooks and memories. Thread environment access and capability roots
are empty. Unexpected tool requests fail the decision and terminate the child.
The loopback endpoint is `/v1/chat/completions`; set the actor model to
`gpt-6.1-terra`. Support-model endpoints remain separate.

Codex reasoning effort defaults to `medium`. Chat Completions `temperature`,
`top_p`, `seed`, token caps and `response_format: {"type":"json_object"}`
are unsupported and are **not applied**. Other response formats are rejected. Requests
containing them fail unless the explicit acknowledgement flag above is set.
Responses report these original controls as `codex_adapter.unapplied_controls`,
together with requested/resolved model, effort, transport and actual token
usage. This is not a sampling-equivalent replacement for another model.
No model or message-body logging is performed by this HTTP adapter. Codex's
own service and client behavior remains that of the installed Codex release.

```sh
python -m unittest discover -s examples/codex_actor -p test_adapter.py -v
```

Configure the actor client separately from support models:

```sh
export TARSPHERE_AGENT_MODEL=gpt-6.1-terra
export TARSPHERE_AGENT_API_BASE=http://127.0.0.1:18093/v1
export TARSPHERE_AGENT_API_KEY=codex-local
```

`codex-local` is a placeholder required by the OpenAI SDK, not a credential.
Codex manages authentication through its existing login. The adapter inherits
the configured Codex provider and reports the resolved provider in its response.
See the official [Codex app-server documentation](https://learn.chatgpt.com/docs/app-server).

Two public synthetic observation checks verified the transport with Codex
0.167.0 and `gpt-6.1-terra` at medium reasoning effort. A direct app-server call
completed in 6.33 seconds with 11,116 input and 139 output tokens; an HTTP
adapter call completed in 6.67 seconds with 11,116 input and 151 output tokens.
Both returned an envelope accepted by the strict decision parser; the HTTP
result also passed `MoveBodyHandler` acceptance. These checks establish model
transport and decision parsing. They do not establish physical execution or
episode success. Raw probe artifacts remain outside the repository.

The actor model can be selected explicitly with `--model`. To select the
`gpt-5.6-sol` default from the author model configuration through the same Codex
account, launch a separate
adapter with `--model gpt-5.6-sol` and set `TARSPHERE_AGENT_MODEL=gpt-5.6-sol`.
The request model must match the adapter selection; thread and turn requests
pin that model and disable provider fallback. The resolved model is checked
and reported. This option does not itself establish an inference or physical
result for Sol; the verification readings above apply to Terra.
