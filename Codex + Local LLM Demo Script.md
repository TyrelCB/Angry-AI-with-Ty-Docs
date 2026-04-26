# Codex + Local LLM Demo Script

## Opening

Today I want to show something pretty wild: a fully local coding workflow where a terminal agent talks to a local model stack, uses a local web search tool, and stays mostly on hardware I control.

The goal for this session was to wire Codex to llama.cpp, give the local model web access through MCP, and compare that experience with OpenCode.

## Part 1: The original problem

The first issue was that Codex and llama.cpp do not speak exactly the same API.

Codex now expects the OpenAI **Responses API**.

Upstream llama.cpp exposes **Chat Completions**.

So if you point Codex directly at vanilla llama.cpp, it does not just work.

## Part 2: The bridge

To fix that, I built a local adapter.

That adapter listens on a local port and accepts Codex Responses API calls.

Then it translates them into llama.cpp chat completion requests, forwards them to the Qwen 3.6 model, and maps the response back into the shape Codex expects.

So the live flow became:

1. Codex talks to a local proxy
2. the proxy talks to llama.cpp
3. llama.cpp serves the local Qwen 3.6 model

That made `codex --profile llama` work against the local model stack.

## Part 3: What model we used

The model in this setup is:

`Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`

And it is served through llama.cpp on port `9090`.

So at this point, Codex could already reason through the local model instead of relying on the hosted default path.

## Part 4: Adding web access

The next challenge was web access.

And this is where it gets interesting.

Codex has built-in web search as a product feature, but that is not the same thing as giving a local model its own local web tool chain.

So instead of relying on hosted search, I built a **local MCP server** called `local_web`.

That server exposes two tools:

- `search_web`
- `fetch_url`

And it is backed by a local **SearxNG** instance running on `127.0.0.1:18910`.

So now the search path is:

1. the agent asks the MCP server to search
2. the MCP server queries local SearxNG
3. the MCP server returns structured web results
4. the agent can also fetch and read pages through `fetch_url`

## Part 5: What worked

The MCP server itself worked.

We confirmed:

- SearxNG search works
- the MCP server exposes both tools correctly
- direct MCP client calls to `search_web` and `fetch_url` succeed

Inside Codex, the `/mcp` screen also showed the server correctly.

It listed:

- `local_web`
- `search_web`
- `fetch_url`

So the server was definitely registered and visible.

## Part 6: The Codex twist

But then we hit the real plot twist.

Even though Codex could see the MCP server in its MCP inventory, the model session itself still acted like those tools were unavailable during real turns.

So the server existed.

The tools existed.

The UI showed them.

But the actual model turn still said, essentially, “I don’t have those tools.”

That means the control plane was working, but the tools were not getting mounted into the active model session correctly.

And that appears to match current open Codex bugs around MCP tools disappearing when you use a custom provider.

So the short version is:

- Codex sees the MCP server
- Codex does not reliably hand those MCP tools to the local-model session

## Part 7: Testing the same idea in OpenCode

To make sure the MCP server was not the problem, I tested the exact same local web tool through **OpenCode**.

I added the `local_web` MCP server to OpenCode.

Then I added the Qwen 3.6 model under the llama provider as:

`llama/qwen3.6-35b`

And when I ran OpenCode with that model, it actually called the web tool.

Not in theory.

Not just in a status screen.

It really used the tool.

It invoked:

`local_web_search_web`

And returned the expected repository URL for the OpenAI Codex GitHub repo.

So that was the key proof point:

the local MCP server works, the local search backend works, and a local-model agent can use them successfully.

## Part 8: The conclusion

By the end of the session, we had four clear results.

First, **Codex plus llama.cpp** works through a local Responses adapter.

Second, the **local web MCP server** works.

Third, **OpenCode plus Qwen 3.6 plus local_web** works end to end.

Fourth, **Codex plus custom local provider plus MCP tool use** is still blocked by what looks like a current Codex bug.

## Closing

So if I wanted a dramatic one-line ending for this demo, it would be this:

We got the local model stack working, we got local web tools working, and we proved the idea works end to end — but in Codex, the last mile is currently blocked not by the model, not by MCP, and not by SearxNG, but by Codex’s own custom-provider MCP mounting behavior.

And that is exactly why this was such a useful session.
