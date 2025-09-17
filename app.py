import asyncio
import sys
import os
import json
import warnings
from dotenv import load_dotenv
from typing import List, Optional, Any, Dict
from contextlib import AsyncExitStack
import google.generativeai as genai
import google.generativeai.types as genai_types

from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.history import InMemoryHistory

# --- Suppress Windows asyncio pipe warnings ---
warnings.filterwarnings("ignore", category=ResourceWarning)


# --- Gemini Service ---
class GeminiService:
    def __init__(self, model: str):
        self.model = model
        self.client = genai.GenerativeModel(self.model)

    def add_user_message(self, messages: list, message: Any):
        if isinstance(message, list):
            user_message = {"role": "user", "parts": message}
        elif isinstance(message, dict) and 'functionResponse' in message:
            user_message = {"role": "user", "parts": [message]}
        else:
            content = message.text if hasattr(message, 'text') else message
            user_message = {"role": "user", "parts": [content]}
        messages.append(user_message)

    def add_assistant_message(self, messages: list, message: Any):
        assistant_message = {"role": "model", "parts": message.parts}
        messages.append(assistant_message)

    def text_from_message(self, message: genai_types.GenerateContentResponse) -> str:
        text_parts = [
            part.text
            for part in message.parts
            if hasattr(part, 'text') and part.text
        ]
        return "\n".join(text_parts)

    def chat(self, messages, tool_config: Optional[list] = None) -> genai_types.GenerateContentResponse:
        response = self.client.generate_content(
            contents=messages,
            tools=tool_config,
            generation_config=genai_types.GenerationConfig(temperature=1.0)
        )
        return response


# --- MCP Client ---
class MCPClient:
    def __init__(self, command: str, args: list[str], env: Optional[dict] = None):
        self._command = command
        self._args = args
        self._env = env
        self._session: Optional[ClientSession] = None
        self._exit_stack: AsyncExitStack = AsyncExitStack()

    async def connect(self):
        server_params = StdioServerParameters(command=self._command, args=self._args, env=self._env)
        stdio_transport = await self._exit_stack.enter_async_context(stdio_client(server_params))
        _stdio, _write = stdio_transport
        self._session = await self._exit_stack.enter_async_context(ClientSession(_stdio, _write))
        await self._session.initialize()

    def session(self) -> ClientSession:
        if self._session is None:
            raise ConnectionError("Client session not initialized. Call connect() first.")
        return self._session

    async def list_tools(self) -> list[types.Tool]:
        result = await self.session().list_tools()
        return result.tools

    async def call_tool(self, tool_name: str, tool_input: dict) -> types.CallToolResult | None:
        return await self.session().call_tool(tool_name, tool_input)

    async def read_resource(self, uri: str) -> Any:
        return await self.session().read_resource(uri)

    async def list_prompts(self) -> list[types.Prompt]:
        return await self.session().list_prompts()
    
    async def get_prompt(self, prompt_name: str, args: dict[str, str]):
        return await self.session().get_prompt(prompt_name, args)

    async def cleanup(self):
        await self._exit_stack.aclose()
        self._session = None

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup()


# --- Chat Agent ---
class CliChat:
    def __init__(self, gemini_service: GeminiService, doc_client: Any, clients: dict[str, Any]):
        self.gemini_service = gemini_service
        self.doc_client = doc_client
        self.clients = clients
        self.messages: List[Dict[str, Any]] = []

    async def _extract_resources(self, query: str) -> List[dict]:
        mentions = [word[1:] for word in query.split() if word.startswith("@")]
        if not mentions:
            return []
        try:
            resources = await self.doc_client.read_resource("docs://documents")
            if hasattr(resources, "contents") and resources.contents:
                resources = json.loads(resources.contents[0].text)

            mentioned_docs = []
            for mention in mentions:
                if mention in resources:
                    content = await self.doc_client.read_resource(f"docs://documents/{mention}")
                    if hasattr(content, "contents") and content.contents:
                        text_content = content.contents[0].text
                        mentioned_docs.append((mention, text_content))
            return [{"text": f'\n<document id="{doc_id}">\n{content}\n</document>\n'} for doc_id, content in mentioned_docs]
        except Exception as e:
            print(f"Error extracting resources: {e}")
            return []

    async def run(self, query: str) -> str:
        added_resources = await self._extract_resources(query) if "@" in query else []
        parts = [{"text": query}] + added_resources
        self.messages.append({"role": "user", "parts": parts})
        response = self.gemini_service.chat(messages=self.messages)

        text = self.gemini_service.text_from_message(response)
        if text and text.strip():
            return text
        return "The model did not return usable text."


# --- Unified Completer for / and @ ---
class UnifiedCompleter(Completer):
    def __init__(self):
        self.prompts = []
        self.resources = []

    def update_prompts(self, prompts: List): self.prompts = prompts
    def update_resources(self, resources: List): self.resources = resources

    def get_completions(self, document, complete_event):
        text_before_cursor = document.text_before_cursor
        if "@" in text_before_cursor:
            prefix = text_before_cursor[text_before_cursor.rfind("@") + 1:]
            for resource_id in self.resources:
                if resource_id.lower().startswith(prefix.lower()):
                    yield Completion(resource_id, start_position=-len(prefix), display=resource_id)
            return
        if text_before_cursor.startswith("/"):
            cmd_prefix = text_before_cursor[1:]
            for prompt in self.prompts:
                pname = getattr(prompt, "name", None) or (prompt[0] if isinstance(prompt, (tuple, list)) else None)
                if pname and pname.startswith(cmd_prefix):
                    yield Completion(pname, start_position=-len(cmd_prefix), display=f"/{pname}")


# --- CLI App ---
class CliApp:
    def __init__(self, agent: CliChat):
        self.agent = agent
        self.completer = UnifiedCompleter()
        self.history = InMemoryHistory()
        self.kb = KeyBindings()
        self.session = PromptSession(completer=self.completer, history=self.history, key_bindings=self.kb)

        @self.kb.add("@")
        def _(event):
            buffer = event.app.current_buffer
            buffer.insert_text("@")
            buffer.start_completion(select_first=False)

        @self.kb.add("/")
        def _(event):
            buffer = event.app.current_buffer
            buffer.insert_text("/")
            buffer.start_completion(select_first=False)

    async def initialize(self):
        try:
            resources = await self.agent.doc_client.read_resource("docs://documents")
            if hasattr(resources, "contents") and resources.contents:
                resources = json.loads(resources.contents[0].text)

            if isinstance(resources, list):
                print("Available documents:")
                for resource in resources:
                    print(f"  - {resource}")
                self.completer.update_resources(resources)

            prompts_resp = await self.agent.doc_client.list_prompts()
            filtered_prompts = []

            # Works for both ListPromptsResult and dict
            raw_prompts = getattr(prompts_resp, "prompts", prompts_resp) or []

            if raw_prompts:
                print("\nAvailable prompts:")
                for prompt in raw_prompts:
                    if hasattr(prompt, "name") and hasattr(prompt, "description"):
                        print(f"  - /{prompt.name}: {prompt.description}")
                        filtered_prompts.append(prompt)
                    elif isinstance(prompt, dict) and "name" in prompt:
                        name, desc = prompt["name"], prompt.get("description", "")
                        print(f"  - /{name}: {desc}")
                        filtered_prompts.append(types.Prompt(name=name, description=desc, arguments=[]))

                self.completer.update_prompts(filtered_prompts)


        except Exception as e:
            print(f"Error during initialization: {e}")

    async def run(self):
        print("\nUse @document_name to reference documents, or /prompt_name for prompts.")
        print("Type your queries below (Ctrl+C to exit):\n")
        while True:
            try:
                user_input = await self.session.prompt_async("> ")
                if not user_input.strip():
                    continue
                response = await self.agent.run(user_input)
                print(f"\nResponse:\n{response}\n")
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}\n")


# --- Main ---
async def main():
    load_dotenv()
    gemini_model = os.getenv("GEMINI_MODEL", "")
    gemini_api_key = os.getenv("GEMINI_API_KEY", "")
    assert gemini_model, "Error: GEMINI_MODEL cannot be empty."
    assert gemini_api_key, "Error: GEMINI_API_KEY cannot be empty."
    genai.configure(api_key=gemini_api_key)

    gemini_service = GeminiService(model=gemini_model)
    clients = {}
    command, args = ("uv", ["run", "mcp_server.py"]) if os.getenv("USE_UV", "0") == "1" else ("python", ["mcp_server.py"])

    async with AsyncExitStack() as stack:
        doc_client = await stack.enter_async_context(MCPClient(command=command, args=args))
        clients["doc_client"] = doc_client
        chat = CliChat(gemini_service=gemini_service, doc_client=doc_client, clients=clients)
        cli = CliApp(chat)
        await cli.initialize()
        await cli.run()


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    asyncio.run(main())
