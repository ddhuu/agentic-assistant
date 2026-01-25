import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from langgraph.prebuilt import create_react_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from config.llm import gemini
from config.prompt import filesystem_agent_prompt
from agents.base import BaseAgent
from schemas.agent_schema import ResponseFormat
from langgraph.checkpoint.memory import MemorySaver
import asyncio
from utils.get_agent_response import get_agent_response
from typing import Any, AsyncIterable, Dict
from langchain_core.messages import AIMessage, HumanMessage



class FilesystemAgent(BaseAgent):
    def __init__(self, graph):
        super().__init__(
            agent_name='FilesystemAgent',
            description='Interact with the file system to find and manage files',
            content_types=['file', 'directory']
        )
        self.graph = graph
        
    @property
    def name(self):
        return self.agent_name

    @classmethod
    async def create(cls, mcp_client=None):
        """Create FilesystemAgent with MCP client and tools
        
        Args:
            mcp_client: Optional MCP client to use. If None, a new client will be created.
        """
        try:
        
            if mcp_client is None:
                print("Creating new MCP client for FilesystemAgent")
                client = MultiServerMCPClient({
                    "document_search": {
                        "command": "cmd",
                        "args": [
                            "/c",
                            "npx",
                            "-y",
                            "@modelcontextprotocol/server-filesystem",
                            "C:\\Users\\dhuu3\\Desktop\\local-classify-docs-ai-agent\\data",
                        ],
                        "transport": "stdio",
                    }
                })
            else:
                print("Using provided MCP client for FilesystemAgent")
                client = mcp_client
            
            # MemorySaver
            memory = MemorySaver()
            
            # Get tools from MCP client
            tools = await client.get_tools()
            
            # Create the graph
            graph = create_react_agent(
                model=gemini,
                tools=tools,
                prompt=filesystem_agent_prompt,
                name="Filesystem Agent",
                response_format=ResponseFormat,
                checkpointer=memory,
            )
            
            return cls(graph)
            
        except Exception as e:
            print(f"Error creating FilesystemAgent: {e}")
            raise
    
    async def run(self, query: str, session_id: str = "default"):
        """Run the agent with a query"""
        try:
            config = {"recursion_limit": 50, "configurable": {"thread_id": session_id}}
            response = await self.graph.ainvoke(
                {"messages": [HumanMessage(content=query)]},
                config=config
            )
            
            # Get response and format it
            result = get_agent_response(self.graph, response['messages'][-1], config)
            
            # Always return dict format for consistency
            if isinstance(result, str):
                return {
                    'response_type': 'data',
                    'is_task_complete': True,
                    'require_user_input': False,
                    'content': result
                }
            return result
            
        except Exception as e:
            print(f"Error running FilesystemAgent: {e}")
            return {
                'response_type': 'error',
                'is_task_complete': True,
                'require_user_input': False,
                'content': f"Error: {str(e)}"
            }
        
    async def stream(
        self, query: str, sessionId: str, task_id: str = None
    ) -> AsyncIterable[Dict[str, Any]]:
        """Stream responses from the agent"""
        try:
            inputs = {'messages': [HumanMessage(content=query)]}
            config = {'configurable': {'thread_id': sessionId}, 'recursion_limit': 50}

            async for item in self.graph.astream(inputs, config, stream_mode='values'):
                message = item['messages'][-1]
                if isinstance(message, AIMessage):
                    yield {
                        'response_type': 'text',
                        'is_task_complete': False,
                        'require_user_input': False,
                        'content': message.content,
                    }
        except Exception as e:
            yield {
                'response_type': 'error',
                'is_task_complete': True,
                'require_user_input': False,
                'content': f"Error in streaming: {str(e)}",
            }
    
    async def ainvoke(self, input_data, config=None):
        """Required method for langgraph-supervisor compatibility - FIXED VERSION"""
        try:
            if config is None:
                config = {"configurable": {"thread_id": "default"}}
                
            # Extract the message content from the input data
            messages = input_data.get("messages", [])
            if not messages:
                return {"messages": [AIMessage(content="No input provided")]}
            
            last_message = messages[-1]
            
            # Handle different message types
            if hasattr(last_message, 'content'):
                query = last_message.content
            elif isinstance(last_message, dict) and 'content' in last_message:
                query = last_message['content']
            elif isinstance(last_message, tuple) and len(last_message) == 2:
                query = last_message[1]
            else:
                query = str(last_message)
                
            session_id = config.get("configurable", {}).get("thread_id", "default")
            

            result = await self.run(query, session_id)
            
            # Format response cho supervisor
            if isinstance(result, dict) and 'content' in result:
                content = result['content']
                
                # Extract file paths if present
                import re
                file_pattern = r'[A-Z]:\\(?:[^\\/:*?"<>|\r\n]+\\)*[^\\/:*?"<>|\r\n]*\.[a-zA-Z0-9]+'
                file_matches = re.findall(file_pattern, content)
                
                if file_matches:
                    # Clean up paths
                    cleaned_paths = [path.replace('\\\\', '\\') for path in file_matches]
                    if len(cleaned_paths) == 1:
                        formatted_content = f"Tôi đã tìm thấy file: {cleaned_paths[0]}"
                        return {"messages": [AIMessage(content=formatted_content)]}
                    else:
                        formatted_content = f"Tôi đã tìm thấy {len(cleaned_paths)} files:\n"
                        for i, path in enumerate(cleaned_paths, 1):
                            formatted_content += f"{i}. {path}\n"
                        return {"messages": [AIMessage(content=formatted_content)]}
                
                # Check if we need to force a direct search
                if result.get('require_user_input', False):
                    # If we're still getting a require_user_input response, force a direct file search
                    print("[DEBUG] Forcing direct file search")
                    direct_result = await self.run("List all files in the data directory related to LLM", session_id)
                    if isinstance(direct_result, dict) and 'content' in direct_result:
                        return {"messages": [AIMessage(content=direct_result['content'])]}
                    else:
                        return {"messages": [AIMessage(content=str(direct_result))]}
                
                # Default case: return the content
                return {"messages": [AIMessage(content=content)]}
            elif isinstance(result, dict):
                return {"messages": [AIMessage(content=str(result))]}
            else:
                return {"messages": [AIMessage(content=str(result))]}
        except Exception as e:
            print(f"Error in FilesystemAgent.ainvoke: {e}")
            return {"messages": [AIMessage(content=f"Lỗi khi tìm kiếm file: {str(e)}")]}


async def main():
    """Test function for the FilesystemAgent"""
    try:
        agent = await FilesystemAgent.create()
        result = await agent.run("Tìm file có tên liên quan đến finance", session_id="123")
        print("Result:", result)
    except Exception as e:
        print(f"Error in main: {e}")


if __name__ == "__main__":
    asyncio.run(main())