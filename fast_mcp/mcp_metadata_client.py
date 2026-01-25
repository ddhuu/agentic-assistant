import mcp
from mcp.client.stdio import stdio_client
from mcp import StdioServerParameters
import asyncio
from typing import Dict, Any, Optional

# Configure the MCP server parameters
params = StdioServerParameters(
    command="python",
    args=["-m", "mcp_metadata_server"],
    env=None
)

class MetadataClient:
    """Client for interacting with the MCP Metadata Server."""
    
    @staticmethod
    async def _call_tool(tool_name: str, tool_args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call a tool on the MCP server.
        
        Args:
            tool_name: Name of the tool to call
            tool_args: Arguments to pass to the tool
            
        Returns:
            Dict containing the tool's response
        """
        async with stdio_client(params) as streams:
            async with mcp.ClientSession(*streams) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, tool_args)
                return result
    
    @classmethod
    async def save_metadata(
        cls,
        filename: str,
        label: str,
        content: Optional[str] = None,
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Save file metadata to the MCP server.
        
        Args:
            filename: Name of the file
            label: Classification label
            content: Optional file content or summary
            additional_metadata: Any additional metadata
            
        Returns:
            Dict containing the save operation result
        """
        return await cls._call_tool(
            "save_metadata_to_json",
            {
                "filename": filename,
                "label": label,
                "content": content,
                "additional_metadata": additional_metadata or {}
            }
        )
    
    @classmethod
    async def get_metadata(cls, metadata_id: str) -> Dict[str, Any]:
        """
        Retrieve metadata by ID.
        
        Args:
            metadata_id: ID of the metadata to retrieve
            
        Returns:
            Dict containing the metadata or error message
        """
        return await cls._call_tool("get_metadata", {"metadata_id": metadata_id})
    
    @classmethod
    async def search_metadata(
        cls,
        filename: Optional[str] = None,
        label: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Search for metadata by filename or label.
        
        Args:
            filename: Optional filename to filter by
            label: Optional label to filter by
            
        Returns:
            Dict containing matching metadata entries
        """
        return await cls._call_tool("search_metadata", {
            "filename": filename,
            "label": label
        })

# Example usage
async def example_usage():
    # Save some metadata
    save_result = await MetadataClient.save_metadata(
        filename="document.pdf",
        label="Giáo dục",
        content="Tài liệu về giáo dục",
        additional_metadata={"department": "Education", "year": 2024}
    )
    print("Save result:", save_result)
    
    # Search for metadata
    search_result = await MetadataClient.search_metadata(label="Giáo dục")
    print("\nSearch result:", search_result)

if __name__ == "__main__":
    asyncio.run(example_usage())
