from medical_science.src.mcp.protocol import MCPMessage

if __name__ == "__main__":
    msg = MCPMessage(sender="agentA", recipient="agentB", content="Hello")
    print(msg) 