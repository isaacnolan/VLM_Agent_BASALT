"""
QWEN VLM Policy Server - Entry Point

This is the main entry point for the QWEN VLM Policy Server.
The server logic has been modularized into the server/ directory.

To run: python qwen_policy_server.py
"""

if __name__ == "__main__":
    from server.app import app
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
