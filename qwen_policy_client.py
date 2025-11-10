"""
QWEN VLM Policy Client - Entry Point

This is the main entry point for the QWEN VLM Policy Client.
The client logic has been modularized into the client/ directory.

To run: python qwen_policy_client.py --task <task_name>
"""

if __name__ == "__main__":
    from client.run_agent import main
    import sys
    sys.exit(main())
