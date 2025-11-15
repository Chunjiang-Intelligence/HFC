import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import asyncio
from hfc.node.service import NodeService
from hfc.utils.config_loader import load_config
from hfc.utils.logger import setup_logging

def main():
    parser = argparse.ArgumentParser(description="HFC Trainer Node")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the configuration YAML file.")
    parser.add_argument("--node-ip", type=str, required=True, help="IP address of this node (e.g., its ZeroTier IP).")
    parser.add_argument("--node-port", type=int, default=29501, help="Port for this node's RPC server.")
    parser.add_argument("--orchestrator-addr", type=str, required=True, help="Address of the orchestrator (e.g., IP:PORT).")
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(config.log_level)

    orchestrator_host, orchestrator_port = args.orchestrator_addr.split(":")
    
    node_service = NodeService(
        config=config.node,
        node_ip=args.node_ip,
        node_port=args.node_port,
        orchestrator_host=orchestrator_host,
        orchestrator_port=int(orchestrator_port)
    )

    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(node_service.start())
        loop.run_forever()
    except KeyboardInterrupt:
        print("Node shutting down.")
    finally:
        loop.run_until_complete(node_service.stop())
        loop.close()


if __name__ == "__main__":
    main()
