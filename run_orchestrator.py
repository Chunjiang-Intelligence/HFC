import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import asyncio
from hfc.orchestrator.rpc_server import OrchestratorRPCServer
from hfc.orchestrator.service import OrchestratorService
from utils.config_loader import load_config
from utils.logger import setup_logging

def main():
    parser = argparse.ArgumentParser(description="HFC Trainer Orchestrator")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the configuration YAML file.")
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(config.log_level)

    service = OrchestratorService(config.orchestrator)
    rpc_server = OrchestratorRPCServer(service, host=config.orchestrator.host, port=config.orchestrator.port)
    
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(rpc_server.start())
        loop.run_forever()
    except KeyboardInterrupt:
        print("Orchestrator shutting down.")
    finally:
        loop.run_until_complete(rpc_server.stop())
        loop.close()

if __name__ == "__main__":
    main()
