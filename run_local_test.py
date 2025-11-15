import multiprocessing
import os
import sys
import time
import subprocess
import logging

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [LAUNCHER] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

NUM_NODES = 2
ORCHESTRATOR_IP = "127.0.0.1"
ORCHESTRATOR_PORT = 29500
DIST_INIT_PORT = 29400
NODE_BASE_PORT = 29501
CONFIG_FILE = "test_config.yaml"

def run_process(target_script, args, env=None):

    command = [sys.executable, target_script] + args
    
    process_env = os.environ.copy()
    if env:
        process_env.update(env)

    process = subprocess.Popen(command, env=process_env, stdout=sys.stdout, stderr=sys.stderr)
    return process

def main():
    processes = []

    logging.info("--- HFC Local Test Launcher ---")
    logging.info(f"Starting 1 orchestrator and {NUM_NODES} nodes.")

    # --- 1. 設置 torch.distributed 的主節點環境變量 ---
    dist_env = {
        "MASTER_ADDR": ORCHESTRATOR_IP,
        "MASTER_PORT": str(DIST_INIT_PORT)
    }
    logging.info(f"Setting distributed init env: MASTER_ADDR={dist_env['MASTER_ADDR']}, MASTER_PORT={dist_env['MASTER_PORT']}")

    # --- 2. 啟動協調器 ---
    try:
        orch_args = ["--config", CONFIG_FILE]
        orch_process = run_process("run_orchestrator.py", orch_args, env=dist_env)
        processes.append(("Orchestrator", orch_process))
        logging.info("Orchestrator process launched.")
        
        # 等待協調器完全啟動
        time.sleep(5)

        # --- 3. 啟動所有節點 ---
        for i in range(NUM_NODES):
            node_port = NODE_BASE_PORT + i
            node_ip = "127.0.0.1"
            
            node_args = [
                "--config", CONFIG_FILE,
                "--node-ip", node_ip,
                "--node-port", str(node_port),
                "--orchestrator-addr", f"{ORCHESTRATOR_IP}:{ORCHESTRATOR_PORT}"
            ]
            
            node_process = run_process("run_node.py", node_args, env=dist_env)
            processes.append((f"Node-{i}", node_process))
            logging.info(f"Node-{i} process launched (IP: {node_ip}, Port: {node_port}).")
            # 錯開啟動，模擬真實場景
            time.sleep(2)

        logging.info("All processes launched. Monitoring for completion...")
        
        # --- 4. 監控進程 ---
        # 由於訓練步數很少，我們可以等待所有節點進程結束
        while any(p.poll() is None for name, p in processes if "Node" in name):
            time.sleep(5)

        logging.info("All node processes seem to have completed.")

    except Exception as e:
        logging.error(f"An error occurred: {e}")
    finally:
        # --- 5. 清理所有進程 ---
        logging.info("--- Tearing down all processes ---")
        for name, p in processes:
            if p.poll() is None: # 如果進程還在運行
                logging.info(f"Terminating {name} (PID: {p.pid})...")
                p.terminate()
                p.wait(timeout=5) # 等待最多5秒
        logging.info("Cleanup complete.")

if __name__ == "__main__":
    # 設置 multiprocessing 的啟動方法，對於 macOS 和 Windows 是必需的
    multiprocessing.set_start_method("spawn", force=True)
    main()
