import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

import subprocess
import sys
import os
import time
import logging
import multiprocessing
import socket
from contextlib import closing
import threading 

LOG_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=DATE_FORMAT)
logger = logging.getLogger("LAUNCHER")

NUM_NODES = 2
CONFIG_FILE = "test_config.yaml"
BASE_NODE_RPC_PORT = 29600

def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]

class ProcessRunner:
    def __init__(self, name, command, env):
        self.name = name
        self.command = command
        self.env = env
        self.process = None
        self.log_thread = None
        
        self.logger = logging.getLogger(self.name)

        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(f"[%(asctime)s] [%(levelname)s] [{self.name}] %(message)s", DATE_FORMAT)
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
            self.logger.propagate = False

    def start(self):
        self.logger.info(f"Launching command: {' '.join(self.command)}")
        self.process = subprocess.Popen(
            self.command,
            env=self.env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        self.log_thread = threading.Thread(target=self._log_output, args=(self.process.stdout,))
        self.log_thread.daemon = True
        self.log_thread.start()

    def _log_output(self, pipe):
        try:
            for line in iter(pipe.readline, ''):
                # 使用主進程的 logger 實例來打印
                self.logger.info(line.strip())
        except Exception as e:
            # 在主進程中記錄線程錯誤
            logger.error(f"Log reader thread for {self.name} failed: {e}")
        finally:
            pipe.close()

    def is_alive(self):
        return self.process is not None and self.process.poll() is None

    def terminate(self):
        if self.is_alive():
            self.logger.warning(f"Terminating process...")
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.logger.error(f"Process did not terminate gracefully, killing.")
                self.process.kill()
        
        if self.log_thread and self.log_thread.is_alive():
            # 線程是守護線程，會隨主進程退出，無需手動終止
            pass
        self.logger.info(f"Process stopped.")

def main():
    """
    主測試函數，用於啟動和管理所有進程。
    """
    processes = []
    
    logger.info("Finding free ports for test environment...")
    orchestrator_ip = "127.0.0.1"
    orchestrator_port = find_free_port()
    dist_init_port = find_free_port()
    
    logger.info(f"Orchestrator RPC Port: {orchestrator_port}")
    logger.info(f"Distributed Init Port: {dist_init_port}")

    shared_env = os.environ.copy()
    shared_env["MASTER_ADDR"] = orchestrator_ip
    shared_env["MASTER_PORT"] = str(dist_init_port)
    shared_env["WORLD_SIZE"] = str(NUM_NODES + 1)
    shared_env["PYTHONUNBUFFERED"] = "1"
    
    # 創建臨時配置文件
    with open(CONFIG_FILE, 'r') as f:
        config_data = f.read()
    config_data = config_data.replace("port: 29500", f"port: {orchestrator_port}")
    temp_config_path = "temp_test_config.yaml"
    with open(temp_config_path, 'w') as f:
        f.write(config_data)

    try:
        # 啟動協調器
        orch_command = [sys.executable, "run_orchestrator.py", "--config", temp_config_path]
        orchestrator_runner = ProcessRunner("Orchestrator", orch_command, shared_env)
        orchestrator_runner.start()
        processes.append(orchestrator_runner)
        
        logger.info("Waiting for orchestrator to initialize...")
        time.sleep(10)

        # 啟動節點
        for i in range(NUM_NODES):
            node_port = BASE_NODE_RPC_PORT + i
            node_ip = "127.0.0.1"
            
            node_command = [
                sys.executable, "run_node.py", "--config", temp_config_path,
                "--node-ip", node_ip, "--node-port", str(node_port),
                "--orchestrator-addr", f"{orchestrator_ip}:{orchestrator_port}"
            ]
            
            node_runner = ProcessRunner(f"Node-{i}", node_command, shared_env)
            node_runner.start()
            processes.append(node_runner)
            time.sleep(5)

        logger.info("All processes launched. Monitoring training progress...")
        
        node_processes = [p for p in processes if "Node" in p.name]
        while any(p.is_alive() for p in node_processes):
            time.sleep(5)
            if not processes[0].is_alive():
                logger.error("Orchestrator process died unexpectedly. Tearing down.")
                break

        logger.info("Test finished. All node processes have completed their tasks.")

    except KeyboardInterrupt:
        logger.warning("Test interrupted by user.")
    except Exception as e:
        logger.error(f"An error occurred during test execution: {e}", exc_info=True)
    finally:
        logger.info("--- Tearing down all processes ---")
        for runner in reversed(processes):
            runner.terminate()
        
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)

        logger.info("Cleanup complete. Test finished.")


if __name__ == "__main__":
    if sys.platform in ["darwin", "win32"] and multiprocessing.get_start_method() != "spawn":
        multiprocessing.set_start_method("spawn", force=True)
    main()
