import subprocess
import sys
import os
import time
import logging
import multiprocessing
import socket
from contextlib import closing

LOG_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=DATE_FORMAT)
logger = logging.getLogger("LAUNCHER")

NUM_NODES = 2
CONFIG_FILE = "test_config.yaml"
BASE_NODE_RPC_PORT = 29600 # 使用一個新的端口範圍以避免衝突

def find_free_port():
    """查找一個未被佔用的端口。"""
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
        self.logger = self._setup_logger()

    def _setup_logger(self):
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(f"[%(asctime)s] [%(levelname)s] [{self.name}] %(message)s", DATE_FORMAT)
        handler.setFormatter(formatter)
        
        logger_instance = logging.getLogger(self.name)
        logger_instance.setLevel(logging.INFO)
        logger_instance.addHandler(handler)
        logger_instance.propagate = False # 防止日誌重複輸出
        return logger_instance

    def start(self):
        self.logger.info(f"Launching command: {' '.join(self.command)}")
        # 我們將 stdout 和 stderr 都重定向到 PIPE，以便在主進程中處理
        self.process = subprocess.Popen(
            self.command,
            env=self.env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1 # Line-buffered
        )
        
        # 創建一個線程來異步讀取和打印日誌
        self.log_thread = multiprocessing.Process(target=self._log_output, args=(self.process.stdout,))
        self.log_thread.daemon = True
        self.log_thread.start()

    def _log_output(self, pipe):
        # 這個函數在一個單獨的線程中運行
        try:
            for line in iter(pipe.readline, ''):
                self.logger.info(line.strip())
        except Exception as e:
            self.logger.error(f"Error in log reader thread: {e}")
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
            self.log_thread.terminate()
        self.logger.info(f"Process stopped.")

def main():
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
    # 增加 WORLD_SIZE，因為 torch.distributed.rpc 初始化需要它
    # 它應該是所有參與者的總和：1個協調器 + NUM_NODES個節點
    shared_env["WORLD_SIZE"] = str(NUM_NODES + 1)
    
    # 設置 Python 的緩衝，確保日誌及時輸出
    shared_env["PYTHONUNBUFFERED"] = "1"
    
    try:
        orch_command = [
            sys.executable, "run_orchestrator.py",
            "--config", CONFIG_FILE,
        ]
        # 修改 config.yaml 中的端口
        # 這是一個 hack，更優雅的方式是通過命令行傳遞所有配置
        # 但為了簡單，我們在這裡動態修改
        with open(CONFIG_FILE, 'r') as f:
            config_data = f.read()
        config_data = config_data.replace("port: 29500", f"port: {orchestrator_port}")
        
        temp_config_path = "temp_test_config.yaml"
        with open(temp_config_path, 'w') as f:
            f.write(config_data)

        orch_command = [
            sys.executable, "run_orchestrator.py",
            "--config", temp_config_path,
        ]
        
        orchestrator_runner = ProcessRunner("Orchestrator", orch_command, shared_env)
        orchestrator_runner.start()
        processes.append(orchestrator_runner)
        
        logger.info("Waiting for orchestrator to initialize...")
        time.sleep(10) # 等待RPC服務器完全啟動

        for i in range(NUM_NODES):
            node_port = BASE_NODE_RPC_PORT + i
            node_ip = "127.0.0.1"
            
            node_command = [
                sys.executable, "run_node.py",
                "--config", temp_config_path,
                "--node-ip", node_ip,
                "--node-port", str(node_port),
                "--orchestrator-addr", f"{orchestrator_ip}:{orchestrator_port}"
            ]
            
            node_runner = ProcessRunner(f"Node-{i}", node_command, shared_env)
            node_runner.start()
            processes.append(node_runner)
            time.sleep(5) # 錯開節點啟動

        logger.info("All processes launched. Monitoring training progress...")
        
        # 由於訓練步數很少，我們可以等待所有節點進程自然結束
        node_processes = [p for p in processes if "Node" in p.name]
        while any(p.is_alive() for p in node_processes):
            time.sleep(5)
            # 如果協調器意外掛掉，也終止測試
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
        
        # 清理臨時配置文件
        if os.path.exists("temp_test_config.yaml"):
            os.remove("temp_test_config.yaml")

        logger.info("Cleanup complete. Test finished.")


if __name__ == "__main__":
    # 對於 macOS 和 Windows，'spawn' 是更安全的多處理啟動方法
    if sys.platform in ["darwin", "win32"]:
        multiprocessing.set_start_method("spawn", force=True)
    main()
