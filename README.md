# HFC Trainer: 面向廣域網的大模型分布式訓練框架

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)

**HFC Trainer** (Hierarchical Federated Clustering Trainer) 是一個專為在**非理想網絡環境**（如廣域網、跨地域、消費級硬件組成的集群）下進行大規模語言模型（LLM）訓練而設計的實驗性分布式訓練框架。

與傳統的、依賴高帶寬/低延遲局域網的數據中心訓練方案不同，HFC Trainer 旨在利用地理上分散的計算資源（例如，由多台 MacBook M4 或帶有消費級 GPU 的 PC 組成的集群），通過創新的分層通信和異步調度策略來克服網絡瓶頸。

## 核心特性

-   分層通信架構
    -   自動測量節點間的網絡延遲，構建網絡拓撲圖。
    -   基於網絡拓撲（例如，使用譜聚類算法）將延遲相近的節點劃分為通信高效的小組（Group）。
    -   在小組內部使用高效的集體通信（如 Ring All-Reduce）聚合梯度；僅在每個組的 Leader 節點之間進行跨公網的稀疏通信，極大減少廣域網帶寬壓力。

-   極致的梯度通信壓縮
    -   GaLore (Gradient Low-Rank Projection)：集成生產級的 [GaLore](https://arxiv.org/abs/2403.03507) 優化器，通過低秩投影從源頭上減少梯度數據的維度。
    -   Top-K 稀疏化：在跨組通信前，僅選取梯度中最重要的 Top-K 更新量進行傳輸。
    -   低精度量化：將稀疏梯度量化為 INT8 等低精度格式，進一步壓縮數據體積。

-   異步 DAG 任務調度
    -   將傳統的串行訓練步驟（`Fwd -> Bwd -> Optim`）分解為一個有向無環圖 (DAG)。
    -   每個操作（如模型的一層前向/反向、梯度通信、參數更新）都是一個獨立的異步任務。
    -   由異步調度器並行執行這些任務，最大化計算、通信和 I/O 的重疊，榨乾異構硬件（CPU/GPU/ANE）的每一分性能。

-   生產級工程實踐
    -   去中心化協調：由一個輕量級的協調器（Orchestrator）負責節點發現、分組和任務啟動，而訓練過程中的通信盡可能是去中心化的。
    -   彈性與容錯：支持節點動態加入，並通過心跳檢測識別掉線節點（未來的版本將支持從斷點恢復）。
    -   與 FSDP 兼*：基於 PyTorch 的完全分片數據並行（FSDP）進行內存管理，使得單個節點無需加載完整模型，極大降低了對單機內存的要求。
    -   零信任網絡支持：完美兼容 [ZeroTier](https://www.zerotier.com/) 或 [Tailscale](https://tailscale.com/)，讓身處不同物理網絡的設備能像在同一個局域網一樣安全、直接地通信。


## 安裝

1.  克隆倉庫
    ```bash
    git clone https://github.com/Chunjiang-Intelligence/HFC.git
    cd HFC
    ```

2.  創建 Python 環境
    建議使用 Conda 或 venv 創建一個新的 Python 3.10+ 環境。
    ```bash
    conda create -n hfc python=3.10
    conda activate hfc
    ```

3.  安裝依賴
    首先，根據你的硬件環境安裝 PyTorch。
    *   **對於 NVIDIA GPU (CUDA):**
        ```bash
        # 請訪問 https://pytorch.org/ 獲取適合你 CUDA 版本的命令
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        ```
    *   **對於 macOS (CPU/MPS):**
        ```bash
        pip install torch torchvision torchaudio
        ```

    然後，安裝本項目的其他依賴：
    ```bash
    pip install -r requirements.txt
    ```
    如果你在 macOS 上，請確保 `bitsandbytes` 沒有被安裝。

## 單機本地測試

在進行多機部署之前，你可以在本地通過多進程模擬一個完整的集群，以驗證系統的端到端流程。

1.  **配置測試環境**
    `test_config.yaml` 文件已經被配置為使用一個非常小的模型 (`distilgpt2`) 和極少的訓練步數，以便快速完成測試。

2.  **運行測試腳本**
    在項目根目錄下，直接運行：
    ```bash
    python run_local_test.py
    ```
    你將會看到一個 `[LAUNCHER]` 進程依次啟動一個 `[Orchestrator]` 和多個 `[Node-N]` 進程。日誌會顯示節點註冊、分組、指令下發和短暫的訓練過程，最後自動清理所有進程。

## 多機廣域網訓練

以下是在真實的多機環境中部署 HFC Trainer 的步驟。

1.  **網絡設置 (重要！)**
    *   在所有機器（協調器和計算節點）上安裝並配置 **ZeroTier** 或 **Tailscale**。
    -   創建一個私有網絡，並確保所有機器都已加入並被授權。
    *   記下每台機器在該虛擬網絡中的 IP 地址。

2.  **配置 `config.yaml`**
    *   打開 `config.yaml` 文件。
    *   修改 `node.model_name_or_path` 為你想要訓練的模型，例如 `meta-llama/Llama-2-7b-hf`。
    *   根據你的需求調整 `max_train_steps`、`learning_rate` 等訓練參數。
    *   確保 `orchestrator.min_nodes_to_start` 符合你的集群規模。

3.  **設置主節點環境變量**
    選擇一台機器作為 `torch.distributed` 的主節點（通常是協調器所在的機器）。在**所有**參與訓練的機器的終端中，設置以下環境變量：
    ```bash
    export MASTER_ADDR=<協調器的ZeroTier IP>
    export MASTER_PORT=29400 # 一個未被佔用的端口
    ```

4.  **啟動協調器 (Orchestrator)**
    在你選定的協調器機器上運行：
    ```bash
    python run_orchestrator.py --config config.yaml
    ```
    協調器將會啟動並等待計算節點的加入。

5.  **啟動計算節點 (Node)**
    在**每一台**計算機器（例如，你的 MacBook M4）上，運行以下命令。確保為每個節點提供**唯一**的 `--node-port`。

    *   在節點 1 上：
        ```bash
        python run_node.py --config config.yaml \
                           --node-ip <本機的ZeroTier IP> \
                           --node-port 29501 \
                           --orchestrator-addr <協調器的ZeroTier IP>:29500
        ```
    *   在節點 2 上：
        ```bash
        python run_node.py --config config.yaml \
                           --node-ip <本機的ZeroTier IP> \
                           --node-port 29502 \
                           --orchestrator-addr <協調器的ZeroTier IP>:29500
        ```
    *   ... 以此類推。

    當註冊的節點數達到 `min_nodes_to_start` 後，協調器將自動開始拓撲測量、分組，並下發訓練指令，整個集群將開始協同訓練。

## 未來方向

HFC Trainer 是一個雄心勃勃的項目，還有許多可以探索和完善的方向：
-   [ ] **全異步通信**：將集體通信操作（如 All-Reduce）分解為更細粒度的 P2P Send/Recv 任務，並集成到 DAG 調度器中，以實現極致的計算/通信重疊。
-   [ ] **動態 K 值與自適應壓縮**：根據訓練階段（例如，訓練初期 vs. 趨於收斂）和網絡帶寬監測，動態調整 Top-K 稀疏化的 `k` 值。
-   [ ] **健壯的容錯與彈性**：實現完整的斷點續傳機制，並能在節點掉線或新節點加入時，動態地重新分組並恢復訓練。
-   [ ] **硬件異構性支持**：優化調度器，使其能夠感知節點的硬件差異（例如，M4 Max vs. M4 Pro），並分配相應的計算負載。
-   [ ] **安全性增強**：在 RPC 通信中加入認證和加密機制。

## 貢獻

歡迎對此項目感興趣的開發者和研究者進行貢獻。你可以通過提交 Pull Request 或創建 Issue 來參與。

## 致謝

本項目的靈感來源於分布式計算、聯邦學習以及對大模型訓練民主化的追求。特別感謝 GaLore 論文的作者們，他們的工作為在有限資源下進行高效訓練提供了可能。
