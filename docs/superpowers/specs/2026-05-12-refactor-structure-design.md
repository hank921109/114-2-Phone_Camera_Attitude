# 設計文件：相機標定專案模組化重構 (Modular Package Refactoring)

## 1. 背景與動圖 (Background)
目前的專案結構較為扁平且雜亂，原始碼與資源檔案（測試影像、輸出結果）混合在根目錄，不利於維護與擴充。本設計旨在將其轉化為標準的 Python 專案結構，提升程式碼的可讀性與組織性。

## 2. 目標結構 (Target Structure)
```plaintext
VanishingPointCameraCalibration/
├── main.py                 # 專案啟動入口
├── requirements.txt        # 依賴庫清單
├── .gitignore              # 版本控制排除清單
├── README.md               # 專案說明
├── src/                    # 原始碼目錄
│   └── vp_calib/           # 核心套件
│       ├── __init__.py
│       ├── engine.py       # 消失點演算法模組 (原 vp_dete_cali.py)
│       └── gui.py          # GUI 介面模組 (原 qt_ui.py)
├── data/                   # 靜態資料目錄
│   ├── samples/            # 測試影像 (原 RealTest/)
│   └── assets/             # UI 靜態資源 (原 pic/)
└── outputs/                # 標定結果輸出目錄 (原 RunResult/)
```

## 3. 變動細節 (Changes)

### 3.1 程式碼重構
*   **路徑管理**：將所有硬編碼的路徑（如 `./RunResult/`）替換為相對於專案根目錄的動態路徑。
*   **模組解耦**：
    *   `engine.py`：封裝 `run_line_ransac` 與 `calculate_camera_attitude` 等核心演算法。
    *   `gui.py`：封裝 PyQt5 類別，負責與使用者互動。
*   **啟動腳本**：`main.py` 負責配置 `sys.path` 並啟動應用程式。

### 3.2 資料與資源
*   將 `RealTest/` 移動至 `data/samples/`。
*   將 `pic/` 中開發相關的影像移動至 `data/assets/`。
*   建立空目錄 `outputs/` 作為標定結果的預設存儲路徑。

### 3.3 依賴與配置
*   產出 `requirements.txt`。
*   更新 `.gitignore` 排除 `outputs/` 中的生成檔案與 `__pycache__`。

## 4. 驗證計畫 (Validation)
*   執行 `python main.py` 確保 GUI 能正常啟動。
*   測試影像開啟功能，確認其能正確讀取 `data/samples/`。
*   執行標定流程，驗證輸出結果是否正確儲存於 `outputs/` 目錄中。
