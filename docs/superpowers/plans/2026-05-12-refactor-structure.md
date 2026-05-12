# 相機標定專案模組化重構實作計畫

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 將專案重構為標準 Python 模組化結構，分離原始碼、資料與輸出。

**Architecture:** 採用 `src/` 佈局，將核心演算法封裝於 `vp_calib.engine`，UI 邏輯封裝於 `vp_calib.gui`，並提供 `main.py` 作為進入點。

**Tech Stack:** Python 3.10, PyQt5, Scikit-Image, NumPy, Matplotlib

---

### Task 1: 建立目錄結構與初始化環境

**Files:**
- Create: `src/vp_calib/__init__.py`
- Create: `data/samples/`
- Create: `data/assets/`
- Create: `outputs/`
- Modify: `.gitignore`

- [ ] **Step 1: 建立新目錄**
- [ ] **Step 2: 建立套件初始化檔案**
- [ ] **Step 3: 更新 .gitignore**
- [ ] **Step 4: Commit**

---

### Task 2: 遷移資源檔案

**Files:**
- Modify: `data/samples/`
- Modify: `data/assets/`

- [ ] **Step 1: 移動測試影像**
- [ ] **Step 2: 移動 UI 資源**
- [ ] **Step 3: Commit**

---

### Task 3: 重構核心演算法模組 (engine.py)

**Files:**
- Create: `src/vp_calib/engine.py` (由 `vp_dete_cali.py` 遷移並修改)
- Modify: `src/vp_calib/engine.py`

- [ ] **Step 1: 遷移內容並修正硬編碼路徑**
- [ ] **Step 2: 刪除舊檔案**
- [ ] **Step 3: Commit**

---

### Task 4: 重構介面模組 (gui.py)

**Files:**
- Create: `src/vp_calib/gui.py` (由 `qt_ui.py` 遷移並修改)
- Modify: `src/vp_calib/gui.py`

- [ ] **Step 1: 修正導入路徑**
- [ ] **Step 2: 修正樣本讀取路徑**
- [ ] **Step 3: 刪除舊檔案**
- [ ] **Step 4: Commit**

---

### Task 5: 建立啟動入口 (main.py)

**Files:**
- Create: `main.py`

- [ ] **Step 1: 撰寫啟動邏輯**
- [ ] **Step 2: Commit**

---

### Task 6: 建立依賴清單與更新文件

**Files:**
- Create: `requirements.txt`
- Modify: `README.md`

- [ ] **Step 1: 生成 requirements.txt**
- [ ] **Step 2: 更新 README.md 中的圖片路徑**
- [ ] **Step 3: Commit**

---

### Task 7: 最終驗證

- [ ] **Step 1: 執行程式**
- [ ] **Step 2: 驗證功能**
