~~@src/vp_calib/@main.py 1. 處理 的影片 左上角要印 FPS  2. [Image            
   rt1_axes.png] 左上角的文字要放大2倍  3. 簡查codeʼ以表列建議~~
   
   
   
~~[Image result_rt4.jpg] [Image result_rt7.jpg] 發現 座標起點都在中間(錯誤)  
   ，應當讓 影像下半部的 RANSEC 3 種線段 的 相交點 視為 地平線起點(座標起點)~~	
   
   
~~3. 幫我 git push, msg :  室內圖 座標起點都正常 除了 outputs/result_rt0.jpg ，空拍ʼoffice 的 RANSEC 
不準確，.....~~

### 5/24
~~1. 檢查 @README.md Breakdown 的模組名稱 要與 dataflow diagram, API table 的 模組名稱一樣~~
~~2. 針對 outputs/result_rt0.jpg 室外風景 ，構思 要追加的影像處理算法or參數~~


你是 嵌入式 影像處理工程師，修改 @READ
1. BREAKDWON 下方要追加 列出 各個 子模組的 What , why, how ，以表格呈現
2. 檢查  Breakdown 的模組名稱 要與 dataflow diagram, API table 的 模組名稱一樣
3. 給我建議


我需要以 KITTI 為 驗證的資料集
1. 請查 KITTI 專門測試 y, r, p 的資料集 ，直接 當前最受歡迎的 KITTI  下載方式，社群公認最穩定且推薦的方法是使用 valgur 的下載腳本（配合 AWS S3  鏡像）。
2. 對著 資料集 執行測試ʼ統計 當前 系統的 準確度ʼ效能


1. 1. 列當前 file tree給我  2. 建議可以rm的file  3. 簡易 file tree            
   改良SOP，以便你下次更快了解此專案
   
   
   @README.md 1.  #### 模組拆解 (Breakdown) 請改為 算法的名稱，ex： Canny,    
   Hough, RANSEC, ...  2. #### 模組拆解 (Breakdown) 的後方追加 每個算法的     
   What , why, how ，以表格呈現
 
