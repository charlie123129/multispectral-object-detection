# 碩士論文
## 基於LoRALayer的跨模態融合Transformer用於多光譜物體檢測及其在邊緣設備上的實現
## LoRALayer Based Cross-Modality Fusion Transformer for Multispectral Object Detection and Its Implementation on Edge Devices
#### 參考文件與成果展示:[論文連結](https://etheses.lib.ntust.edu.tw/thesis/detail/7bdd9833c2702daa9945b20db8c1b98b/?seq=1)

### 先下載 yolov5
```
git clone https://github.com/ultralytics/yolov5
```
### 建立虛擬環境
```
conda create --name yolov5 python=3.9
```

## 切換到虛擬環境
```
conda activate yolov5
```

### 到yolov5資料內安裝所有需要的套件
```
pip install -r requirements.txt
```

### 下載 multispectral
>### 把multispectral資料夾內的資料複製到yolov5資料夾
```
git clone https://github.com/charlie123129/multispectral-object-detection
```

