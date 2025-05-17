<h1> squat_coach</h1>

## 目录结构

```plaintext
squat_coach/
├── code
│   ├── videos
│   ├── chat.py
│   ├── fewshot.json
│   ├── main.py
│   ├── output.mp3
│   ├── pose.py
│   └── predict_standard.py
├── Qwen-1_8B-Chat
└── yolo11n-pose.pt
```

## 文件说明

### code
- **videos**: 存放模型评估和分析的视频。  
- **chat.py**: 处理自然语言交互的脚本。  
- **fewshot.json**: few-shot学习样本数据。  
- **main.py**: 项目主程序入口。  
- **output.mp3**: 语音输出文件。  
- **pose.py**: 关键点检测与姿势分析模块。  
- **predict_standard.py**: 预测标准数据生成脚本。  

### Qwen-1_8B-Chat
- 预训练模型文件，基于Qwen-1.8B-Chat。  

### yolo11n-pose.pt
- 关键点检测模型权重文件。  
