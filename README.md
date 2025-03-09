# Fast Trans - 自部署极速翻译服务

基于Helsinki-NLP/opus-mt模型的英汉互译API服务，支持容器化部署和高性能推理。

## 核心功能

- 英译汉 (/translate/en-to-zh)
- 汉译英 (/translate/zh-to-en)
- 健康状态检查 (/health)
- 支持GPU加速推理
- 请求缓存优化
- 详细的日志记录

## 技术栈

- **框架**: FastAPI + Uvicorn
- **模型**: Helsinki-NLP/opus-mt-en-zh 和 Helsinki-NLP/opus-mt-zh-en
- **推理引擎**: PyTorch + Helsinki-NLP/opus-mt
- **容器化**: Docker

## 快速开始

### 环境要求

- Python 3.10+
- pip 23.0+
- Docker 20.10+ (可选)
- CUDA 11.8+ (GPU支持需要)

### 安装步骤

```bash
git clone https://github.com/yourrepo/fast-trans.git
cd fast-trans

# 安装依赖
pip install -r src/requirements.txt
```

### 配置环境变量
```bash
# 基本配置
export PORT=8000
export HOST=0.0.0.0

# 高级配置 
export MAX_LENGTH=200       # 最大生成文本长度
export MAX_TEXT_LENGTH=500  # 输入文本最大长度
```

### 启动服务
```bash
python -m src.main
```

## Docker部署
```bash
# 构建镜像 (包含模型下载)
docker build -t fast-trans .

# 运行容器 (映射8000端口)
docker run -p 8000:8000 \
  -e PORT=8000 \
  -e HOST=0.0.0.0 \
  fast-trans
```

## API文档

### 翻译请求示例
```bash
curl -X POST "http://localhost:8000/translate/en-to-zh" \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, world!", "source_lang": "en", "target_lang": "zh"}'
```

### 响应格式
```json
{
  "translation": "你好，世界！"
}
```

## 项目结构
```
fast-trans/
├── src/                 # 源代码
│   ├── main.py          # 服务入口
│   ├── requirements.txt # 依赖清单
├── Dockerfile           # 容器配置
├── .gitignore           # 版本控制排除项
└── README.md            # 项目文档
```

## 开发注意事项

1. 模型自动下载于首次启动时
2. 日志文件生成于 translation_service.log
3. 使用LRU缓存加速重复请求
4. 单Worker运行保证模型单例

## 许可证
Apache License 2.0
