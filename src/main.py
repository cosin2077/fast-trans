from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from fastapi import FastAPI, Request, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Optional
import logging
import uvicorn
from functools import lru_cache
from starlette.middleware.cors import CORSMiddleware
import os
from contextlib import asynccontextmanager
import time

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("translation_service.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 环境变量配置
class Settings:
    MODEL_EN_ZH = os.getenv("MODEL_EN_ZH", "Helsinki-NLP/opus-mt-en-zh")
    MODEL_ZH_EN = os.getenv("MODEL_ZH_EN", "Helsinki-NLP/opus-mt-zh-en")
    MAX_LENGTH = int(os.getenv("MAX_LENGTH", 100))
    MAX_TEXT_LENGTH = int(os.getenv("MAX_TEXT_LENGTH", 1000))
    PORT = int(os.getenv("PORT", 8000))
    HOST = os.getenv("HOST", "0.0.0.0")

settings = Settings()

# 模型加载的生命周期管理
models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Current device: {torch.cuda.current_device() if torch.cuda.is_available() else 'CPU'}")
    print(f"Device name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
    logger.info(f"Using device: {device}")
    
    logger.info("Loading models...")
    start_time = time.time()
    
    models["en_zh_tokenizer"] = AutoTokenizer.from_pretrained(settings.MODEL_EN_ZH)
    models["en_zh_model"] = AutoModelForSeq2SeqLM.from_pretrained(settings.MODEL_EN_ZH).to(device)
    models["zh_en_tokenizer"] = AutoTokenizer.from_pretrained(settings.MODEL_ZH_EN)
    models["zh_en_model"] = AutoModelForSeq2SeqLM.from_pretrained(settings.MODEL_ZH_EN).to(device)
    
    logger.info(f"Models loaded in {time.time() - start_time:.2f} seconds")
    yield
    # 关闭时清理
    models.clear()
    torch.cuda.empty_cache()

# FastAPI应用
app = FastAPI(
    title="Translation Service",
    description="English-Chinese Translation API",
    version="1.0.0",
    lifespan=lifespan
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 请求模型
class TranslateRequest(BaseModel):
    text: str = Field(..., max_length=settings.MAX_TEXT_LENGTH)
    source_lang: Optional[str] = None
    target_lang: Optional[str] = None

    class Config:
        schema_extra = {
            "example": {
                "text": "Hello, world!",
                "source_lang": "en",
                "target_lang": "zh"
            }
        }

# 翻译核心函数
@lru_cache(maxsize=1000)
def translate(text: str, model, tokenizer, max_length: int = settings.MAX_LENGTH) -> str:
    try:
        start_time = time.time()
        input_ids = tokenizer(text, return_tensors="pt", padding=True, truncation=True).input_ids.to(model.device)
        outputs = model.generate(input_ids=input_ids, max_length=max_length, num_beams=4, early_stopping=True)
        translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"Translation completed in {time.time() - start_time:.3f}s")
        return translated_text
    except Exception as e:
        logger.error(f"Translation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")

# 依赖注入获取模型
def get_en_zh_models():
    return {"model": models["en_zh_model"], "tokenizer": models["en_zh_tokenizer"]}

def get_zh_en_models():
    return {"model": models["zh_en_model"], "tokenizer": models["zh_en_tokenizer"]}

# API端点
@app.post("/translate/en-to-zh", response_model=dict)
async def translate_en_to_zh(
    request: TranslateRequest,
    models_dict: dict = Depends(get_en_zh_models)
):
    try:
        result = translate(
            text=request.text,
            model=models_dict["model"],
            tokenizer=models_dict["tokenizer"]
        )
        return {"translation": result}
    except Exception as e:
        logger.error(f"EN-ZH translation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/translate/zh-to-en", response_model=dict)
async def translate_zh_to_en(
    request: TranslateRequest,
    models_dict: dict = Depends(get_zh_en_models)
):
    try:
        result = translate(
            text=request.text,
            model=models_dict["model"],
            tokenizer=models_dict["tokenizer"]
        )
        return {"translation": result}
    except Exception as e:
        logger.error(f"ZH-EN translation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

# 健康检查端点
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "en_zh_device": str(models["en_zh_model"].device),
        "zh_en_device": str(models["zh_en_model"].device),
        "timestamp": time.time()
    }

if __name__ == "__main__":
    uvicorn.run(
        app,
        host=settings.HOST,
        port=settings.PORT,
        log_level="info",
        workers=1  # 单worker避免模型重复加载
    )