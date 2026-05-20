"""
部署模块
Deployment Module

包含:
- 模型服务封装
- REST API接口
- Docker配置生成
- 健康检查
- 批量推理
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Any
import os
import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime
import logging


@dataclass
class InferenceRequest:
    """推理请求"""
    request_id: str
    image_path: Optional[str] = None
    image_data: Optional[np.ndarray] = None
    patient_id: Optional[str] = None
    return_visualization: bool = False
    return_uncertainty: bool = True


@dataclass
class InferenceResult:
    """推理结果"""
    request_id: str
    success: bool
    diagnosis: Optional[str] = None
    confidence: float = 0.0
    lesion_type: Optional[str] = None
    is_active: Optional[bool] = None
    uncertainty: float = 0.0
    needs_review: bool = False
    processing_time_ms: float = 0.0
    error_message: Optional[str] = None
    visualization_path: Optional[str] = None
    detailed_results: Optional[Dict] = None


class ModelServer:
    """模型服务器"""
    
    def __init__(self, model_path: str, config: Dict = None, device: str = 'cuda'):
        self.config = config or {}
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.logger = logging.getLogger(__name__)
        
        self.model = self._load_model(model_path)
        self.model.eval()
        
        self.window_width = config.get('window_width', 160)
        self.window_level = config.get('window_level', 60)
        self.class_names = ['良性', '恶性', '囊型包虫病', '泡型包虫病']
        
        self.stats = {'total_requests': 0, 'successful_requests': 0, 'failed_requests': 0, 'total_processing_time': 0.0}
        self.logger.info(f"Model server initialized on {self.device}")
    
    def _load_model(self, model_path: str) -> nn.Module:
        """加载模型"""
        checkpoint = torch.load(model_path, map_location=self.device)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            # 假设有预定义的模型类
            model = checkpoint.get('model', checkpoint)
        else:
            model = checkpoint
        return model.to(self.device) if hasattr(model, 'to') else model
    
    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """预处理图像"""
        if image.ndim == 4:
            image = image[0]
        
        min_hu = self.window_level - self.window_width / 2
        max_hu = self.window_level + self.window_width / 2
        image = np.clip(image, min_hu, max_hu)
        image = (image - min_hu) / (max_hu - min_hu)
        
        image = image[np.newaxis, np.newaxis, ...]
        tensor = torch.from_numpy(image.astype(np.float32)).to(self.device)
        return tensor
    
    def postprocess(self, outputs: Dict) -> Dict:
        """后处理模型输出"""
        results = {}
        
        if 'classification' in outputs:
            cls_out = outputs['classification']
            logits = cls_out.get('logits')
            
            if logits is not None:
                probs = torch.softmax(logits, dim=1)
                confidence, pred = probs.max(dim=1)
                
                results['diagnosis'] = self.class_names[pred.item()]
                results['confidence'] = confidence.item()
                results['probabilities'] = {name: probs[0, i].item() for i, name in enumerate(self.class_names)}
            
            if 'uncertainty' in cls_out:
                results['uncertainty'] = cls_out['uncertainty'].mean().item()
                results['needs_review'] = results['uncertainty'] > 0.3
        
        if 'segmentation' in outputs:
            seg_out = outputs['segmentation']
            results['liver_mask'] = seg_out['logits'].argmax(dim=1).cpu().numpy()
        
        if 'detection' in outputs:
            det_out = outputs['detection']
            results['lesion_boxes'] = det_out.get('box_pred', [])
        
        return results
    
    @torch.no_grad()
    def predict(self, request: InferenceRequest) -> InferenceResult:
        """执行预测"""
        start_time = time.time()
        self.stats['total_requests'] += 1
        
        try:
            # 加载图像
            if request.image_data is not None:
                image = request.image_data
            elif request.image_path:
                image = self._load_image(request.image_path)
            else:
                raise ValueError("No image provided")
            
            # 预处理
            tensor = self.preprocess(image)
            
            # 推理
            outputs = self.model(tensor)
            
            # 后处理
            results = self.postprocess(outputs)
            
            processing_time = (time.time() - start_time) * 1000
            self.stats['successful_requests'] += 1
            self.stats['total_processing_time'] += processing_time
            
            return InferenceResult(
                request_id=request.request_id,
                success=True,
                diagnosis=results.get('diagnosis'),
                confidence=results.get('confidence', 0),
                uncertainty=results.get('uncertainty', 0),
                needs_review=results.get('needs_review', False),
                processing_time_ms=processing_time,
                detailed_results=results
            )
            
        except Exception as e:
            self.stats['failed_requests'] += 1
            return InferenceResult(
                request_id=request.request_id,
                success=False,
                error_message=str(e),
                processing_time_ms=(time.time() - start_time) * 1000
            )
    
    def _load_image(self, path: str) -> np.ndarray:
        """加载图像文件"""
        if path.endswith('.nii') or path.endswith('.nii.gz'):
            import nibabel as nib
            return nib.load(path).get_fdata()
        elif path.endswith('.npy'):
            return np.load(path)
        else:
            raise ValueError(f"Unsupported file format: {path}")
    
    def predict_batch(self, requests: List[InferenceRequest]) -> List[InferenceResult]:
        """批量预测"""
        return [self.predict(req) for req in requests]
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        avg_time = self.stats['total_processing_time'] / max(self.stats['successful_requests'], 1)
        return {
            **self.stats,
            'average_processing_time_ms': avg_time,
            'success_rate': self.stats['successful_requests'] / max(self.stats['total_requests'], 1)
        }
    
    def health_check(self) -> Dict:
        """健康检查"""
        try:
            dummy_input = torch.randn(1, 1, 32, 64, 64).to(self.device)
            _ = self.model(dummy_input)
            return {'status': 'healthy', 'device': str(self.device), 'timestamp': datetime.now().isoformat()}
        except Exception as e:
            return {'status': 'unhealthy', 'error': str(e), 'timestamp': datetime.now().isoformat()}


class FastAPIApp:
    """FastAPI应用生成器"""
    
    @staticmethod
    def generate_app_code(model_path: str, output_path: str = 'app.py'):
        """生成FastAPI应用代码"""
        code = '''"""
肝脏病灶智能分析系统 - REST API
Liver Lesion Analysis System - REST API
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List
import numpy as np
import tempfile
import os
import uuid

app = FastAPI(
    title="肝脏病灶智能分析系统",
    description="AI-powered liver lesion analysis system",
    version="1.0.0"
)

# 初始化模型服务器
from deployment import ModelServer
server = ModelServer("''' + model_path + '''")


class PredictionRequest(BaseModel):
    patient_id: Optional[str] = None
    return_visualization: bool = False


class PredictionResponse(BaseModel):
    request_id: str
    success: bool
    diagnosis: Optional[str] = None
    confidence: float = 0.0
    uncertainty: float = 0.0
    needs_review: bool = False
    processing_time_ms: float = 0.0
    error_message: Optional[str] = None


@app.get("/health")
async def health_check():
    """健康检查"""
    return server.health_check()


@app.get("/stats")
async def get_stats():
    """获取统计信息"""
    return server.get_stats()


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...), patient_id: Optional[str] = None):
    """
    执行预测
    
    上传CT图像文件进行分析
    """
    # 保存上传的文件
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        from deployment import InferenceRequest
        
        request = InferenceRequest(
            request_id=str(uuid.uuid4()),
            image_path=tmp_path,
            patient_id=patient_id
        )
        
        result = server.predict(request)
        
        return PredictionResponse(
            request_id=result.request_id,
            success=result.success,
            diagnosis=result.diagnosis,
            confidence=result.confidence,
            uncertainty=result.uncertainty,
            needs_review=result.needs_review,
            processing_time_ms=result.processing_time_ms,
            error_message=result.error_message
        )
    finally:
        os.unlink(tmp_path)


@app.post("/predict/batch")
async def predict_batch(files: List[UploadFile] = File(...)):
    """批量预测"""
    results = []
    for file in files:
        result = await predict(file)
        results.append(result)
    return results


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(code)
        
        return output_path


class DockerGenerator:
    """Docker配置生成器"""
    
    @staticmethod
    def generate_dockerfile(output_path: str = 'Dockerfile'):
        """生成Dockerfile"""
        dockerfile = '''# 肝脏病灶智能分析系统 Docker镜像
FROM nvidia/cuda:11.8-cudnn8-runtime-ubuntu22.04

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 安装系统依赖
RUN apt-get update && apt-get install -y \\
    python3.10 \\
    python3-pip \\
    libgl1-mesa-glx \\
    libglib2.0-0 \\
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 复制依赖文件
COPY requirements.txt .

# 安装Python依赖
RUN pip3 install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 暴露端口
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# 启动命令
CMD ["python3", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
'''
        
        with open(output_path, 'w') as f:
            f.write(dockerfile)
        
        return output_path
    
    @staticmethod
    def generate_docker_compose(output_path: str = 'docker-compose.yml'):
        """生成docker-compose配置"""
        compose = '''version: '3.8'

services:
  liver-lesion-api:
    build: .
    container_name: liver-lesion-api
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
      - ./data:/app/data
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - MODEL_PATH=/app/models/best_model.pth
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  nginx:
    image: nginx:alpine
    container_name: liver-lesion-nginx
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - liver-lesion-api
    restart: unless-stopped
'''
        
        with open(output_path, 'w') as f:
            f.write(compose)
        
        return output_path
    
    @staticmethod
    def generate_requirements(output_path: str = 'requirements.txt'):
        """生成requirements.txt"""
        requirements = '''# 核心依赖
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
scipy>=1.10.0
scikit-learn>=1.2.0

# 医学图像处理
nibabel>=5.0.0
SimpleITK>=2.2.0
pydicom>=2.3.0

# Web框架
fastapi>=0.100.0
uvicorn>=0.22.0
python-multipart>=0.0.6

# 可视化
matplotlib>=3.7.0
plotly>=5.14.0

# 工具
pyyaml>=6.0
tqdm>=4.65.0
requests>=2.28.0

# 可选：ONNX Runtime
onnxruntime-gpu>=1.15.0

# 可选：报告生成
reportlab>=4.0.0
'''
        
        with open(output_path, 'w') as f:
            f.write(requirements)
        
        return output_path


class DeploymentManager:
    """部署管理器"""
    
    def __init__(self, output_dir: str = 'deployment'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_all(self, model_path: str):
        """生成所有部署文件"""
        files = {}
        
        # FastAPI应用
        files['app.py'] = FastAPIApp.generate_app_code(
            model_path, os.path.join(self.output_dir, 'app.py')
        )
        
        # Docker文件
        files['Dockerfile'] = DockerGenerator.generate_dockerfile(
            os.path.join(self.output_dir, 'Dockerfile')
        )
        files['docker-compose.yml'] = DockerGenerator.generate_docker_compose(
            os.path.join(self.output_dir, 'docker-compose.yml')
        )
        files['requirements.txt'] = DockerGenerator.generate_requirements(
            os.path.join(self.output_dir, 'requirements.txt')
        )
        
        # 生成部署说明
        files['README.md'] = self._generate_readme()
        
        return files
    
    def _generate_readme(self) -> str:
        """生成部署说明"""
        readme = '''# 肝脏病灶智能分析系统 - 部署指南

## 快速开始

### 1. 使用Docker部署

```bash
# 构建镜像
docker-compose build

# 启动服务
docker-compose up -d

# 查看日志
docker-compose logs -f
```

### 2. API使用

#### 健康检查
```bash
curl http://localhost:8000/health
```

#### 执行预测
```bash
curl -X POST "http://localhost:8000/predict" \\
     -F "file=@/path/to/ct_image.nii.gz"
```

### 3. 性能监控

访问 http://localhost:8000/stats 查看性能统计

## 配置说明

- `MODEL_PATH`: 模型文件路径
- `CUDA_VISIBLE_DEVICES`: 使用的GPU设备

## 技术支持

如有问题，请联系技术支持团队。
'''
        
        readme_path = os.path.join(self.output_dir, 'README.md')
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme)
        
        return readme_path
