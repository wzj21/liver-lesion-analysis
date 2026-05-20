"""
模型量化与推理优化
Model Quantization and Inference Optimization

包含:
- 动态量化
- 静态量化
- 量化感知训练 (QAT)
- ONNX导出
- TensorRT优化
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import os
import logging


class ModelQuantizer:
    """模型量化器"""
    
    def __init__(self, model: nn.Module, config: Dict = None, device: torch.device = torch.device('cpu')):
        self.model = model
        self.config = config or {}
        self.device = device
        self.logger = logging.getLogger(__name__)
    
    def dynamic_quantize(self, dtype: torch.dtype = torch.qint8) -> nn.Module:
        """动态量化"""
        self.logger.info("Applying dynamic quantization...")
        quantized_model = torch.quantization.quantize_dynamic(
            self.model.cpu(), {nn.Linear, nn.Conv3d}, dtype=dtype
        )
        self.logger.info("Dynamic quantization completed")
        return quantized_model
    
    def static_quantize(self, calibration_dataloader, num_batches: int = 100, backend: str = 'fbgemm') -> nn.Module:
        """静态量化"""
        self.logger.info(f"Applying static quantization with backend: {backend}")
        
        self.model.cpu()
        self.model.eval()
        self.model.qconfig = torch.quantization.get_default_qconfig(backend)
        
        model_prepared = torch.quantization.prepare(self.model)
        
        self.logger.info("Running calibration...")
        with torch.no_grad():
            for i, batch in enumerate(calibration_dataloader):
                if i >= num_batches:
                    break
                images = batch['image']
                model_prepared(images)
        
        quantized_model = torch.quantization.convert(model_prepared)
        self.logger.info("Static quantization completed")
        return quantized_model
    
    def compare_models(self, original_model: nn.Module, quantized_model: nn.Module, test_dataloader, num_batches: int = 10) -> Dict[str, Any]:
        """比较原始模型和量化模型"""
        import time
        
        original_model.eval()
        quantized_model.eval()
        
        original_times, quantized_times, output_diffs = [], [], []
        
        with torch.no_grad():
            for i, batch in enumerate(test_dataloader):
                if i >= num_batches:
                    break
                
                images = batch['image'].cpu()
                
                start = time.time()
                original_out = original_model(images)
                original_times.append(time.time() - start)
                
                start = time.time()
                quantized_out = quantized_model(images)
                quantized_times.append(time.time() - start)
                
                if isinstance(original_out, dict) and isinstance(quantized_out, dict):
                    for key in original_out:
                        if torch.is_tensor(original_out[key]) and torch.is_tensor(quantized_out[key]):
                            diff = (original_out[key] - quantized_out[key]).abs().mean().item()
                            output_diffs.append(diff)
        
        original_size = self._get_model_size(original_model)
        quantized_size = self._get_model_size(quantized_model)
        
        return {
            'original_avg_time': np.mean(original_times),
            'quantized_avg_time': np.mean(quantized_times),
            'speedup': np.mean(original_times) / np.mean(quantized_times),
            'original_size_mb': original_size,
            'quantized_size_mb': quantized_size,
            'compression_ratio': original_size / quantized_size,
            'avg_output_diff': np.mean(output_diffs) if output_diffs else 0
        }
    
    def _get_model_size(self, model: nn.Module) -> float:
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        return (param_size + buffer_size) / (1024 * 1024)


class ONNXExporter:
    """ONNX模型导出器"""
    
    def __init__(self, model: nn.Module, config: Dict = None):
        self.model = model
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
    
    def export(self, output_path: str, input_shape: Tuple[int, ...] = (1, 1, 128, 256, 256),
               opset_version: int = 14, dynamic_axes: bool = True, simplify: bool = True) -> str:
        """导出ONNX模型"""
        self.logger.info(f"Exporting model to ONNX: {output_path}")
        
        self.model.eval()
        self.model.cpu()
        
        dummy_input = torch.randn(input_shape)
        
        dynamic_axes_config = None
        if dynamic_axes:
            dynamic_axes_config = {
                'input': {0: 'batch_size', 2: 'depth', 3: 'height', 4: 'width'},
                'output': {0: 'batch_size'}
            }
        
        torch.onnx.export(
            self.model, dummy_input, output_path,
            export_params=True, opset_version=opset_version,
            do_constant_folding=True, input_names=['input'],
            output_names=['output'], dynamic_axes=dynamic_axes_config
        )
        
        if simplify:
            try:
                import onnx
                from onnxsim import simplify as onnx_simplify
                model_onnx = onnx.load(output_path)
                model_simplified, check = onnx_simplify(model_onnx)
                if check:
                    onnx.save(model_simplified, output_path)
                    self.logger.info("ONNX model simplified successfully")
            except ImportError:
                self.logger.warning("onnx-simplifier not installed")
        
        self.logger.info(f"ONNX export completed: {output_path}")
        return output_path
    
    def verify(self, onnx_path: str, test_input: torch.Tensor) -> bool:
        """验证ONNX模型"""
        try:
            import onnx
            import onnxruntime as ort
            
            model_onnx = onnx.load(onnx_path)
            onnx.checker.check_model(model_onnx)
            
            self.model.eval()
            with torch.no_grad():
                pytorch_output = self.model(test_input)
            
            ort_session = ort.InferenceSession(onnx_path)
            ort_inputs = {ort_session.get_inputs()[0].name: test_input.numpy()}
            ort_output = ort_session.run(None, ort_inputs)[0]
            
            if isinstance(pytorch_output, dict):
                pytorch_output = list(pytorch_output.values())[0]
            
            diff = np.abs(pytorch_output.numpy() - ort_output).mean()
            self.logger.info(f"ONNX verification - Mean difference: {diff:.6f}")
            return diff < 1e-4
        except Exception as e:
            self.logger.error(f"ONNX verification failed: {e}")
            return False


class TensorRTOptimizer:
    """TensorRT优化器"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
    
    def optimize(self, onnx_path: str, output_path: str, precision: str = 'fp16',
                 workspace_size: int = 1 << 30, max_batch_size: int = 8) -> str:
        """使用TensorRT优化模型"""
        try:
            import tensorrt as trt
        except ImportError:
            self.logger.error("TensorRT not installed")
            return None
        
        self.logger.info(f"Optimizing with TensorRT (precision: {precision})")
        
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(TRT_LOGGER)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, TRT_LOGGER)
        
        with open(onnx_path, 'rb') as f:
            if not parser.parse(f.read()):
                for error in range(parser.num_errors):
                    self.logger.error(parser.get_error(error))
                return None
        
        config = builder.create_builder_config()
        config.max_workspace_size = workspace_size
        
        if precision == 'fp16' and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        elif precision == 'int8' and builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
        
        self.logger.info("Building TensorRT engine...")
        engine = builder.build_engine(network, config)
        
        if engine is None:
            self.logger.error("Failed to build TensorRT engine")
            return None
        
        with open(output_path, 'wb') as f:
            f.write(engine.serialize())
        
        self.logger.info(f"TensorRT engine saved to: {output_path}")
        return output_path


class InferenceEngine:
    """统一推理引擎"""
    
    def __init__(self, model_path: str, backend: str = 'pytorch', device: str = 'cuda'):
        self.model_path = model_path
        self.backend = backend
        self.device = device
        self.logger = logging.getLogger(__name__)
        self.model = self._load_model()
    
    def _load_model(self):
        if self.backend == 'pytorch':
            model = torch.load(self.model_path, map_location=self.device)
            model.eval()
            return model
        elif self.backend == 'onnx':
            import onnxruntime as ort
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.device == 'cuda' else ['CPUExecutionProvider']
            return ort.InferenceSession(self.model_path, providers=providers)
        elif self.backend == 'tensorrt':
            import tensorrt as trt
            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
            with open(self.model_path, 'rb') as f:
                return trt.Runtime(TRT_LOGGER).deserialize_cuda_engine(f.read())
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self.backend == 'pytorch':
            with torch.no_grad():
                x_tensor = torch.from_numpy(x).to(self.device)
                output = self.model(x_tensor)
                if isinstance(output, dict):
                    return {k: v.cpu().numpy() for k, v in output.items()}
                return output.cpu().numpy()
        elif self.backend == 'onnx':
            input_name = self.model.get_inputs()[0].name
            return self.model.run(None, {input_name: x})[0]
        return None
    
    def benchmark(self, input_shape: Tuple[int, ...], num_runs: int = 100, warmup_runs: int = 10) -> Dict[str, float]:
        import time
        
        x = np.random.randn(*input_shape).astype(np.float32)
        
        for _ in range(warmup_runs):
            _ = self(x)
        
        times = []
        for _ in range(num_runs):
            start = time.time()
            _ = self(x)
            times.append(time.time() - start)
        
        return {
            'mean_latency_ms': np.mean(times) * 1000,
            'std_latency_ms': np.std(times) * 1000,
            'min_latency_ms': np.min(times) * 1000,
            'max_latency_ms': np.max(times) * 1000,
            'throughput_fps': 1 / np.mean(times)
        }
