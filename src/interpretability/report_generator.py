"""
解释报告生成模块
Explanation Report Generator Module

包含:
- 可解释性报告生成
- 自然语言解释
- 不确定性分析报告
- 临床建议生成
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
import json
import os


@dataclass
class InterpretabilityReport:
    """可解释性报告数据结构"""
    
    report_id: str
    generated_time: str
    patient_id: Optional[str] = None
    
    diagnosis: str = ""
    diagnosis_confidence: float = 0.0
    lesion_type: str = ""
    is_active: Optional[bool] = None
    
    total_uncertainty: float = 0.0
    epistemic_uncertainty: float = 0.0
    aleatoric_uncertainty: float = 0.0
    needs_review: bool = False
    review_reason: str = ""
    
    attention_regions: List[Dict] = field(default_factory=list)
    key_slices: List[int] = field(default_factory=list)
    
    top_positive_features: List[Dict] = field(default_factory=list)
    top_negative_features: List[Dict] = field(default_factory=list)
    feature_group_contributions: Dict[str, float] = field(default_factory=dict)
    
    similar_cases: List[Dict] = field(default_factory=list)
    contrast_explanation: str = ""
    recommendations: List[str] = field(default_factory=list)
    
    cam_visualization_path: Optional[str] = None
    attention_visualization_path: Optional[str] = None
    uncertainty_map_path: Optional[str] = None


class ExplanationGenerator:
    """解释生成器"""
    
    def __init__(self, language: str = 'zh', style: str = 'clinical'):
        self.language = language
        self.style = style
        self.templates = self._load_templates()
    
    def _load_templates(self) -> Dict[str, Dict]:
        return {
            'clinical': {
                'diagnosis_intro': "AI辅助诊断系统分析结果如下：",
                'diagnosis_result': "诊断结论：{diagnosis}（置信度：{confidence:.1%}）",
                'uncertainty_low': "模型对该诊断具有较高的确定性。",
                'uncertainty_medium': "模型对该诊断存在一定的不确定性，建议结合临床表现综合判断。",
                'uncertainty_high': "模型对该诊断存在较高的不确定性，强烈建议人工复核。",
                'attention_intro': "模型主要关注以下区域：",
                'feature_intro': "影响诊断的主要因素：",
                'similar_case_intro': "参考相似病例：",
                'recommendation_intro': "临床建议：",
            }
        }
    
    def generate_full_explanation(self, report: InterpretabilityReport) -> str:
        """生成完整解释"""
        sections = [
            self._generate_header(report),
            self._generate_diagnosis_section(report),
            self._generate_uncertainty_section(report),
            self._generate_attention_section(report),
            self._generate_feature_section(report),
            self._generate_similar_cases_section(report),
            self._generate_recommendations_section(report),
        ]
        return "\n".join(sections)
    
    def _generate_header(self, report: InterpretabilityReport) -> str:
        lines = [
            "=" * 60,
            "肝脏病灶智能分析报告",
            "=" * 60,
            f"报告编号：{report.report_id}",
            f"生成时间：{report.generated_time}",
        ]
        if report.patient_id:
            lines.append(f"患者编号：{report.patient_id}")
        lines.append("")
        return "\n".join(lines)
    
    def _generate_diagnosis_section(self, report: InterpretabilityReport) -> str:
        lines = [
            "【诊断结果】",
            f"  诊断：{report.diagnosis}",
            f"  置信度：{report.diagnosis_confidence:.1%}",
            f"  病灶类型：{report.lesion_type}",
        ]
        if report.is_active is not None:
            lines.append(f"  活性判定：{'活动期' if report.is_active else '静止期'}")
        lines.append("")
        return "\n".join(lines)
    
    def _generate_uncertainty_section(self, report: InterpretabilityReport) -> str:
        lines = ["【不确定性分析】"]
        
        level = "低" if report.total_uncertainty < 0.2 else "中" if report.total_uncertainty < 0.4 else "高"
        lines.append(f"  总体不确定性：{report.total_uncertainty:.1%}（{level}）")
        lines.append(f"  - 模型不确定性：{report.epistemic_uncertainty:.1%}")
        lines.append(f"  - 数据不确定性：{report.aleatoric_uncertainty:.1%}")
        
        if report.needs_review:
            lines.append(f"  ⚠ 建议人工复核：{report.review_reason}")
        lines.append("")
        return "\n".join(lines)
    
    def _generate_attention_section(self, report: InterpretabilityReport) -> str:
        if not report.attention_regions:
            return ""
        
        lines = ["【关注区域分析】"]
        for i, region in enumerate(report.attention_regions[:5], 1):
            lines.append(f"  {i}. {region.get('location', 'N/A')} - 重要性：{region.get('importance', 0):.1%}")
        
        if report.key_slices:
            lines.append(f"  关键切片：第 {', '.join(map(str, report.key_slices))} 层")
        lines.append("")
        return "\n".join(lines)
    
    def _generate_feature_section(self, report: InterpretabilityReport) -> str:
        if not report.top_positive_features and not report.top_negative_features:
            return ""
        
        lines = ["【特征贡献分析】"]
        
        if report.top_positive_features:
            lines.append("  支持诊断的特征：")
            for feat in report.top_positive_features[:3]:
                lines.append(f"    + {feat.get('name', 'N/A')}：{feat.get('contribution', 0):.1%}")
        
        if report.top_negative_features:
            lines.append("  不支持诊断的特征：")
            for feat in report.top_negative_features[:3]:
                lines.append(f"    - {feat.get('name', 'N/A')}：{feat.get('contribution', 0):.1%}")
        
        if report.feature_group_contributions:
            lines.append("  特征组贡献：")
            for group, contrib in report.feature_group_contributions.items():
                lines.append(f"    {group}：{contrib:.1%}")
        lines.append("")
        return "\n".join(lines)
    
    def _generate_similar_cases_section(self, report: InterpretabilityReport) -> str:
        if not report.similar_cases:
            return ""
        
        lines = ["【相似病例参考】"]
        for case in report.similar_cases[:3]:
            lines.append(f"  - 病例 {case.get('case_id', 'N/A')}：{case.get('diagnosis', 'N/A')}（相似度：{case.get('similarity', 0):.1%}）")
        lines.append("")
        return "\n".join(lines)
    
    def _generate_recommendations_section(self, report: InterpretabilityReport) -> str:
        if not report.recommendations:
            return ""
        
        lines = ["【临床建议】"]
        for i, rec in enumerate(report.recommendations, 1):
            lines.append(f"  {i}. {rec}")
        lines.append("")
        return "\n".join(lines)


class RecommendationEngine:
    """临床建议引擎"""
    
    def __init__(self):
        self.rules = self._load_rules()
    
    def _load_rules(self) -> List[Dict]:
        return [
            {
                'condition': lambda r: r.total_uncertainty > 0.4,
                'recommendation': '不确定性较高，建议进行增强CT或MRI检查进一步确认'
            },
            {
                'condition': lambda r: r.lesion_type == '恶性' and r.diagnosis_confidence > 0.8,
                'recommendation': '高度怀疑恶性病变，建议穿刺活检明确病理诊断'
            },
            {
                'condition': lambda r: r.lesion_type in ['囊型包虫病', '泡型包虫病'],
                'recommendation': '建议行包虫病血清学检测辅助诊断'
            },
            {
                'condition': lambda r: r.is_active == True,
                'recommendation': '病灶处于活动期，建议尽早治疗干预'
            },
            {
                'condition': lambda r: r.is_active == False,
                'recommendation': '病灶处于静止期，建议定期随访观察'
            },
            {
                'condition': lambda r: r.epistemic_uncertainty > r.aleatoric_uncertainty * 2,
                'recommendation': '模型不确定性较高，可能为罕见病例类型，建议专家会诊'
            },
        ]
    
    def generate_recommendations(self, report: InterpretabilityReport) -> List[str]:
        recommendations = []
        for rule in self.rules:
            try:
                if rule['condition'](report):
                    recommendations.append(rule['recommendation'])
            except:
                pass
        return recommendations


class ReportExporter:
    """报告导出器"""
    
    def __init__(self, output_dir: str = 'reports'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def export_json(self, report: InterpretabilityReport, filename: str = None) -> str:
        if filename is None:
            filename = f"report_{report.report_id}.json"
        
        path = os.path.join(self.output_dir, filename)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(asdict(report), f, ensure_ascii=False, indent=2)
        return path
    
    def export_html(self, report: InterpretabilityReport, filename: str = None) -> str:
        if filename is None:
            filename = f"report_{report.report_id}.html"
        
        generator = ExplanationGenerator()
        content = generator.generate_full_explanation(report)
        
        html = f"""
<!DOCTYPE html>
<html lang="zh">
<head>
    <meta charset="UTF-8">
    <title>肝脏病灶分析报告 - {report.report_id}</title>
    <style>
        body {{ font-family: 'Microsoft YaHei', sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
        .section {{ margin: 20px 0; padding: 15px; background: #f8f9fa; border-radius: 5px; }}
        .section-title {{ color: #2980b9; font-weight: bold; margin-bottom: 10px; }}
        .confidence-high {{ color: #27ae60; }}
        .confidence-medium {{ color: #f39c12; }}
        .confidence-low {{ color: #e74c3c; }}
        .uncertainty-warning {{ background: #fff3cd; padding: 10px; border-radius: 5px; border-left: 4px solid #ffc107; }}
        pre {{ white-space: pre-wrap; font-family: inherit; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🏥 肝脏病灶智能分析报告</h1>
        <pre>{content}</pre>
        
        <div class="section">
            <div class="section-title">📊 可视化结果</div>
            {"<img src='" + report.cam_visualization_path + "' style='max-width:100%'/>" if report.cam_visualization_path else "<p>无可视化图像</p>"}
        </div>
        
        <div style="text-align:center; color:#999; margin-top:30px;">
            <p>本报告由AI辅助诊断系统自动生成，仅供参考，不能替代医生的专业诊断。</p>
            <p>生成时间：{report.generated_time}</p>
        </div>
    </div>
</body>
</html>
"""
        
        path = os.path.join(self.output_dir, filename)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(html)
        return path


class InterpretabilityPipeline:
    """
    可解释性完整管道
    
    整合所有可解释性组件
    """
    
    def __init__(
        self,
        model,
        cam_generator=None,
        attention_visualizer=None,
        shap_analyzer=None,
        case_retriever=None,
        config: Dict = None
    ):
        self.model = model
        self.cam_generator = cam_generator
        self.attention_visualizer = attention_visualizer
        self.shap_analyzer = shap_analyzer
        self.case_retriever = case_retriever
        self.config = config or {}
        
        self.explanation_generator = ExplanationGenerator()
        self.recommendation_engine = RecommendationEngine()
        self.exporter = ReportExporter()
    
    def analyze(
        self,
        input_tensor: torch.Tensor,
        patient_id: Optional[str] = None,
        generate_visualizations: bool = True
    ) -> InterpretabilityReport:
        """
        完整分析管道
        """
        import uuid
        
        # 创建报告
        report = InterpretabilityReport(
            report_id=str(uuid.uuid4())[:8],
            generated_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            patient_id=patient_id
        )
        
        # 1. 模型预测
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_tensor)
        
        self._extract_predictions(outputs, report)
        
        # 2. CAM可视化
        if self.cam_generator and generate_visualizations:
            self._generate_cam(input_tensor, report)
        
        # 3. 注意力分析
        if self.attention_visualizer:
            self._analyze_attention(input_tensor, report)
        
        # 4. SHAP分析
        if self.shap_analyzer:
            self._analyze_shap(input_tensor, report)
        
        # 5. 相似病例检索
        if self.case_retriever:
            results = self.case_retriever.retrieve(input_tensor)
            report.similar_cases = results
        
        # 6. 生成建议
        report.recommendations = self.recommendation_engine.generate_recommendations(report)
        
        return report
    
    def _extract_predictions(self, outputs: Dict, report: InterpretabilityReport):
        """提取预测结果"""
        if isinstance(outputs, dict):
            # 分类结果
            if 'classification' in outputs:
                cls_out = outputs['classification']
                logits = cls_out.get('logits')
                if logits is not None:
                    probs = torch.softmax(logits, dim=1)
                    conf, pred = probs.max(dim=1)
                    report.diagnosis_confidence = conf.item()
                    
                    # 映射类别名称
                    class_names = ['良性', '恶性', '囊型包虫病', '泡型包虫病']
                    report.diagnosis = class_names[pred.item()] if pred.item() < len(class_names) else f"类别{pred.item()}"
                    report.lesion_type = report.diagnosis
                
                # 不确定性
                if 'uncertainty' in cls_out:
                    report.total_uncertainty = cls_out['uncertainty'].mean().item()
                    report.epistemic_uncertainty = report.total_uncertainty * 0.6  # 简化估计
                    report.aleatoric_uncertainty = report.total_uncertainty * 0.4
                    
                    if report.total_uncertainty > 0.3:
                        report.needs_review = True
                        report.review_reason = "模型不确定性超过阈值"
    
    def _generate_cam(self, input_tensor: torch.Tensor, report: InterpretabilityReport):
        """生成CAM可视化"""
        try:
            cams = self.cam_generator.generate(input_tensor)
            
            # 从CAM中提取关注区域
            for layer_name, cam in cams.items():
                # 找到高激活区域
                threshold = 0.5
                high_activation = cam > threshold
                
                if high_activation.any():
                    coords = np.where(high_activation[0] if cam.ndim > 2 else high_activation)
                    center = tuple(int(c.mean()) for c in coords)
                    
                    report.attention_regions.append({
                        'location': f"层{layer_name}，中心位置{center}",
                        'importance': float(cam.max()),
                        'layer': layer_name
                    })
        except Exception as e:
            print(f"CAM generation failed: {e}")
    
    def _analyze_attention(self, input_tensor: torch.Tensor, report: InterpretabilityReport):
        """分析注意力"""
        pass  # 根据具体注意力可视化器实现
    
    def _analyze_shap(self, input_tensor: torch.Tensor, report: InterpretabilityReport):
        """SHAP分析"""
        try:
            shap_values = self.shap_analyzer.explain(input_tensor)
            
            # 简化：使用通道平均作为特征贡献
            if shap_values is not None:
                channel_importance = np.abs(shap_values).mean(axis=(0, 2, 3, 4) if shap_values.ndim == 5 else (0, 2, 3))
                
                for i, imp in enumerate(channel_importance):
                    feature_name = f"特征通道{i}"
                    if imp > 0:
                        report.top_positive_features.append({
                            'name': feature_name,
                            'contribution': float(imp)
                        })
                
                # 排序取top
                report.top_positive_features.sort(key=lambda x: x['contribution'], reverse=True)
                report.top_positive_features = report.top_positive_features[:5]
        except Exception as e:
            print(f"SHAP analysis failed: {e}")
    
    def generate_report(
        self,
        report: InterpretabilityReport,
        format: str = 'html'
    ) -> str:
        """生成报告文件"""
        if format == 'html':
            return self.exporter.export_html(report)
        elif format == 'json':
            return self.exporter.export_json(report)
        else:
            return self.explanation_generator.generate_full_explanation(report)
