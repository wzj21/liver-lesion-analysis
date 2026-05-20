"""
结构化报告生成模块
Structured Report Generator Module

包含:
- DICOM SR生成
- HL7 FHIR报告
- PDF报告生成
- 临床结构化模板
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
import uuid


@dataclass
class LesionMeasurement:
    """病灶测量数据"""
    lesion_id: str
    location: str  # 肝段位置
    size_mm: Tuple[float, float, float]  # (长, 宽, 深)
    volume_mm3: float
    hu_mean: float
    hu_std: float
    boundary_type: str  # 清晰/模糊
    enhancement_pattern: Optional[str] = None


@dataclass  
class DiagnosisResult:
    """诊断结果"""
    primary_diagnosis: str
    diagnosis_code: str  # ICD-10
    confidence: float
    differential_diagnoses: List[Dict] = field(default_factory=list)
    lesion_type: str = ""
    is_malignant: Optional[bool] = None
    activity_status: Optional[str] = None  # 活动期/静止期


@dataclass
class StructuredReport:
    """结构化报告"""
    # 报告元信息
    report_id: str
    report_time: str
    report_type: str = "AI辅助诊断报告"
    
    # 患者信息
    patient_id: Optional[str] = None
    patient_name: Optional[str] = None
    patient_age: Optional[int] = None
    patient_gender: Optional[str] = None
    
    # 检查信息
    study_id: Optional[str] = None
    study_date: Optional[str] = None
    modality: str = "CT"
    body_part: str = "肝脏"
    
    # 影像发现
    liver_volume_ml: Optional[float] = None
    lesion_count: int = 0
    lesions: List[LesionMeasurement] = field(default_factory=list)
    
    # 诊断结果
    diagnosis: Optional[DiagnosisResult] = None
    
    # AI分析
    ai_confidence: float = 0.0
    uncertainty_level: str = "低"
    needs_review: bool = False
    review_reason: Optional[str] = None
    
    # 可解释性
    key_findings: List[str] = field(default_factory=list)
    attention_regions: List[Dict] = field(default_factory=list)
    similar_cases: List[Dict] = field(default_factory=list)
    
    # 建议
    recommendations: List[str] = field(default_factory=list)
    follow_up: Optional[str] = None
    
    # 可视化路径
    visualization_paths: Dict[str, str] = field(default_factory=dict)


from typing import Tuple


class StructuredReportGenerator:
    """结构化报告生成器"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.templates_dir = config.get('templates_dir', 'templates')
        
        # ICD-10编码映射
        self.icd10_codes = {
            '良性': 'D13.4',
            '恶性': 'C22.0',
            '囊型包虫病': 'B67.8',
            '泡型包虫病': 'B67.5',
            '肝血管瘤': 'D18.0',
            '肝囊肿': 'K76.89'
        }
    
    def generate_report(
        self,
        analysis_results: Dict,
        patient_info: Dict = None,
        study_info: Dict = None
    ) -> StructuredReport:
        """
        生成结构化报告
        
        Args:
            analysis_results: AI分析结果
            patient_info: 患者信息
            study_info: 检查信息
            
        Returns:
            结构化报告对象
        """
        report = StructuredReport(
            report_id=str(uuid.uuid4())[:8].upper(),
            report_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        
        # 填充患者信息
        if patient_info:
            report.patient_id = patient_info.get('id')
            report.patient_name = patient_info.get('name')
            report.patient_age = patient_info.get('age')
            report.patient_gender = patient_info.get('gender')
        
        # 填充检查信息
        if study_info:
            report.study_id = study_info.get('id')
            report.study_date = study_info.get('date')
            report.modality = study_info.get('modality', 'CT')
        
        # 填充分析结果
        self._fill_analysis_results(report, analysis_results)
        
        return report
    
    def _fill_analysis_results(self, report: StructuredReport, results: Dict):
        """填充分析结果"""
        # 肝脏信息
        if 'liver' in results:
            report.liver_volume_ml = results['liver'].get('volume_ml')
        
        # 病灶信息
        if 'lesions' in results:
            lesions_data = results['lesions']
            report.lesion_count = len(lesions_data)
            
            for i, lesion in enumerate(lesions_data):
                measurement = LesionMeasurement(
                    lesion_id=f"L{i+1:02d}",
                    location=lesion.get('location', '未知'),
                    size_mm=tuple(lesion.get('size', [0, 0, 0])),
                    volume_mm3=lesion.get('volume', 0),
                    hu_mean=lesion.get('hu_mean', 0),
                    hu_std=lesion.get('hu_std', 0),
                    boundary_type=lesion.get('boundary', '清晰')
                )
                report.lesions.append(measurement)
        
        # 诊断结果
        if 'diagnosis' in results:
            diag = results['diagnosis']
            primary = diag.get('primary', '未确定')
            
            report.diagnosis = DiagnosisResult(
                primary_diagnosis=primary,
                diagnosis_code=self.icd10_codes.get(primary, 'R93.2'),
                confidence=diag.get('confidence', 0),
                lesion_type=diag.get('lesion_type', ''),
                is_malignant=diag.get('is_malignant'),
                activity_status=diag.get('activity_status'),
                differential_diagnoses=diag.get('differential', [])
            )
        
        # AI置信度和不确定性
        report.ai_confidence = results.get('confidence', 0)
        uncertainty = results.get('uncertainty', 0)
        
        if uncertainty < 0.2:
            report.uncertainty_level = "低"
        elif uncertainty < 0.4:
            report.uncertainty_level = "中"
        else:
            report.uncertainty_level = "高"
            report.needs_review = True
            report.review_reason = "模型不确定性超过阈值"
        
        # 关键发现
        report.key_findings = results.get('key_findings', [])
        report.attention_regions = results.get('attention_regions', [])
        report.similar_cases = results.get('similar_cases', [])
        
        # 建议
        report.recommendations = results.get('recommendations', [])
        report.follow_up = results.get('follow_up')
        
        # 可视化
        report.visualization_paths = results.get('visualizations', {})
    
    def to_json(self, report: StructuredReport) -> str:
        """转换为JSON"""
        data = asdict(report)
        # 处理嵌套dataclass
        if report.diagnosis:
            data['diagnosis'] = asdict(report.diagnosis)
        data['lesions'] = [asdict(l) if hasattr(l, '__dataclass_fields__') else l for l in report.lesions]
        return json.dumps(data, ensure_ascii=False, indent=2)
    
    def to_dicom_sr(self, report: StructuredReport) -> Dict:
        """
        转换为DICOM SR格式
        
        Returns:
            DICOM SR结构
        """
        sr = {
            'SOPClassUID': '1.2.840.10008.5.1.4.1.1.88.22',  # Enhanced SR
            'Modality': 'SR',
            'SeriesDescription': 'AI Liver Lesion Analysis Report',
            'ContentDate': datetime.now().strftime('%Y%m%d'),
            'ContentTime': datetime.now().strftime('%H%M%S'),
            'ContentSequence': []
        }
        
        # 添加患者信息
        if report.patient_id:
            sr['PatientID'] = report.patient_id
        if report.patient_name:
            sr['PatientName'] = report.patient_name
        
        # 添加发现
        findings = {
            'ConceptNameCodeSequence': {
                'CodeValue': '121070',
                'CodingSchemeDesignator': 'DCM',
                'CodeMeaning': 'Findings'
            },
            'TextValue': self._format_findings_text(report)
        }
        sr['ContentSequence'].append(findings)
        
        # 添加诊断
        if report.diagnosis:
            diagnosis = {
                'ConceptNameCodeSequence': {
                    'CodeValue': '121073',
                    'CodingSchemeDesignator': 'DCM', 
                    'CodeMeaning': 'Impression'
                },
                'TextValue': f"{report.diagnosis.primary_diagnosis} (ICD-10: {report.diagnosis.diagnosis_code})"
            }
            sr['ContentSequence'].append(diagnosis)
        
        return sr
    
    def _format_findings_text(self, report: StructuredReport) -> str:
        """格式化发现文本"""
        lines = []
        
        if report.liver_volume_ml:
            lines.append(f"肝脏体积：{report.liver_volume_ml:.1f} ml")
        
        lines.append(f"病灶数量：{report.lesion_count}")
        
        for lesion in report.lesions:
            lines.append(f"- {lesion.lesion_id}: 位置{lesion.location}, "
                        f"大小{lesion.size_mm[0]:.1f}x{lesion.size_mm[1]:.1f}x{lesion.size_mm[2]:.1f}mm, "
                        f"体积{lesion.volume_mm3:.1f}mm³")
        
        return "\n".join(lines)
    
    def to_fhir(self, report: StructuredReport) -> Dict:
        """
        转换为HL7 FHIR DiagnosticReport格式
        
        Returns:
            FHIR DiagnosticReport资源
        """
        fhir_report = {
            'resourceType': 'DiagnosticReport',
            'id': report.report_id,
            'status': 'final',
            'category': [{
                'coding': [{
                    'system': 'http://terminology.hl7.org/CodeSystem/v2-0074',
                    'code': 'RAD',
                    'display': 'Radiology'
                }]
            }],
            'code': {
                'coding': [{
                    'system': 'http://loinc.org',
                    'code': '24531-6',
                    'display': 'CT Abdomen'
                }]
            },
            'effectiveDateTime': report.report_time,
            'issued': report.report_time,
            'conclusion': '',
            'conclusionCode': [],
            'presentedForm': []
        }
        
        # 患者引用
        if report.patient_id:
            fhir_report['subject'] = {
                'reference': f'Patient/{report.patient_id}'
            }
        
        # 诊断结论
        if report.diagnosis:
            fhir_report['conclusion'] = report.diagnosis.primary_diagnosis
            fhir_report['conclusionCode'].append({
                'coding': [{
                    'system': 'http://hl7.org/fhir/sid/icd-10',
                    'code': report.diagnosis.diagnosis_code,
                    'display': report.diagnosis.primary_diagnosis
                }]
            })
        
        return fhir_report


class PDFReportGenerator:
    """PDF报告生成器"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
    
    def generate(
        self,
        report: StructuredReport,
        output_path: str,
        include_images: bool = True
    ) -> str:
        """
        生成PDF报告
        
        Args:
            report: 结构化报告
            output_path: 输出路径
            include_images: 是否包含图像
            
        Returns:
            PDF文件路径
        """
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import mm
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
            from reportlab.lib import colors
            from reportlab.pdfbase import pdfmetrics
            from reportlab.pdfbase.ttfonts import TTFont
            
            # 注册中文字体（如果可用）
            try:
                pdfmetrics.registerFont(TTFont('SimHei', 'SimHei.ttf'))
                chinese_font = 'SimHei'
            except:
                chinese_font = 'Helvetica'
            
            doc = SimpleDocTemplate(output_path, pagesize=A4)
            styles = getSampleStyleSheet()
            
            # 自定义样式
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontName=chinese_font,
                fontSize=18,
                spaceAfter=30
            )
            
            heading_style = ParagraphStyle(
                'CustomHeading',
                parent=styles['Heading2'],
                fontName=chinese_font,
                fontSize=14,
                spaceAfter=12
            )
            
            normal_style = ParagraphStyle(
                'CustomNormal',
                parent=styles['Normal'],
                fontName=chinese_font,
                fontSize=10,
                spaceAfter=6
            )
            
            story = []
            
            # 标题
            story.append(Paragraph("肝脏病灶智能分析报告", title_style))
            story.append(Spacer(1, 12))
            
            # 基本信息表格
            info_data = [
                ['报告编号', report.report_id, '报告时间', report.report_time],
                ['患者ID', report.patient_id or '-', '患者姓名', report.patient_name or '-'],
                ['年龄', str(report.patient_age) if report.patient_age else '-', 
                 '性别', report.patient_gender or '-'],
            ]
            
            info_table = Table(info_data, colWidths=[60, 120, 60, 120])
            info_table.setStyle(TableStyle([
                ('FONTNAME', (0, 0), (-1, -1), chinese_font),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
                ('BACKGROUND', (2, 0), (2, -1), colors.lightgrey),
            ]))
            story.append(info_table)
            story.append(Spacer(1, 20))
            
            # 诊断结果
            story.append(Paragraph("诊断结果", heading_style))
            if report.diagnosis:
                story.append(Paragraph(
                    f"主要诊断：{report.diagnosis.primary_diagnosis} "
                    f"(置信度：{report.diagnosis.confidence:.1%})",
                    normal_style
                ))
                story.append(Paragraph(
                    f"ICD-10编码：{report.diagnosis.diagnosis_code}",
                    normal_style
                ))
            story.append(Spacer(1, 12))
            
            # 病灶信息
            if report.lesions:
                story.append(Paragraph("病灶详情", heading_style))
                
                lesion_data = [['编号', '位置', '大小(mm)', '体积(mm³)', 'HU值']]
                for lesion in report.lesions:
                    lesion_data.append([
                        lesion.lesion_id,
                        lesion.location,
                        f"{lesion.size_mm[0]:.1f}×{lesion.size_mm[1]:.1f}×{lesion.size_mm[2]:.1f}",
                        f"{lesion.volume_mm3:.1f}",
                        f"{lesion.hu_mean:.1f}±{lesion.hu_std:.1f}"
                    ])
                
                lesion_table = Table(lesion_data, colWidths=[40, 80, 100, 80, 80])
                lesion_table.setStyle(TableStyle([
                    ('FONTNAME', (0, 0), (-1, -1), chinese_font),
                    ('FONTSIZE', (0, 0), (-1, -1), 9),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                    ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                ]))
                story.append(lesion_table)
                story.append(Spacer(1, 12))
            
            # AI分析
            story.append(Paragraph("AI分析", heading_style))
            story.append(Paragraph(f"置信度：{report.ai_confidence:.1%}", normal_style))
            story.append(Paragraph(f"不确定性：{report.uncertainty_level}", normal_style))
            
            if report.needs_review:
                story.append(Paragraph(
                    f"⚠ 建议人工复核：{report.review_reason}",
                    normal_style
                ))
            story.append(Spacer(1, 12))
            
            # 建议
            if report.recommendations:
                story.append(Paragraph("临床建议", heading_style))
                for i, rec in enumerate(report.recommendations, 1):
                    story.append(Paragraph(f"{i}. {rec}", normal_style))
            
            # 添加图像
            if include_images and report.visualization_paths:
                story.append(Spacer(1, 20))
                story.append(Paragraph("可视化结果", heading_style))
                
                for name, path in report.visualization_paths.items():
                    if os.path.exists(path):
                        try:
                            img = Image(path, width=400, height=300)
                            story.append(img)
                            story.append(Paragraph(name, normal_style))
                            story.append(Spacer(1, 12))
                        except:
                            pass
            
            # 免责声明
            story.append(Spacer(1, 30))
            disclaimer = ParagraphStyle(
                'Disclaimer',
                parent=styles['Normal'],
                fontName=chinese_font,
                fontSize=8,
                textColor=colors.grey
            )
            story.append(Paragraph(
                "本报告由AI辅助诊断系统自动生成，仅供参考，不能替代医生的专业诊断。",
                disclaimer
            ))
            
            doc.build(story)
            return output_path
            
        except ImportError:
            # 如果reportlab不可用，生成简单的HTML
            return self._generate_html_fallback(report, output_path.replace('.pdf', '.html'))
    
    def _generate_html_fallback(self, report: StructuredReport, output_path: str) -> str:
        """HTML备用方案"""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>肝脏病灶分析报告</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #2c3e50; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <h1>肝脏病灶智能分析报告</h1>
    <p>报告编号：{report.report_id}</p>
    <p>报告时间：{report.report_time}</p>
    
    <h2>诊断结果</h2>
    <p>{report.diagnosis.primary_diagnosis if report.diagnosis else '未确定'}</p>
    
    <h2>临床建议</h2>
    <ul>
        {''.join(f'<li>{rec}</li>' for rec in report.recommendations)}
    </ul>
</body>
</html>
"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        return output_path
