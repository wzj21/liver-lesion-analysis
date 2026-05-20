"""
3D可视化模块
3D Visualization Module

包含:
- VTK体绘制
- 肝脏/病灶3D渲染
- 不确定性叠加显示
- 交互式可视化
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import os


class VolumeRenderer:
    """
    体绘制渲染器
    
    使用VTK进行3D医学图像可视化
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.renderer = None
        self.render_window = None
        self.interactor = None
        
        self._check_vtk()
    
    def _check_vtk(self):
        """检查VTK是否可用"""
        try:
            import vtk
            self.vtk = vtk
            self.vtk_available = True
        except ImportError:
            self.vtk_available = False
            print("Warning: VTK not installed. 3D visualization will be limited.")
    
    def create_volume_from_numpy(
        self,
        volume_data: np.ndarray,
        spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    ):
        """
        从numpy数组创建VTK体数据
        
        Args:
            volume_data: 3D numpy数组 [D, H, W]
            spacing: 体素间距 (z, y, x)
            
        Returns:
            VTK ImageData对象
        """
        if not self.vtk_available:
            return None
        
        vtk = self.vtk
        
        # 创建vtkImageData
        image_data = vtk.vtkImageData()
        image_data.SetDimensions(volume_data.shape[2], volume_data.shape[1], volume_data.shape[0])
        image_data.SetSpacing(spacing[2], spacing[1], spacing[0])
        
        # 转换数据
        flat_data = volume_data.flatten(order='F').astype(np.float32)
        
        vtk_array = vtk.vtkFloatArray()
        vtk_array.SetNumberOfComponents(1)
        vtk_array.SetNumberOfTuples(len(flat_data))
        
        for i, val in enumerate(flat_data):
            vtk_array.SetValue(i, val)
        
        image_data.GetPointData().SetScalars(vtk_array)
        
        return image_data
    
    def render_liver_and_lesions(
        self,
        ct_volume: np.ndarray,
        liver_mask: np.ndarray,
        lesion_mask: Optional[np.ndarray] = None,
        uncertainty_map: Optional[np.ndarray] = None,
        spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        output_path: Optional[str] = None
    ) -> Optional[Any]:
        """
        渲染肝脏和病灶的3D可视化
        
        Args:
            ct_volume: CT图像 [D, H, W]
            liver_mask: 肝脏分割掩码 [D, H, W]
            lesion_mask: 病灶分割掩码 [D, H, W]
            uncertainty_map: 不确定性图 [D, H, W]
            spacing: 体素间距
            output_path: 输出图像路径
            
        Returns:
            渲染结果或None
        """
        if not self.vtk_available:
            return self._render_matplotlib_fallback(
                ct_volume, liver_mask, lesion_mask, output_path
            )
        
        vtk = self.vtk
        
        # 创建渲染器
        renderer = vtk.vtkRenderer()
        renderer.SetBackground(0.1, 0.1, 0.1)
        
        # 1. 渲染肝脏（半透明）
        liver_actor = self._create_surface_actor(
            liver_mask, spacing,
            color=(0.8, 0.4, 0.4),
            opacity=0.3
        )
        if liver_actor:
            renderer.AddActor(liver_actor)
        
        # 2. 渲染病灶（不透明）
        if lesion_mask is not None:
            lesion_actor = self._create_surface_actor(
                lesion_mask, spacing,
                color=(1.0, 0.0, 0.0),
                opacity=1.0
            )
            if lesion_actor:
                renderer.AddActor(lesion_actor)
        
        # 3. 不确定性叠加
        if uncertainty_map is not None:
            uncertainty_actor = self._create_uncertainty_overlay(
                uncertainty_map, spacing
            )
            if uncertainty_actor:
                renderer.AddActor(uncertainty_actor)
        
        # 设置相机
        renderer.ResetCamera()
        camera = renderer.GetActiveCamera()
        camera.Elevation(30)
        camera.Azimuth(30)
        
        # 创建渲染窗口
        render_window = vtk.vtkRenderWindow()
        render_window.SetSize(800, 800)
        render_window.AddRenderer(renderer)
        render_window.SetOffScreenRendering(1)
        
        # 渲染到图像
        if output_path:
            render_window.Render()
            
            # 保存图像
            window_to_image = vtk.vtkWindowToImageFilter()
            window_to_image.SetInput(render_window)
            window_to_image.Update()
            
            writer = vtk.vtkPNGWriter()
            writer.SetFileName(output_path)
            writer.SetInputConnection(window_to_image.GetOutputPort())
            writer.Write()
            
            return output_path
        
        return render_window
    
    def _create_surface_actor(
        self,
        mask: np.ndarray,
        spacing: Tuple[float, float, float],
        color: Tuple[float, float, float],
        opacity: float
    ):
        """创建表面actor"""
        vtk = self.vtk
        
        # 创建ImageData
        image_data = self.create_volume_from_numpy(mask.astype(np.float32), spacing)
        
        # Marching Cubes提取等值面
        contour = vtk.vtkMarchingCubes()
        contour.SetInputData(image_data)
        contour.SetValue(0, 0.5)
        contour.Update()
        
        # 平滑
        smoother = vtk.vtkSmoothPolyDataFilter()
        smoother.SetInputConnection(contour.GetOutputPort())
        smoother.SetNumberOfIterations(50)
        smoother.Update()
        
        # 创建mapper和actor
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(smoother.GetOutputPort())
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*color)
        actor.GetProperty().SetOpacity(opacity)
        
        return actor
    
    def _create_uncertainty_overlay(
        self,
        uncertainty_map: np.ndarray,
        spacing: Tuple[float, float, float]
    ):
        """创建不确定性叠加层"""
        vtk = self.vtk
        
        # 只显示高不确定性区域
        threshold = 0.3
        high_uncertainty = (uncertainty_map > threshold).astype(np.float32)
        
        if high_uncertainty.sum() == 0:
            return None
        
        image_data = self.create_volume_from_numpy(high_uncertainty, spacing)
        
        contour = vtk.vtkMarchingCubes()
        contour.SetInputData(image_data)
        contour.SetValue(0, 0.5)
        contour.Update()
        
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(contour.GetOutputPort())
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(1.0, 1.0, 0.0)  # 黄色表示不确定
        actor.GetProperty().SetOpacity(0.5)
        
        return actor
    
    def _render_matplotlib_fallback(
        self,
        ct_volume: np.ndarray,
        liver_mask: np.ndarray,
        lesion_mask: Optional[np.ndarray],
        output_path: Optional[str]
    ):
        """Matplotlib备用渲染"""
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # 下采样以减少点数
        step = 4
        
        # 肝脏表面点
        liver_points = np.array(np.where(liver_mask[::step, ::step, ::step] > 0))
        if liver_points.shape[1] > 0:
            ax.scatter(
                liver_points[2], liver_points[1], liver_points[0],
                c='coral', alpha=0.1, s=1, label='Liver'
            )
        
        # 病灶点
        if lesion_mask is not None:
            lesion_points = np.array(np.where(lesion_mask[::step, ::step, ::step] > 0))
            if lesion_points.shape[1] > 0:
                ax.scatter(
                    lesion_points[2], lesion_points[1], lesion_points[0],
                    c='red', alpha=0.8, s=5, label='Lesion'
                )
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        ax.set_title('3D Liver and Lesion Visualization')
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            return output_path
        
        return fig
    
    def create_animation(
        self,
        ct_volume: np.ndarray,
        liver_mask: np.ndarray,
        lesion_mask: Optional[np.ndarray],
        output_path: str,
        num_frames: int = 36,
        spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    ):
        """
        创建旋转动画
        
        Args:
            output_path: 输出GIF路径
            num_frames: 帧数
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.animation as animation
        
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        step = 4
        liver_points = np.array(np.where(liver_mask[::step, ::step, ::step] > 0))
        
        def update(frame):
            ax.clear()
            ax.view_init(elev=20, azim=frame * 360 / num_frames)
            
            if liver_points.shape[1] > 0:
                ax.scatter(
                    liver_points[2], liver_points[1], liver_points[0],
                    c='coral', alpha=0.1, s=1
                )
            
            if lesion_mask is not None:
                lesion_points = np.array(np.where(lesion_mask[::step, ::step, ::step] > 0))
                if lesion_points.shape[1] > 0:
                    ax.scatter(
                        lesion_points[2], lesion_points[1], lesion_points[0],
                        c='red', alpha=0.8, s=5
                    )
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            return ax,
        
        anim = animation.FuncAnimation(fig, update, frames=num_frames, interval=100)
        anim.save(output_path, writer='pillow', fps=10)
        plt.close()
        
        return output_path


class SliceViewer:
    """
    多平面切片查看器
    
    支持轴向、矢状、冠状视图
    """
    
    def __init__(self):
        pass
    
    def create_multi_planar_view(
        self,
        volume: np.ndarray,
        masks: Dict[str, np.ndarray] = None,
        slice_indices: Dict[str, int] = None,
        output_path: Optional[str] = None
    ):
        """
        创建多平面视图
        
        Args:
            volume: 3D体数据 [D, H, W]
            masks: 掩码字典 {'liver': mask1, 'lesion': mask2}
            slice_indices: 切片索引 {'axial': z, 'sagittal': x, 'coronal': y}
            output_path: 输出路径
        """
        import matplotlib.pyplot as plt
        
        D, H, W = volume.shape
        
        if slice_indices is None:
            slice_indices = {
                'axial': D // 2,
                'sagittal': W // 2,
                'coronal': H // 2
            }
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 轴向视图 (Axial)
        ax = axes[0]
        axial_slice = volume[slice_indices['axial'], :, :]
        ax.imshow(axial_slice, cmap='gray')
        if masks:
            for name, mask in masks.items():
                mask_slice = mask[slice_indices['axial'], :, :]
                color = 'red' if 'lesion' in name.lower() else 'cyan'
                ax.contour(mask_slice, levels=[0.5], colors=color, linewidths=1)
        ax.set_title(f'Axial (Z={slice_indices["axial"]})')
        ax.axis('off')
        
        # 矢状视图 (Sagittal)
        ax = axes[1]
        sagittal_slice = volume[:, :, slice_indices['sagittal']]
        ax.imshow(sagittal_slice, cmap='gray', aspect='auto')
        if masks:
            for name, mask in masks.items():
                mask_slice = mask[:, :, slice_indices['sagittal']]
                color = 'red' if 'lesion' in name.lower() else 'cyan'
                ax.contour(mask_slice, levels=[0.5], colors=color, linewidths=1)
        ax.set_title(f'Sagittal (X={slice_indices["sagittal"]})')
        ax.axis('off')
        
        # 冠状视图 (Coronal)
        ax = axes[2]
        coronal_slice = volume[:, slice_indices['coronal'], :]
        ax.imshow(coronal_slice, cmap='gray', aspect='auto')
        if masks:
            for name, mask in masks.items():
                mask_slice = mask[:, slice_indices['coronal'], :]
                color = 'red' if 'lesion' in name.lower() else 'cyan'
                ax.contour(mask_slice, levels=[0.5], colors=color, linewidths=1)
        ax.set_title(f'Coronal (Y={slice_indices["coronal"]})')
        ax.axis('off')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            return output_path
        
        return fig
    
    def create_slice_montage(
        self,
        volume: np.ndarray,
        mask: Optional[np.ndarray] = None,
        num_slices: int = 16,
        output_path: Optional[str] = None
    ):
        """创建切片蒙太奇"""
        import matplotlib.pyplot as plt
        
        D = volume.shape[0]
        indices = np.linspace(0, D-1, num_slices, dtype=int)
        
        cols = 4
        rows = (num_slices + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(16, 4*rows))
        axes = axes.flatten()
        
        for i, idx in enumerate(indices):
            ax = axes[i]
            ax.imshow(volume[idx], cmap='gray')
            
            if mask is not None:
                ax.contour(mask[idx], levels=[0.5], colors='red', linewidths=1)
            
            ax.set_title(f'Slice {idx}')
            ax.axis('off')
        
        # 隐藏多余的子图
        for i in range(len(indices), len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            return output_path
        
        return fig
