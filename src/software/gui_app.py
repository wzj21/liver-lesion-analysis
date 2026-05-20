"""
Windows desktop GUI for the liver lesion analysis software.

The GUI is intentionally thin: it collects paths, runs the clinical inference
wrapper in a background thread, and opens the generated report.
"""

from __future__ import annotations

import os
import queue
import sys
import threading
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

import tkinter as tk
from tkinter import filedialog, messagebox, ttk


APP_TITLE = "肝脏病灶AI辅助分析系统"


class LiverLesionDesktopApp(tk.Tk):
    """Simple Windows desktop application."""

    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("840x600")
        self.minsize(760, 540)

        self.project_root = self._resolve_project_root()
        self.message_queue: "queue.Queue[tuple[str, Any]]" = queue.Queue()
        self.last_outputs: Dict[str, Any] = {}

        self.input_path = tk.StringVar()
        self.output_path = tk.StringVar(value=str(self.project_root / "outputs" / "desktop_case"))
        self.config_path = tk.StringVar(value=str(self.project_root / "configs" / "software_inference.yaml"))
        self.patient_id = tk.StringVar()
        self.allow_missing_weights = tk.BooleanVar(value=False)
        self.enable_llm = tk.BooleanVar(value=False)

        self._build_ui()
        self.after(150, self._poll_messages)

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        header = ttk.Frame(self, padding=(18, 14, 18, 8))
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)

        title = ttk.Label(header, text=APP_TITLE, font=("Microsoft YaHei UI", 18, "bold"))
        title.grid(row=0, column=0, sticky="w")
        subtitle = ttk.Label(
            header,
            text="选择DICOM序列文件夹或NIfTI文件，运行分割、分类和报告生成。",
            font=("Microsoft YaHei UI", 10),
        )
        subtitle.grid(row=1, column=0, sticky="w", pady=(6, 0))

        body = ttk.Frame(self, padding=(18, 8, 18, 10))
        body.grid(row=1, column=0, sticky="nsew")
        body.columnconfigure(1, weight=1)
        body.rowconfigure(7, weight=1)

        self._path_row(body, 0, "输入数据", self.input_path, self._choose_input, "选择DICOM文件夹或NIfTI文件")
        self._path_row(body, 1, "输出目录", self.output_path, self._choose_output, "选择输出目录")
        self._path_row(body, 2, "配置文件", self.config_path, self._choose_config, "选择software_inference.yaml")

        ttk.Label(body, text="病例ID").grid(row=3, column=0, sticky="w", pady=8)
        ttk.Entry(body, textvariable=self.patient_id).grid(row=3, column=1, columnspan=2, sticky="ew", pady=8)

        options = ttk.Frame(body)
        options.grid(row=4, column=1, columnspan=2, sticky="w", pady=(2, 8))
        ttk.Checkbutton(
            options,
            text="流程调试：允许缺失权重",
            variable=self.allow_missing_weights,
        ).grid(row=0, column=0, sticky="w", padx=(0, 18))
        ttk.Checkbutton(
            options,
            text="启用大模型报告建议",
            variable=self.enable_llm,
        ).grid(row=0, column=1, sticky="w")

        actions = ttk.Frame(body)
        actions.grid(row=5, column=0, columnspan=3, sticky="ew", pady=(8, 10))
        actions.columnconfigure(4, weight=1)
        self.run_button = ttk.Button(actions, text="开始分析", command=self._start_analysis)
        self.run_button.grid(row=0, column=0, padx=(0, 10))
        ttk.Button(actions, text="打开报告", command=self._open_report).grid(row=0, column=1, padx=(0, 10))
        ttk.Button(actions, text="打开输出目录", command=self._open_output_dir).grid(row=0, column=2, padx=(0, 10))
        ttk.Button(actions, text="清空日志", command=self._clear_log).grid(row=0, column=3)

        self.progress = ttk.Progressbar(body, mode="indeterminate")
        self.progress.grid(row=6, column=0, columnspan=3, sticky="ew", pady=(0, 8))

        log_frame = ttk.LabelFrame(body, text="运行日志", padding=8)
        log_frame.grid(row=7, column=0, columnspan=3, sticky="nsew")
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        self.log_text = tk.Text(log_frame, height=14, wrap="word", font=("Consolas", 10))
        self.log_text.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.log_text.configure(yscrollcommand=scrollbar.set)

        footer = ttk.Label(
            self,
            text="声明：本软件为AI辅助分析工具，不能替代医生诊断。临床使用需完成验证、审批和人工复核流程。",
            padding=(18, 0, 18, 12),
            foreground="#555555",
        )
        footer.grid(row=2, column=0, sticky="ew")

    @staticmethod
    def _resolve_project_root() -> Path:
        if getattr(sys, "frozen", False):
            executable_dir = Path(sys.executable).resolve().parent
            internal_dir = executable_dir / "_internal"
            if (internal_dir / "configs" / "software_inference.yaml").exists():
                return internal_dir
            return executable_dir
        return Path(__file__).resolve().parents[2]

    def _path_row(self, parent, row: int, label: str, variable: tk.StringVar, command, placeholder: str) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=8)
        entry = ttk.Entry(parent, textvariable=variable)
        entry.grid(row=row, column=1, sticky="ew", pady=8, padx=(0, 8))
        button = ttk.Button(parent, text="浏览...", command=command)
        button.grid(row=row, column=2, sticky="e", pady=8)
        entry.insert(0, variable.get())
        if not variable.get():
            entry.insert(0, placeholder)
            entry.delete(0, "end")

    def _choose_input(self) -> None:
        choice = messagebox.askyesno("选择输入", "输入是DICOM文件夹吗？\n选择“否”可选择NIfTI/DICOM单文件。")
        if choice:
            path = filedialog.askdirectory(title="选择DICOM序列文件夹")
        else:
            path = filedialog.askopenfilename(
                title="选择影像文件",
                filetypes=[
                    ("Medical images", "*.nii *.nii.gz *.dcm"),
                    ("All files", "*.*"),
                ],
            )
        if path:
            self.input_path.set(path)
            if not self.patient_id.get():
                self.patient_id.set(Path(path).stem.replace(".nii", ""))

    def _choose_output(self) -> None:
        path = filedialog.askdirectory(title="选择输出目录")
        if path:
            self.output_path.set(path)

    def _choose_config(self) -> None:
        path = filedialog.askopenfilename(
            title="选择配置文件",
            filetypes=[("YAML", "*.yaml *.yml"), ("All files", "*.*")],
        )
        if path:
            self.config_path.set(path)

    def _start_analysis(self) -> None:
        if not self.input_path.get():
            messagebox.showwarning("缺少输入", "请先选择DICOM文件夹或NIfTI/DICOM文件。")
            return
        if not self.output_path.get():
            messagebox.showwarning("缺少输出目录", "请先选择输出目录。")
            return

        self.run_button.configure(state="disabled")
        self.progress.start(12)
        self._log("开始分析...")

        worker = threading.Thread(target=self._run_analysis_worker, daemon=True)
        worker.start()

    def _run_analysis_worker(self) -> None:
        try:
            from .inference_app import LiverLesionSoftware, SoftwareInferenceConfig

            config = SoftwareInferenceConfig.from_yaml(self.config_path.get())
            if self.allow_missing_weights.get():
                config.require_checkpoints = False
            if self.enable_llm.get():
                config.llm["enabled"] = True

            app = LiverLesionSoftware(config)
            result = app.analyze(
                input_path=self.input_path.get(),
                output_dir=self.output_path.get(),
                patient_id=self.patient_id.get() or None,
            )
            self.message_queue.put(("done", result))
        except Exception as exc:
            self.message_queue.put(("error", f"{exc}\n\n{traceback.format_exc()}"))

    def _poll_messages(self) -> None:
        try:
            while True:
                kind, payload = self.message_queue.get_nowait()
                if kind == "done":
                    self.last_outputs = payload.get("outputs", {})
                    self._log("分析完成。")
                    self._log(f"JSON: {self.last_outputs.get('result_json')}")
                    self._log(f"报告: {self.last_outputs.get('report_html')}")
                    self.progress.stop()
                    self.run_button.configure(state="normal")
                    messagebox.showinfo("完成", "分析完成，已生成报告和结果文件。")
                elif kind == "error":
                    self._log("分析失败：")
                    self._log(str(payload))
                    self.progress.stop()
                    self.run_button.configure(state="normal")
                    messagebox.showerror("分析失败", str(payload).splitlines()[0])
        except queue.Empty:
            pass
        self.after(150, self._poll_messages)

    def _open_report(self) -> None:
        report = self.last_outputs.get("report_html")
        if not report:
            candidate = Path(self.output_path.get()) / "report.html"
            report = str(candidate) if candidate.exists() else None
        self._open_path(report, "尚未找到报告文件。")

    def _open_output_dir(self) -> None:
        path = self.output_path.get()
        if path:
            Path(path).mkdir(parents=True, exist_ok=True)
        self._open_path(path, "尚未设置输出目录。")

    @staticmethod
    def _open_path(path: Optional[str], missing_message: str) -> None:
        if not path or not Path(path).exists():
            messagebox.showwarning("无法打开", missing_message)
            return
        os.startfile(str(Path(path).resolve()))

    def _clear_log(self) -> None:
        self.log_text.delete("1.0", "end")

    def _log(self, message: str) -> None:
        self.log_text.insert("end", message + "\n")
        self.log_text.see("end")


def main() -> None:
    app = LiverLesionDesktopApp()
    app.mainloop()


if __name__ == "__main__":
    main()
