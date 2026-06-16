"""Мониторинг живости и нагрузки во время прогона LLM-судей.

Снимает CPU/RAM раз в шаг и пишет временную метку для каждого
полученного токена/чанка от модели -- чтобы по логу было видно,
что модель не "зависла", а постепенно генерирует ответ.
"""
import psutil
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from pathlib import Path
from typing import List, Dict


class SystemMonitor:
    def __init__(self, output_dir: str = "results/monitoring"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.snapshots: List[Dict] = []
        self.heartbeats: List[Dict] = []
        self._start = datetime.now()

    def snapshot(self, label: str = "") -> Dict:
        cpu = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory()
        s = {
            "time": datetime.now(),
            "elapsed_sec": (datetime.now() - self._start).total_seconds(),
            "cpu_percent": cpu,
            "ram_percent": mem.percent,
            "ram_used_gb": mem.used / (1024 ** 3),
            "label": label,
        }
        self.snapshots.append(s)
        return s

    def heartbeat(self, query_idx: int, model: str, tokens_so_far: int):
        t = datetime.now()
        self.heartbeats.append({
            "time": t,
            "elapsed_sec": (t - self._start).total_seconds(),
            "query_idx": query_idx,
            "model": model,
            "tokens_so_far": tokens_so_far,
        })

    def save_plot(self, filename: str = "system_load.jpeg") -> str:
        if len(self.snapshots) < 2:
            return ""
        times = [s["time"] for s in self.snapshots]
        cpus = [s["cpu_percent"] for s in self.snapshots]
        rams = [s["ram_percent"] for s in self.snapshots]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        fig.suptitle("Нагрузка системы во время прогона мультимодального LLM-судьи",
                      fontsize=13, fontweight="bold")

        ax1.plot(times, cpus, "b-", linewidth=1.2)
        ax1.fill_between(times, cpus, alpha=0.3, color="blue")
        ax1.set_ylabel("CPU, %")
        ax1.set_ylim(0, 100)
        ax1.grid(True, alpha=0.3)

        ax2.plot(times, rams, "g-", linewidth=1.2)
        ax2.fill_between(times, rams, alpha=0.3, color="green")
        ax2.set_ylabel("RAM, %")
        ax2.set_ylim(0, 100)
        ax2.set_xlabel("Время")
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))

        for hb in self.heartbeats:
            if hb["tokens_so_far"] == 0:
                ax1.axvline(hb["time"], color="red", alpha=0.15, linewidth=0.8)

        plt.xticks(rotation=45)
        plt.tight_layout()

        path = self.output_dir / filename
        fig.savefig(path, dpi=130, bbox_inches="tight", format="jpeg")
        plt.close(fig)
        return str(path)

    def summary(self) -> Dict:
        if not self.snapshots:
            return {}
        return {
            "duration_sec": self.snapshots[-1]["elapsed_sec"],
            "avg_cpu": sum(s["cpu_percent"] for s in self.snapshots) / len(self.snapshots),
            "max_cpu": max(s["cpu_percent"] for s in self.snapshots),
            "avg_ram": sum(s["ram_percent"] for s in self.snapshots) / len(self.snapshots),
            "max_ram": max(s["ram_percent"] for s in self.snapshots),
            "snapshots": len(self.snapshots),
            "heartbeats": len(self.heartbeats),
        }
