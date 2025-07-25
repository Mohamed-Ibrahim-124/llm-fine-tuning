import json
import statistics
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import psutil
import torch

from ..utils.logger import setup_logger

logger = setup_logger(__name__)

@dataclass
class PerformanceMetrics:
    """Data class for storing performance metrics."""
    latency_ms: float
    throughput_requests_per_second: float
    memory_usage_mb: float
    gpu_memory_usage_mb: Optional[float] = None
    cpu_usage_percent: float = 0.0
    timestamp: str = ""

class PerformanceMonitor:
    """Monitor and track performance metrics for the LLM pipeline."""
    
    def __init__(self):
        self.metrics_history: List[PerformanceMetrics] = []
        self.start_time = None
        
    def start_monitoring(self):
        """Start performance monitoring session."""
        self.start_time = time.time()
        logger.info("Performance monitoring started")
        
    def measure_inference_latency(self, model, input_text: str, max_length: int = 50) -> float:
        """Measure inference latency for a single prediction."""
        start_time = time.time()
        
        # Perform inference
        with torch.no_grad():
            if hasattr(model, 'generate'):
                # For transformer models
                outputs = model.generate(
                    input_text, 
                    max_length=max_length, 
                    num_return_sequences=1,
                    do_sample=False
                )
            else:
                # For pipeline models
                outputs = model(input_text, max_length=max_length)
        
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000
        
        return latency_ms
    
    def measure_throughput(self, model, test_inputs: List[str], batch_size: int = 1) -> float:
        """Measure throughput (requests per second)."""
        start_time = time.time()
        
        for i in range(0, len(test_inputs), batch_size):
            batch = test_inputs[i:i + batch_size]
            for input_text in batch:
                self.measure_inference_latency(model, input_text)
        
        end_time = time.time()
        total_time = end_time - start_time
        throughput = len(test_inputs) / total_time
        
        return throughput
    
    def get_system_metrics(self) -> Dict[str, float]:
        """Get current system resource usage."""
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)
        
        metrics = {
            "memory_usage_mb": memory.used / (1024 * 1024),
            "memory_percent": memory.percent,
            "cpu_usage_percent": cpu_percent,
        }
        
        # GPU metrics if available
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / (1024 * 1024)
            metrics["gpu_memory_usage_mb"] = gpu_memory
            metrics["gpu_memory_percent"] = (gpu_memory / torch.cuda.get_device_properties(0).total_memory * 1024 * 1024) * 100
        
        return metrics
    
    def record_metrics(self, latency_ms: float, throughput: float, system_metrics: Dict[str, float]):
        """Record performance metrics."""
        metric = PerformanceMetrics(
            latency_ms=latency_ms,
            throughput_requests_per_second=throughput,
            memory_usage_mb=system_metrics.get("memory_usage_mb", 0),
            gpu_memory_usage_mb=system_metrics.get("gpu_memory_usage_mb"),
            cpu_usage_percent=system_metrics.get("cpu_usage_percent", 0),
            timestamp=datetime.now().isoformat()
        )
        
        self.metrics_history.append(metric)
        logger.info("Recorded metrics: latency=%.2fms, throughput=%.2f req/s", 
                    latency_ms, throughput)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get a summary of performance metrics."""
        if not self.metrics_history:
            return {"error": "No metrics recorded"}
        
        latencies = [m.latency_ms for m in self.metrics_history]
        throughputs = [m.throughput_requests_per_second for m in self.metrics_history]
        memory_usage = [m.memory_usage_mb for m in self.metrics_history]
        
        summary = {
            "total_requests": len(self.metrics_history),
            "latency": {
                "mean_ms": statistics.mean(latencies),
                "median_ms": statistics.median(latencies),
                "min_ms": min(latencies),
                "max_ms": max(latencies),
                "std_ms": statistics.stdev(latencies) if len(latencies) > 1 else 0
            },
            "throughput": {
                "mean_rps": statistics.mean(throughputs),
                "median_rps": statistics.median(throughputs),
                "min_rps": min(throughputs),
                "max_rps": max(throughputs)
            },
            "memory": {
                "mean_mb": statistics.mean(memory_usage),
                "max_mb": max(memory_usage)
            },
            "monitoring_duration_seconds": time.time() - self.start_time if self.start_time else 0
        }
        
        # Add GPU metrics if available
        gpu_memory_usage = [m.gpu_memory_usage_mb for m in self.metrics_history if m.gpu_memory_usage_mb is not None]
        if gpu_memory_usage:
            summary["gpu_memory"] = {
                "mean_mb": statistics.mean(gpu_memory_usage),
                "max_mb": max(gpu_memory_usage)
            }
        
        return summary
    
    def save_metrics(self, filepath: str = "logs/performance_metrics.json"):
        """Save performance metrics to file."""
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        data = {
            "summary": self.get_performance_summary(),
            "detailed_metrics": [
                {
                    "latency_ms": m.latency_ms,
                    "throughput_requests_per_second": m.throughput_requests_per_second,
                    "memory_usage_mb": m.memory_usage_mb,
                    "gpu_memory_usage_mb": m.gpu_memory_usage_mb,
                    "cpu_usage_percent": m.cpu_usage_percent,
                    "timestamp": m.timestamp
                }
                for m in self.metrics_history
            ]
        }
        
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        
        logger.info("Performance metrics saved to %s", filepath)
    
    def compare_models(self, baseline_model, fine_tuned_model, test_inputs: List[str]) -> Dict[str, Any]:
        """Compare performance between baseline and fine-tuned models."""
        logger.info("Comparing performance between baseline and fine-tuned models")
        
        # Test baseline model
        self.start_monitoring()
        baseline_latencies = []
        for input_text in test_inputs:
            latency = self.measure_inference_latency(baseline_model, input_text)
            baseline_latencies.append(latency)
        baseline_summary = self.get_performance_summary()
        
        # Test fine-tuned model
        self.metrics_history = []  # Reset for fine-tuned model
        self.start_monitoring()
        fine_tuned_latencies = []
        for input_text in test_inputs:
            latency = self.measure_inference_latency(fine_tuned_model, input_text)
            fine_tuned_latencies.append(latency)
        fine_tuned_summary = self.get_performance_summary()
        
        # Calculate improvements
        baseline_avg_latency = statistics.mean(baseline_latencies)
        fine_tuned_avg_latency = statistics.mean(fine_tuned_latencies)
        
        latency_improvement = ((baseline_avg_latency - fine_tuned_avg_latency) / baseline_avg_latency) * 100
        
        comparison = {
            "baseline": {
                "avg_latency_ms": baseline_avg_latency,
                "summary": baseline_summary
            },
            "fine_tuned": {
                "avg_latency_ms": fine_tuned_avg_latency,
                "summary": fine_tuned_summary
            },
            "improvements": {
                "latency_improvement_percent": latency_improvement,
                "faster": fine_tuned_avg_latency < baseline_avg_latency
            }
        }
        
        logger.info("Performance comparison completed: %.2f%% latency improvement", 
                    latency_improvement)
        
        return comparison

def create_performance_monitor() -> PerformanceMonitor:
    """Create and return a performance monitor instance."""
    return PerformanceMonitor()

if __name__ == "__main__":
    # Example usage
    monitor = PerformanceMonitor()
    monitor.start_monitoring()
    
    # Simulate some metrics
    system_metrics = monitor.get_system_metrics()
    monitor.record_metrics(100.0, 10.0, system_metrics)
    
    summary = monitor.get_performance_summary()
    print(json.dumps(summary, indent=2)) 