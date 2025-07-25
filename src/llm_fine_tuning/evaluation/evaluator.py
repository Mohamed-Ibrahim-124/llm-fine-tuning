import json
import time
from typing import Any, Dict, List

from evaluate import load
from transformers import AutoModelForCausalLM, pipeline

from ..utils.logger import setup_logger
from .benchmark_generator import create_benchmark_dataset
from .performance_monitor import create_performance_monitor

logger = setup_logger(__name__)


def evaluate_model(model, model_name, tokenized_val):
    """Comprehensive evaluation of fine-tuned model against baseline."""
    logger.info("Starting comprehensive model evaluation")

    # Load evaluation metrics
    rouge = load("rouge")
    bleu = load("bleu")

    # Create benchmark dataset if it doesn't exist
    try:
        with open("data/evaluation/benchmark_dataset.json", "r") as f:
            benchmark_data = json.load(f)
    except FileNotFoundError:
        logger.info("Benchmark dataset not found, creating new one")
        benchmark_data = create_benchmark_dataset()

    # Extract questions and expected answers from benchmark
    benchmark_questions = [item["question"] for item in benchmark_data]
    expected_answers = [item["expected_answer"] for item in benchmark_data]

    # Initialize performance monitor
    performance_monitor = create_performance_monitor()

    # Baseline evaluation
    logger.info("Evaluating baseline model")
    baseline_model = AutoModelForCausalLM.from_pretrained(model_name)
    baseline_generator = pipeline("text-generation", model=baseline_model)

    baseline_preds = []
    baseline_start_time = time.time()
    for question in benchmark_questions:
        try:
            pred = baseline_generator(question, max_length=100, do_sample=False)[0][
                "generated_text"
            ]
            baseline_preds.append(pred)
        except Exception as e:
            logger.warning("Baseline prediction failed for question: %s", e)
            baseline_preds.append("")
    baseline_time = time.time() - baseline_start_time

    # Fine-tuned evaluation
    logger.info("Evaluating fine-tuned model")
    fine_tuned_generator = pipeline("text-generation", model=model)

    fine_tuned_preds = []
    fine_tuned_start_time = time.time()
    for question in benchmark_questions:
        try:
            pred = fine_tuned_generator(question, max_length=100, do_sample=False)[0][
                "generated_text"
            ]
            fine_tuned_preds.append(pred)
        except Exception as e:
            logger.warning("Fine-tuned prediction failed for question: %s", e)
            fine_tuned_preds.append("")
    fine_tuned_time = time.time() - fine_tuned_start_time

    # Calculate metrics
    baseline_rouge = rouge.compute(
        predictions=baseline_preds, references=expected_answers
    )
    baseline_bleu = bleu.compute(
        predictions=baseline_preds, references=expected_answers
    )

    fine_tuned_rouge = rouge.compute(
        predictions=fine_tuned_preds, references=expected_answers
    )
    fine_tuned_bleu = bleu.compute(
        predictions=fine_tuned_preds, references=expected_answers
    )

    # Performance metrics
    baseline_throughput = len(benchmark_questions) / baseline_time
    fine_tuned_throughput = len(benchmark_questions) / fine_tuned_time

    # Record performance metrics
    system_metrics = performance_monitor.get_system_metrics()
    performance_monitor.record_metrics(
        baseline_time * 1000 / len(benchmark_questions),  # avg latency in ms
        baseline_throughput,
        system_metrics,
    )
    performance_monitor.record_metrics(
        fine_tuned_time * 1000 / len(benchmark_questions),  # avg latency in ms
        fine_tuned_throughput,
        system_metrics,
    )

    # Calculate improvements
    rouge_improvement = {
        "rouge1": (
            (fine_tuned_rouge["rouge1"] - baseline_rouge["rouge1"])
            / baseline_rouge["rouge1"]
        )
        * 100,
        "rouge2": (
            (fine_tuned_rouge["rouge2"] - baseline_rouge["rouge2"])
            / baseline_rouge["rouge2"]
        )
        * 100,
        "rougeL": (
            (fine_tuned_rouge["rougeL"] - baseline_rouge["rougeL"])
            / baseline_rouge["rougeL"]
        )
        * 100,
    }

    bleu_improvement = (
        (fine_tuned_bleu["bleu"] - baseline_bleu["bleu"]) / baseline_bleu["bleu"]
    ) * 100

    # Compile results
    results = {
        "baseline": {
            "rouge": baseline_rouge,
            "bleu": baseline_bleu,
            "throughput_requests_per_second": baseline_throughput,
            "total_time_seconds": baseline_time,
        },
        "fine_tuned": {
            "rouge": fine_tuned_rouge,
            "bleu": fine_tuned_bleu,
            "throughput_requests_per_second": fine_tuned_throughput,
            "total_time_seconds": fine_tuned_time,
        },
        "improvements": {
            "rouge_improvement_percent": rouge_improvement,
            "bleu_improvement_percent": bleu_improvement,
            "throughput_improvement_percent": (
                (fine_tuned_throughput - baseline_throughput) / baseline_throughput
            )
            * 100,
        },
        "performance_summary": performance_monitor.get_performance_summary(),
    }

    # Save detailed results
    save_evaluation_results(results, benchmark_data, baseline_preds, fine_tuned_preds)

    logger.info("Evaluation completed successfully")
    logger.info(
        "ROUGE improvements: ROUGE-1: %.2f%%, ROUGE-2: %.2f%%, ROUGE-L: %.2f%%",
        rouge_improvement["rouge1"],
        rouge_improvement["rouge2"],
        rouge_improvement["rougeL"],
    )
    logger.info("BLEU improvement: %.2f%%", bleu_improvement)

    return results


def save_evaluation_results(
    results: Dict[str, Any],
    benchmark_data: List[Dict],
    baseline_preds: List[str],
    fine_tuned_preds: List[str],
):
    """Save detailed evaluation results to file."""
    import os

    os.makedirs("logs", exist_ok=True)

    detailed_results = {"summary": results, "detailed_predictions": []}

    for i, (benchmark_item, baseline_pred, fine_tuned_pred) in enumerate(
        zip(benchmark_data, baseline_preds, fine_tuned_preds)
    ):
        detailed_results["detailed_predictions"].append(
            {
                "id": benchmark_item["id"],
                "category": benchmark_item["category"],
                "difficulty": benchmark_item["difficulty"],
                "question": benchmark_item["question"],
                "expected_answer": benchmark_item["expected_answer"],
                "baseline_prediction": baseline_pred,
                "fine_tuned_prediction": fine_tuned_pred,
            }
        )

    with open("logs/evaluation_results.json", "w") as f:
        json.dump(detailed_results, f, indent=2)

    logger.info("Detailed evaluation results saved to logs/evaluation_results.json")
