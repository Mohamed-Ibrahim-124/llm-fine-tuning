import mlflow
import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments

from ..utils.logger import setup_logger

logger = setup_logger(__name__)


def fine_tune_model(model_name, tokenized_data):
    logger.info("Starting fine-tuning for model: %s", model_name)
    with mlflow.start_run():
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("batch_size", 1)
        mlflow.log_param("epochs", 1)

        model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", torch_dtype=torch.float16
        )
        config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, config)

        training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=1,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            fp16=True,
            logging_steps=10,
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_data,
        )
        trainer.train()
        mlflow.log_metric("train_loss", trainer.state.log_history[-1]["loss"])
        logger.info("Fine-tuning completed")
        return model
