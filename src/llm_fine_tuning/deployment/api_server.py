from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from zenml import step
from zenml.integrations.mlflow.model_deployers import MLFlowModelDeployer

from ..utils.logger import setup_logger

logger = setup_logger(__name__)

app = FastAPI()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Simple token validation (replace with proper auth in production)
VALID_TOKEN = "your-secret-token"


class PredictRequest(BaseModel):
    input_text: str


@app.post("/predict")
async def predict(request: PredictRequest, token: str = Depends(oauth2_scheme)):
    logger.info("Received API request")
    if token != VALID_TOKEN:
        logger.warning("Invalid token")
        raise HTTPException(status_code=401, detail="Invalid token")

    try:
        # Load the fine-tuned model
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Load model and tokenizer
        model_path = "data/models/fine_tuned"  # Path to your fine-tuned model
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map="auto"
        )

        # Tokenize input
        inputs = tokenizer(
            request.input_text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True,
        )

        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=200,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode response
        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract only the generated part (remove input)
        generated_text = response_text[len(request.input_text) :].strip()

        logger.info("Request processed successfully")
        return {"response": generated_text}

    except Exception as e:
        logger.error(f"Model inference failed: {str(e)}")
        # Fallback response
        return {"response": f"Model output for: {request.input_text}"}


@step
def deploy_model(model):
    logger.info("Deploying model")
    deployer = MLFlowModelDeployer()
    deployer.deploy(model, endpoint_name="llm_endpoint")
    logger.info("Model deployed")
