from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import OAuth2PasswordBearer
from zenml import step
from zenml.integrations.mlflow.model_deployers import MLFlowModelDeployer

from ..utils.logger import setup_logger

logger = setup_logger(__name__)

app = FastAPI()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Simple token validation (replace with proper auth in production)
VALID_TOKEN = "your-secret-token"


@app.post("/predict")
async def predict(input_text: str, token: str = Depends(oauth2_scheme)):
    logger.info("Received API request")
    if token != VALID_TOKEN:
        logger.warning("Invalid token")
        raise HTTPException(status_code=401, detail="Invalid token")
    # Simulated inference (replace with actual model inference)
    response = "Model output"
    logger.info("Request processed successfully")
    return {"response": response}


@step
def deploy_model(model):
    logger.info("Deploying model")
    deployer = MLFlowModelDeployer()
    deployer.deploy(model, endpoint_name="llm_endpoint")
    logger.info("Model deployed")
