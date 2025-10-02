# Azure OpenAI model provider
# Provides support for Azure OpenAI models and endpoints

from enum import Enum
from typing import Optional, Dict, Any, Union
from dataclasses import dataclass

from agno.models.openai import OpenAIChat


@dataclass
class AzureOpenAIConfig:
    """Configuration for Azure OpenAI endpoints"""
    
    endpoint: str
    api_key: str
    api_version: str = "2024-08-01-preview"
    deployment_name: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for agno configuration"""
        config = {
            "endpoint": self.endpoint,
            "api_key": self.api_key,
            "api_version": self.api_version,
        }
        if self.deployment_name:
            config["deployment_name"] = self.deployment_name
        return config


class AzureOpenAIModels(Enum):
    """Azure OpenAI model identifiers and deployment names"""

    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"
    GPT_4 = "gpt-4"
    GPT_4_32K = "gpt-4-32k"
    GPT_35_TURBO = "gpt-35-turbo"
    GPT_35_TURBO_16K = "gpt-35-turbo-16k"
    TEXT_EMBEDDING_ADA_002 = "text-embedding-ada-002"
    TEXT_EMBEDDING_3_SMALL = "text-embedding-3-small"
    TEXT_EMBEDDING_3_LARGE = "text-embedding-3-large"


class AzureOpenAIParams(Enum):
    """Azure OpenAI parameter constants"""

    # Temperature settings
    CREATIVE_TEMP = 0.7
    BALANCED_TEMP = 0.5
    PRECISE_TEMP = 0.1
    DETERMINISTIC_TEMP = 0.0

    # Max tokens settings
    DEFAULT_MAX_TOKENS = 4096
    LARGE_MAX_TOKENS = 8192
    SMALL_MAX_TOKENS = 1024

    # Azure-specific settings
    DEFAULT_API_VERSION = "2024-08-01-preview"
    LATEST_API_VERSION = "2024-10-01-preview"


# Model metadata for Azure OpenAI models
AZURE_OPENAI_MODEL_METADATA = {
    AzureOpenAIModels.GPT_4O: {
        "cost_per_1k_input": 0.005,
        "cost_per_1k_output": 0.015,
        "context_window": 128000,
        "max_output_tokens": 4096,
        "supports_vision": True,
        "supports_function_calling": True,
    },
    AzureOpenAIModels.GPT_4O_MINI: {
        "cost_per_1k_input": 0.00015,
        "cost_per_1k_output": 0.0006,
        "context_window": 128000,
        "max_output_tokens": 16384,
        "supports_vision": True,
        "supports_function_calling": True,
    },
    AzureOpenAIModels.GPT_4: {
        "cost_per_1k_input": 0.03,
        "cost_per_1k_output": 0.06,
        "context_window": 8192,
        "max_output_tokens": 4096,
        "supports_vision": False,
        "supports_function_calling": True,
    },
    AzureOpenAIModels.GPT_4_32K: {
        "cost_per_1k_input": 0.06,
        "cost_per_1k_output": 0.12,
        "context_window": 32768,
        "max_output_tokens": 4096,
        "supports_vision": False,
        "supports_function_calling": True,
    },
    AzureOpenAIModels.GPT_35_TURBO: {
        "cost_per_1k_input": 0.0015,
        "cost_per_1k_output": 0.002,
        "context_window": 4096,
        "max_output_tokens": 4096,
        "supports_vision": False,
        "supports_function_calling": True,
    },
    AzureOpenAIModels.GPT_35_TURBO_16K: {
        "cost_per_1k_input": 0.003,
        "cost_per_1k_output": 0.004,
        "context_window": 16384,
        "max_output_tokens": 4096,
        "supports_vision": False,
        "supports_function_calling": True,
    },
}


def create_azure_openai_model(
    model: Union[str, AzureOpenAIModels],
    azure_config: AzureOpenAIConfig,
    **kwargs
) -> OpenAIChat:
    """
    Create an Azure OpenAI model instance.
    
    Args:
        model: Model identifier (enum or string)
        azure_config: Azure OpenAI configuration
        **kwargs: Additional parameters for the model
        
    Returns:
        Configured OpenAIChat instance for Azure
    """
    if isinstance(model, AzureOpenAIModels):
        model_id = model.value
    else:
        model_id = model
    
    # Azure OpenAI uses different base URL structure
    base_url = f"{azure_config.endpoint}/openai/deployments/{azure_config.deployment_name or model_id}"
    
    # Configure for Azure OpenAI
    model_config = {
        "id": model_id,  # OpenAIChat expects 'id', not 'model'
        "api_key": azure_config.api_key,
        "base_url": base_url,
        "extra_query": {"api-version": azure_config.api_version},
        **kwargs
    }
    
    return OpenAIChat(**model_config)


def register_azure_openai_models(model_factory_class) -> None:
    """
    Register Azure OpenAI models with the ModelFactory.
    
    Args:
        model_factory_class: The ModelFactory class to register models with
    """
    for model_enum in AzureOpenAIModels:
        # Create the model creation function
        creation_function = lambda m=model_enum, **kwargs: create_azure_openai_model(
            m, 
            get_default_azure_config(), 
            **kwargs
        )
        
        # Get metadata if available
        metadata = AZURE_OPENAI_MODEL_METADATA.get(model_enum)
        
        # Register with ModelFactory using the same pattern as OpenAI
        model_factory_class.register_model(
            model_enum=model_enum,
            creation_function=creation_function,
            metadata=metadata,
            is_default=False,  # Azure OpenAI models are not defaults
            provider=None,
        )


def get_default_azure_config() -> AzureOpenAIConfig:
    """
    Get default Azure OpenAI configuration from environment variables.
    
    Returns:
        AzureOpenAIConfig with values from environment or defaults
    """
    import os
    
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "https://your-resource.openai.azure.com")
    api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", AzureOpenAIParams.DEFAULT_API_VERSION.value)
    deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    
    return AzureOpenAIConfig(
        endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
        deployment_name=deployment_name
    )


def get_azure_model_by_name(model_name: str) -> Optional[AzureOpenAIModels]:
    """
    Get Azure OpenAI model enum by string name.
    
    Args:
        model_name: String name of the model
        
    Returns:
        AzureOpenAIModels enum or None if not found
    """
    for model in AzureOpenAIModels:
        if model.value == model_name:
            return model
    return None


def list_available_models() -> Dict[str, Dict[str, Any]]:
    """
    List all available Azure OpenAI models with their metadata.
    
    Returns:
        Dictionary mapping model names to metadata
    """
    models = {}
    for model_enum in AzureOpenAIModels:
        models[model_enum.value] = {
            "enum": model_enum,
            "metadata": AZURE_OPENAI_MODEL_METADATA.get(model_enum, {}),
        }
    return models