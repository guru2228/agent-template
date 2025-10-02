# Azure OpenAI Provider Configuration

This document explains how to configure and use the Azure OpenAI provider in the agent template.

## Overview

The Azure OpenAI provider enables agents to use OpenAI models deployed through Microsoft Azure OpenAI Service. This provides enterprise-grade features like:

- Private endpoints
- Enterprise security and compliance
- Regional data residency
- Advanced monitoring and logging
- Integration with Azure AD

## Configuration

### Environment Variables

Set these environment variables for Azure OpenAI configuration:

```bash
# Required
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
AZURE_OPENAI_API_KEY=your-api-key

# Optional
AZURE_OPENAI_API_VERSION=2024-08-01-preview
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o-deployment
```

### Agent Configuration

#### Option 1: Direct Provider Configuration
```yaml
llm:
  provider: "azure_openai"
  model: "gpt-4o"
  deployment: "gpt-4o-deployment"
  endpoint: "${AZURE_OPENAI_ENDPOINT}"
  apiKey:
    secretRef:
      keyName: "azure-openai-api-key"
  parameters:
    temperature: 0.3
    maxTokens: 2000
    apiVersion: "2024-08-01-preview"
```

#### Option 2: Using Provider Alias
```yaml
llm:
  provider: "azure"  # Synonym for azure_openai
  model: "gpt-4o"
  deployment: "my-gpt4-deployment"
  parameters:
    temperature: 0.5
```

## Available Models

The Azure OpenAI provider supports these models:

| Model | Identifier | Context Window | Features |
|-------|------------|----------------|----------|
| GPT-4o | `gpt-4o` | 128K | Vision, Function Calling |
| GPT-4o Mini | `gpt-4o-mini` | 128K | Vision, Function Calling |
| GPT-4 | `gpt-4` | 8K | Function Calling |
| GPT-4 32K | `gpt-4-32k` | 32K | Function Calling |
| GPT-3.5 Turbo | `gpt-35-turbo` | 4K | Function Calling |
| GPT-3.5 Turbo 16K | `gpt-35-turbo-16k` | 16K | Function Calling |

## Programmatic Usage

### Using ModelFactory

```python
from app.models.factory import ModelFactory

# Get Azure OpenAI provider
providers = ModelFactory.providers()
azure_model = ModelFactory.get(providers.AZURE_OPENAI)

# Get specific model
gpt4_model = ModelFactory.get_model("gpt-4o", provider="azure_openai")
```

### Direct Configuration

```python
from app.models.providers.azure_openai import (
    AzureOpenAIConfig, 
    create_azure_openai_model,
    AzureOpenAIModels
)

# Create configuration
config = AzureOpenAIConfig(
    endpoint="https://your-resource.openai.azure.com",
    api_key="your-api-key",
    api_version="2024-08-01-preview",
    deployment_name="gpt-4o-deployment"
)

# Create model
model = create_azure_openai_model(
    AzureOpenAIModels.GPT_4O,
    config,
    temperature=0.3
)
```

## Deployment Considerations

### Security
- Store API keys in secure secret management systems
- Use Azure Key Vault for production deployments
- Configure private endpoints for enhanced security
- Enable Azure AD authentication when possible

### Monitoring
- Monitor API usage and costs through Azure portal
- Set up alerts for unusual usage patterns
- Track model performance and response times

### Cost Optimization
- Choose appropriate models for your use case
- Monitor token usage and optimize prompts
- Use cheaper models for simpler tasks
- Implement caching where appropriate

## Example Agent

See `azure-support-agent.agent.yaml` for a complete example of an agent configured to use Azure OpenAI.

## Troubleshooting

### Common Issues

1. **Authentication Errors**
   - Verify API key is correct
   - Check endpoint URL format
   - Ensure deployment exists

2. **Model Not Found**
   - Verify deployment name matches your Azure configuration
   - Check model is deployed in your Azure resource

3. **Rate Limiting**
   - Check your Azure OpenAI quota limits
   - Implement retry logic with backoff

### Support

For Azure OpenAI specific issues, consult:
- [Azure OpenAI Documentation](https://docs.microsoft.com/azure/cognitive-services/openai/)
- [Azure OpenAI Service Limits](https://docs.microsoft.com/azure/cognitive-services/openai/quotas-limits)
- Azure Support for service-level issues