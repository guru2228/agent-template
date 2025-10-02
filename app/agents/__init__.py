"""
Agent Discovery System - YAML-Only Public API.

This module provides the interface for the YAML-based agent discovery system.
External code should import from this module to access agent functionality.

Main exports:
- AgentRegistry: Direct registry access for discovery and creation
- AgentBuilder: Fluent API for creating agents

Discovery Sources:
- YAML files in app/agents/definitions/ (default)
- Additional folder paths containing *.agent.yaml files
- Environment variables containing YAML arrays

Usage:
    from app.agents import AgentRegistry, AgentBuilder, discover_and_create_all

    # Discover and create all agents from default sources
    agents = discover_and_create_all()
    
    # Discover from additional paths
    agents = discover_and_create_all(
        additional_paths=["/path/to/agents", "/another/path"]
    )
    
    # Discover from environment variables
    agents = discover_and_create_all(
        env_vars=["AGENTS_CONFIG", "EXTRA_AGENTS"]
    )

    # Direct registry access for advanced usage
    registry = AgentRegistry()
    registry.discover_agents(
        additional_paths=["/custom/path"],
        env_vars=["MY_AGENTS"]
    )

    # Create agent programmatically
    agent = (AgentBuilder("My Agent")
             .with_model("openai")
             .with_mcp("docs")
             .build())

Environment Variable Formats:
    # JSON array of agent objects
    AGENTS_CONFIG='[{"name": "agent1", "model": "openai"}, {"name": "agent2", "model": "anthropic"}]'
    
    # YAML array
    AGENTS_CONFIG='
    - name: agent1
      model: openai
    - name: agent2
      model: anthropic
    '
    
    # Single YAML object
    SINGLE_AGENT='name: my-agent\nmodel: openai\ninstructions: You are a helpful assistant'
"""

# Core YAML-only API
from .registry import AgentRegistry
from .builder import AgentBuilder

# Base types that might be useful for type annotations
from .base import (
    AgentDefinition,
    AgentMetadata,
    AgentFilter,
    DiscoveryPattern
)

# Version info
__version__ = "3.0.0"  # Bumped for YAML-only API

# Public API
__all__ = [
    # Primary API
    "AgentRegistry",
    "AgentBuilder",

    # Types for annotations
    "AgentDefinition",
    "AgentMetadata", 
    "AgentFilter",
    "DiscoveryPattern",

    # Metadata
    "__version__"
]

# Convenience functions
def discover_and_create_all(
    env_vars=None
):
    """
    Discover and create all enabled agents from environment variables only.
    
    Args:
        env_vars: Environment variable names containing YAML arrays
        
    Returns:
        List of Agent instances
        
    Example:
        # Discover from default environment variable
        agents = discover_and_create_all()
        
        # Discover from specific environment variables
        agents = discover_and_create_all(
            env_vars=["AGENT_DEFINITIONS", "EXTRA_AGENTS"]
        )
    """
    return AgentRegistry.discover_and_create_all(
        env_vars=env_vars
    )