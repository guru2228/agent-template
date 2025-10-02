"""
AgentRegistry - YAML-based agent discovery and management.

Registry that handles YAML-based agent discovery:
- YAML-based agent definitions (CRD format)
- Agent registration and metadata tracking
- Simple discovery path management
- Graceful error handling

Follows KISS principles with YAML-only agent definitions.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any, Union

from agno.agent import Agent

from .base import (
    AgentDefinition,
    AgentMetadata,
    DiscoveryPattern,
    AgentFilter,
    discovery_logger,
)


class AgentRegistry:
    """
    Singleton registry for YAML-based agent discovery.

    Handles YAML-based agent discovery:
    - YAML agent definitions in CRD format  
    - Agent registration and metadata tracking
    - Discovery path management
    - Graceful error handling
    """

    _instance = None
    _created_agents_cache: List[Agent] = []  # Global cache for created agents

    def __new__(cls):
        """Singleton pattern implementation."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize registry if not already initialized."""
        if self._initialized:
            return

        # Core registry state - simplified
        self._agent_definitions: Dict[str, AgentDefinition] = {}
        self._discovery_paths: List[Path] = []
        self._discovery_completed: bool = False

        # Set default discovery paths
        self._setup_default_discovery_paths()

        self._initialized = True
        # Registry initialization - silent

    def _setup_default_discovery_paths(self):
        """Setup default discovery paths for YAML agent definitions."""
        # Primary discovery path: app/agents/definitions/
        definitions_path = Path(__file__).parent / "definitions"
        if definitions_path.exists():
            self._discovery_paths.append(definitions_path)

    def discover_agents(
        self, 
        force_refresh: bool = False,
        env_vars: Optional[List[str]] = None
    ) -> None:
        """
        Discover agents using environment variables only (YAML-based definitions).

        Args:
            force_refresh: Whether to force rediscovery of all agents
            env_vars: Environment variable names containing YAML arrays
        """
        if self._discovery_completed and not force_refresh:
            return

        # Silent discovery start

        if force_refresh:
            self.clear_registry()

        # Discover agents from environment variables only
        try:
            self._discover_from_env_vars(env_vars)
        except Exception as e:
            discovery_logger.error(f"Error in agent discovery: {e}")

        self._discovery_completed = True

        # Only log if agents found
        if self._agent_definitions:
            agent_names = list(self._agent_definitions.keys())
            discovery_logger.info(f"Discovered {len(agent_names)} agent(s): {', '.join(agent_names)}")

    def _discover_from_env_vars(
        self, 
        env_vars: Optional[List[str]] = None
    ) -> None:
        """
        Discover agents defined in environment variables only.
        
        Args:
            env_vars: Environment variable names containing YAML arrays
        """
        # Discover from environment variables
        if env_vars:
            for env_var in env_vars:
                self._discover_from_env_var(env_var)
        else:
            discovery_logger.debug("No environment variables specified for agent discovery")

    def _discover_from_env_var(self, env_var: str) -> None:
        """
        Discover agents from environment variable containing YAML data.        2. YAML array: '- name: agent1\n  ...\n- name: agent2\n  ...'
        3. Single YAML object: 'name: agent1\n...'
        
        Args:
            env_var: Environment variable name containing YAML data
        """
        env_value = os.getenv(env_var)
        if not env_value:
            discovery_logger.debug(f"Environment variable {env_var} is not set or empty")
            return
        
        try:
            # Try to parse as YAML first (handles both YAML arrays and single objects)
            yaml_data = yaml.safe_load(env_value)
            
            if not yaml_data:
                discovery_logger.warning(f"Environment variable {env_var} contains empty YAML")
                return
            
            # Handle different data structures
            if isinstance(yaml_data, list):
                # Array of agent definitions
                discovery_logger.debug(f"Found {len(yaml_data)} agent(s) in {env_var}")
                for idx, agent_data in enumerate(yaml_data):
                    if isinstance(agent_data, dict):
                        self._load_yaml_agent_from_dict(agent_data, f"{env_var}[{idx}]")
                    else:
                        discovery_logger.warning(f"Invalid agent data at index {idx} in {env_var}: expected dict, got {type(agent_data)}")
            elif isinstance(yaml_data, dict):
                # Single agent definition
                discovery_logger.debug(f"Found single agent in {env_var}")
                self._load_yaml_agent_from_dict(yaml_data, env_var)
            else:
                discovery_logger.error(f"Environment variable {env_var} contains invalid YAML structure: expected dict or list, got {type(yaml_data)}")
                
        except yaml.YAMLError as e:
            discovery_logger.error(f"Invalid YAML in environment variable {env_var}: {e}")
        except Exception as e:
            discovery_logger.error(f"Failed to process environment variable {env_var}: {e}")

    def _load_yaml_agent_from_dict(self, yaml_data: dict, source_ref: str) -> None:
        """
        Load agent definition from YAML dictionary data.
        
        Args:
            yaml_data: Dictionary containing agent configuration
            source_ref: Reference to the source (for logging)
        """
        try:
            if not yaml_data:
                discovery_logger.warning(f"Empty YAML data from: {source_ref}")
                return
            
            # Determine format and extract agent name
            if self._is_crd_agent(yaml_data):
                agent_name = self._extract_crd_agent_name(yaml_data)
                metadata = self._extract_crd_metadata(yaml_data)
            else:
                # Legacy simple format
                if 'name' not in yaml_data:
                    discovery_logger.error(f"YAML agent definition missing 'name' field from: {source_ref}")
                    return
                
                agent_name = yaml_data['name']
                metadata = self._extract_simple_metadata(yaml_data)
            
            if not agent_name:
                discovery_logger.error(f"Could not extract agent name from: {source_ref}")
                return
            
            # Create factory function from YAML configuration
            def yaml_agent_factory():
                from .builder import AgentBuilder
                if self._is_crd_agent(yaml_data):
                    return self._build_agent_from_crd(yaml_data)
                else:
                    return self._build_agent_from_yaml(yaml_data)
            
            # Set factory function metadata for consistency
            yaml_agent_factory._agent_metadata = metadata
            
            # Register the agent definition
            self._register_agent_definition(
                agent_name,
                yaml_agent_factory,
                metadata,
                f"env:{source_ref}",
                source_ref
            )
            
        except Exception as e:
            discovery_logger.error(f"Failed to load YAML agent definition from {source_ref}: {e}")

    def _is_crd_agent(self, yaml_data: dict) -> bool:
        """Check if YAML data represents a Kubernetes CRD Agent."""
        return (
            yaml_data.get('apiVersion') == 'agents.enterprise.com/v1alpha9' and
            yaml_data.get('kind') == 'Agent' and
            'spec' in yaml_data
        )

    def _extract_crd_agent_name(self, yaml_data: dict) -> str:
        """Extract agent name from CRD format."""
        # Try metadata.name first, then spec.identity.displayName
        if 'metadata' in yaml_data and 'name' in yaml_data['metadata']:
            return yaml_data['metadata']['name']
        
        spec = yaml_data.get('spec', {})
        identity = spec.get('identity', {})
        return identity.get('displayName', '')

    def _extract_crd_metadata(self, yaml_data: dict) -> 'AgentMetadata':
        """Extract discovery metadata from CRD format."""
        spec = yaml_data.get('spec', {})
        deployment_context = spec.get('deploymentContext', {})
        
        # Extract agent name
        agent_name = self._extract_crd_agent_name(yaml_data)
        
        # Extract tags from deployment context
        tags = deployment_context.get('tags', [])
        
        # Extract priority (use lifecycle stage as priority indicator)
        lifecycle = deployment_context.get('lifecycle', 'development')
        priority_map = {
            'prod': 10,
            'staging': 20,
            'testing': 30,
            'development': 40,
            'prototype': 50,
            'ideation': 60
        }
        priority = priority_map.get(lifecycle, 50)
        
        # Check if enabled (assume enabled unless explicitly disabled)
        enabled = deployment_context.get('enabled', True)
        
        # Extract custom attributes from various CRD fields
        custom_attributes = {
            'crd_version': yaml_data.get('apiVersion', ''),
            'kind': yaml_data.get('kind', ''),
            'environment': deployment_context.get('environment', ''),
            'lifecycle': lifecycle,
            'tenant_id': deployment_context.get('tenantId', ''),
            'workspace_id': deployment_context.get('workspaceId', ''),
        }
        
        # Add identity information
        identity = spec.get('identity', {})
        if identity:
            custom_attributes.update({
                'urn': identity.get('urn', ''),
                'uuid': identity.get('uuid', ''),
                'version': identity.get('version', ''),
                'owner_contacts': identity.get('ownerContacts', []),
            })
        
        # Add ownership information
        ownership = spec.get('ownership', {})
        if ownership:
            custom_attributes.update({
                'organization': ownership.get('organization', ''),
                'team': ownership.get('team', ''),
                'user': ownership.get('user', ''),
            })
        
        return AgentMetadata(
            name=agent_name,
            pattern=DiscoveryPattern.YAML,
            tags=tags,
            priority=priority,
            enabled=enabled,
            dependencies=[],  # CRD doesn't have direct dependencies concept
            custom_attributes=custom_attributes
        )

    def _extract_simple_metadata(self, yaml_data: dict) -> 'AgentMetadata':
        """Extract discovery metadata from simple YAML format."""
        agent_name = yaml_data['name']
        
        # Create metadata from YAML
        return AgentMetadata(
            name=agent_name,
            pattern=DiscoveryPattern.YAML,
            tags=yaml_data.get('metadata', {}).get('tags', []),
            priority=yaml_data.get('metadata', {}).get('priority', 50),
            enabled=yaml_data.get('metadata', {}).get('enabled', True),
            dependencies=yaml_data.get('metadata', {}).get('dependencies', []),
            custom_attributes=yaml_data.get('metadata', {}).get('custom_attributes', {})
        )

    def _build_agent_from_yaml(self, yaml_data: dict) -> 'Agent':
        """
        Build an agent from YAML configuration data.
        
        Args:
            yaml_data: Dictionary containing agent configuration
            
        Returns:
            Configured Agent instance
        """
        from .builder import AgentBuilder
        
        # Start with agent name
        builder = AgentBuilder(yaml_data['name'])
        
        # Configure model
        if 'model' in yaml_data:
            model_config = yaml_data['model']
            if isinstance(model_config, str):
                # Simple string - could be provider or model ID
                # Check if it looks like a model ID (contains hyphens/versions)
                if any(model_id in model_config.lower() for model_id in ['gpt-', 'claude-', 'gemini-', 'llama-']):
                    # Treat as model ID
                    builder = builder.with_model(provider=None, model_id=model_config)
                else:
                    # Treat as provider
                    builder = builder.with_model(model_config)
            elif isinstance(model_config, dict):
                # Detailed model configuration
                provider = model_config.get('provider')
                model_id = model_config.get('model_id')
                kwargs = {k: v for k, v in model_config.items() if k not in ('provider', 'model_id')}
                
                if model_id:
                    builder = builder.with_model(provider, model_id=model_id, **kwargs)
                else:
                    builder = builder.with_model(provider, **kwargs)
        
        # Configure database
        if yaml_data.get('database', {}).get('enabled', False):
            db_config = yaml_data.get('database', {}).get('config')
            builder = builder.with_db(db_config)
        
        # Configure vector database
        if yaml_data.get('vector_db', {}).get('enabled', False):
            vector_config = yaml_data.get('vector_db', {}).get('config')
            builder = builder.with_vector_db(vector_config)
        
        # Configure memory
        if yaml_data.get('memory', {}).get('enabled', False):
            memory_config = yaml_data.get('memory', {}).get('config')
            builder = builder.with_memory(memory_config)
        
        # Configure MCP tools
        if 'mcp_tools' in yaml_data:
            mcp_tools = yaml_data['mcp_tools']
            if isinstance(mcp_tools, list):
                builder = builder.with_mcp(*mcp_tools)
        
        # Configure custom tools (not supported in YAML for security reasons)
        # Custom tools should use decorator pattern
        
        # Configure instructions
        if 'instructions' in yaml_data:
            builder = builder.with_instructions(yaml_data['instructions'])
        
        # Configure agent settings
        if 'config' in yaml_data:
            config = yaml_data['config']
            if isinstance(config, dict):
                builder = builder.with_config(**config)
        
        # Configure metadata
        if 'metadata' in yaml_data:
            metadata = yaml_data['metadata']
            if isinstance(metadata, dict):
                builder = builder.with_metadata(**metadata)
        
        return builder.build()

    def _build_agent_from_crd(self, yaml_data: dict) -> 'Agent':
        """
        Build an agent from Kubernetes CRD Agent specification.
        
        Args:
            yaml_data: Dictionary containing CRD Agent specification
            
        Returns:
            Configured Agent instance
        """
        from .builder import AgentBuilder
        
        spec = yaml_data.get('spec', {})
        
        # Extract agent name
        agent_name = self._extract_crd_agent_name(yaml_data)
        builder = AgentBuilder(agent_name)
        
        # Configure LLM
        llm_config = spec.get('llm', {})
        if llm_config:
            provider = llm_config.get('provider', 'openai').lower()
            model = llm_config.get('model', '')
            parameters = llm_config.get('parameters', {})
            
            # Map CRD LLM parameters to builder parameters
            model_kwargs = {}
            if 'temperature' in parameters:
                model_kwargs['temperature'] = parameters['temperature']
            if 'maxTokens' in parameters:
                model_kwargs['max_tokens'] = parameters['maxTokens']
            if 'topP' in parameters:
                model_kwargs['top_p'] = parameters['topP']
            
            # Handle provider mapping
            provider_map = {
                'openai': 'openai',
                'azureopenai': 'openai',  # Map Azure OpenAI to openai provider
                'anthropic': 'anthropic',
                'googlevertexai': 'google',
                'cohere': 'cohere',
            }
            
            mapped_provider = provider_map.get(provider, 'openai')
            
            if model:
                builder = builder.with_model(mapped_provider, model_id=model, **model_kwargs)
            else:
                builder = builder.with_model(mapped_provider, **model_kwargs)
        
        # Extract and configure tools from MCP servers
        tools = spec.get('tools', [])
        mcp_servers = spec.get('mcpServers', [])
        
        if tools and mcp_servers:
            # Build a map of server references to server names
            server_map = {}
            for server in mcp_servers:
                server_identity = server.get('identity', {})
                server_urn = server_identity.get('urn')
                server_uuid = server_identity.get('uuid')
                server_name = server_identity.get('displayName', server.get('name', ''))
                
                if server_urn:
                    server_map[server_urn] = server_name
                if server_uuid:
                    server_map[server_uuid] = server_name
            
            # Extract MCP tool names
            mcp_tool_names = []
            for tool in tools:
                mcp_config = tool.get('mcp', {})
                server_ref = mcp_config.get('serverRef', {})
                tool_name = mcp_config.get('toolName', '')
                
                # Find the server name
                server_urn = server_ref.get('urn')
                server_uuid = server_ref.get('uuid')
                server_name = ''
                
                if server_urn in server_map:
                    server_name = server_map[server_urn]
                elif server_uuid in server_map:
                    server_name = server_map[server_uuid]
                
                # Use server name or tool name as MCP tool reference
                if server_name:
                    mcp_tool_names.append(server_name)
                elif tool_name:
                    mcp_tool_names.append(tool_name)
            
            if mcp_tool_names:
                builder = builder.with_mcp(*mcp_tool_names)
        
        # Configure instructions from systemPrompt or role/goal
        instructions = spec.get('systemPrompt', '')
        if not instructions:
            # Build instructions from role and goal
            role = spec.get('role', '')
            goal = spec.get('goal', '')
            backstory = spec.get('backstory', '')
            
            instruction_parts = []
            if role:
                instruction_parts.append(f"You are a {role}.")
            if goal:
                instruction_parts.append(f"Your primary goal is: {goal}")
            if backstory:
                instruction_parts.append(f"Background: {backstory}")
            
            instructions = '\n\n'.join(instruction_parts)
        
        if instructions:
            builder = builder.with_instructions(instructions)
        
        # Configure agent settings from behavior and context
        config = {}
        
        # Extract behavior settings
        behavior = spec.get('behavior', {})
        if behavior:
            # Map behavior settings to agent config
            # Note: 'seed' is for LLM determinism, not Agent constructor
            if 'determinism' in behavior:
                determinism = behavior['determinism']
                # Skip 'seed' as it's not a valid Agent parameter
                # Seed should be handled at the model level, not agent level
                pass
        
        # Extract deployment context settings
        deployment_context = spec.get('deploymentContext', {})
        if deployment_context:
            tenant_id = deployment_context.get('tenantId', '1')
            workspace_id = deployment_context.get('workspaceId', '1')
            
            config.update({
                'user_id': tenant_id,
                'session_id': f"{tenant_id}_{workspace_id}_session",
            })
        
        # Default agent settings
        config.update({
            'add_history_to_context': True,
            'num_history_sessions': 5,
            'num_history_runs': 20,
            'enable_session_summaries': True,
            'markdown': True,
        })
        
        # Set agent ID from identity
        identity = spec.get('identity', {})
        agent_id = identity.get('urn') or identity.get('uuid') or agent_name.lower().replace(' ', '_')
        config['id'] = agent_id
        
        if config:
            builder = builder.with_config(**config)
        
        # Add database and memory if knowledge bases are configured
        knowledge_bases = spec.get('knowledgeBases', [])
        if knowledge_bases:
            builder = builder.with_db()
            builder = builder.with_memory()
        
        return builder.build()



    def _register_agent_definition(
        self,
        name: str,
        factory_func: Callable,
        metadata: AgentMetadata,
        module_path: str,
        source_file: str,
    ) -> None:
        """Register a complete agent definition."""
        try:
            # Simple priority check - SIMPLIFIED
            if name in self._agent_definitions:
                existing_priority = self._agent_definitions[name].metadata.priority
                new_priority = metadata.priority
                if existing_priority >= new_priority:
                    discovery_logger.debug(
                        f"Skipping agent {name} - existing priority {existing_priority} >= new priority {new_priority}"
                    )
                    return

            agent_def = AgentDefinition(
                name=name,
                factory_function=factory_func,
                metadata=metadata,
                module_path=module_path,
                source_file=source_file,
            )
            self._agent_definitions[name] = agent_def
        except Exception as e:
            discovery_logger.error(f"Failed to register agent {name}: {e}")

    def get_agent_definition(self, name: str) -> Optional[AgentDefinition]:
        """Get agent definition by name."""
        return self._agent_definitions.get(name)

    def list_agents(
        self, filter_obj: Optional[AgentFilter] = None
    ) -> List[AgentDefinition]:
        """
        List all agent definitions, optionally filtered.

        Args:
            filter_obj: Optional filter criteria

        Returns:
            List of agent definitions matching the filter
        """
        agents = list(self._agent_definitions.values())

        if filter_obj:
            agents = [agent for agent in agents if filter_obj.matches(agent)]

        # Sort by priority (higher priority first) then by name
        agents.sort(key=lambda a: (-a.metadata.priority, a.name))
        return agents

    def enable_agent(self, name: str) -> bool:
        """Enable an agent."""
        agent_def = self._agent_definitions.get(name)
        if agent_def:
            agent_def.metadata.enabled = True
            return True
        return False

    def disable_agent(self, name: str) -> bool:
        """Disable an agent."""
        agent_def = self._agent_definitions.get(name)
        if agent_def:
            agent_def.metadata.enabled = False
            return True
        return False

    def clear_registry(self) -> None:
        """Clear all discovered agents and reset discovery state."""
        self._agent_definitions.clear()
        self._discovery_completed = False

    def add_discovery_path(self, path: Union[str, Path]) -> bool:
        """
        Add a new discovery path to the registry.
        
        Args:
            path: Directory path to scan for YAML agent files
            
        Returns:
            True if path was added successfully, False if path doesn't exist
        """
        path_obj = Path(path)
        if path_obj.exists() and path_obj.is_dir():
            if path_obj not in self._discovery_paths:
                self._discovery_paths.append(path_obj)
                discovery_logger.debug(f"Added discovery path: {path_obj}")
                # Reset discovery state to include new path
                self._discovery_completed = False
            return True
        else:
            discovery_logger.warning(f"Cannot add discovery path - does not exist or is not a directory: {path}")
            return False

    def remove_discovery_path(self, path: Union[str, Path]) -> bool:
        """
        Remove a discovery path from the registry.
        
        Args:
            path: Directory path to remove
            
        Returns:
            True if path was removed, False if path was not found
        """
        path_obj = Path(path)
        if path_obj in self._discovery_paths:
            self._discovery_paths.remove(path_obj)
            discovery_logger.debug(f"Removed discovery path: {path_obj}")
            # Reset discovery state to exclude removed path
            self._discovery_completed = False
            return True
        return False

    def list_discovery_paths(self) -> List[Path]:
        """
        List all current discovery paths.
        
        Returns:
            List of Path objects representing discovery paths
        """
        return self._discovery_paths.copy()

    @classmethod
    def discover_and_create_all(
        cls,
        env_vars: Optional[List[str]] = None
    ) -> List[Agent]:
        """
        One-shot discovery and creation of all enabled agents from environment variables.

        This is the simplified public API for server integration.
        Returns cached agents if already created.
        
        Args:
            env_vars: Environment variable names containing YAML arrays
        """
        # Return cached agents if already created - PERFORMANCE OPTIMIZATION
        if cls._created_agents_cache:
            return cls._created_agents_cache

        registry = cls()

        # Only perform discovery once
        if not registry._discovery_completed:
            registry.discover_agents(
                env_vars=env_vars
            )

        # Get enabled agents and create instances
        enabled_agents = [
            agent_def
            for agent_def in registry._agent_definitions.values()
            if agent_def.metadata.enabled
        ]

        agents = []
        for agent_def in enabled_agents:
            try:
                agent = agent_def.factory_function()
                agents.append(agent)
            except Exception as e:
                discovery_logger.error(f"Failed to create agent {agent_def.name}: {e}")
                # Continue with other agents

        # Cache the created agents
        cls._created_agents_cache = agents

        # Single summary log
        if agents:
            agent_names = [getattr(a, 'name', 'Unknown') for a in agents]
            discovery_logger.info(f"Loaded {len(agents)} agent(s): {', '.join(agent_names)}")

        return agents
