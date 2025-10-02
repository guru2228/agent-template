import os
from agno.os import AgentOS
from fastapi import FastAPI

from app.agents import discover_and_create_all
from app.utils.log import logger

# Phoenix AI observability setup
try:
    import phoenix as px
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    # AgnoAI instrumentation for automatic tracing
    from openinference.instrumentation.agno import AgnoInstrumentor
    PHOENIX_AVAILABLE = True
except ImportError:
    logger.warning("Phoenix AI not available. Run: pip install arize-phoenix-otel to enable observability.")
    PHOENIX_AVAILABLE = False


class Server:
    agent_os: AgentOS
    app: FastAPI
    tracer_provider = None

    def __init__(self):
        # Initialize Phoenix AI observability
        if PHOENIX_AVAILABLE:
            logger.info("Initializing Phoenix AI observability...")
            try:
                                # Start Phoenix session (for UI dashboard)\n                phoenix_session = px.launch_app(\n                    host=\"localhost\",\n                    port=6006\n                )
                
                # Create resource for the tracer
                resource = Resource.create({
                    "service.name": "sage-agent-os",
                    "service.version": "1.0.0"
                })
                
                # Create TracerProvider with our resource
                self.tracer_provider = TracerProvider(resource=resource)
                
                # Create OTLP exporter for Phoenix (points to Phoenix collector)
                otlp_exporter = OTLPSpanExporter(
                    endpoint="http://localhost:4317",
                    insecure=True
                )
                
                # Create BatchSpanProcessor with production-optimized settings
                batch_processor = BatchSpanProcessor(
                    otlp_exporter,
                    max_queue_size=2048,
                    max_export_batch_size=512,
                    export_timeout_millis=30000,
                    schedule_delay_millis=5000
                )
                
                # Add the batch processor to the tracer provider
                self.tracer_provider.add_span_processor(batch_processor)
                
                # Set as global tracer provider
                trace.set_tracer_provider(self.tracer_provider)
                
                # Auto-instrument AgnoAI
                AgnoInstrumentor().instrument()
                    
                logger.info("Phoenix AI observability initialized successfully with BatchSpanProcessor")
            except Exception as e:
                logger.warning(f"Failed to initialize Phoenix AI observability: {e}")
                self.tracer_provider = None
        
        # Discover and create all enabled agents from environment variables only
        try:
            # Always check for AGENT_DEFINITIONS environment variable
            # Default to AGENT_DEFINITIONS if no specific env vars provided
            env_vars = ['AGENT_DEFINITIONS']
            agents = discover_and_create_all(env_vars=env_vars)
        except Exception as e:
            logger.error(f"Failed to discover and create agents: {e}")
            # Fallback to empty list if agent loading fails
            agents = []

        # Create the AgentOS
        self.agent_os = AgentOS(
            os_id="sage-agent-os",
            name="Sage Agent OS",
            description="Sage Agent OS powered by Agno",
            agents=agents,
        )

        # Get the FastAPI app
        self.app = self.agent_os.get_app()

    def serve(self):
        # use fastapi dev main.py to run the app with fastapi only
        # use python main.py to run the app with full agent os support
        host = os.getenv("HOST", "0.0.0.0")
        port = int(os.getenv("PORT", 8000))
        
        try:
            self.agent_os.serve(app="main:app", reload=True, port=port, host=host)
        except KeyboardInterrupt:
            logger.info("Server shutdown requested")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup resources on server shutdown."""
        if PHOENIX_AVAILABLE and self.tracer_provider:
            logger.info("Cleaning up Phoenix AI observability...")
            # Phoenix cleanup is handled automatically
        logger.info("Server cleanup completed")
