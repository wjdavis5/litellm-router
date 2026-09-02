"""litellm CustomLLM instance for the claude-gateway (LXC 109).

All logic lives in cli_gateway_handler.CliGatewayLLM; this module binds the
provider name, env prefix (CLAUDECODE_BASE / CLAUDECODE_TOKEN), and default base.
"""

from handlers.cli_gateway_handler import CliGatewayLLM

claudecode_llm = CliGatewayLLM(
    provider="claudecode",
    env_prefix="CLAUDECODE",
    default_base="http://192.168.0.94:8100",
)
