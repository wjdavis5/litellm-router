"""litellm CustomLLM instance for the opencode-gateway (LXC 108).

All logic lives in cli_gateway_handler.CliGatewayLLM; this module binds the
provider name, env prefix (OPENCODE_BASE / OPENCODE_TOKEN), and default base.
"""

from handlers.cli_gateway_handler import CliGatewayLLM

opencode_llm = CliGatewayLLM(
    provider="opencode",
    env_prefix="OPENCODE",
    default_base="http://192.168.0.93:8100",
)
