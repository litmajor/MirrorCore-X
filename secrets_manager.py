"""
Secure Secrets Management for MirrorCore-X
Replaces hardcoded API keys with environment variable management
"""

import os
import json
import logging
from typing import Dict, Optional, Any
from pathlib import Path
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

logger = logging.getLogger(__name__)

class SecretsManager:
    """Secure secrets management with environment variables and optional encryption"""

    def __init__(self, encryption_key: Optional[str] = None):
        self.encryption_key = encryption_key
        self.fernet = None

        if encryption_key:
            self._setup_encryption(encryption_key)

        # Environment variable prefixes for different services
        self.prefixes = {
            'exchange': 'EXCHANGE_',
            'database': 'DB_',
            'api': 'API_',
            'notification': 'NOTIFY_',
            'webhook': 'WEBHOOK_'
        }

        logger.info("Secrets Manager initialized")

    def _setup_encryption(self, password: str):
        """Setup Fernet encryption with password-based key derivation"""
        try:
            # Generate key from password
            password_bytes = password.encode()
            salt = b'stable_salt_for_mirrorcore'  # In production, use random salt stored securely

            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )

            key = base64.urlsafe_b64encode(kdf.derive(password_bytes))
            self.fernet = Fernet(key)

            logger.info("Encryption setup complete")

        except Exception as e:
            logger.error(f"Failed to setup encryption: {e}")
            self.fernet = None

    def get_exchange_credentials(self, exchange_name: str) -> Dict[str, str]:
        """Get exchange API credentials from environment"""
        prefix = f"{self.prefixes['exchange']}{exchange_name.upper()}_"

        credentials = {
            'apiKey': self._get_secret(f"{prefix}API_KEY"),
            'secret': self._get_secret(f"{prefix}SECRET"),
            'password': self._get_secret(f"{prefix}PASSPHRASE", required=False),
            'sandbox': self._get_secret(f"{prefix}SANDBOX", default="true").lower() == "true",
            'testnet': self._get_secret(f"{prefix}TESTNET", default="true").lower() == "true"
        }

        # Validate required credentials
        if not credentials['apiKey'] or not credentials['secret']:
            raise ValueError(f"Missing API credentials for {exchange_name}. "
                           f"Set {prefix}API_KEY and {prefix}SECRET environment variables.")

        logger.info(f"Loaded credentials for {exchange_name} exchange")
        return credentials

    def get_database_config(self) -> Dict[str, str]:
        """Get database configuration from environment"""
        prefix = self.prefixes['database']

        config = {
            'host': self._get_secret(f"{prefix}HOST", default="localhost"),
            'port': self._get_secret(f"{prefix}PORT", default="5432"),
            'database': self._get_secret(f"{prefix}NAME", default="mirrorcore"),
            'username': self._get_secret(f"{prefix}USER", default="postgres"),
            'password': self._get_secret(f"{prefix}PASSWORD", required=False),
            'url': self._get_secret(f"{prefix}URL", required=False)
        }

        logger.info("Database configuration loaded")
        return config

    def get_api_keys(self) -> Dict[str, str]:
        """Get various API keys"""
        prefix = self.prefixes['api']

        api_keys = {
            'openai': self._get_secret(f"{prefix}OPENAI", required=False),
            'anthropic': self._get_secret(f"{prefix}ANTHROPIC", required=False),
            'telegram': self._get_secret(f"{prefix}TELEGRAM", required=False),
            'discord': self._get_secret(f"{prefix}DISCORD", required=False),
            'slack': self._get_secret(f"{prefix}SLACK", required=False),
            'fear_greed': self._get_secret(f"{prefix}FEAR_GREED", required=False)
        }

        return {k: v for k, v in api_keys.items() if v}

    def get_notification_config(self) -> Dict[str, str]:
        """Get notification service configuration"""
        prefix = self.prefixes['notification']

        config = {
            'email_smtp_host': self._get_secret(f"{prefix}EMAIL_SMTP_HOST", required=False),
            'email_smtp_port': self._get_secret(f"{prefix}EMAIL_SMTP_PORT", default="587"),
            'email_username': self._get_secret(f"{prefix}EMAIL_USER", required=False),
            'email_password': self._get_secret(f"{prefix}EMAIL_PASS", required=False),
            'email_from': self._get_secret(f"{prefix}EMAIL_FROM", required=False),
            'email_to': self._get_secret(f"{prefix}EMAIL_TO", required=False),
            'webhook_emergency': self._get_secret(f"{prefix}WEBHOOK_EMERGENCY", required=False),
            'webhook_trades': self._get_secret(f"{prefix}WEBHOOK_TRADES", required=False)
        }

        return {k: v for k, v in config.items() if v}

    def _get_secret(self, key: str, default: Optional[str] = None, required: bool = True) -> Optional[str]:
        """Get secret from environment with optional decryption"""
        value = os.getenv(key, default)

        if required and not value:
            raise ValueError(f"Required environment variable {key} not found")

        if not value:
            return None

        # Check if value is encrypted (starts with 'enc:')
        if value.startswith('enc:') and self.fernet:
            try:
                encrypted_value = value[4:]  # Remove 'enc:' prefix
                decrypted_bytes = self.fernet.decrypt(encrypted_value.encode())
                return decrypted_bytes.decode()
            except Exception as e:
                logger.error(f"Failed to decrypt {key}: {e}")
                return None

        return value

    def encrypt_secret(self, plaintext: str) -> str:
        """Encrypt a secret value"""
        if not self.fernet:
            raise ValueError("Encryption not setup")

        encrypted_bytes = self.fernet.encrypt(plaintext.encode())
        return f"enc:{encrypted_bytes.decode()}"

    def validate_credentials(self, exchange_name: str = None) -> bool:
        """Validate that all required credentials are available"""
        try:
            if exchange_name:
                self.get_exchange_credentials(exchange_name)

            # Check if any API keys are available
            api_keys = self.get_api_keys()

            logger.info(f"Credential validation passed. Available API keys: {list(api_keys.keys())}")
            return True

        except Exception as e:
            logger.error(f"Credential validation failed: {e}")
            return False

    @staticmethod
    def create_env_template(filename: str = ".env.template"):
        """Create environment variable template file"""
        template = """
# MirrorCore-X Environment Variables Template
# Copy this file to .env and fill in your values

# Exchange API Credentials
EXCHANGE_BINANCE_API_KEY=your_binance_api_key_here
EXCHANGE_BINANCE_SECRET=your_binance_secret_here
EXCHANGE_BINANCE_SANDBOX=true
EXCHANGE_BINANCE_TESTNET=true

EXCHANGE_KUCOIN_API_KEY=your_kucoin_api_key_here
EXCHANGE_KUCOIN_SECRET=your_kucoin_secret_here
EXCHANGE_KUCOIN_PASSPHRASE=your_kucoin_passphrase_here
EXCHANGE_KUCOIN_SANDBOX=true

# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=mirrorcore
DB_USER=postgres
DB_PASSWORD=your_database_password
DB_URL=postgresql://user:pass@localhost:5432/mirrorcore

# API Keys
API_OPENAI=your_openai_api_key
API_ANTHROPIC=your_anthropic_api_key
API_TELEGRAM=your_telegram_bot_token
API_DISCORD=your_discord_webhook_url
API_SLACK=your_slack_webhook_url

# Notification Configuration
NOTIFY_EMAIL_SMTP_HOST=smtp.gmail.com
NOTIFY_EMAIL_SMTP_PORT=587
NOTIFY_EMAIL_USER=your_email@gmail.com
NOTIFY_EMAIL_PASS=your_email_password
NOTIFY_EMAIL_FROM=mirrorcore@yourcompany.com
NOTIFY_EMAIL_TO=alerts@yourcompany.com
NOTIFY_WEBHOOK_EMERGENCY=https://hooks.slack.com/emergency
NOTIFY_WEBHOOK_TRADES=https://hooks.slack.com/trades

# System Configuration
MIRRORCORE_ENVIRONMENT=development
MIRRORCORE_LOG_LEVEL=INFO
MIRRORCORE_MAX_DRAWDOWN=15.0
MIRRORCORE_INITIAL_CAPITAL=100000.0
"""

        with open(filename, 'w') as f:
            f.write(template.strip())

        logger.info(f"Environment template created: {filename}")

def load_secrets_from_env_file(env_file: str = ".env"):
    """Load environment variables from .env file"""
    env_path = Path(env_file)

    if not env_path.exists():
        logger.warning(f"Environment file {env_file} not found")
        return

    try:
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()

        logger.info(f"Environment variables loaded from {env_file}")

    except Exception as e:
        logger.error(f"Failed to load environment file {env_file}: {e}")

# Usage example
if __name__ == "__main__":
    # Create template
    SecretsManager.create_env_template()

    # Load from .env file
    load_secrets_from_env_file()

    # Initialize secrets manager
    secrets = SecretsManager()

    # Test credential loading
    try:
        binance_creds = secrets.get_exchange_credentials('binance')
        print("Binance credentials loaded successfully")

        api_keys = secrets.get_api_keys()
        print(f"Available API keys: {list(api_keys.keys())}")

    except Exception as e:
        print(f"Error: {e}")
        print("Make sure to set up your .env file with proper credentials")