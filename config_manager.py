
import json
import logging
from typing import Dict, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass, asdict, fields
import os
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class TradingSystemConfig:
    """Main configuration for the trading system"""
    # System settings
    system_name: str = "MirrorCore-X"
    version: str = "2.0.0"
    environment: str = "development"  # development, staging, production
    
    # Trading settings
    initial_capital: float = 100000.0
    max_drawdown: float = 0.15
    max_position_size: float = 0.1
    risk_free_rate: float = 0.02
    transaction_cost: float = 0.001
    slippage: float = 0.0005
    
    # Scanner settings
    scanner_enabled: bool = True
    scanner_interval: float = 30.0  # seconds
    top_n_symbols: int = 50
    min_volume_usd: float = 500000.0
    quote_currency: str = "USDT"
    
    # Strategy settings
    strategy_trainer_enabled: bool = True
    min_strategy_weight: float = 0.1
    max_strategy_weight: float = 1.0
    strategy_lookback: int = 20
    
    # Risk management
    risk_management_enabled: bool = True
    circuit_breakers_enabled: bool = True
    stress_testing_enabled: bool = True
    var_confidence: float = 0.95
    
    # Dashboard settings
    dashboard_enabled: bool = True
    dashboard_port: int = 5000
    dashboard_host: str = "0.0.0.0"
    websocket_enabled: bool = True
    
    # Exchange settings
    exchange_name: str = "kucoinfutures"
    exchange_sandbox: bool = True
    exchange_rate_limit: bool = True
    
    # Logging settings
    log_level: str = "INFO"
    log_file: str = "mirrorcore.log"
    log_max_size: int = 10485760  # 10MB
    log_backup_count: int = 5
    
    # AI/ML settings
    rl_enabled: bool = False
    bayesian_enabled: bool = False
    imagination_engine_enabled: bool = False
    sentiment_analysis_enabled: bool = False
    
    # Performance settings
    max_concurrent_requests: int = 50
    timeout_seconds: float = 10.0
    retry_attempts: int = 3
    
    # Data settings
    data_directory: str = "data"
    backup_enabled: bool = True
    backup_interval: int = 3600  # seconds
    
    # Notification settings
    notifications_enabled: bool = True
    email_notifications: bool = False
    webhook_url: Optional[str] = None

class ConfigManager:
    """Centralized configuration management"""
    
    def __init__(self, config_file: str = "config.json"):
        self.config_file = Path(config_file)
        self.config = TradingSystemConfig()
        self.user_overrides: Dict[str, Any] = {}
        self.environment_overrides: Dict[str, Any] = {}
        
    def load_config(self) -> TradingSystemConfig:
        """Load configuration from file and environment"""
        # Load from file if exists
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    file_config = json.load(f)
                self._apply_config_dict(file_config)
                logger.info(f"Configuration loaded from {self.config_file}")
            except Exception as e:
                logger.error(f"Error loading config file: {e}")
        
        # Load environment overrides
        self._load_environment_variables()
        
        # Apply user overrides
        self._apply_config_dict(self.user_overrides)
        
        # Validate configuration
        self._validate_config()
        
        return self.config
    
    def save_config(self) -> bool:
        """Save current configuration to file"""
        try:
            config_dict = asdict(self.config)
            with open(self.config_file, 'w') as f:
                json.dump(config_dict, f, indent=2, default=str)
            logger.info(f"Configuration saved to {self.config_file}")
            return True
        except Exception as e:
            logger.error(f"Error saving config: {e}")
            return False
    
    def update_config(self, updates: Dict[str, Any]) -> bool:
        """Update configuration with new values"""
        try:
            self.user_overrides.update(updates)
            self._apply_config_dict(updates)
            self._validate_config()
            return True
        except Exception as e:
            logger.error(f"Error updating config: {e}")
            return False
    
    def get_config(self) -> TradingSystemConfig:
        """Get current configuration"""
        return self.config
    
    def get_config_dict(self) -> Dict[str, Any]:
        """Get configuration as dictionary"""
        return asdict(self.config)
    
    def reset_to_defaults(self) -> bool:
        """Reset configuration to defaults"""
        try:
            self.config = TradingSystemConfig()
            self.user_overrides.clear()
            logger.info("Configuration reset to defaults")
            return True
        except Exception as e:
            logger.error(f"Error resetting config: {e}")
            return False
    
    def create_environment_config(self, environment: str) -> bool:
        """Create environment-specific configuration"""
        try:
            env_file = Path(f"config_{environment}.json")
            
            if environment == "production":
                prod_config = asdict(self.config)
                prod_config.update({
                    "environment": "production",
                    "exchange_sandbox": False,
                    "log_level": "WARNING",
                    "dashboard_enabled": False,
                    "max_drawdown": 0.1,
                    "risk_management_enabled": True,
                    "circuit_breakers_enabled": True
                })
            elif environment == "staging":
                prod_config = asdict(self.config)
                prod_config.update({
                    "environment": "staging",
                    "exchange_sandbox": True,
                    "log_level": "INFO",
                    "initial_capital": 10000.0
                })
            else:  # development
                prod_config = asdict(self.config)
                prod_config.update({
                    "environment": "development",
                    "exchange_sandbox": True,
                    "log_level": "DEBUG",
                    "dashboard_enabled": True
                })
            
            with open(env_file, 'w') as f:
                json.dump(prod_config, f, indent=2, default=str)
            
            logger.info(f"Environment config created: {env_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating environment config: {e}")
            return False
    
    def _apply_config_dict(self, config_dict: Dict[str, Any]):
        """Apply configuration dictionary to config object"""
        for key, value in config_dict.items():
            if hasattr(self.config, key):
                # Type conversion based on the field type
                field_type = None
                for field in fields(TradingSystemConfig):
                    if field.name == key:
                        field_type = field.type
                        break
                if field_type:
                    try:
                        # Handle Optional types (Union, Optional)
                        origin = getattr(field_type, "__origin__", None)
                        if origin is Union and hasattr(field_type, "__args__"):
                            # This is Optional[T], get T (ignore NoneType)
                            args = getattr(field_type, "__args__", None)
                            if args:
                                non_none_types = [t for t in args if t is not type(None)]
                                if non_none_types:
                                    field_type = non_none_types[0]
                        # Convert value to correct type
                        if field_type == bool and isinstance(value, str):
                            value = value.lower() in ('true', '1', 'yes', 'on')
                        elif field_type in (int, float) and isinstance(value, str):
                            value = field_type(value)
                        setattr(self.config, key, value)
                    except (ValueError, TypeError) as err:
                        logger.warning(f"Could not convert config value {key}={value}: {err}")
    
    def _load_environment_variables(self):
        """Load configuration from environment variables"""
        env_prefix = "MIRRORCORE_"
        
        for key in os.environ:
            if key.startswith(env_prefix):
                config_key = key[len(env_prefix):].lower()
                value = os.environ[key]
                
                if hasattr(self.config, config_key):
                    self.environment_overrides[config_key] = value
                    logger.debug(f"Environment override: {config_key}={value}")
        
        if self.environment_overrides:
            self._apply_config_dict(self.environment_overrides)
    
    def _validate_config(self):
        """Validate configuration values"""
        errors = []
        
        # Validate numeric ranges
        if not 0 < self.config.max_drawdown <= 1:
            errors.append("max_drawdown must be between 0 and 1")
        
        if not 0 < self.config.max_position_size <= 1:
            errors.append("max_position_size must be between 0 and 1")
        
        if self.config.initial_capital <= 0:
            errors.append("initial_capital must be positive")
        
        if not 0 < self.config.var_confidence < 1:
            errors.append("var_confidence must be between 0 and 1")
        
        if self.config.scanner_interval <= 0:
            errors.append("scanner_interval must be positive")
        
        # Validate string values
        if self.config.environment not in ["development", "staging", "production"]:
            errors.append("environment must be development, staging, or production")
        
        if self.config.log_level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            errors.append("log_level must be valid Python logging level")
        
        # Validate ports
        if not 1024 <= self.config.dashboard_port <= 65535:
            errors.append("dashboard_port must be between 1024 and 65535")
        
        if errors:
            error_msg = "Configuration validation errors: " + "; ".join(errors)
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    def get_exchange_config(self) -> Dict[str, Any]:
        """Get exchange-specific configuration"""
        return {
            'exchange': self.config.exchange_name,
            'sandbox': self.config.exchange_sandbox,
            'enableRateLimit': self.config.exchange_rate_limit,
            'timeout': self.config.timeout_seconds * 1000,  # Convert to milliseconds
            'retries': self.config.retry_attempts
        }
    
    def get_scanner_config(self) -> Dict[str, Any]:
        """Get scanner-specific configuration"""
        return {
            'enabled': self.config.scanner_enabled,
            'interval': self.config.scanner_interval,
            'top_n': self.config.top_n_symbols,
            'min_volume_usd': self.config.min_volume_usd,
            'quote_currency': self.config.quote_currency
        }
    
    def get_risk_config(self) -> Dict[str, Any]:
        """Get risk management configuration"""
        return {
            'enabled': self.config.risk_management_enabled,
            'max_drawdown': self.config.max_drawdown,
            'max_position_size': self.config.max_position_size,
            'circuit_breakers': self.config.circuit_breakers_enabled,
            'stress_testing': self.config.stress_testing_enabled,
            'var_confidence': self.config.var_confidence
        }
    
    def get_dashboard_config(self) -> Dict[str, Any]:
        """Get dashboard configuration"""
        return {
            'enabled': self.config.dashboard_enabled,
            'host': self.config.dashboard_host,
            'port': self.config.dashboard_port,
            'websocket': self.config.websocket_enabled
        }
    
    def export_config_template(self, filename: str = "config_template.json") -> bool:
        """Export configuration template with descriptions"""
        try:
            template = {
                "_description": "MirrorCore-X Configuration Template",
                "_version": self.config.version,
                "_created": datetime.now().isoformat(),
                "_fields": {}
            }
            
            # Add field descriptions
            field_descriptions = {
                "system_name": "Name of the trading system",
                "environment": "Environment: development, staging, or production",
                "initial_capital": "Starting capital in USD",
                "max_drawdown": "Maximum allowed drawdown (0.0 to 1.0)",
                "max_position_size": "Maximum position size as fraction of portfolio",
                "scanner_enabled": "Enable market scanning",
                "scanner_interval": "Scanner update interval in seconds",
                "risk_management_enabled": "Enable risk management system",
                "dashboard_enabled": "Enable web dashboard",
                "dashboard_port": "Dashboard web server port",
                "exchange_sandbox": "Use exchange sandbox mode"
            }
            
            config_dict = asdict(self.config)
            for key, value in config_dict.items():
                template["_fields"][key] = {
                    "value": value,
                    "type": type(value).__name__,
                    "description": field_descriptions.get(key, "No description available")
                }
            
            # Add actual config
            template.update(config_dict)
            
            with open(filename, 'w') as f:
                json.dump(template, f, indent=2, default=str)
            
            logger.info(f"Configuration template exported to {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting config template: {e}")
            return False

# Global configuration instance
config_manager = ConfigManager()

def get_config() -> TradingSystemConfig:
    """Get global configuration"""
    return config_manager.get_config()

def update_config(updates: Dict[str, Any]) -> bool:
    """Update global configuration"""
    return config_manager.update_config(updates)

def save_config() -> bool:
    """Save global configuration"""
    return config_manager.save_config()

# Example usage
if __name__ == "__main__":
    # Initialize config manager
    config_mgr = ConfigManager()
    
    # Load configuration
    config = config_mgr.load_config()
    print(f"Loaded config for {config.system_name} v{config.version}")
    
    # Update some settings
    config_mgr.update_config({
        "max_drawdown": 0.12,
        "dashboard_port": 5001,
        "scanner_interval": 45.0
    })
    
    # Save configuration
    config_mgr.save_config()
    
    # Create environment configs
    config_mgr.create_environment_config("production")
    config_mgr.create_environment_config("staging")
    
    # Export template
    config_mgr.export_config_template()
    
    print("Configuration management demo completed")
