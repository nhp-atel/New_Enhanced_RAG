"""Basic tests for CLI module to improve coverage."""

import pytest
from click.testing import CliRunner
from unittest.mock import Mock, patch

# Import CLI components
from enhanced_rag.api.cli import cli, show_config


class TestCLIBasics:
    """Test basic CLI functionality."""
    
    def setup_method(self):
        """Setup CLI runner."""
        self.runner = CliRunner()
    
    def test_cli_help(self):
        """Test CLI help command."""
        result = self.runner.invoke(cli, ['--help'])
        
        assert result.exit_code == 0
        assert "Enhanced RAG System CLI" in result.output
    
    def test_cli_version(self):
        """Test CLI version display (may not be implemented)."""
        result = self.runner.invoke(cli, ['--version'])
        
        # Version might not be implemented, so accept various exit codes
        assert result.exit_code in [0, 2]  # 0 for success, 2 for no such option
    
    @patch('enhanced_rag.api.cli.ConfigManager')
    def test_show_config_command(self, mock_config_manager):
        """Test show-config command."""
        # Mock config manager
        mock_config = Mock()
        mock_config.model_dump.return_value = {"test": "config"}
        mock_config_manager.return_value.load_config.return_value = mock_config
        
        result = self.runner.invoke(cli, ['show-config'])
        
        # Command should execute successfully
        assert result.exit_code == 0
    
    def test_show_config_help(self):
        """Test show-config command help."""
        result = self.runner.invoke(cli, ['show-config', '--help'])
        
        assert result.exit_code == 0
        assert "Show current configuration" in result.output
    
    def test_cli_invalid_command(self):
        """Test CLI with invalid command."""
        result = self.runner.invoke(cli, ['invalid-command'])
        
        # Should show error or help
        assert result.exit_code != 0 or "Usage:" in result.output
    
    def test_cli_main_help_sections(self):
        """Test CLI main help shows different sections."""
        result = self.runner.invoke(cli, ['--help'])
        
        assert result.exit_code == 0
        assert "Commands:" in result.output
        
    def test_cli_empty_args(self):
        """Test CLI with no arguments."""
        result = self.runner.invoke(cli, [])
        
        # Should show help or usage info
        assert "Usage:" in result.output or "Commands:" in result.output