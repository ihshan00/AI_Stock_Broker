import yaml

def load_config(config_path: str) -> dict:
    """Loads configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        A dictionary containing the loaded configuration.
        Returns an empty dictionary if the file is not found or if there's an error during loading.
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            return config if config else {}  # Handle empty YAML files
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        return {}
    except yaml.YAMLError as e:
        print(f"Error loading YAML configuration: {e}")
        return {}

def load_text_file(file_path: str) -> str:
    """Loads content from a text file.

    Args:
        file_path: Path to the text file.

    Returns:
        A string containing the loaded content.
        Returns an empty string if the file is not found or if there's an error during loading.
    """
    try:
        with open(file_path, 'r') as file:
            content = file.read()
            return content
    except FileNotFoundError:
        print(f"Error: Text file not found at {file_path}")
        return ""
    except Exception as e:
        print(f"Error loading text file: {e}")
        return ""
