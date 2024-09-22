#!/usr/bin/env bash

# bootstrap.sh - Set up the project environment

# Exit immediately if a command exits with a non-zero status
set -e

# Function to check if a command exists
command_exists () {
    command -v "$1" >/dev/null 2>&1 ;
}

echo "Starting bootstrap process..."

# 1. Check for required dependencies
echo "Checking for required dependencies..."

# Check for Python 3
if ! command_exists python3 ; then
    echo "Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Check for pip
if ! command_exists pip3 ; then
    echo "pip is not installed. Please install pip."
    exit 1
fi

# Check for DVC
if ! command_exists dvc ; then
    echo "DVC is not installed. Installing DVC..."
    pip3 install --upgrade dvc[all]
fi

# 2. Set up the Python virtual environment
echo "Setting up Python virtual environment..."

# Create a virtual environment in the .venv directory
python3 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# 3. Install Python dependencies
echo "Installing Python dependencies..."

pip install -r requirements.txt

# 4. Prompt the user for necessary tokens securely
echo "Configuring authentication..."

# Provide instructions on how to obtain the Hugging Face token
echo ""
echo "To run this project, you need a Hugging Face access token."
echo "If you don't have one, follow these steps:"
echo "1. Go to https://huggingface.co/settings/tokens"
echo "2. Log in or create an account if you haven't already."
echo "3. Click on 'New token' to create a new access token."
echo "4. Give it a name and select the appropriate scopes (e.g., 'api', 'read', 'write')."
echo "5. Click 'Generate' and copy the token."
echo ""

# Check if HF_TOKEN is already set
if [ -z "$HF_TOKEN" ]; then
    read -s -p "Enter your Hugging Face token (HF_TOKEN): " HF_TOKEN
    echo ""
fi

# Provide instructions on how to obtain the DVC Studio token
echo ""
echo "To send live experiment updates to DVC Studio, you need a DVC Studio access token."
echo "If you don't have one, follow these steps:"
echo "1. Go to https://studio.iterative.ai/"
echo "2. Log in or create an account if you haven't already."
echo "3. Click on your profile icon in the top-right corner and select 'Account settings'."
echo "4. Navigate to 'Access tokens' and click 'Generate new token'."
echo "5. Provide a name and select the appropriate scopes (e.g., 'Experiment operations')."
echo "6. Click 'Generate' and copy the token."
echo ""

# Check if DVC_STUDIO_TOKEN is already set
if [ -z "$DVC_STUDIO_TOKEN" ]; then
    read -s -p "Enter your DVC Studio token (DVC_STUDIO_TOKEN): " DVC_STUDIO_TOKEN
    echo ""
fi

# Export tokens for the current session
export HF_TOKEN
export DVC_STUDIO_TOKEN

# 5. Configure DVC remotes with authentication
echo "Configuring DVC remotes..."

# Remove any existing local config to prevent duplication
rm -f .dvc/config.local

# Configure models_remote
dvc remote modify models_remote auth custom
dvc remote modify models_remote --local custom_auth_header "Authorization: Bearer $HF_TOKEN"

# Configure dataset_remote
dvc remote modify dataset_remote auth custom
dvc remote modify dataset_remote --local custom_auth_header "Authorization: Bearer $HF_TOKEN"

# 6. Initialize DVC (if not already initialized)
if [ ! -d ".dvc" ]; then
    dvc init
fi

# 7. Pull data and models from DVC remotes
echo "Pulling data and models from DVC remotes..."

dvc pull

# 8. Final instructions
echo ""
echo "Bootstrap process completed successfully!"
echo "To start working on the project:"
echo "1. Activate the virtual environment with:"
echo "   source .venv/bin/activate"
echo "2. Ensure that HF_TOKEN and DVC_STUDIO_TOKEN are exported in your environment when running experiments."
echo ""
echo "Note: You can add the following lines to your shell profile (~/.bashrc or ~/.zshrc) to persist the tokens (optional):"
echo "export HF_TOKEN=\"$HF_TOKEN\""
echo "export DVC_STUDIO_TOKEN=\"$DVC_STUDIO_TOKEN\""
echo ""
echo "Alternatively, you can run 'source bootstrap-env' to export the tokens for your session."

# Create a script to export tokens for future sessions (optional)
echo "export HF_TOKEN=\"$HF_TOKEN\"" > bootstrap-env
echo "export DVC_STUDIO_TOKEN=\"$DVC_STUDIO_TOKEN\"" >> bootstrap-env

# Deactivate virtual environment to prevent accidental usage
deactivate