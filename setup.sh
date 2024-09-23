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

# Check for AWS CLI
if ! command_exists aws ; then
    echo "AWS CLI is not installed. Installing AWS CLI..."

    # Determine the OS and install AWS CLI accordingly
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # For Debian/Ubuntu-based distributions
        sudo apt update
        sudo apt install -y awscli
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        # For macOS
        brew install awscli
    else
        echo "Unsupported OS. Please install AWS CLI manually."
        exit 1
    fi
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

# 4. Configure AWS credentials
echo "Configuring AWS credentials..."

# Check if AWS credentials are configured
if ! aws sts get-caller-identity >/dev/null 2>&1 ; then
    echo ""
    echo "AWS credentials are not configured."
    echo "Please configure your AWS credentials by running 'aws configure' and then re-run this script."
    deactivate
    exit 1
fi

echo "AWS credentials are configured."

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

# Export token for the current session
export DVC_STUDIO_TOKEN

# 5. Configure DVC remotes
echo "Configuring DVC remotes..."

# Remove any existing local config to prevent duplication
rm -f .dvc/config.local

# Optionally set AWS region for DVC remotes
# Uncomment and set your AWS region if needed
# AWS_REGION='your-aws-region'
# dvc remote modify models_remote region $AWS_REGION
# dvc remote modify datasets_remote region $AWS_REGION

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
echo "2. Ensure that DVC_STUDIO_TOKEN is exported in your environment when running experiments."
echo ""
echo "Note: You can add the following line to your shell profile (~/.bashrc or ~/.zshrc) to persist the token (optional):"
echo "export DVC_STUDIO_TOKEN=\"$DVC_STUDIO_TOKEN\""
echo ""
echo "Alternatively, you can run 'source bootstrap-env' to export the token for your session."

# Create a script to export token for future sessions (optional)
echo "export DVC_STUDIO_TOKEN=\"$DVC_STUDIO_TOKEN\"" > bootstrap-env

# Deactivate virtual environment to prevent accidental usage
deactivate