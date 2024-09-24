#!/usr/bin/env bash
#===============================================================================
# Title           :install-terraform.sh
# Description     :This script installs Terraform if it is not already installed.
# Version         :1.0
# Usage           :./install-terraform.sh [OPTIONS]
# Notes           :Requires sudo privileges to install packages
#===============================================================================

set -Eeuo pipefail
trap cleanup SIGINT SIGTERM ERR EXIT

#---------------------------------------
# Configuration Variables
#---------------------------------------
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd -P)
DEFAULT_TERRAFORM_VERSION="latest"
TERRAFORM_VERSION="${TERRAFORM_VERSION:-$DEFAULT_TERRAFORM_VERSION}"

#---------------------------------------
# Function: usage
# Description:
#   Prints the help message.
#---------------------------------------
usage() {
  cat <<EOF
Usage: install-terraform.sh [OPTIONS]

This script installs Terraform if it is not already installed.

Options:
  -v, --version VERSION  Specify the Terraform version to install (default: $DEFAULT_TERRAFORM_VERSION).
  -h, --help             Display this help message and exit.

Environment Variables:
  TERRAFORM_VERSION      Specify the Terraform version to install (default: $DEFAULT_TERRAFORM_VERSION).

Examples:
  install-terraform.sh
  install-terraform.sh --version 1.0.0
  TERRAFORM_VERSION=1.0.0 install-terraform.sh
EOF
}

#---------------------------------------
# Function: msg
# Description:
#   Prints a message to the console with a timestamp.
# Arguments:
#   $1 - Message to print
#---------------------------------------
msg() {
  echo >&2 -e "[$(date +'%Y-%m-%d %H:%M:%S')] ${1-}"
}

#---------------------------------------
# Function: die
# Description:
#   Prints an error message and exits the script.
# Arguments:
#   $1 - Error message
#   $2 - Exit code (optional, default: 1)
#---------------------------------------
die() {
  local msg=$1
  local code=${2-1} # default exit status 1
  msg "$msg"
  exit "$code"
}

#---------------------------------------
# Function: cleanup
# Description:
#   Cleans up temporary files and processes.
#---------------------------------------
cleanup() {
  trap - SIGINT SIGTERM ERR EXIT
  # Add cleanup code here if needed
}

#---------------------------------------
# Function: check_dependencies
# Description:
#   Checks if required dependencies are installed.
#---------------------------------------
check_dependencies() {
  if ! command -v wget &>/dev/null; then
    die "wget is not installed. Please install it and try again."
  fi
  if ! command -v gpg &>/dev/null; then
    die "gpg is not installed. Please install it and try again."
  fi
  if ! command -v lsb_release &>/dev/null; then
    die "lsb_release is not installed. Please install it and try again."
  fi
}

#---------------------------------------
# Function: add_hashicorp_repo
# Description:
#   Adds the HashiCorp repository to the system.
#---------------------------------------
add_hashicorp_repo() {
  msg "Adding HashiCorp GPG key..."
  wget -O- https://apt.releases.hashicorp.com/gpg | sudo gpg --dearmor -o /usr/share/keyrings/hashicorp-archive-keyring.gpg

  msg "Adding HashiCorp repository..."
  echo "deb [signed-by=/usr/share/keyrings/hashicorp-archive-keyring.gpg] https://apt.releases.hashicorp.com $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/hashicorp.list
}

#---------------------------------------
# Function: install_terraform
# Description:
#   Installs Terraform if it is not already installed.
#---------------------------------------
install_terraform() {
  if ! terraform_installed; then
    add_hashicorp_repo

    msg "Updating package lists..."
    sudo apt update

    if [[ "$TERRAFORM_VERSION" == "latest" ]]; then
      msg "Installing latest version of Terraform..."
      sudo apt install -y terraform
    else
      msg "Installing Terraform version $TERRAFORM_VERSION..."
      sudo apt install -y terraform=$TERRAFORM_VERSION
    fi

    if [[ $? -ne 0 ]]; then
      die "Failed to install Terraform."
    fi
  fi

  sudo ln -sf /usr/bin/terraform /usr/local/bin/terraform
  if [[ $? -ne 0 ]]; then
    die "Failed to create symbolic link for Terraform."
  fi

  local version=$(print_version)
  msg "INSTALLED: Terraform $version"
}

#---------------------------------------
# Function: terraform_installed
# Description:
#   Checks if Terraform is already installed.
#---------------------------------------
terraform_installed() {
  command -v terraform &>/dev/null
}

#---------------------------------------
# Function: print_version
# Description:
#   Prints the installed Terraform version.
#---------------------------------------
print_version() {
  terraform version | head -n 1 | cut -d 'v' -f 2
}

#---------------------------------------
# Function: parse_params
# Description:
#   Parses command-line parameters.
#---------------------------------------
parse_params() {
  while :; do
    case "${1-}" in
    -h | --help)
      usage
      exit 0
      ;;
    -v | --version)
      TERRAFORM_VERSION="${2-}"
      shift
      ;;
    -?*)
      die "Unknown option: $1"
      ;;
    *)
      break
      ;;
    esac
    shift
  done

  return 0
}

#---------------------------------------
# Main Script Logic
#---------------------------------------
main() {
  parse_params "$@"
  check_dependencies
  install_terraform
}

# Execute main function
main "$@"
