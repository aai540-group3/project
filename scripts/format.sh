#!/usr/bin/env bash
#===============================================================================
# Title           :format.sh
# Description     :This script formats various code and text files.
# Version         :1.0
# Usage           :./format.sh
# Notes           :Requires installation of formatting tools (see TOOLS arrays).
#===============================================================================

#---------------------------------------
# Array of APT tools to install
#---------------------------------------
APT_TOOLS=(
  jq
  yq
  shfmt
  terraform
)

#---------------------------------------
# Array of PIP tools to install
#---------------------------------------
PIP_TOOLS=(
  autopep8
  isort
  docformatter
  black
)

#---------------------------------------
# Function: install_apt_package
# Description:
#   Installs a specified package using apt if the command is not found.
# Arguments:
#   $1 - Package name
#---------------------------------------
install_apt_package() {
  local package_name="$1"
  if ! command -v "$package_name" >/dev/null 2>&1; then
    echo "Command '$package_name' not found. Installing..."
    sudo apt-get update
    sudo apt-get install -y --no-install-recommends "$package_name"
  fi
}

#---------------------------------------
# Function: install_pip_package
# Description:
#   Installs a specified package using pip if the command is not found.
# Arguments:
#   $1 - Package name
#---------------------------------------
install_pip_package() {
  local package_name="$1"
  if ! command -v "$package_name" >/dev/null 2>&1; then
    echo "Command '$package_name' not found. Installing..."
    pip install "$package_name"
  fi
}

#---------------------------------------
# Function: format_python
# Description:
#   Formats Python files using black, autopep8, isort, and docformatter.
#---------------------------------------
format_python() {
  echo "Formatting Python files..."
  find . -name "*.py" -print0 | while IFS= read -r -d $'\0' file; do
    black "$file" -l 9999
    autopep8 -i "$file"
    isort "$file"
    docformatter -i "$file"
  done
}

#---------------------------------------
# Function: format_json
# Description:
#   Formats JSON files using jq.
#---------------------------------------
format_json() {
  echo "Formatting JSON files..."
  find . -name "*.json" -print0 | while IFS= read -r -d $'\0' file; do
    jq . "$file" >"$file.tmp" && mv "$file.tmp" "$file"
  done
}

#---------------------------------------
# Function: format_yaml
# Description:
#   Formats YAML/YML files using yq.
#---------------------------------------
format_yaml() {
  echo "Formatting YAML/YML files..."
  find . -name "*.yaml" -o -name "*.yml" -print0 | while IFS= read -r -d $'\0' file; do
    yq -i . "$file"
  done
}

#---------------------------------------
# Function: format_markdown
# Description:
#   Formats Markdown files using prettier.
#---------------------------------------
format_markdown() {
  echo "Formatting Markdown files..."
  find . -name "*.md" -print0 | while IFS= read -r -d $'\0' file; do
    npx prettier --write "$file"
  done
}

#---------------------------------------
# Function: format_shell_scripts
# Description:
#   Formats shell scripts using shfmt.
#---------------------------------------
format_shell_scripts() {
  echo "Formatting shell scripts..."
  find . -name "*.sh" -print0 | while IFS= read -r -d $'\0' file; do
    shfmt -i 2 -w "$file"
  done
}

#---------------------------------------
# Function: format_terraform
# Description:
#   Formats Terraform files using terraform fmt.
#---------------------------------------
format_terraform() {
  echo "Formatting Terraform files..."
  find . -name "*.tf" -print0 | while IFS= read -r -d $'\0' file; do
    terraform fmt "$file"
  done
  find . -name "*.tfvars" -print0 | while IFS= read -r -d $'\0' file; do
    terraform fmt "$file"
  done
}

#---------------------------------------
# Main Script Logic
#---------------------------------------

echo "---- Starting Formatting Process ----"

# Install APT tools if not found
for tool in "${APT_TOOLS[@]}"; do
  install_apt_package "$tool"
done

# Install PIP tools if not found
for tool in "${PIP_TOOLS[@]}"; do
  install_pip_package "$tool"
done

format_python
format_json
format_yaml
format_markdown
format_shell_scripts
format_terraform

echo "---- Formatting Process Complete ----"
