#!/usr/bin/env bash
#===============================================================================
# Title           :install_latex.sh
# Description     :This script installs LaTeX and related dependencies
# Version         :1.0
# Usage           :sudo ./install_latex.sh
# Notes           :Requires sudo privileges to install packages
#===============================================================================

set -euo pipefail

#---------------------------------------
# Function: install_package
# Description:
#   Installs a specified package using apt
# Arguments:
#   $1 - Package name
#---------------------------------------
install_package() {
    local package_name="$1"
    echo "Installing $package_name..."
    sudo apt install -y "$package_name"
    echo "$package_name installed successfully."
}

#---------------------------------------
# Main Script Logic
#---------------------------------------

echo "---- Starting LaTeX Dependencies Installation ----"

# Install latexmk
install_package "latexmk"

# Install texlive-fonts-extra
install_package "texlive-fonts-extra"

# Install texlive-bibtex-extra and biber
sudo apt-get install -y texlive-bibtex-extra biber

echo "---- LaTeX Dependencies Installation Complete ----"