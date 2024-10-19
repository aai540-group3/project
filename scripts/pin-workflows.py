#!/usr/bin/env python3
"""This script updates GitHub Actions workflow files, pinning action references
to specific commit SHAs and adding comments with the latest version name (tag
or branch). It enhances security by ensuring workflows use immutable code
versions and stay up-to-date with the latest versions of actions.

Usage:
    Run the script to update all `.yml` and `.yaml` files in the
    GitHub workflows directory (`.github/workflows`).

Requirements:
    - Python 3.x
    - `requests` library
    - Set the `GITHUB_TOKEN` environment variable.
"""

import concurrent.futures
import glob
import logging
import os
import re
import subprocess
import sys
from typing import List, Optional, Tuple

import requests

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Global configuration
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
if not GITHUB_TOKEN:
    print("Error: GITHUB_TOKEN environment variable is not set.")
    sys.exit(1)

# Determine the workflows directory using git rev-parse
try:
    repo_root = subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"], text=True
    ).strip()
    WORKFLOWS_DIR = os.path.join(repo_root, ".github", "workflows")
except subprocess.CalledProcessError as e:
    logger.error(f"Error determining repository root: {e}")
    sys.exit(1)

if not os.path.exists(WORKFLOWS_DIR):
    logger.error(f"No workflows directory found at '{WORKFLOWS_DIR}'")
    sys.exit(1)


class GitHubAPI:
    """A helper class to interact with the GitHub API."""

    def __init__(self, token: Optional[str] = None):
        """Initialize the GitHubAPI instance."""
        self.session = requests.Session()
        self.headers = {}
        if token:
            self.headers["Authorization"] = f"token {token}"
        self.session.headers.update(self.headers)

    def get_latest_version(
        self, owner: str, repo: str, log_messages: List[str]
    ) -> Optional[Tuple[str, str]]:
        """Retrieve the latest version (release or tag) and its commit SHA."""
        # Try to get the latest release
        latest_release = self.get_latest_release(owner, repo, log_messages)
        if latest_release:
            return latest_release

        # Try to get the latest tag
        latest_tag = self.get_latest_tag(owner, repo, log_messages)
        if latest_tag:
            return latest_tag

        # Get the latest commit SHA from the default branch
        default_branch_commit = self.get_default_branch_commit(
            owner, repo, log_messages
        )
        if default_branch_commit:
            sha, branch_name = default_branch_commit
            return (sha, branch_name)

        log_messages.append(
            f"  Could not find any releases, tags, or default branch for {owner}/{repo}"
        )
        return None

    def get_latest_release(
        self, owner: str, repo: str, log_messages: List[str]
    ) -> Optional[Tuple[str, str]]:
        """Retrieve the latest release tag and its commit SHA for a
        repository."""
        releases_url = f"https://api.github.com/repos/{owner}/{repo}/releases/latest"
        response = self.session.get(releases_url)
        if response.status_code == 200:
            data = response.json()
            tag_name = data.get("tag_name")
            sha = self._get_sha_for_tag(owner, repo, tag_name, log_messages)
            if sha:
                log_messages.append(
                    f"  Using latest release '{tag_name}' for {owner}/{repo}"
                )
                return (sha, tag_name)
        elif response.status_code == 404:
            log_messages.append(f"  No releases found for {owner}/{repo}")
        else:
            log_messages.append(
                f"  Failed to fetch latest release for {owner}/{repo}: {response.status_code}"
            )
        return None

    def get_latest_tag(
        self, owner: str, repo: str, log_messages: List[str]
    ) -> Optional[Tuple[str, str]]:
        """Retrieve the most recent tag and its commit SHA for a repository."""
        tags_url = f"https://api.github.com/repos/{owner}/{repo}/tags"
        params = {"per_page": 1}
        response = self.session.get(tags_url, params=params)
        if response.status_code == 200:
            tags = response.json()
            if tags:
                tag = tags[0]
                tag_name = tag.get("name")
                sha = self._get_sha_for_tag(owner, repo, tag_name, log_messages)
                if sha:
                    log_messages.append(
                        f"  Using latest tag '{tag_name}' for {owner}/{repo}"
                    )
                    return (sha, tag_name)
            else:
                log_messages.append(f"  No tags found for {owner}/{repo}")
        else:
            log_messages.append(
                f"  Failed to fetch tags for {owner}/{repo}: {response.status_code}"
            )
        return None

    def get_default_branch_commit(
        self, owner: str, repo: str, log_messages: List[str]
    ) -> Optional[Tuple[str, str]]:
        """Retrieve the latest commit SHA from the default branch."""
        repo_url = f"https://api.github.com/repos/{owner}/{repo}"
        response = self.session.get(repo_url)
        if response.status_code == 200:
            data = response.json()
            default_branch = data.get("default_branch", "main")
            sha = self._get_sha_from_branch(owner, repo, default_branch, log_messages)
            if sha:
                log_messages.append(
                    f"  Using default branch '{default_branch}' for {owner}/{repo}"
                )
                return (sha, default_branch)
        else:
            log_messages.append(
                f"  Failed to fetch repository info for {owner}/{repo}: {response.status_code}"
            )
        return None

    def _get_sha_for_tag(
        self, owner: str, repo: str, tag_name: str, log_messages: List[str]
    ) -> Optional[str]:
        """Get the commit SHA associated with a tag name."""
        ref_url = f"https://api.github.com/repos/{owner}/{repo}/git/ref/tags/{tag_name}"
        response = self.session.get(ref_url)
        if response.status_code == 200:
            ref_data = response.json()
            object_data = ref_data.get("object", {})
            sha = object_data.get("sha")
            if sha and object_data.get("type") == "tag":
                # Handle annotated tags
                sha = self._get_sha_for_annotated_tag(owner, repo, sha, log_messages)
            return sha
        else:
            log_messages.append(
                f"  Failed to fetch tag '{tag_name}' for {owner}/{repo}"
            )
        return None

    def _get_sha_for_annotated_tag(
        self, owner: str, repo: str, tag_sha: str, log_messages: List[str]
    ) -> Optional[str]:
        """Resolve the commit SHA for an annotated tag."""
        tag_url = f"https://api.github.com/repos/{owner}/{repo}/git/tags/{tag_sha}"
        response = self.session.get(tag_url)
        if response.status_code == 200:
            return response.json().get("object", {}).get("sha")
        return None

    def _get_sha_from_branch(
        self, owner: str, repo: str, branch: str, log_messages: List[str]
    ) -> Optional[str]:
        """Get the commit SHA from a branch name."""
        branch_url = f"https://api.github.com/repos/{owner}/{repo}/branches/{branch}"
        response = self.session.get(branch_url)
        if response.status_code == 200:
            branch_data = response.json()
            sha = branch_data.get("commit", {}).get("sha")
            return sha
        else:
            log_messages.append(
                f"  Failed to fetch branch '{branch}' for {owner}/{repo}: {response.status_code}"
            )
        return None


def update_workflow_file(file_path: str) -> List[str]:
    """Update a workflow file by pinning actions to the latest version SHA and
    adding version comments.

    :param file_path: The path to the workflow file to update.
    :return: List of log messages.
    """
    github_api = GitHubAPI(token=GITHUB_TOKEN)

    # Use a local list to collect log messages for this file
    log_messages = []
    log_messages.append(f"Processing workflow file: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    pattern = r"(uses:\s*([^\s@]+)@([^\s#\n]+)(\s*#.*)?)"

    def replace_match(match):
        full_match = match.group(0)
        action = match.group(2)
        existing_ref = match.group(3)
        existing_comment = match.group(4) or ""

        # Collect logging per action
        log_messages.append(
            f"  Found action: {action}@{existing_ref}{existing_comment}"
        )

        # Ensure action includes owner and repo
        if "/" not in action:
            action = f"actions/{action}"
        owner_repo_parts = action.split("/")
        if len(owner_repo_parts) < 2:
            log_messages.append(f"  Invalid action format: {action}. Skipping.")
            return full_match

        owner, repo = owner_repo_parts[:2]
        path = "/".join(owner_repo_parts[2:]) if len(owner_repo_parts) > 2 else ""

        latest_version = github_api.get_latest_version(owner, repo, log_messages)
        if latest_version:
            latest_sha, version_name = latest_version
            new_action = f"{owner}/{repo}"
            if path:
                new_action += f"/{path}"
            new_line = f"uses: {new_action}@{latest_sha} # {version_name}"
            if new_line != full_match:
                log_messages.append(f"    Updated action: {full_match} -> {new_line}")
            else:
                log_messages.append(f"    Action is already up to date.")
            return new_line
        else:
            log_messages.append(
                f"  Could not retrieve latest version for {owner}/{repo}. Skipping update."
            )
            return full_match

    updated_content = re.sub(pattern, replace_match, content)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(updated_content)

    log_messages.append(f"Workflow file '{file_path}' updated successfully.\n")

    # Return the log messages for this file
    return log_messages


def main():
    """Main function to update workflow files."""

    yaml_files = glob.glob(os.path.join(WORKFLOWS_DIR, "*.yml")) + glob.glob(
        os.path.join(WORKFLOWS_DIR, "*.yaml")
    )

    if not yaml_files:
        logger.error(f"No workflow files found in '{WORKFLOWS_DIR}'")
        sys.exit(1)

    # Dictionary to collect logs per file
    logs_per_file = {}

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_file = {
            executor.submit(update_workflow_file, yaml_file): yaml_file
            for yaml_file in yaml_files
        }

        # Wait for all updates to complete and collect logs
        for future in concurrent.futures.as_completed(future_to_file):
            yaml_file = future_to_file[future]
            try:
                logs = future.result()
                logs_per_file[yaml_file] = logs
            except Exception as exc:
                logger.error(f"{yaml_file} generated an exception: {exc}")

    # Output the log messages in order of files
    for yaml_file in sorted(yaml_files):
        logs = logs_per_file.get(yaml_file, [])
        for message in logs:
            logger.info(message)


if __name__ == "__main__":
    main()
