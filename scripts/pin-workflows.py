#!/usr/bin/env python3
"""
.. module:: scripts.pin-workflows
   :synopsis: Pin GitHub Actions workflow action references.

This script updates GitHub Actions workflow files, pinning action
references to specific commit SHAs or the latest release tag. It
enhances security by ensuring workflows use immutable code versions
or the most up-to-date versions within a major release.

Usage:
    Run the script to update all `.yml` and `.yaml` files in the
    GitHub workflows directory (`.github/workflows`).

Requirements:
    - Python 3.x
    - `requests` library
    - `PyYAML` library (optional, for future enhancements)
    - Set the `GITHUB_TOKEN` environment variable.
"""

import concurrent.futures
import glob
import logging
import os
import re
import subprocess
import sys
import queue
from typing import Dict, List, Optional

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
    repo_root = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
    WORKFLOWS_DIR = os.path.join(repo_root, ".github", "workflows")
except subprocess.CalledProcessError as e:
    logger.error(f"Error determining repository root: {e}")
    sys.exit(1)

if not os.path.exists(WORKFLOWS_DIR):
    logger.error(f"No workflows directory found at '{WORKFLOWS_DIR}'")
    sys.exit(1)

# Create a queue for logging messages
log_queue = queue.Queue()


class GitHubAPI:
    """
    A helper class to interact with the GitHub API.
    """

    def __init__(self, token: Optional[str] = None):
        """
        Initialize the GitHubAPI instance.

        :param token: GitHub personal access token.
        :type token: str or None
        """
        self.session = requests.Session()
        self.headers = {}
        if token:
            self.headers["Authorization"] = f"token {token}"
        self.session.headers.update(self.headers)

    def get_latest_release_sha(self, owner: str, repo: str) -> Optional[str]:
        """
        Retrieve the latest release tag for a repository.

        :param owner: The GitHub username or organization name.
        :type owner: str
        :param repo: The repository name.
        :type repo: str
        :return: The tag name of the latest release if found, otherwise None.
        :rtype: str or None
        """
        releases_url = f"https://api.github.com/repos/{owner}/{repo}/releases/latest"
        response = self.session.get(releases_url)
        if response.status_code == 200:
            return response.json().get("tag_name")
        log_queue.put(f"Failed to fetch latest release for {owner}/{repo}: {response.status_code}")
        return None

    def get_tags(self, owner: str, repo: str) -> List[Dict]:
        """
        Retrieve all tags for a given GitHub repository.

        :param owner: The GitHub username or organization name.
        :type owner: str
        :param repo: The repository name.
        :type repo: str
        :return: A list of tag objects retrieved from the GitHub API.
        :rtype: list
        """
        tags = []
        page = 1
        per_page = 100
        while True:
            tags_url = f"https://api.github.com/repos/{owner}/{repo}/tags"
            params = {"per_page": per_page, "page": page}
            response = self.session.get(tags_url, params=params)
            if response.status_code == 200:
                page_tags = response.json()
                if not page_tags:
                    break
                tags.extend(page_tags)
                page += 1
            else:
                log_queue.put(f"Failed to fetch tags for {owner}/{repo}: {response.status_code}")
                break
        return tags

    def get_tag_for_sha(self, owner: str, repo: str, sha: str) -> Optional[str]:
        """
        Find the tag name associated with a commit SHA.

        :param owner: The GitHub username or organization name.
        :type owner: str
        :param repo: The repository name.
        :type repo: str
        :param sha: The commit SHA to search for.
        :type sha: str
        :return: The tag name if found, otherwise None.
        :rtype: str or None
        """
        tags = self.get_tags(owner, repo)
        for tag in tags:
            if tag.get("commit", {}).get("sha") == sha:
                return tag.get("name")
        return None

    def get_sha_for_ref(self, owner: str, repo: str, ref: str) -> Optional[str]:
        """
        Resolve a reference (branch, tag, or SHA) to a commit SHA.

        :param owner: The GitHub username or organization name.
        :type owner: str
        :param repo: The repository name.
        :type repo: str
        :param ref: The reference to resolve.
        :type ref: str
        :return: The commit SHA if resolved, otherwise None.
        :rtype: str or None
        """

        # Use latest release if ref is not pinned
        if not re.match(r"^[a-f0-9]+$", ref):
            latest_release_tag = self.get_latest_release_sha(owner, repo)
            if latest_release_tag:
                ref = latest_release_tag
                log_queue.put(f"Using latest release tag '{ref}' for {owner}/{repo}")
            else:
                log_queue.put(f"Could not find latest release for {owner}/{repo}. Using original ref '{ref}'")

        # Try resolving as a branch
        sha = self._get_sha_from_ref_type(owner, repo, ref_type="heads", ref=ref)
        if sha:
            return sha

        # Try resolving as a tag
        sha = self._get_sha_from_ref_type(owner, repo, ref_type="tags", ref=ref)
        if sha:
            return sha

        # Try validating as a SHA
        sha = self._validate_sha(owner, repo, sha_candidate=ref)
        if sha:
            return sha

        log_queue.put(f"Failed to resolve ref '{ref}' in {owner}/{repo}")
        return None

    def _get_sha_from_ref_type(self, owner: str, repo: str, ref_type: str, ref: str) -> Optional[str]:
        """
        Helper to get SHA from a specific ref type (branch or tag).

        :param owner: The GitHub username or organization name.
        :type owner: str
        :param repo: The repository name.
        :type repo: str
        :param ref_type: The type of ref ('heads' for branches, 'tags' for tags).
        :type ref_type: str
        :param ref: The reference name.
        :type ref: str
        :return: The commit SHA if found, otherwise None.
        :rtype: str or None
        """
        ref_url = f"https://api.github.com/repos/{owner}/{repo}/git/refs/{ref_type}/{ref}"
        response = self.session.get(ref_url)
        if response.status_code == 200:
            ref_data = response.json()
            sha = ref_data.get("object", {}).get("sha")
            # Handle annotated tags
            if ref_type == "tags" and ref_data.get("object", {}).get("type") == "tag":
                sha = self._get_sha_for_annotated_tag(owner, repo, sha)
            return sha
        return None

    def _get_sha_for_annotated_tag(self, owner: str, repo: str, tag_sha: str) -> Optional[str]:
        """
        Resolve the commit SHA for an annotated tag.

        :param owner: The GitHub username or organization name.
        :type owner: str
        :param repo: The repository name.
        :type repo: str
        :param tag_sha: The SHA of the tag object.
        :type tag_sha: str
        :return: The commit SHA if resolved, otherwise None.
        :rtype: str or None
        """
        tag_url = f"https://api.github.com/repos/{owner}/{repo}/git/tags/{tag_sha}"
        response = self.session.get(tag_url)
        if response.status_code == 200:
            return response.json().get("object", {}).get("sha")
        return None

    def _validate_sha(self, owner: str, repo: str, sha_candidate: str) -> Optional[str]:
        """
        Validate if a given string is a valid commit SHA.

        :param owner: The GitHub username or organization name.
        :type owner: str
        :param repo: The repository name.
        :type repo: str
        :param sha_candidate: The SHA candidate to validate.
        :type sha_candidate: str
        :return: The commit SHA if valid, otherwise None.
        :rtype: str or None
        """
        commit_url = f"https://api.github.com/repos/{owner}/{repo}/commits/{sha_candidate}"
        response = self.session.get(commit_url)
        if response.status_code == 200:
            return response.json().get("sha")
        return None


def update_workflow_file(file_path: str) -> None:
    """
    Update a workflow file by pinning actions and adding comments.

    :param file_path: The path to the workflow file to update.
    :type file_path: str
    """
    github_api = GitHubAPI(token=GITHUB_TOKEN)

    log_queue.put(f"\nProcessing workflow file: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    pattern = r"(uses:\s*([^\s@]+)@([^\s#\n]+)(\s*#.*)?)"

    def replace_match(match):
        full_match = match.group(0)
        action = match.group(2)
        ref = match.group(3)
        existing_comment = match.group(4) or ""

        # Group related logging statements together
        log_queue.put(f"Found action: {action}@{ref}{existing_comment}")

        owner_repo = action
        if "/" not in owner_repo:
            owner_repo = "actions/" + owner_repo
        owner_repo_parts = owner_repo.split("/")
        if len(owner_repo_parts) < 2:
            log_queue.put(f"Invalid action format: {action}. Skipping.")
            return full_match

        (owner, repo) = owner_repo_parts[:2]

        sha = github_api.get_sha_for_ref(owner, repo, ref)
        if sha:
            latest_release_tag = github_api.get_latest_release_sha(owner, repo)

            if latest_release_tag and not re.match(r"^[a-f0-9]+$", ref):
                version_info = github_api.get_tag_for_sha(owner, repo, sha)
                if not version_info or not re.match(r"v\d+\.\d+\.\d+", version_info):
                    version_info = latest_release_tag
            else:
                version_info = github_api.get_tag_for_sha(owner, repo, sha) or ref

            new_line = f"uses: {action}@{sha} # {version_info}"
            log_queue.put(f"BEFORE: {full_match}")
            log_queue.put(f"AFTER:  {new_line}\n")
            return new_line
        else:
            log_queue.put(f"Could not resolve ref '{ref}' for {action}. Skipping update.\n")
            return full_match

    updated_content = re.sub(pattern, replace_match, content)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(updated_content)
    log_queue.put(f"Workflow file '{file_path}' updated successfully.")


def main():
    """
    Main function to update workflow files.
    """

    yaml_files = glob.glob(os.path.join(WORKFLOWS_DIR, "*.yml")) + glob.glob(os.path.join(WORKFLOWS_DIR, "*.yaml"))

    if not yaml_files:
        logger.error(f"No workflow files found in '{WORKFLOWS_DIR}'")
        sys.exit(1)

    # Function to process and print log messages from the queue
    def process_log_queue():
        while True:
            try:
                message = log_queue.get_nowait()
                logger.info(message)
            except queue.Empty:
                break

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_file = {executor.submit(update_workflow_file, yaml_file): yaml_file for yaml_file in yaml_files}

        # Wait for all updates to complete
        executor.shutdown(wait=True)

        # Process and print the log messages in order
        process_log_queue()


if __name__ == "__main__":
    main()
