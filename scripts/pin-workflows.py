#!/usr/bin/env python3
"""
This script scans GitHub Actions workflow files and updates action references by pinning them to specific
commit SHAs. It replaces the action references with their corresponding commit SHAs and adds any associated
tags as comments. This practice enhances security by ensuring that workflows use immutable code versions.

Usage:
    Run the script to automatically update all `.yml` and `.yaml` files in the GitHub workflows
    directory (`.github/workflows`).

Requirements:
    - Python 3.x
    - `requests` library
    - `PyYAML` library (optional, for future enhancements)
    - Set the `GITHUB_TOKEN` environment variable with a valid GitHub personal access token.

Note:
    Ensure that you have the necessary permissions for the repositories being accessed. The GitHub token must
    have at least `public_repo` scope for public repositories and additional scopes if accessing private repositories.
"""

import glob
import logging
import os
import re
import sys
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

WORKFLOWS_DIR = os.path.join("..", ".github", "workflows")


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
                logger.error(
                    f"Failed to fetch tags for {owner}/{repo}: {response.status_code}"
                )
                break
        return tags

    def get_tag_for_sha(self, owner: str, repo: str, sha: str) -> Optional[str]:
        """
        Find the tag name associated with a specific commit SHA in a repository.

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
        Resolve a reference (branch name, tag name, or SHA) to a commit SHA.

        :param owner: The GitHub username or organization name.
        :type owner: str
        :param repo: The repository name.
        :type repo: str
        :param ref: The reference to resolve.
        :type ref: str
        :return: The commit SHA if resolved, otherwise None.
        :rtype: str or None
        """
        # Attempt to resolve the ref as a branch
        sha = self._get_sha_from_ref_type(owner, repo, ref_type="heads", ref=ref)
        if sha:
            return sha

        # Attempt to resolve the ref as a tag
        sha = self._get_sha_from_ref_type(owner, repo, ref_type="tags", ref=ref)
        if sha:
            return sha

        # Attempt to validate the ref as a SHA
        sha = self._validate_sha(owner, repo, sha_candidate=ref)
        if sha:
            return sha

        logger.warning(f"Failed to resolve ref '{ref}' in {owner}/{repo}")
        return None

    def _get_sha_from_ref_type(
        self, owner: str, repo: str, ref_type: str, ref: str
    ) -> Optional[str]:
        """
        Helper method to get SHA from a specific ref type (branch or tag).

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
        ref_url = (
            f"https://api.github.com/repos/{owner}/{repo}/git/refs/{ref_type}/{ref}"
        )
        response = self.session.get(ref_url)
        if response.status_code == 200:
            ref_data = response.json()
            sha = ref_data.get("object", {}).get("sha")
            # Handle annotated tags
            if ref_type == "tags" and ref_data.get("object", {}).get("type") == "tag":
                sha = self._get_sha_for_annotated_tag(owner, repo, sha)
            return sha
        return None

    def _get_sha_for_annotated_tag(
        self, owner: str, repo: str, tag_sha: str
    ) -> Optional[str]:
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
            tag_data = response.json()
            return tag_data.get("object", {}).get("sha")
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
        commit_url = (
            f"https://api.github.com/repos/{owner}/{repo}/commits/{sha_candidate}"
        )
        response = self.session.get(commit_url)
        if response.status_code == 200:
            commit_data = response.json()
            return commit_data.get("sha")
        return None


def update_workflow_file(file_path: str, github_api: GitHubAPI) -> None:
    """
    Update a GitHub Actions workflow file by pinning actions to commit SHAs and adding version comments.

    :param file_path: The path to the workflow file to update.
    :type file_path: str
    :param github_api: An instance of the GitHubAPI class.
    :type github_api: GitHubAPI
    """
    logger.info(f"\nProcessing workflow file: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Improved Regex pattern to handle existing comments
    pattern = r"(uses:\s*([^\s@]+)@([^\s#\n]+)(\s*#.*)?)"

    def replace_match(match):
        full_match = match.group(0)
        action = match.group(2)
        ref = match.group(3)
        existing_comment = match.group(4) or ""  # Get existing comment or empty string
        logger.info(f"Found action: {action}@{ref}{existing_comment}")

        owner_repo = action
        if "/" not in owner_repo:
            owner_repo = (
                "actions/" + owner_repo
            )  # Default to 'actions' org if not specified
        owner_repo_parts = owner_repo.split("/")
        if len(owner_repo_parts) < 2:
            logger.warning(f"Invalid action format: {action}. Skipping.")
            return full_match

        owner, repo = owner_repo_parts[:2]

        sha = github_api.get_sha_for_ref(owner, repo, ref)
        if sha:
            tag = github_api.get_tag_for_sha(owner, repo, sha)
            version_info = tag if tag else "no tag found"

            new_line = f"uses: {action}@{sha}  # {version_info}"
            logger.info(f"BEFORE: {full_match}")
            logger.info(f"AFTER:  {new_line}\n")
            return new_line
        else:
            logger.warning(
                f"Could not resolve ref '{ref}' for {action}. Skipping update.\n"
            )
            return full_match

    updated_content = re.sub(pattern, replace_match, content)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(updated_content)
    logger.info(f"Workflow file '{file_path}' updated successfully.")


def main():
    """
    Main function to update GitHub Actions workflow files by pinning action references to commit SHAs.
    """
    if not os.path.exists(WORKFLOWS_DIR):
        logger.error(f"No workflows directory found at '{WORKFLOWS_DIR}'")
        sys.exit(1)

    # Get all .yml and .yaml files in the workflows directory
    yaml_files = glob.glob(os.path.join(WORKFLOWS_DIR, "*.yml")) + glob.glob(
        os.path.join(WORKFLOWS_DIR, "*.yaml")
    )

    if not yaml_files:
        logger.error(f"No workflow files found in '{WORKFLOWS_DIR}'")
        sys.exit(1)

    github_api = GitHubAPI(token=GITHUB_TOKEN)

    for yaml_file in yaml_files:
        try:
            update_workflow_file(yaml_file, github_api)
        except Exception as e:
            logger.error(f"An error occurred while processing '{yaml_file}': {e}")


if __name__ == "__main__":
    main()
