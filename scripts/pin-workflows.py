import re
import os
import sys
import glob
import requests

GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')

def get_tags(owner, repo):
    headers = {}
    if GITHUB_TOKEN:
        headers['Authorization'] = f'token {GITHUB_TOKEN}'

    tags = []
    page = 1
    per_page = 100
    while True:
        tags_url = f'https://api.github.com/repos/{owner}/{repo}/tags'
        params = {'per_page': per_page, 'page': page}
        response = requests.get(tags_url, headers=headers, params=params)
        if response.status_code == 200:
            page_tags = response.json()
            if not page_tags:
                break
            tags.extend(page_tags)
            page += 1
        else:
            print(f"Failed to fetch tags for {owner}/{repo}: {response.status_code}")
            break
    return tags

def get_tag_for_sha(owner, repo, sha):
    tags = get_tags(owner, repo)
    for tag in tags:
        if tag['commit']['sha'] == sha:
            return tag['name']
    return None

def get_sha_for_ref(owner, repo, ref):
    headers = {}
    if GITHUB_TOKEN:
        headers['Authorization'] = f'token {GITHUB_TOKEN}'

    # Try to get the SHA directly (might be a SHA, tag, or branch)
    ref_url = f'https://api.github.com/repos/{owner}/{repo}/git/refs/heads/{ref}'
    response = requests.get(ref_url, headers=headers)
    if response.status_code == 200:
        ref_data = response.json()
        sha = ref_data['object']['sha']
        return sha
    else:
        ref_url = f'https://api.github.com/repos/{owner}/{repo}/git/refs/tags/{ref}'
        response = requests.get(ref_url, headers=headers)
        if response.status_code == 200:
            ref_data = response.json()
            sha = ref_data['object']['sha']
            # Handle annotated tags
            if ref_data['object']['type'] == 'tag':
                tag_url = f'https://api.github.com/repos/{owner}/{repo}/git/tags/{sha}'
                tag_response = requests.get(tag_url, headers=headers)
                if tag_response.status_code == 200:
                    tag_data = tag_response.json()
                    sha = tag_data['object']['sha']
            return sha
        else:
            # It's possibly a SHA
            commit_url = f'https://api.github.com/repos/{owner}/{repo}/git/commits/{ref}'
            commit_response = requests.get(commit_url, headers=headers)
            if commit_response.status_code == 200:
                commit_data = commit_response.json()
                sha = commit_data['sha']
                return sha
            else:
                print(f"Failed to find ref '{ref}' in {owner}/{repo}")
                return None

def update_workflow_file(file_path):
    print(f"\nProcessing workflow file: {file_path}")

    with open(file_path, 'r') as f:
        content = f.read()

    # Regex pattern to find uses: statements with any ref (SHA, tag, or branch)
    pattern = r'(uses:\s*([^\s@]+)@([^\s#\n]+)(\s*#.*)?)'
    matches = re.findall(pattern, content)

    updated_content = content

    for full_match, action, ref, comment in matches:
        print(f"Found action: {action}@{ref}")
        owner_repo = action
        if '/' not in owner_repo:
            owner_repo = 'actions/' + owner_repo
        # Extract owner and repo
        owner_repo_parts = owner_repo.split('/')
        if len(owner_repo_parts) < 2:
            print(f"Invalid action format: {action}")
            continue
        owner = owner_repo_parts[0]
        repo = owner_repo_parts[1]

        # Get the commit SHA for the ref
        sha = get_sha_for_ref(owner, repo, ref)

        if sha:
            # Get the semantic version tag for the SHA
            tag = get_tag_for_sha(owner, repo, sha)
            if tag:
                version_info = tag
            else:
                version_info = "no tag found"

            # Prepare the original and new uses lines
            original_line = full_match.strip()
            new_line = f'uses: {action}@{sha}  # {version_info}'

            # Print BEFORE and AFTER lines
            print(f"BEFORE: {original_line}")
            print(f"AFTER:  {new_line}\n")

            # Replace in the content
            updated_content = updated_content.replace(
                original_line,
                new_line
            )
        else:
            print(f"Could not resolve ref '{ref}' for {action}. Skipping update.\n")

    with open(file_path, 'w') as f:
        f.write(updated_content)
    print(f"Workflow file '{file_path}' updated successfully.")

if __name__ == '__main__':
    # Set the workflows directory
    workflows_dir = '../.github/workflows'
    if not os.path.exists(workflows_dir):
        print(f"No workflows directory found at '{workflows_dir}'")
        sys.exit(1)

    # Get all .yml and .yaml files in the workflows directory
    yaml_files = glob.glob(os.path.join(workflows_dir, '*.yml')) + \
                 glob.glob(os.path.join(workflows_dir, '*.yaml'))

    if not yaml_files:
        print(f"No workflow files found in '{workflows_dir}'")
        sys.exit(1)

    for yaml_file in yaml_files:
        update_workflow_file(yaml_file)