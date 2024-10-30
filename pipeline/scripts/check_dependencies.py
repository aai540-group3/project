#!/usr/bin/env python3
import pkg_resources
import subprocess
from pathlib import Path
from typing import Dict, List, Set

def get_installed_packages() -> Set[str]:
    """Get set of installed packages."""
    return {pkg.key for pkg in pkg_resources.working_set}

def parse_requirements(file_path: Path) -> Set[str]:
    """Parse requirements file and return set of package names."""
    packages = set()
    with open(file_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                # Remove version specifiers
                package = line.split('>=')[0].split('<=')[0].split('==')[0].split('<')[0].split('>')[0]
                packages.add(package)
    return packages

def check_conflicts(requirements_files: List[Path]) -> Dict[str, List[str]]:
    """Check for conflicting dependencies between requirement files."""
    conflicts = {}
    all_requirements = {}

    for req_file in requirements_files:
        packages = parse_requirements(req_file)
        for pkg in packages:
            if pkg in all_requirements:
                if pkg not in conflicts:
                    conflicts[pkg] = [all_requirements[pkg]]
                conflicts[pkg].append(req_file.name)
            else:
                all_requirements[pkg] = req_file.name

    return conflicts

def main():
    # Get all requirements files
    requirements_files = list(Path().glob('requirements-*.txt'))
    requirements_files.append(Path('requirements.txt'))

    # Check installed packages
    installed = get_installed_packages()

    print("Checking dependencies...")

    # Check each requirements file
    for req_file in requirements_files:
        print(f"\nChecking {req_file}...")
        required = parse_requirements(req_file)
        missing = required - installed

        if missing:
            print(f"Missing packages in {req_file}:")
            for pkg in missing:
                print(f"  - {pkg}")
        else:
            print(f"All packages in {req_file} are installed")

    # Check for conflicts
    conflicts = check_conflicts(requirements_files)
    if conflicts:
        print("\nFound conflicting dependencies:")
        for pkg, files in conflicts.items():
            print(f"  - {pkg} appears in: {', '.join(files)}")
    else:
        print("\nNo conflicting dependencies found")

    # Check for outdated packages
    print("\nChecking for outdated packages...")
    subprocess.run(["pip", "list", "--outdated"])

if __name__ == '__main__':
    main()
