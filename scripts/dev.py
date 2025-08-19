#!/usr/bin/env python3
"""
Development script for Kartezio.
Provides common development tasks using uv and modern Python tooling.
"""

import subprocess
import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


def run_command(cmd, description=None):
    """Run a command and handle errors."""
    if description:
        print(f"🔄 {description}...")

    try:
        result = subprocess.run(cmd, shell=True, check=True, cwd=PROJECT_ROOT)
        if description:
            print(f"✅ {description} completed")
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        if description:
            print(f"❌ {description} failed")
        print(f"Error: {e}")
        return False


def lint_code():
    """Run code linting with ruff."""
    print("🧹 Running code linting...")
    success = True

    # Run ruff linting
    if not run_command("uv run ruff check src tests", "Ruff linting"):
        success = False

    # Run ruff formatting check
    if not run_command(
        "uv run ruff format --check src tests", "Ruff format check"
    ):
        success = False

    return success


def format_code():
    """Format code with ruff."""
    print("🎨 Formatting code...")
    success = True

    # Fix linting issues
    if not run_command("uv run ruff check --fix src tests", "Ruff auto-fix"):
        success = False

    # Format code
    if not run_command("uv run ruff format src tests", "Ruff formatting"):
        success = False

    return success


def run_tests():
    """Run the test suite."""
    print("🧪 Running tests...")

    # Run our custom test validation
    if not run_command("uv run python validate_tests.py", "Test validation"):
        return False

    # Run pytest if available
    print("🔬 Running pytest...")
    return run_command("uv run pytest tests/ -v", "Pytest")


def run_security_tests():
    """Run security-focused tests."""
    print("🔒 Running security tests...")
    return run_command(
        "uv run python -m unittest tests.test_security -v", "Security tests"
    )


def type_check():
    """Run type checking with mypy."""
    print("🎯 Running type checking...")
    return run_command("uv run mypy src", "MyPy type checking")


def check_dependencies():
    """Check for dependency issues."""
    print("📦 Checking dependencies...")
    success = True

    # Check for dependency conflicts
    if not run_command("uv lock --check", "Dependency lock check"):
        success = False

    # Try to import the package
    if not run_command(
        "uv run python -c 'import kartezio; print(\"✅ Package imports successfully\")'",
        "Package import test",
    ):
        success = False

    return success


def build_package():
    """Build the package."""
    print("🏗️ Building package...")
    return run_command("uv build", "Package build")


def clean():
    """Clean build artifacts."""
    print("🧽 Cleaning build artifacts...")
    success = True

    # Remove common build artifacts
    artifacts = [
        "build/",
        "dist/",
        "*.egg-info/",
        ".pytest_cache/",
        "htmlcov/",
        ".coverage",
        "__pycache__/",
        ".mypy_cache/",
    ]

    for artifact in artifacts:
        if not run_command(f"rm -rf {artifact}", f"Removing {artifact}"):
            success = False

    return success


def dev_install():
    """Install package in development mode."""
    print("🔧 Installing package in development mode...")
    return run_command("uv sync --dev", "Development installation")


def main():
    parser = argparse.ArgumentParser(description="Kartezio Development Tools")
    parser.add_argument(
        "command",
        choices=[
            "lint",
            "format",
            "test",
            "security",
            "typecheck",
            "deps",
            "build",
            "clean",
            "install",
            "all",
        ],
        help="Command to run",
    )
    parser.add_argument(
        "--fast", action="store_true", help="Skip slow operations"
    )

    args = parser.parse_args()

    success = True

    if args.command == "lint":
        success = lint_code()
    elif args.command == "format":
        success = format_code()
    elif args.command == "test":
        success = run_tests()
    elif args.command == "security":
        success = run_security_tests()
    elif args.command == "typecheck":
        success = type_check()
    elif args.command == "deps":
        success = check_dependencies()
    elif args.command == "build":
        success = build_package()
    elif args.command == "clean":
        success = clean()
    elif args.command == "install":
        success = dev_install()
    elif args.command == "all":
        print("🚀 Running full development workflow...")
        commands = [
            ("Installing dependencies", dev_install),
            ("Linting code", lint_code),
            ("Running security tests", run_security_tests),
        ]

        if not args.fast:
            commands.extend(
                [
                    ("Type checking", type_check),
                    ("Running full tests", run_tests),
                    ("Checking dependencies", check_dependencies),
                    ("Building package", build_package),
                ]
            )

        for description, func in commands:
            print(f"\n{'=' * 50}")
            print(f"{description}...")
            print(f"{'=' * 50}")
            if not func():
                print(f"❌ {description} failed")
                success = False
                break

    if success:
        print(f"\n🎉 Command '{args.command}' completed successfully!")
        sys.exit(0)
    else:
        print(f"\n💥 Command '{args.command}' failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
