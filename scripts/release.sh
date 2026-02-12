#!/bin/bash
set -e

# Colors and formatting
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'
BOLD='\033[1m'

print_step() { echo -e "\n${BOLD}${GREEN}=== $1 ===${NC}\n"; }
print_warning() { echo -e "${YELLOW}WARNING: $1${NC}"; }
print_error() { echo -e "${RED}ERROR: $1${NC}"; }
confirm() {
  echo -e -n "${BOLD}$1 (y/N) ${NC}"
  read -r response
  [[ "$response" == "y" ]]
}

# Header
echo -e "${BOLD}${GREEN}"
echo "╔════════════════════════════════════╗"
echo "║        pymde release script        ║"
echo "╚════════════════════════════════════╝"
echo -e "${NC}"

# Check if version type is provided
if [ -z "$1" ]; then
  echo -e "\nAvailable version types:"
  echo "  - minor (0.x.0)"
  echo "  - patch (0.0.x)"
  print_error "Please specify version type: ./scripts/release.sh <minor|patch>"
  exit 1
fi

VERSION_TYPE=$1

if [[ ! "$VERSION_TYPE" =~ ^(minor|patch)$ ]]; then
  print_error "Invalid version type. Use: minor or patch"
  exit 1
fi

# Check if on main branch
print_step "Checking git branch"
BRANCH=$(git branch --show-current)
if [ "$BRANCH" != "main" ]; then
  print_error "Not on main branch. Current branch: $BRANCH"
  exit 1
fi

# Check git state
print_step "Checking git status"
if [ -n "$(git status --porcelain)" ]; then
  print_error "Git working directory is not clean"
  git status
  exit 1
fi

# Pull latest
print_step "Pulling latest changes"
git pull origin main

# Read current version
VERSION_FILE="pymde/__init__.py"
CURRENT_VERSION=$(python -c "
import re
with open('$VERSION_FILE') as f:
    match = re.search(r'__version__\s*=\s*[\"'\''](.*?)[\"'\'']', f.read())
    print(match.group(1))
")

IFS='.' read -r MAJOR MINOR PATCH <<< "$CURRENT_VERSION"

if [ "$VERSION_TYPE" == "minor" ]; then
  MINOR=$((MINOR + 1))
  PATCH=0
elif [ "$VERSION_TYPE" == "patch" ]; then
  PATCH=$((PATCH + 1))
fi

NEW_VERSION="$MAJOR.$MINOR.$PATCH"

# Update version in file
print_step "Updating version"
sed -i.bak "s/__version__ = .*/__version__ = \"$NEW_VERSION\"/" "$VERSION_FILE"
rm -f "${VERSION_FILE}.bak"
echo "  $CURRENT_VERSION -> $NEW_VERSION"

# Summary
echo -e "\n${BOLD}Release Summary:${NC}"
echo "  Version: $CURRENT_VERSION -> $NEW_VERSION"
echo "  Tag: v$NEW_VERSION"
echo "  File: $VERSION_FILE"

if ! confirm "Proceed with release?"; then
  git checkout "$VERSION_FILE"
  print_warning "Release cancelled"
  exit 1
fi

# Commit
print_step "Committing version change"
git add "$VERSION_FILE"
git commit -m "v$NEW_VERSION"

# Push
PUSHED=false
if confirm "Push changes to remote?"; then
  git push origin main
  echo -e "${GREEN}Changes pushed${NC}"
  PUSHED=true
fi

# Tag
TAGGED=false
if confirm "Create and push tag v$NEW_VERSION?"; then
  git tag -a "v$NEW_VERSION" -m "v$NEW_VERSION"
  git push origin "v$NEW_VERSION"
  echo -e "${GREEN}Tag pushed — release workflow will start${NC}"
  TAGGED=true
fi

if $TAGGED; then
  echo -e "\n${BOLD}${GREEN}Release v$NEW_VERSION complete!${NC}\n"
  echo "  Monitor: https://github.com/cvxgrp/pymde/actions"
  echo "  PyPI:    https://pypi.org/project/pymde/$NEW_VERSION/"
else
  echo -e "\n${YELLOW}Version bumped to $NEW_VERSION locally but not released.${NC}"
  if ! $PUSHED; then
    echo "  Run: git push origin main"
  fi
  echo "  Run: git tag -a v$NEW_VERSION -m 'v$NEW_VERSION' && git push origin v$NEW_VERSION"
fi
