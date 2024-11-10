#!/usr/bin/env bash
#
# Optimized and Parallelized Build and Push Docker Images using Docker Buildx.
# Supports optional caching and cleanup of Docker resources post-build.

set -Eeuo pipefail
trap cleanup SIGINT SIGTERM ERR EXIT

#######################################
# Constants and Environment Variables
#######################################
readonly REGISTRY="ghcr.io"
readonly REPOSITORY="aai540-group3/pipeline"
readonly MAX_JOBS=4
readonly DEFAULT_BUILDER_NAME="default"

#######################################
# Global Variables
#######################################
CLEANUP_AFTER_BUILD=false
ENABLE_CACHE=false
CUSTOM_BUILDER=""
PARAM=""
FLAG=0
VERBOSE=0
args=()

readonly IMAGES=(
  base
  infrastruct
  ingest
  preprocess
  explore
  featurize
  optimize
  autogluon
  logisticregression
  neuralnetwork
  evaluate
  register
  deploy
  serve
  monitor
)

#######################################
# Functions
#######################################

usage() {
  cat <<EOF
Usage: $(basename "${BASH_SOURCE[0]}") [OPTIONS]

Optimized and Parallelized Build and Push Docker Images using Docker Buildx.

Options:
  -h, --help             Print this help and exit
  -v, --verbose          Enable verbose output
  --enable-cache         Enable Docker build caching
  --cleanup-after-build  Clean up Docker resources after build
  -b, --builder NAME     Specify a custom Docker Buildx builder
EOF
  exit
}

cleanup() {
  trap - SIGINT SIGTERM ERR EXIT
  if [[ "${CLEANUP_AFTER_BUILD}" == "true" ]]; then
    msg "${YELLOW}Initiating additional cleanup steps...${NOFORMAT}"
    prune_buildx_cache
    prune_dangling_images
    prune_unused_builders
    prune_all_docker_resources
    msg "${GREEN}Additional cleanup steps completed.${NOFORMAT}"
  fi
}

setup_colors() {
  if [[ -t 2 ]] && [[ -z "${NO_COLOR-}" ]] && [[ "${TERM-}" != "dumb" ]]; then
    NOFORMAT='\033[0m'
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    ORANGE='\033[0;33m'
    BLUE='\033[0;34m'
    PURPLE='\033[0;35m'
    CYAN='\033[0;36m'
    YELLOW='\033[1;33m'
  else
    NOFORMAT='' RED='' GREEN='' ORANGE='' BLUE='' PURPLE='' CYAN='' YELLOW=''
  fi
}

msg() {
  echo >&2 -e "${1-}"
}

die() {
  local msg="$1"
  local code="${2-1}"
  msg "${RED}${msg}${NOFORMAT}"
  exit "${code}"
}

parse_params() {
  CLEANUP_AFTER_BUILD=false
  ENABLE_CACHE=false
  CUSTOM_BUILDER=""
  FLAG=0
  PARAM=""

  while :; do
    case "${1-}" in
    -h | --help)
      usage
      ;;
    -v | --verbose)
      VERBOSE=1
      set -x # Enables verbose output for debugging
      ;;
    --enable-cache)
      ENABLE_CACHE=true
      ;;
    --cleanup-after-build)
      CLEANUP_AFTER_BUILD=true
      ;;
    -b | --builder)
      CUSTOM_BUILDER="${2-}"
      shift
      ;;
    --no-color)
      NO_COLOR=1
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

  args=("$@")
}

authenticate() {
  if ! docker buildx imagetools inspect "${REGISTRY}/${REPOSITORY}-base:latest" >/dev/null 2>&1; then
    msg "${CYAN}Authenticating to ${REGISTRY}...${NOFORMAT}"
    echo "${REGISTRY_PASSWORD}" | docker login "${REGISTRY}" -u "${REGISTRY_USERNAME}" --password-stdin || die "Authentication failed"
    msg "${GREEN}Authentication successful.${NOFORMAT}"
  else
    msg "${GREEN}Already authenticated.${NOFORMAT}"
  fi
}

builder_exists() {
  docker buildx inspect "${BUILDER_NAME}" >/dev/null 2>&1
}

builder_running() {
  docker buildx inspect "${BUILDER_NAME}" | grep -q "Status: running"
}

create_builder() {
  msg "${ORANGE}Creating new builder '${BUILDER_NAME}'...${NOFORMAT}"
  docker buildx create --name "${BUILDER_NAME}" --use || die "Failed to create builder '${BUILDER_NAME}'"
}

bootstrap_builder() {
  msg "${ORANGE}Bootstrapping builder '${BUILDER_NAME}'...${NOFORMAT}"
  docker buildx inspect "${BUILDER_NAME}" --bootstrap || die "Failed to bootstrap builder '${BUILDER_NAME}'"
}

ensure_builder() {
  if ! builder_exists; then
    create_builder
    bootstrap_builder
  elif ! builder_running; then
    bootstrap_builder
  fi
  docker buildx use "${BUILDER_NAME}" || die "Failed to use builder '${BUILDER_NAME}'"
}

build_and_push_image() {
  local image="$1"
  local full_tag="${REGISTRY}/${REPOSITORY}-${image}:latest"

  msg "${BLUE}Building and pushing image: ${image} (${full_tag})${NOFORMAT}"

  local build_cmd=(
    docker buildx build
    --builder "${BUILDER_NAME}"
    --file "docker/Dockerfile.${image}"
    --tag "${full_tag}"
    --push
    .
  )

  if [[ "${ENABLE_CACHE}" == "true" ]]; then
    local cache_tag="${REGISTRY}/${REPOSITORY}-${image}-cache:latest"
    build_cmd+=(--cache-from "type=registry,ref=${cache_tag}")
    build_cmd+=(--cache-to "type=registry,ref=${cache_tag},mode=max")
    msg "${YELLOW}Caching is enabled for image: ${image}${NOFORMAT}"
  else
    msg "${YELLOW}Caching is disabled for image: ${image}${NOFORMAT}"
  fi

  "${build_cmd[@]}" || die "Failed to build and push image: ${image}"

  msg "${GREEN}Successfully built and pushed: ${full_tag}${NOFORMAT}"

  msg "${ORANGE}Removing local image: ${full_tag}${NOFORMAT}"
  docker image rm "${full_tag}" || msg "${RED}Warning: Failed to remove local image: ${full_tag}${NOFORMAT}"

}

process_image() {
  local image="$1"
  build_and_push_image "${image}"
}

run_in_parallel() {
  local max_jobs="$1"
  shift
  local cmds=("$@")
  local job_count=0

  for cmd in "${cmds[@]}"; do
    eval "${cmd}" &
    ((job_count++))

    if ((job_count >= max_jobs)); then
      wait -n
      ((job_count--))
    fi
  done

  wait # Wait for all remaining jobs
}

prune_buildx_cache() {
  msg "${CYAN}Pruning Docker Buildx cache...${NOFORMAT}"
  docker buildx prune --builder "${BUILDER_NAME}" --force || msg "${RED}Warning: Failed to prune Buildx cache for builder '${BUILDER_NAME}'${NOFORMAT}"
}

prune_dangling_images() {
  msg "${CYAN}Pruning dangling Docker images...${NOFORMAT}"
  docker image prune -f || msg "${RED}Warning: Failed to prune dangling Docker images${NOFORMAT}"
}

prune_unused_builders() {
  msg "${CYAN}Pruning unused Docker Buildx builders...${NOFORMAT}"
  docker buildx prune --all --force || msg "${RED}Warning: Failed to prune unused Docker Buildx builders${NOFORMAT}"
}

prune_all_docker_resources() {
  msg "${CYAN}Pruning all unused Docker resources...${NOFORMAT}"
  docker system prune -af || msg "${RED}Warning: Failed to prune all unused Docker resources${NOFORMAT}"
}

main() {
  parse_params "$@"
  setup_colors

  msg "${RED}Read parameters:${NOFORMAT}"
  msg "- Verbose: ${VERBOSE}"
  msg "- Enable Cache: ${ENABLE_CACHE}"
  msg "- Cleanup After Build: ${CLEANUP_AFTER_BUILD}"
  msg "- Custom Builder: ${CUSTOM_BUILDER}"
  msg "- Flag: ${FLAG}"
  msg "- Param: ${PARAM}"
  msg "- Arguments: ${args[*]-}"

  local builder_name="${CUSTOM_BUILDER:-${DEFAULT_BUILDER_NAME}}"
  BUILDER_NAME="${builder_name}"
  readonly BUILDER_NAME

  if [[ -z "${REGISTRY_USERNAME-}" ]]; then
    die "Required environment variable REGISTRY_USERNAME is not set."
  fi

  if [[ -z "${REGISTRY_PASSWORD-}" ]]; then
    die "Required environment variable REGISTRY_PASSWORD is not set."
  fi

  authenticate
  ensure_builder

  msg "${GREEN}Build Configuration:${NOFORMAT}"
  msg "- Builder: ${BUILDER_NAME}"
  msg "- Max Parallel Jobs: ${MAX_JOBS}"
  msg "- Caching Enabled: ${ENABLE_CACHE}"

  local build_commands=()
  for image in "${IMAGES[@]}"; do
    build_commands+=("process_image '${image}'")
  done

  run_in_parallel "${MAX_JOBS}" "${build_commands[@]}"

  msg "${GREEN}All Docker images have been processed successfully.${NOFORMAT}"

}

main "$@"
