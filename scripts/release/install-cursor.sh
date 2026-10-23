#!/bin/sh
# Install this fork's cursor-cli GitHub Release.
#
# The release workflow stamps @PRIME_VERSION@ and @CURSOR_CLI_VERSION@ and
# uploads the result as install.sh on the cursor-cli-<version> release.
# curl -fsSL https://github.com/rycerzes/prime-agent/releases/latest/download/install.sh | sh
set -eu

PRIME_VERSION="${PRIME_AGENT_VERSION_OVERRIDE:-@PRIME_VERSION@}"
CURSOR_CLI_VERSION="${CURSOR_CLI_VERSION_OVERRIDE:-@CURSOR_CLI_VERSION@}"
REPO="${PRIME_AGENT_GITHUB_REPO:-rycerzes/prime-agent}"

case "$PRIME_VERSION" in
  @PRIME_VERSION@)
    echo "install.sh was not stamped with the prime-agent version" >&2
    exit 1
    ;;
esac
case "$CURSOR_CLI_VERSION" in
  @CURSOR_CLI_VERSION@)
    echo "install.sh was not stamped with the cursor-cli version" >&2
    exit 1
    ;;
esac

tag="cursor-cli-${CURSOR_CLI_VERSION}"
base="https://github.com/${REPO}/releases/download/${tag}"

os=$(uname -s)
arch=$(uname -m)
case "${os}:${arch}" in
  Linux:x86_64|Linux:amd64) platform=linux-x64 ;;
  MINGW64_NT*:x86_64|MSYS_NT*:x86_64|CYGWIN_NT*:x86_64) platform=win32-x64 ;;
  *)
    echo "unsupported platform ${os} ${arch}; this release ships linux-x64 and win32-x64" >&2
    exit 1
    ;;
esac

asset="prime-agent-${PRIME_VERSION}-${platform}+cursor-cli-${CURSOR_CLI_VERSION}.tar.gz"
# '+' in a URL path is easy to misread as a space. Percent-encode it.
url_asset=$(printf '%s' "$asset" | sed 's/+/%2B/g')

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

echo "downloading ${asset}"
curl -fL --retry 3 --retry-delay 2 -o "$tmp/$asset" "${base}/${url_asset}"
curl -fL --retry 3 --retry-delay 2 -o "$tmp/SHA256SUMS" "${base}/SHA256SUMS"

line=$(grep "  ${asset}\$" "$tmp/SHA256SUMS" || true)
[ -n "$line" ] || {
  echo "SHA256SUMS has no line for ${asset}" >&2
  exit 1
}
printf '%s\n' "$line" > "$tmp/SHA256SUMS.check"
if command -v sha256sum >/dev/null 2>&1; then
  (cd "$tmp" && sha256sum -c SHA256SUMS.check)
elif command -v shasum >/dev/null 2>&1; then
  (cd "$tmp" && shasum -a 256 -c SHA256SUMS.check)
else
  echo "sha256sum or shasum is required to verify the download" >&2
  exit 1
fi

tar -xOf "$tmp/$asset" install.sh > "$tmp/install-rust.sh"
sh "$tmp/install-rust.sh" --archive "$tmp/$asset"
