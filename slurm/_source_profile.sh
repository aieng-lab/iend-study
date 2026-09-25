# Source a profile after stripping CR (Windows rsync). Safe for `a=b&c` constraints.
source_profile() {
  local f="$1"
  if [[ ! -f "$f" ]]; then
    echo "Missing profile: $f" >&2
    return 1
  fi
  # shellcheck disable=SC1090
  source <(tr -d '\r' <"$f")
}
