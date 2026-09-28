#!/usr/bin/env bash
# Fail when private or experiment data is tracked by Git.
# Rule source: .trellis/tasks/09-28-ferrots-implementation/prd.md (X1).
set -euo pipefail

status=0

tracked_data=$(git ls-files | grep -E '^(data|runs)/' || true)
if [ -n "$tracked_data" ]; then
  echo "error: tracked files under data/ or runs/:" >&2
  echo "$tracked_data" >&2
  status=1
fi

while IFS= read -r f; do
  [ -f "$f" ] || continue
  header=$(head -n 1 "$f")
  case "$header" in
    *窑头煤反馈* | *窑喂料反馈*)
      echo "error: fixture has a cement data header: $f" >&2
      status=1
      ;;
  esac
done < <(git ls-files 'tests/fixtures/*.csv' 'tests/fixtures/**/*.csv')

if [ "$status" -eq 0 ]; then
  echo "ok: no private data tracked"
fi
exit "$status"
