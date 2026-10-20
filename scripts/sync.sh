#!/usr/bin/env bash
# Two-step sync. 1) Every commit goes to your private git remote (a copy of all your work). 2) Only if the sanitize check passes does
# main go on to the public remote (GitHub). Uncommitted changes are not synced: commit first.
#   PRIVATE_REMOTE (default: gitserver)  PUBLIC_REMOTE (default: origin)  PRIVATE_AWAY (optional fallback URL when away from home)
cd "$(dirname "$0")/.."
PRIVATE_REMOTE=${PRIVATE_REMOTE:-gitserver}; PUBLIC_REMOTE=${PUBLIC_REMOTE:-origin}
[ -n "$(git status --porcelain)" ] && echo "note: there are uncommitted changes; only commits are synced."
if git remote get-url "$PRIVATE_REMOTE" >/dev/null 2>&1; then
  if git push "$PRIVATE_REMOTE" --all -q && git push "$PRIVATE_REMOTE" --tags -q; then echo "private copy: up to date ($PRIVATE_REMOTE)"
  elif [ -n "$PRIVATE_AWAY" ] && git push "$PRIVATE_AWAY" --all -q; then echo "private copy: up to date (away route)"
  else echo "private copy: could not reach $PRIVATE_REMOTE (not on your network?). Not publishing until the private copy is safe."; exit 2; fi
else echo "private copy: no remote called $PRIVATE_REMOTE, skipped"; fi
bash scripts/sanitize_check.sh || { echo "NOT published to $PUBLIC_REMOTE: fix the findings above, commit, and run this again."; exit 1; }
git push "$PUBLIC_REMOTE" main && echo "published: $PUBLIC_REMOTE is up to date"
