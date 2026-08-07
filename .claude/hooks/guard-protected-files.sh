#!/usr/bin/env bash
# PreToolUse guard for arc-whest.
#
# Two classes of file must never be rewritten by an agent:
#
#   docs/HUMAN_MODEL.md          Samuel's own beliefs. An agent that "corrects"
#                                this file destroys the only record of whether the
#                                project's model of itself is any good. Agents may
#                                read it, quiz against it, and propose changes in
#                                chat -- never write it.
#
#   experiments/vault/manifest.json
#   experiments/vault/touch_log.jsonl
#                                The seal on the 500-network holdout. Its whole
#                                value is that there is no quiet way to run
#                                against it: the counter and the log must only
#                                ever be moved by local_sim/vault.py, which is a
#                                Bash call and is unaffected by this guard.
#
# Reads the hook payload on stdin; matches the file_path field only, so file
# *contents* that mention these names are not affected. No jq dependency.

set -u
payload=$(cat)

deny() {
    printf '{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"deny","permissionDecisionReason":"%s"}}\n' "$1"
    exit 0
}

# file_path value only, tolerant of Windows backslash escaping in the JSON.
path=$(printf '%s' "$payload" \
    | sed -n 's/.*"file_path"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p')

ask() {
    printf '{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"ask","permissionDecisionReason":"%s"}}\n' "$1"
    exit 0
}

# --- shell commands: the two irreversible actions in this project -------------
# Both were prose rules in CLAUDE.md. Prose does not stop a tool call.
cmd=$(printf '%s' "$payload" \
    | sed -n 's/.*"command"[[:space:]]*:[[:space:]]*"\(.*\)".*/\1/p')

case "$cmd" in
    *"whest submit"*)
        ask "SUBMISSION. This is outward-facing and cannot be undone: it consumes a daily submission slot and a failed MLP scores as zeros. Project rule: never submit without Samuel's explicit approval, and only after whest validate plus a clean mini run with n_failed_mlps=0. Confirm the file, the version, and that Samuel has approved this specific submission."
        ;;
    *"vault.py evaluate"*)
        ask "VAULT TOUCH. This spends one of three permitted evaluations of the 500-network sealed holdout and increments a counter that cannot be decremented. Permitted reasons: the baseline, one per estimator change, designation day. If this is to browse or to re-check a number already measured, cancel."
        ;;
esac

case "$path" in
    *HUMAN_MODEL.md)
        deny "docs/HUMAN_MODEL.md is Samuel-owned and write-protected by the project harness. Agents may read it, quiz against it, and PROPOSE corrections in chat -- but must not edit it. If a belief in it is contradicted by evidence, say so and cite the evidence; let Samuel make the edit."
        ;;
    *vault*manifest.json|*vault*touch_log.jsonl)
        deny "The vault seal is write-protected. manifest.json and touch_log.jsonl may only be modified by local_sim/vault.py, which increments the touch counter and appends a log line together. Hand-editing either one breaks the guarantee that the holdout baseline was not chosen by re-rolling. Permitted touches: baseline, one per estimator change, designation day."
        ;;
esac

exit 0
