# Tunnel-token remediation status

The runnable application and revised notebook no longer use ngrok or require a token. No secret is included in their current sources. .env files are excluded from Git.

An ngrok auth token was exposed in an earlier public notebook revision. Removing it from the current branch does not revoke it or erase cached/history copies. The token owner must revoke/rotate that token in the ngrok dashboard. That account action has not been performed by this maintenance pass.

Git history has not been force-rewritten. A coordinated history-cleanup operation may be considered after revocation, but it changes commit IDs and cannot retract existing clones or third-party caches. This repository does not claim that the historical exposure has been fully resolved.
