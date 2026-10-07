# RunPod access (verified 2026-10-07)

The approved overflow box for Lane V/M (PLAN: "RunPod approved when the local GPU is busy > 2 h").

| what | value |
|---|---|
| pod | `arc-whest-twin`, id `acc7entl4kttqk`, RTX PRO 5000 Blackwell (54 GB VRAM), 20 vCPU, community cloud, **$0.82/hr while RUNNING** |
| image | `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` (torch 2.8.0+cu128, CUDA OK, `runpodctl` preinstalled) |
| storage | `/workspace`, 250 GB, persists across stop/start (holds an earlier arc-whest bundle); everything else is wiped on stop |
| SSH | proxy only (no public IP): `ssh -i ~/.ssh/arc3_devbox -o IdentitiesOnly=yes -tt acc7entl4kttqk-64410ffd@ssh.runpod.io` |
| files | `runpodctl send <file>` here (prints a one-time code), `runpodctl receive <code>` on the pod; ~1 MB/s via relay |

**Start / stop.** Use the RunPod MCP tools (`pod-action` start/stop on `acc7entl4kttqk`) or the
console. Local `runpodctl` (v2.14.0 in `~/.local/bin`, checksum-verified) has **no API key**
(`~/.runpod/config.toml` carries an empty `apikey`), so it can transfer files but not manage pods.
To give it full control, a human runs `runpodctl doctor` or sets `RUNPOD_API_KEY`.

**Keys.** The proxy authenticates the key registered on the account, `~/.ssh/arc3_devbox`. The
pod's `PUBLIC_KEY` env holds `~/.ssh/id_ed25519`, which the proxy **refuses**. Always pass
`-o IdentitiesOnly=yes -i ~/.ssh/arc3_devbox`.

**Running commands.** The proxy needs a PTY and ignores a command argument: pipe the commands
on stdin with `-tt` and end with `exit`:

    printf 'cd /workspace && nvidia-smi -L; exit\n' | ssh -i ~/.ssh/arc3_devbox -o IdentitiesOnly=yes \
        -tt acc7entl4kttqk-64410ffd@ssh.runpod.io

sshd answered on the first try ~60 s after `start`. Long jobs: launch under `nohup … &` with output
in `/workspace`, so they survive the SSH session ending.

**File transfer.** `runpodctl send` blocks until the receiver connects. Run it in the background,
capture the code from its first output line, then `runpodctl receive <code>` on the pod (in
`/workspace`). Each `send` mints a new code. Verified: a 2 MB file arrived with matching sha256. For bulk data
(hundreds of MB) the relay is slow; tar code + small labels only.

**Cost hygiene.** Stop the pod the moment a job's outputs are pulled back. Stopped pods still
bill for disk: this account has three EXITED pods with 250 GB volumes each (`arc-whest-twin`,
`biohub-p10`, `biohub-deliverables-retrain`) and a 175 GB network volume `arc3-devbox` (EU-RO-1).
Deleting any of them destroys data and is the owner's call.
