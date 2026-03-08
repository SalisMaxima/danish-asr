# DTU HPC — Access and Login

## Who can access

All DTU students and employees automatically have access — use your DTU credentials (username + password). The system is case-sensitive.

Visitors: a DTU host must create a guest account at https://guest.dtu.dk → select "UNIX Databar".

---

## SSH access

### From on-campus or DTU VPN

Password authentication only — no key needed:

```bash
ssh userid@login.hpc.dtu.dk
# or equivalently:
ssh userid@login1.gbar.dtu.dk
ssh userid@login2.gbar.dtu.dk
```

### From external networks (off-campus, no VPN) — requires SSH key

Since August 24, 2022, external access requires **SSH key + passphrase + DTU password**.

**Step 1: Generate an Ed25519 key pair (on your local machine)**

```bash
cd ~/.ssh
ssh-keygen -t ed25519 -f gbar
# Enter a strong passphrase (different from your DTU password)
```

**Step 2: Deploy the public key to the cluster**

Must be done from DTU network/VPN, or via the transfer node:

```bash
ssh s123456@transfer.gbar.dtu.dk mkdir -m 700 -p .ssh
scp gbar.pub s123456@transfer.gbar.dtu.dk:.ssh/authorized_keys
ssh s123456@transfer.gbar.dtu.dk chmod 600 .ssh/authorized_keys
```

**Step 3: Connect using the key**

```bash
ssh -i ~/.ssh/gbar s123456@login.hpc.dtu.dk
# Prompted for: key passphrase, then DTU password
```

**Optional `~/.ssh/config` shortcut:**

```
Host gbar1
    User s123456
    IdentityFile ~/.ssh/gbar
    Hostname login1.gbar.dtu.dk
```

Then simply: `ssh gbar1`

**PuTTY (Windows):**

```bash
puttygen -t ed25519 -o gbar-putty -O private
puttygen gbar-putty -o gbar-openssh.key -O private-openssh-new
```

### Security

- Verify SSH fingerprints before first connection: https://www.hpc.dtu.dk/fp.txt
- Never upload private keys to repos or share them.
- **Passwordless keys are rejected** for external access — keys must have a passphrase.
- Multi-factor authentication is required for Cisco VPN.

---

## Login nodes

```
login1.gbar.dtu.dk
login2.gbar.dtu.dk
login1.hpc.dtu.dk
login2.hpc.dtu.dk
```

**CRITICAL: Do NOT run any applications on login nodes.** Login nodes are for job submission and file management only. Running code there violates HPC policy.

---

## Interactive compute access

After SSH login, use one of the following to get an interactive shell on a compute node:

```bash
linuxsh        # generic compute node, no GPU
voltash        # shared GPU node: 2× V100, 16 GB each
sxm2sh         # shared GPU node: 4× V100-SXM2 NVLink, 32 GB each
a100sh         # shared GPU node: 2× A100, 40 GB each
```

These nodes are **shared** — other users may be present. Check occupancy with `nvidia-smi` before starting intensive work.

For **exclusive** GPU access, use `bsub -Is` through the scheduler — see [05-interactive-monitoring.md](05-interactive-monitoring.md).

---

## ThinLinc (graphical desktop)

**Server:** `thinlinc.gbar.dtu.dk`

| Network | Authentication |
|---------|---------------|
| On-campus / VPN | Username + password |
| External (no VPN) | SSH key with passphrase (passwordless keys rejected) |

Default desktop: Xfce. To get an LSF application node from within ThinLinc: Applications Menu → DTU → xterm (LSF-application node).

**External ThinLinc config:** Options → Security → Authentication method: "public key" → select your private key file (`gbar`).

> The web-based ThinLinc interface is only accessible from the DTU network.

---

## VS Code Remote

Not officially supported. The last working version with DTU HPC is **VS Code 1.85** — newer versions fail due to glibc/libstdc++ incompatibility with Scientific Linux 7.9.

---

## Operating system

All compute nodes run **Scientific Linux 7.9**.

---

## File transfer

Use the dedicated transfer server — it has both 10 GbE and Infiniband, making it faster than the login nodes for bulk transfers:

```bash
# Upload
scp localfile s123456@transfer.gbar.dtu.dk:/work3/s123456/path/

# Bulk upload (resumable)
rsync -avzP localdir/ s123456@transfer.gbar.dtu.dk:/work3/s123456/dest/

# SFTP interactive session
sftp s123456@transfer.gbar.dtu.dk
```

SFTP clients (FileZilla, Cyberduck) also work — connect to `transfer.gbar.dtu.dk`.

---

## Key gotchas

| Gotcha | Details |
|--------|---------|
| Default walltime is 15 min | Always set `#BSUB -W hh:mm` or your job dies after 15 minutes |
| DOS/Windows line endings | Job scripts with `\r\n` endings fail silently — fix with `dos2unix job.sh` |
| Never modify `.bashrc` blindly | Keep another session open to revert if login breaks |
| Never set `LD_LIBRARY_PATH` in login profiles | Causes security risks, performance issues, subtle wrong results |
| Danish characters in filenames | Avoid æ, ø, å in filenames, job names, and paths |
| Account expiry | All access and data lost immediately on account closure — back up before expiry |
| Quota exceeded | Exceeding home quota (30 GB) kills running jobs and can prevent login |
