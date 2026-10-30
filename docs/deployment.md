# Deployment: churn API on AWS (ECR + EC2)

The scoring API (Stage 6) runs as a Docker container on a single EC2 instance, pulled from a private ECR repository.
This is a demonstration deployment: one small instance, plain HTTP, reachable from one IP address only.

Placeholders used in this document: `<ACCOUNT_ID>`, `<REGION>`, `<OWNER>/<REPO>`, `<YOUR_IP>`, `<INSTANCE_PUBLIC_IP>`,
`<GIT_SHA>`. Nothing account-specific is committed.

## Architecture

```
 GitHub Actions (preferred)                 AWS  <REGION>
 --------------------------                 ---------------------------------------------------------
 push to main -> pytest                     ECR repository  bank-churn-api
              -> docker build                 immutable tags, scan on push, keep last 3 images
              -> container smoke test              |  image pull (instance role: ECR read-only)
              -> push (OIDC role) ------------>    v
                                          Security group --> EC2 instance (Amazon Linux 2023)
 your machine                                 (TCP 8000 only,   docker run  bank-churn-api:<GIT_SHA>
 ------------                                  from <YOUR_IP>/32) --restart unless-stopped, memory cap,
 deploy.sh --use-ecr-image  (creates the         read-only, no capabilities, non-root
 role, security group and instance)           Session Manager <-- IAM role (SSM core): no SSH, no port 22
 browser / smoke test --- HTTP :8000 --------^
```

| Component | Choice |
|---|---|
| Region | `ap-southeast-2` (Sydney). The account's AWS Organization denies EC2/ECR in every other region probed, including `ap-south-1`. |
| Image build | Preferred: GitHub Actions (tests, build, smoke test, push). Fallback: build and push from the developer machine. |
| Registry | ECR private repo `bank-churn-api`; image tagged with the 7-character git SHA (no `latest`); immutable tags |
| Compute | One `t3.micro` (x86_64, 2 vCPU, 1 GiB), Amazon Linux 2023, 8 GiB encrypted gp3 disk |
| Access | Session Manager through the instance role; **no SSH key, no port 22** |
| Network | Default VPC, public IPv4, security group allows **TCP 8000 only, from one /32** |
| Not used | Load balancer, NAT gateway, Elastic IP, CloudWatch logs, TLS |

## Getting the image into ECR

### Preferred: GitHub Actions with OIDC (designed, **blocked by the account's organization guardrails**)

Building in CI is better than building on a laptop: tests gate the image, the build is reproducible, and the push runs
over GitHub's network straight to AWS (a home connection to the region proved too slow, see the fallback below).

**No long-lived AWS keys are stored in GitHub.** The workflow asks GitHub for a short-lived OIDC token and exchanges it
for temporary AWS credentials by assuming an IAM role. That needs an IAM OIDC identity provider for
`token.actions.githubusercontent.com` in the AWS account.

> **Status: not created.** The account belongs to an AWS Organization whose service control policy explicitly denies the
> IAM identity-provider APIs (`iam:ListOpenIDConnectProviders`, `iam:GetOpenIDConnectProvider`, `iam:ListSAMLProviders`
> returned explicit SCP denials). Creating the provider is expected to be denied too, so it was not attempted. Until an
> Organization administrator allows it (or creates the provider once), the CI push job stays inactive and the fallback is used.

**What to ask the Organization administrator:** allow `iam:CreateOpenIDConnectProvider`, `iam:GetOpenIDConnectProvider`,
`iam:ListOpenIDConnectProviders` and `iam:TagOpenIDConnectProvider` for this account, or create the provider themselves with
URL `https://token.actions.githubusercontent.com` and audience (client ID) `sts.amazonaws.com`.

**IAM role `bank-churn-api-github-actions`: trust policy.** Only workflow runs on the `main` branch of this repository can assume it.
Pull-request runs and other branches present a different `sub` claim and are refused.

```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": { "Federated": "arn:aws:iam::<ACCOUNT_ID>:oidc-provider/token.actions.githubusercontent.com" },
    "Action": "sts:AssumeRoleWithWebIdentity",
    "Condition": {
      "StringEquals": {
        "token.actions.githubusercontent.com:aud": "sts.amazonaws.com",
        "token.actions.githubusercontent.com:sub": "repo:<OWNER>/<REPO>:ref:refs/heads/main"
      }
    }
  }]
}
```

**Permissions policy: push to one repository, nothing else.** No delete, no repository settings, no other service.

```json
{
  "Version": "2012-10-17",
  "Statement": [
    { "Sid": "AuthToken", "Effect": "Allow", "Action": "ecr:GetAuthorizationToken", "Resource": "*" },
    { "Sid": "PushOnlyToThisRepository", "Effect": "Allow",
      "Action": [
        "ecr:BatchCheckLayerAvailability", "ecr:InitiateLayerUpload", "ecr:UploadLayerPart", "ecr:CompleteLayerUpload",
        "ecr:PutImage", "ecr:BatchGetImage", "ecr:GetDownloadUrlForLayer",
        "ecr:DescribeImages", "ecr:DescribeImageScanFindings"
      ],
      "Resource": "arn:aws:ecr:<REGION>:<ACCOUNT_ID>:repository/bank-churn-api" }
  ]
}
```

(`ecr:GetAuthorizationToken` cannot be scoped to a repository; AWS requires `Resource: "*"` for it.)

**To activate once the provider is allowed:**

1. Create the OIDC provider (if the administrator has not) and the role with the two policies above.
2. In the GitHub repository, add two **variables** (Settings, Secrets and variables, Actions, Variables), not secrets:
   `AWS_ROLE_ARN` = the role's ARN, and `ECR_PUSH_ENABLED` = `true`.
3. Re-run the workflow (or push to `main`). The `push` job now runs after `image` succeeds.
4. Deploy the image CI pushed: `scripts/aws/deploy.sh --use-ecr-image`.

**Hardening to apply when activating:** protect the `main` branch (anyone who can push to `main` can publish an image);
pin third-party actions in the workflow to commit SHAs instead of version tags; consider also constraining the role to this
workflow file with the `job_workflow_ref` claim if the account's IAM supports it for the GitHub provider.

### In use: GitHub Actions with a push-only IAM user key (because OIDC is blocked)

OIDC is the preferred design, but the organization's SCP denies the identity-provider APIs, so the CI push job authenticates with a
long-lived access key instead. This is a deliberate trade-off: a stored key can leak and does not expire, so it is limited as far as IAM allows.

- **IAM user `bank-churn-ci-pusher`**: no console access, tagged `project=bank-churn-retention`, one inline policy
  (`ecr-push-bank-churn-api-only`) identical to the permissions policy above: `ecr:GetAuthorizationToken` plus the push actions on the
  `bank-churn-api` repository in `<REGION>` only. It cannot delete images, change repository settings or touch any other service.
- **The access key is created by the account owner in the IAM console**, never by Claude or a script, and goes straight into GitHub
  secrets. Nothing key-like is committed or logged.
- **GitHub settings:** secrets `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`; variable `ECR_PUSH_ENABLED` = `true`. The push job
  also requires the `main` branch, and without the variable only `test` and `image` run.
- **Residual risk:** anyone who can push to `main` (or edit workflows) can use the key to push an image to this one repository.
  Protect `main`, and rotate the key after use or deactivate it once the image is in ECR. Tags are immutable, so an existing tag cannot be overwritten.
- **Switch to OIDC** when the Organization administrator allows it (steps above), then delete the user's key and the user.

### Fallback: build and push from the developer machine

```bash
scripts/aws/deploy.sh                    # builds, pushes to ECR, then deploys
```

The script first runs a **network preflight**: it times a 1 MB upload to ECR and stops if that takes more than 10 seconds
(under 100 KB/s), because the image is about 300 MB compressed and a slow path makes `docker push` fail after 5 to 15 minutes
per attempt. On this project's home connection, uploads to the Sydney region ran at roughly 4 KB/s (the same machine uploads
to a CDN at about 440 KB/s), so the push could not succeed; switching network (for example a phone hotspot) is the practical fix.
`SKIP_PREFLIGHT=1` overrides the check.

The Dockerfile installs dependencies in six separate layers (largest about 150 MB) rather than one 557 MB layer, so
uploads are smaller and a retry only repeats the layer that failed.

## What CI does today

`.github/workflows/build-push.yml` runs on every push to `main` (except changes limited to `docs/`, `notebooks/`, `reports/` and
Markdown) and on manual trigger:

1. **`test`**: installs `requirements-ci.txt` (the serving pins plus test tools) and runs `pytest`. Tests that need `data/`
   (not tracked in git) skip themselves; the Spark parity checks are run locally.
2. **`image`**: builds the Docker image, runs it with the production hardening flags (read-only filesystem, all capabilities
   dropped, `no-new-privileges`, 512 MB memory cap), waits for it to answer, runs `scripts/smoke_test.py` against the container,
   and checks it runs as non-root with a healthy HEALTHCHECK.
3. **`push`** (inactive until enabled): runs only when the repository variable `ECR_PUSH_ENABLED` is `true`. It authenticates with the
   `bank-churn-ci-pusher` key from GitHub secrets (OIDC role once allowed), skips the
   push if the immutable tag already exists, pushes the image that passed the smoke test, waits for the ECR scan, and writes the
   compressed size and finding counts to the job summary.

## Prerequisites

- An IAM user or role (not the root user) allowed to use ECR, EC2, IAM and SSM read access in `<REGION>`.
- AWS CLI v2; for the fallback path, Docker running locally. Run the scripts from a bash shell (Git Bash on Windows).
- For `--use-ecr-image`: an image tagged `<GIT_SHA>` already in ECR.
- A committed working tree for the files that go into the image when building locally (the script refuses to tag a dirty tree).

## Deploy

```bash
scripts/aws/deploy.sh                          # build locally and push (fallback path), then deploy
scripts/aws/deploy.sh --use-ecr-image [TAG]    # deploy an image already in ECR; TAG defaults to `git rev-parse --short=7 HEAD`
```

The script is safe to re-run; existing resources are reused. In order, it:

1. Checks you are not the root user and determines the image tag.
2. **Local path only:** creates the ECR repository (immutable tags, scan on push) and a lifecycle policy that keeps the last 3 images,
   runs the network preflight, builds the image, pushes it (retrying on timeouts), waits for the scan, and prints the compressed size
   and finding counts. **With `--use-ecr-image`:** verifies the repository and tag exist (listing the tags that do exist if not) and
   prints the size and findings; nothing is built or pushed.
3. Creates the IAM instance role and profile with `AmazonEC2ContainerRegistryReadOnly` and `AmazonSSMManagedInstanceCore`.
4. Creates the security group with one rule: TCP 8000 from your current public IP as a /32.
5. Looks up the latest Amazon Linux 2023 AMI from the public SSM parameter and launches the instance with IMDSv2 required.
6. Waits until `/health` answers, then prints the `/health` and `/docs` URLs.

The instance's user-data (`scripts/aws/user_data.sh.tpl`) installs Docker, logs in to ECR with the instance role, pulls the image and runs:

```
docker run -d --restart unless-stopped -p 8000:8000 \
  --memory 512m --memory-swap 512m \
  --read-only --cap-drop ALL --security-opt no-new-privileges \
  -e XGB_NTHREADS="$(nproc)" <ACCOUNT_ID>.dkr.ecr.<REGION>.amazonaws.com/bank-churn-api:<GIT_SHA>
```

`XGB_NTHREADS` is set to the instance's vCPU count because Python inside a container reports the host's CPUs.

### Verify

```bash
python scripts/smoke_test.py --base-url http://<INSTANCE_PUBLIC_IP>:8000
python scripts/parity_check.py http://<INSTANCE_PUBLIC_IP>:8000 http://127.0.0.1:8001   # vs a local API on the same 50 test rows
scripts/aws/status.sh                                                                     # state, health, console output
```

`status.sh` reads the instance's console output (lines starting `BANKCHURN:` mark user-data progress), so no SSH or command
access is needed. Console output lags the instance by a few minutes. Interactive access, if needed, is through Session Manager
(`aws ssm start-session --target <INSTANCE_ID>`), which requires `ssm:StartSession` on your IAM user.

### If your IP changes

The security group only admits the IP it was created with. Run `scripts/aws/update_ip.sh` to replace the rule with your current IP.

## Security choices

- **No long-lived AWS keys in CI** (designed, preferred, blocked): GitHub OIDC with a role limited to the `main` branch and to pushing one repository.
  **In use instead:** a push-only IAM user key stored as GitHub secrets (see above), the least-privilege option while OIDC is denied.
- **No SSH, no port 22.** Access is through Session Manager; the instance role has `AmazonSSMManagedInstanceCore`.
- **One /32 on one port.** Only the deployer's current IP can reach the API. There is no public exposure.
- **Least-privilege instance role:** ECR read-only plus SSM core. No broad policies are attached to the instance.
- **IMDSv2 required with a hop limit of 1**, so a process inside the container cannot reach instance metadata or the role's credentials.
- **Container hardening:** non-root user (uid 10001), read-only root filesystem, all Linux capabilities dropped, `no-new-privileges`,
  512 MB memory cap. These flags are tested locally and in CI before anything is deployed.
- **Immutable ECR tags** with scan on push: a tag can never be overwritten, and each push is scanned for known vulnerabilities.
- **Encrypted disk** and a private registry.
- **No secrets in the image, the scripts or the workflow.** The image holds only the model and code. AWS identifiers are placeholders in
  committed files; the role ARN lives in a GitHub variable, and the scripts mask account IDs and ARNs in their output.
- **Logs never contain request values:** the API logs only request id, path, status, latency and model version.
- **Known limits:** traffic is plain HTTP (no TLS), acceptable for a demo that scores synthetic data from one IP but not for real
  customer data. The scan covers OS packages in the base image, not application logic.

## ECR scan findings (image `4228fe6`)

The ECR scan of the deployed image reported 3 HIGH, 9 MEDIUM and 3 LOW findings. **All 3 HIGH findings are Debian OS packages inherited from the
`python:3.11-slim` base image (Debian 13). None comes from the Python dependencies we pin in `requirements-serve.txt`.** ECR listed no fixed
version for any of them at scan time, so rebuilding does not remove them yet.

| CVE | Package | What it is |
|---|---|---|
| CVE-2026-85091 | zlib 1.3.dfsg+really1.3.1-1 | Heap buffer overflow in `gz_vacate()` when `gzprintf()`/`gzvprintf()` is called after a stalled non-blocking `gzwrite()` |
| CVE-2026-5435 | glibc 2.41-12+deb13u4 | Out-of-bounds write in the deprecated resolver print functions `ns_printrrf`, `ns_printrr`, `fp_nquery` (TSIG records) |
| CVE-2026-19499 | glibc 2.41-12+deb13u4 | `strfmon`/`strfmon_l` can write past the output buffer with right-justified width padding |

**Assessment.** Each flaw needs the application to call a specific, rarely used function. The API is Python (FastAPI, scikit-learn, XGBoost) and
does not intentionally call any of them, so exposure looks low, but the code paths in the libraries were not audited. Risk is further reduced by the
demonstration setup: one /32 on one port, non-root user, read-only filesystem, all capabilities dropped, and synthetic data only.
**Action:** none now. Rebuild on a newer base image when Debian ships fixes, and re-check the scan; the image tag changes with each commit, so the
immutable-tag rule does not block this.

## Costs

Estimates, not live prices; check the AWS pricing pages and your Billing console. The EC2 instance type was chosen from the
free-tier-eligible list shown for the account.

| Item | Approximate cost |
|---|---|
| `t3.micro` instance | about $0.013 per hour |
| Public IPv4 address | $0.005 per hour |
| 8 GiB gp3 disk | under $1 per month |
| ECR storage (3 images, shared layers) | about $0.10 per month or less |
| Session Manager, IAM, security group, OIDC role | no charge |
| GitHub Actions (private repository) | free minutes apply; a run takes roughly 10 minutes |

That is roughly $0.02 per hour all in: about $0.45 for a day and $14 per month if left running and not covered by free-tier credits.
Data transfer out is small for a demo. Billing is per hour while the instance exists, so tear it down when finished.

## Tear down

```bash
scripts/aws/teardown.sh        # asks you to type 'delete'; add --yes to skip the prompt
```

It terminates the instance (the disk is deleted with it), deletes the security group, detaches policies and deletes the IAM role and
instance profile, and force-deletes the ECR repository with all its images. It finishes by printing a leftover check; every line should
be empty. Resources are found by their `project=bank-churn-retention` tag and fixed names, so no state file is needed.
The GitHub OIDC provider and role (once created) and the IAM user `bank-churn-ci-pusher` (delete its access key first, then the inline policy, then the user)
are not touched by this script and must be removed separately.

## Troubleshooting

- **`explicit deny in a service control policy`:** the organization blocks that action or region. Stop and ask the organization
  administrator; do not look for a workaround.
- **Network preflight fails or `docker push` times out:** the path to the region is too slow from this machine. Switch networks and
  re-run, or use CI (once OIDC is allowed). The script retries pushes, and layers already uploaded are skipped.
- **`image tag ... is not in ECR` with `--use-ecr-image`:** the tag printed in the error is the commit you are on; pass one of the
  listed tags explicitly, or wait for CI to finish pushing that commit.
- **API unreachable but the instance is running:** your IP may have changed (`update_ip.sh`), or user-data is still running (`status.sh`).
- **The public IP changed:** a stopped and restarted instance gets a new public IPv4 address, so the URL changes. There is no Elastic IP by design.
