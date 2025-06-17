# GitHub Actions Workflows

## Security

To limit our vulnerability to supply chain attacks on GitHub Actions we:

- Restrict Actions to those that we've written, or those published by GitHub or another verified developer
- [Pin versions](https://michaelheap.com/pin-your-github-actions/) to avoid unintentionally pulling in malicious code

### Pinning Versions

Run `npx pin-github-action .github/workflows` from the root of the repo to resolve the SHA for the given version tag of each Action.
This will ensure that we always use the exact version we expect, rather than whatever the latest commit for that tag is.

### Updating Pinned Versions

Dependabot should let us know when the Actions we're using have a new version available.

You can also run `npx pin-github-action .github/workflows` (from the root of the repo) to update the pinned SHAs for the given version tag of each action.