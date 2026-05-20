# Keep Memory Provider — Hermes Plugin

This directory contains the Hermes memory plugin for Keep. It gets
copied into `plugins/memory/keep/` inside your Hermes Agent installation.

## Install

```bash
curl -sSL https://keep.generalbusiness.ai/scripts/install-hermes.sh | bash
```

This finds your Hermes installation, installs the plugin, and runs the setup wizard. Choose your embedding and summarization providers when prompted.

Or manually:

```bash
# Copy the plugin files
cp -r hermes-plugin /path/to/hermes-agent/plugins/memory/keep

# Run setup (installs keep-skill, configures providers)
hermes memory setup
```

New to Keep? https://docs.generalbusiness.ai/guides/hermes/
