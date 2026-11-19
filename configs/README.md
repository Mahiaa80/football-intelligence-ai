# Configuration Files

This directory contains configuration files
for the Football Intelligence AI pipeline.

## Configuration Modules

| File | Purpose |
|---|---|
| data_config.yaml | Dataset and preprocessing settings |
| vision_config.yaml | Computer vision settings |
| graph_config.yaml | Dynamic graph construction settings |
| model_config.yaml | Deep learning model settings |

## Design Principles

- Keep configuration separate from source code.
- Use YAML for experiment settings.
- Make experiments reproducible.
- Avoid hardcoding model parameters.
