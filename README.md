# DISE Airway Analysis System Deployment

This repository contains the DISE airway analysis system using Image Processing and Geometry Rule-based Classification.

## Render Deployment Configurations

To avoid Render memory limits or worker timeout issues during heavy video contour processing, please use the following startup command when configuring the service on Render:

```bash
gunicorn app:app --timeout 180 --workers 1
```

- **--timeout 180**: Extends worker lifetime to 3 minutes to handle longer video scans without getting killed.
- **--workers 1**: Limits concurrency to a single worker to avoid Out Of Memory (OOM) crashes on the Render Free tier (512 MB memory limit).
