"""Paths and keys. Everything is relative to the repo or comes from env vars; no machine-specific paths."""
import os
DATA=os.environ.get('RTCOACH_DATA','data')            # local cache of what lives on the Modal volume
QEVD_RAW=os.environ.get('QEVD_RAW')                  # ingest only: the extracted QEVD folder (Mac, one-time)
MMFIT_RAW=os.environ.get('MMFIT_RAW')                # ingest only: MM-Fit raw folder
VOLUME=os.environ.get('RTCOACH_VOLUME','rtcoach-qevd') # private Modal volume, the only home of QEVD-derived data
def key(name):
    v=os.environ.get(name)
    if not v: raise RuntimeError(f'{name} not set in the environment')
    return v
