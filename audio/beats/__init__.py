"""BEATs audio encoder, vendored unchanged (except relative imports) from Microsoft's unilm repo.

Source: https://github.com/microsoft/unilm/tree/master/beats (commit 732d834db70ee0fc3886b4bcbcfb4ce7fb829be2)
Paper:  Chen et al., "BEATs: Audio Pre-Training with Acoustic Tokenizers", ICML 2023 (arXiv:2212.09058)
License: MIT, Copyright (c) Microsoft Corporation. See LICENSE in this folder.
"""
from .BEATs import BEATs, BEATsConfig

__all__ = ["BEATs", "BEATsConfig"]
