"""Statistical modeling built on top of pipeline output.

Reads the tables the pipeline writes; nothing here runs as a pipeline step. Its
dependencies (scikit-learn, SHAP, ...) live in their own uv groups, so a plain
``uv sync`` does not have to pull them in.

    modeling.unsupervised   models that need no labels
"""
