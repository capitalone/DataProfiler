.. _architecture:

Architecture & Design Overview
******************************

This section describes the design rationale, algorithmic choices, assumptions, testing strategy, and contribution process used in the DataProfiler library.

Overview
--------

DataProfiler computes numeric statistics (e.g., mean, variance, skewness, kurtosis) using **streaming algorithms** that allow efficient, incremental updates without recomputing from raw data. Approximate quantile metrics like the median are calculated using histogram-based estimation, making the system scalable for large or streaming datasets.

Additionally, DataProfiler uses a **Convolutional Neural Network (CNN)** to detect and label entities (e.g., names, emails, credit cards) in unstructured text. This supports critical tasks such as **PII detection**, **schema inference**, and **data quality analysis** across structured and unstructured data.

Algorithm Rationale
-------------------

The algorithms used are designed for **speed, scalability, and flexibility**:

- **Streaming numeric methods** (e.g., Welford's algorithm, moment-based metrics, histogram binning) efficiently summarize data without full recomputation.
- **CNNs for entity detection** are fast, high-throughput, and well-suited for sequence labeling tasks in production environments.

These choices align with the tool's goal of delivering fast, accurate data profiling with minimal configuration.

Assumptions & Limitations
-------------------------

- **Consistent formatting** of sensitive entities is assumed (e.g., standardized credit card or SSN formats).
- **Overlapping entity types** (e.g., phone vs. SSN) may lead to misclassification without context.
- **Synthetic training data** may not fully capture real-world diversity, reducing model accuracy on natural or unstructured text.
- **Quantile estimation** (e.g., median) is approximate and based on binning rather than exact sorting.

Testing & Validation
--------------------

- Comprehensive **unit testing** is performed across Python 3.9, 3.10, and 3.11.
- Tests are executed on every pull request targeting `dev` or `main` branches.
- All pull requests require **two code reviewer approvals** before merging.
- Testing includes correctness, performance, and compatibility checks to ensure production readiness.

Versioning & Contributions
--------------------------

- Versioning and development are managed via **GitHub**.
- Future changes must follow the guidelines in `CONTRIBUTING.md`, including:
  - Forking the repo and branching from `dev` or an active feature branch.
  - Ensuring **80%+ unit test coverage** for all new functionality.
  - Opening a PR and securing **two approvals** prior to merging.
