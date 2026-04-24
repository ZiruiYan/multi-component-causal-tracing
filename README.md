# Multi-component Causal Tracing in Large Language Models

This repository contains the code for ACL 2026 paper **Multi-component Causal Tracing in Large Language Models**.

The paper proposes a unified framework for causal tracing across multiple components of large language models, such as attention heads and MLP neurons. Unlike standard single-component causal tracing, this work studies how groups of components jointly affect target metrics such as gender bias, factual knowledge localization, and variable binding behavior.

## Overview

Causal tracing studies how interventions on internal model representations affect model behavior. Existing methods often analyze one component at a time, which can miss nonlinear interactions between components.

This project introduces **Penalized Gradient-Based Causal Tracing (PGB-CT)**, a scalable method for identifying sparse subsets of components that have high joint impact on a target metric.

The main idea is to replace the original combinatorial subset-selection problem with a continuous relaxation, apply soft interventions, and use a penalty function that encourages both sparsity and binary component selection.
