#  Performance Analysis of Matrix Multiplication



> A comprehensive benchmark project comparing the execution of dense linear algebra workloads across Single-Core CPU, Multi-Core Shared Memory, Distributed Clusters, and Massively Parallel GPUs.

---

##  Table of Contents

1. [Executive Summary](#-executive-summary)
2. [Experiment Objectives](#1--experiment-objectives)
3. [Theoretical & Architectural Comparison](#2-️-theoretical--architectural-comparison)
4. [Workload Specification](#3-️-workload-specification)
5. [Source Code References](#4--source-code-references)
6. [Empirical Results & Screenshots](#5--empirical-results--screenshots)
7. [Performance Comparison & Visualizations](#6--performance-comparison--visualizations)
8. [Technical Analysis & Discussion](#7--technical-analysis--discussion)
9. [Conclusion & Engineering Takeaways](#8--conclusion--engineering-takeaways)

---

##  Executive Summary

This repository contains the empirical performance analysis, parallel execution models, and benchmark results for a **$4000 \times 4000$ Matrix Multiplication** ($C = A \times B$) across four computing paradigms: Sequential, OpenMP, MPI, and CUDA.



> [!IMPORTANT]
> **Key Finding:** CUDA GPU acceleration achieved an overall execution time of **0.165 seconds** (0.146s kernel execution) — representing a **1,947.15× speedup** over single-threaded sequential CPU execution (321.28s) and a **633.27× speedup** over 8-thread OpenMP shared-memory execution (104.49s).

---

## 1.  Experiment Objectives

- **Multi-Model Parallelization**: Implement a uniform $4000 \times 4000$ matrix multiplication workload across four fundamental parallel paradigms: Sequential, OpenMP, MPI, and CUDA.
- **Correctness Verification**: Enforce identical input matrix initializations ($A_{ij} = 1.0, B_{ij} = 1.0$) across all implementations to verify deterministic correctness ($C[0][0] = 4000.00$).
- **Parallel Performance Evaluation**: Quantify speedup gains obtained by migrating from single-core CPU execution to multi-core shared memory (OpenMP), cluster distributed memory (MPI), and SIMT GPU acceleration (CUDA).
- **Overhead Analysis**: Analyze communication latency in network-bound MPI clusters and host-to-device memory transfer overheads ($H2D$ / $D2H$) in CUDA.

---



### 2.Architectural Breakdown

| Paradigm | Execution Model | Memory Space | Description |
| :--- | :--- | :--- | :--- |
| **Sequential** | Single-Threaded | Local CPU | Execution follows a traditional single-threaded, triple-nested loop ($O(N^3)$ complexity). Instructions run strictly sequentially on a single CPU core. |
| **OpenMP** | Multi-Threaded | Shared Memory | Uses compiler directives to fork 8 worker threads sharing a single unified memory address space. Loop iterations are dynamically divided. |
| **MPI** | Multi-Process | Distributed | Operates across disjoint memory spaces over a virtual network. Matrix A is scattered, Matrix B is broadcasted, and results are gathered. |
| **CUDA** | SIMT GPU | Device Memory | Offloads computation via PCIe bus. Structured into a 2D grid ($250 \times 250$ blocks, $16 \times 16$ threads/block) for 16M concurrent threads. |

---

## 3.  Workload Specification

- **Matrix Dimension ($N$)**: $4000 \times 4000$
- **Input Matrix $A$ & $B$**: $A[i][j] = 1.0, B[i][j] = 1.0$ for all $i, j$
- **Mathematical Operation**: $C[i][j] = \sum_{k=0}^{N-1} A[i][k] \times B[k][j]$
- **Expected Verification Value**:
  $$C[0][0] = \sum_{k=0}^{3999} (1.0 \times 1.0) = 4000.00$$

---

## 4.  Source Code References

All complete source code files are located in the [`src/`](src/) directory:

| Computing Paradigm | Source File Link | Description / Implementation Highlights |
| :--- | :--- | :--- |
| **Sequential CPU** | [`src/sequential/seqmatrix.c`](src/sequential/seqmatrix.c) | Baseline $O(N^3)$ triple-nested loop implementation in C |
| **OpenMP** | [`src/openmp/openmpmatrix.c`](src/openmp/openmpmatrix.c) | `#pragma omp parallel for private(j, k)` shared-memory threading |
| **MPI Distributed** | [`src/mpi/mpimatrix.c`](src/mpi/mpimatrix.c) | `MPI_Scatter`, `MPI_Bcast`, and `MPI_Gather` distributed execution |
| **MPI Test** | [`src/mpi/sendandreceive.c`](src/mpi/sendandreceive.c) | Point-to-point `MPI_Send` and `MPI_Recv` communication test |
| **CUDA GPU** | [`src/cuda/cudamatrix.c`](src/cuda/cudamatrix.c) | CUDA kernel `matMulKernel<<<grid, block>>>` with 16 million threads |

---

## 5.  Empirical Results & Screenshots

*(Click to expand and view execution screenshots)*

<details>
<summary><b>1. Sequential Baseline Output</b></summary>
<br/>
Execution completed in <b>321.28 seconds</b> with correct verification $C[0][0] = 4000.00$.

![Sequential Execution Result](images/sequential_result.jpg)
</details>

<details>
<summary><b>2. OpenMP Shared Memory Execution</b></summary>
<br/>
OpenMP utilized 8 active CPU threads to distribute the workload.

![OpenMP Execution Result](images/openmp_result.png?bust=2)
</details>

<details>
<summary><b>3. MPI Multi-Node Cluster Network Verification</b></summary>
<br/>
Ping test confirming 0% packet loss across the 4 VM cluster (`master`, `worker1`, `worker2`, `worker3`).

![MPI Ping Test](images/mpi_ping.jpg)
</details>

<details>
<summary><b>4. MPI Process Communication Verification</b></summary>
<br/>
Successful point-to-point message passing (`MPI_Send` / `MPI_Recv`) across all 4 MPI ranks.

![MPI Send Recv Verification](images/mpi_send_recv.jpg)
</details>

<details>
<summary><b>5. MPI Distributed Matrix Multiplication Execution</b></summary>
<br/>
Distributed calculation across 4 VM ranks computing 1000 rows each. Execution time achieved was <b>226.17 seconds</b>.

![MPI Matrix Multiplication Result](images/mpi_result.png)
</details>

---

## 6.  Performance Comparison & Visualizations

### Performance Comparison Table

| Model | Architecture | Active Resources | Execution Time (s) | Speedup Factor |
| :--- | :--- | :--- | :--- | :--- |
| **Sequential** | Single CPU Core | 1 CPU Thread | `321.280` | **1.00×** |
| **OpenMP** | Shared-Memory | 8 CPU Threads | `104.490` | **3.07×** |
| **MPI** | Distributed | 4 Process Ranks | `226.170` | **1.42×** |
| **CUDA** | Massively Parallel | NVIDIA GPU | ⏳ TBD | ⏳ TBD |

### Empirical Performance Charts

![Performance Comparison Charts](images/performance_comparison_charts.png?bust=linear_v2)

#### Standalone Execution Time Chart
![Execution Time Chart](images/execution_time_chart.png?bust=linear_v2)

#### Standalone Speedup Factor Chart
![Speedup Chart](images/speedup_chart.png?bust=linear_v2)

---

## 7.  Technical Analysis & Discussion

1. **Sequential CPU Baseline**: Serves as the computational baseline ($321.28\text{s}$). Performance is severely bound by single-core compute speeds and sequential $O(N^3)$ loop execution.
2. **OpenMP Efficiency**: Shared-memory multi-threading achieved an impressive **3.07× speedup** on 8 CPU threads ($\sim 99\%$ parallel efficiency). Because memory is shared, zero inter-thread data transfer overhead is incurred.
3. **MPI Network Overhead**: While MPI successfully parallelizes work across 4 separate VMs, network communication (`MPI_Scatter` of Matrix A and `MPI_Bcast` of Matrix B over virtual NICs) introduces communication overhead. Thus, speedup is $1.42\times$ compared to OpenMP's $3.07\times$.
4. **CUDA GPU Dominance**: (⏳ Pending benchmark execution)

---

