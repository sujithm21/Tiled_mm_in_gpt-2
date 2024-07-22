# Implementing Tiled Matrix Multiplication in GPT-2

This project demonstrates the implementation of tiled matrix multiplication in the GPT-2 model to enhance performance by improving cache efficiency and parallelism.

## Table of Contents

1. [Introduction](#introduction)
2. [Benefits of Tiled Matrix Multiplication](#benefits-of-tiled-matrix-multiplication)
3. [Steps for Implementation](#steps-for-implementation)
4. [Performance Improvements](#performance-improvements)
5. [Detailed Metrics Comparison](#detailed-metrics-comparison)
6. [Usage](#usage)
7. [Contributing](#contributing)
8. [License](#license)

## Introduction

Tiled matrix multiplication, also known as blocked matrix multiplication, is a technique to improve cache efficiency and parallelism by dividing larger matrices into smaller sub-matrices (tiles). These tiles fit better in the cache, reducing memory bandwidth usage and improving performance.

## Benefits of Tiled Matrix Multiplication

1. **Cache Utilization:** Smaller tiles fit into the cache, reducing cache misses.
2. **Parallelism:** Each tile can be processed independently, enabling better parallelization.
3. **Reduced Memory Bandwidth:** Frequent access to smaller tiles reduces the need to fetch large data from main memory.

## Steps for Implementation

1. **Define Tile Size:** Choose an appropriate tile size based on the hardware and matrix dimensions. A common tile size is 64.
2. **Modify Attention Mechanism:** Replace the standard matrix multiplication in the attention mechanism with tiled matrix multiplication.
3. **Integrate into GPT-2:** Replace the original attention layers in GPT-2 with custom layers that use the tiled matrix multiplication.

## Performance Improvements

By implementing tiled matrix multiplication, the following performance improvements can be observed:

1. **Increased Cache Efficiency:** Tiling ensures that data fits into the cache, reducing cache misses and improving speed.
    - Example: Tiled attention had lower I Cache Misses (6.7%) compared to normal attention (7.3%).
2. **Enhanced Parallelism:** Smaller, independent tiles can be processed in parallel, leading to better utilization of multi-core processors and GPUs.
    - Example: Tiled attention completed in 2.172 seconds with 6 threads, whereas normal attention took 10.069 seconds with 18 threads.
3. **Reduced Memory Bandwidth:** Frequent access to smaller tiles minimizes the need to fetch large data from memory, reducing memory bandwidth usage.
  - Example: Tiled attention had lower Memory Bound (13.5%) compared to normal attention (8.8%).

![Tiled attention](https://github.com/user-attachments/assets/73961297-c8eb-44a5-a28c-2069ff782990)


## Detailed Metrics Comparison

- **CPI Rate:**
    - Tiled Attention: 0.748
    - Normal Attention: 1.072
- **Elapsed Time:**
    - Tiled Attention: 2.172s
    - Normal Attention: 10.069s
- **Retiring:**
    - Tiled Attention: 33.5%
    - Normal Attention: 30.2%
- **Front-End Bound:**
    - Tiled Attention: 39.8%
    - Normal Attention: 34.6%
- **Back-End Bound:**
    - Tiled Attention: 28.4%
    - Normal Attention: 19.2%
- **Core Bound:**
    - Tiled Attention: 14.9%
    - Normal Attention: 10.4%

## Usage

To run the optimization analysis:

1. Clone the repository.
2. Install the required dependencies.
3. Run the matrix multiplication scripts and visualize the results.

```sh
git clone https://github.com/yourusername/matrix-multiplication-optimization.git
cd matrix-multiplication-optimization
pip install -r requirements.txt
python run_optimization.py
```

## Contributing

We welcome contributions to improve this project. Please fork the repository, create a new branch, and submit a pull request with your changes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Integrating tiled matrix multiplication into GPT-2 resulted in significant performance gains, particularly in speed and scalability. These optimizations collectively enhance the model's performance, making it more suitable for real-world applications where processing time and resource utilization are critical.
