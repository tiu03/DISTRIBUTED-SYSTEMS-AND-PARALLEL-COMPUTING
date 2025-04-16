# DSPC
 DISTRIBUTED SYSTEMS AND PARALLEL COMPUTING
 Optimizing Bicubic Interpolation for Image Processing

 In this project, we investigated the performance of bicubic interpolation against nearest neighbor
interpolation. For simple images, bicubic interpolation produced smoother results than nearest neighbor.
For complex images, the difference was less noticeable, but bicubic still produced better results. Overall,
bicubic interpolation is a superior method for improving image quality during scaling tasks, making it the
better choice when preserving fine details and avoiding artifacts is essential.
We conducted a performance evaluation using three distinct images at five different resolutions (750,
1500, 3000, 6000, and 12000 pixels wide). The evaluation was performed across three different
implementations: serial (single-threaded), OpenMP (multi-threaded on the CPU), and CUDA (parallelized
on the GPU). We decided to repeat each test five times, allowing us to gather reliable data on execution
times.
● Serial: Slowest, especially with large images.
● OpenMP: Faster than serial, but performance gains decrease with larger images.
● CUDA: Fastest, especially with large images.
When analyzing performance gains, OpenMP delivered consistent but moderate improvements due to the
limitations of CPU-based parallelization. Since CPUs have far fewer cores than GPUs, OpenMP can
leverage multi-threading but struggles to scale efficiently for very large workloads. Additionally, the
overhead of managing threads and synchronizing data can limit performance gains, especially in tasks like
bicubic interpolation that require heavy parallelization.
On the other hand, CUDA presents significant performance boosts, especially as the image size increases.
This is because GPUs are designed to handle massive parallel tasks by processing thousands of threads
simultaneously. Meanwhile smaller images may not fully utilize the GPU's capabilities, larger images
benefit substantially from its high-throughput architecture. As image sizes grow, CUDA maintains
efficient performance, making it the best solution for large datasets and complex computations.
