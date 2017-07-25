#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>

#include <my_timer.h>
#include <aligned_allocator.h>

#define NDIM (3)

#ifndef __RESTRICT
#  define __RESTRICT
#endif

#ifdef USE_FLOAT
typedef float ValueType;
#else
typedef double ValueType;
#endif

#define Enable_ArrayOfStructures
#if defined(Enable_ArrayOfStructures) || defined(__AOS)
#  ifndef Enable_ArrayOfStructures
#    define Enable_ArrayOfStructures
#  endif
   /* Array-of-structures (like) format. */
#  define _index(i, j) (NDIM * (i) + (j))
#else
   /* Structure-of-arrays (like) format. */
#  define _index(i, j) ((i) + (j) * n)
#endif

#define h_cent_arr(i, j) h_centroids[_index((i), (j))]
#define h_part_arr(i, j) h_particles[_index((i), (j))]
#define h_dist_arr(i, j) h_distance[i + j * n]

#define d_cent_arr(i, j) d_centroids[_index((i), (j))]
#define d_part_arr(i, j) d_particles[_index((i), (j))]
#define d_dist_arr(i, j) d_distance[i + j * n]

#define blockSize (1024)

/* Generate a random double between 0 and 1. */
ValueType frand(void) { return ((ValueType) rand()) / RAND_MAX; }

void distance(ValueType * __RESTRICT h_distance, ValueType * __RESTRICT h_centroids, ValueType * __RESTRICT h_particles, const int n, const int k)
{
    for (int i = 0; i < n; ++i)
    {
        ValueType x1 = h_part_arr(i, 0);
        ValueType y1 = h_part_arr(i, 1);
        ValueType z1 = h_part_arr(i, 2);
        for (int j = 0; j < k; ++j)
        {
            ValueType x2 = h_cent_arr(j, 0);
            ValueType y2 = h_cent_arr(j, 1);
            ValueType z2 = h_cent_arr(j, 2);
            h_dist_arr(i, j) = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) + (z2 - z1) * (z2 - z1);
        }
    }
}

void assign(ValueType * __RESTRICT h_distance, ValueType * __RESTRICT h_centroids, ValueType * __RESTRICT h_particles, int * __RESTRICT h_cluster, const int n, const int k)
{
    ValueType min_dist = 12;    

    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < k; ++j)
        {
            if (h_dist_arr(i, j) < min_dist)
            {
                min_dist = h_dist_arr(i, j);
                h_cluster[i] = j;
            }
        }
    }
}

void findMean(ValueType * __RESTRICT h_centroids, ValueType * __RESTRICT h_particles, int * __RESTRICT h_cluster, int * __RESTRICT h_num_part, const int n, const int k)
{
    for (int i = 0; i < k; ++i)
    {
        h_cent_arr(i, 0) = 0;
        h_cent_arr(i, 1) = 0;
        h_cent_arr(i, 2) = 0;

        h_num_part[i] = 0;

        for (int j = 0; j < n; ++j)
        {
            if (h_cluster[j] == i)
            {
                h_cent_arr(i, 0) += h_part_arr(j, 0);
                h_cent_arr(i, 1) += h_part_arr(j, 1);
                h_cent_arr(i, 2) += h_part_arr(j, 2);
                h_num_part[i]++;
            }
        }

        if (h_num_part[i] != 0)
        {
            h_cent_arr(i, 0) /= h_num_part[i];
            h_cent_arr(i, 1) /= h_num_part[i];
            h_cent_arr(i, 2) /= h_num_part[i];
        }
    }
}

__global__ void distance_gpu(ValueType * __RESTRICT d_distance, ValueType * __RESTRICT d_centroids, ValueType * __RESTRICT d_particles, const int n, const int k)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    ValueType x1 = d_part_arr(i, 0);
    ValueType y1 = d_part_arr(i, 1);
    ValueType z1 = d_part_arr(i, 2);
    
    if (i < n)
    {
        for (int j = 0; j < k; ++j)
        {
            ValueType x2 = d_cent_arr(j, 0);
            ValueType y2 = d_cent_arr(j, 1);
            ValueType z2 = d_cent_arr(j, 2);
            d_dist_arr(i, j) = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) + (z2 - z1) * (z2 - z1);
        }
    }
}

__global__ void assign_gpu(ValueType * __RESTRICT d_distance, ValueType * __RESTRICT d_centroids, ValueType * __RESTRICT d_particles, int * __RESTRICT d_cluster, const int n, const int k)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    ValueType min_dist = 12;
    
    if (i < n)
    {
        for (int j = 0; j < k; ++j)
        {
            if (d_dist_arr(i, j) < min_dist)
            {
                min_dist = d_dist_arr(i, j);
                d_cluster[i] = j;
            }
        }
    }
}

__global__ void findMean_gpu(ValueType * __RESTRICT d_centroids, ValueType * __RESTRICT d_particles, int * __RESTRICT d_cluster, int * __RESTRICT d_num_part, const int n, const int k)
{
    int i = threadIdx.x;
    
    if (i < k)
    {
        d_cent_arr(i, 0) = 0;
        d_cent_arr(i, 1) = 0;
        d_cent_arr(i, 2) = 0;

        d_num_part[i] = 0;

        for (int j = 0; j < n; ++j)
        {
            if (d_cluster[j] == i)
            {
                d_cent_arr(i, 0) += d_part_arr(j, 0);
                d_cent_arr(i, 1) += d_part_arr(j, 1);
                d_cent_arr(i, 2) += d_part_arr(j, 2);
                d_num_part[i]++;
            }
        }

        if (d_num_part[i] != 0)
        {
            d_cent_arr(i, 0) /= d_num_part[i];
            d_cent_arr(i, 1) /= d_num_part[i];
            d_cent_arr(i, 2) /= d_num_part[i];
        }
    }

}

__host__ void kmeans_gpu(ValueType * __RESTRICT d_distance, ValueType * __RESTRICT d_centroids, ValueType * __RESTRICT d_particles, int * __RESTRICT d_cluster, int * __RESTRICT d_num_part, const int n, const int k, const int numBlocks)
{
    for (int i = 0; i < 10; ++i)
    {
        distance_gpu<<<numBlocks, blockSize>>>(d_distance, d_centroids, d_particles, n, k);
        assign_gpu<<<numBlocks, blockSize>>>(d_distance, d_centroids, d_particles, d_cluster, n, k);
        findMean_gpu<<<1, k>>>(d_centroids, d_particles, d_cluster, d_num_part, n, k);
    }
}

int main()
{
    int n = 10240;
    int k = 16;
    int numBlocks = (n + blockSize - 1) / blockSize;

    // Initialize host variables
    ValueType *h_distance = NULL;
    ValueType *h_centroids = NULL;
    ValueType *h_particles = NULL;
    int *h_num_part = NULL;
    int *h_cluster = NULL;

    // Initialize device variables
    ValueType *d_distance = NULL;
    ValueType *d_centroids = NULL;
    ValueType *d_particles = NULL;
    int *d_cluster = NULL;
    int *d_num_part = NULL;

    // Allocate memory for host variables
    Allocate(h_distance, n * k);
    Allocate(h_centroids, k * NDIM);
    Allocate(h_particles, n * NDIM);
    Allocate(h_cluster, n);
    Allocate(h_num_part, k);

    // Allocate memory for device variables
    cudaMalloc(&d_distance, sizeof(ValueType) * n * k);
    cudaMalloc(&d_centroids, sizeof(ValueType) * k * NDIM);
    cudaMalloc(&d_particles, sizeof(ValueType) * n * NDIM);
    cudaMalloc(&d_cluster, sizeof(int) * n);
    cudaMalloc(&d_num_part, sizeof(int) * k);

    // Seed random number generator for values of particle coordinates
    srand(n);

    // Initialize particle array
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < NDIM; ++j)
            h_part_arr(i, j) = 2 * (frand() - 0.5);
    }

    // Seed random number generator for values of centroid coordinates
    srand(k);

    // Initialize centroid array
    for (int i = 0; i < k; ++i)
    {
        for (int j = 0; j < NDIM; ++j)
            h_cent_arr(i, j) = 2 * (frand() - 0.5);
    }

    /* Perform k-means clustering on CPU */
    /*double t_cpu = 0;

    myTimer_t t_start1 = getTimeStamp();

    for (int i = 0; i < 10; ++i)
    {
        distance(h_distance, h_centroids, h_particles, n, k);
        assign(h_distance, h_centroids, h_particles, h_cluster, n, k);
        findMean(h_centroids, h_particles, h_cluster, h_num_part, n, k); 
    }

    myTimer_t t_end1 = getTimeStamp();

    t_cpu += getElapsedTime(t_start1, t_end1);

    printf("Time for CPU version = %f (ms)\n", t_cpu * 1000);*/

    /* Perform k-means clustering on GPU */    
    double t_gpu = 0;
    double t_memcpy = 0;

    myTimer_t t_start2 = getTimeStamp();
    
    cudaMemcpy(d_centroids, h_centroids, sizeof(ValueType) * k * NDIM, cudaMemcpyHostToDevice);
    cudaMemcpy(d_particles, h_particles, sizeof(ValueType) * n * NDIM, cudaMemcpyHostToDevice); 

    myTimer_t t_middle = getTimeStamp();
    t_memcpy += getElapsedTime(t_start2, t_middle);

    kmeans_gpu(d_distance, d_centroids, d_particles, d_cluster, d_num_part, n, k, numBlocks);

    myTimer_t t_end2 = getTimeStamp(); 
    t_gpu += getElapsedTime(t_middle, t_end2);

    //cudaMemcpy(h_distance, d_distance, sizeof(ValueType) * n * k, cudaMemcpyDeviceToHost);
    //cudaMemcpy(h_cluster, d_cluster, sizeof(int) * n, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_centroids, d_centroids, sizeof(ValueType) * k * NDIM, cudaMemcpyDeviceToHost); 

    t_memcpy += getElapsedTime(t_end2, getTimeStamp());

    
    printf("Time for GPU version (minus memcpy) = %f (ms)\n", t_gpu * 1000);
    printf("Time for memcpy = %f (ms)\n", t_memcpy * 1000);

    /* Test distance, centroid and particle positions */
    /*for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < k; ++j)
        {
            printf("Distance between centroid %d and particle %d: %f\n", j, i, h_distance[i + j * blockSize]);
        }
    }

    for (int i = 0; i < n; ++i)
    {
        printf("Centroid number for particle %d: %d\n", i, h_cluster[i]);
    }

    for (int i = 0; i < k; ++i)
    {
        for (int j = 0; j < NDIM; ++j)
        {
            printf("Value of dimension %d of centroid %d: %f\n", j, i, h_cent_arr(i, j));
        }
    }*/

    free(h_distance);
    free(h_centroids);
    free(h_particles);
    free(h_cluster);

    cudaFree(d_distance);
    cudaFree(d_centroids);
    cudaFree(d_particles);
    cudaFree(d_cluster);
    cudaFree(d_num_part);
}