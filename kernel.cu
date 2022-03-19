#include <math.h>

#include "kernel.h"

__device__ int compute_index(void)
{
  int size_block = blockDim.x * blockDim.y * blockDim.z;

  int id_block =
    gridDim.x * gridDim.y * blockIdx.z
    + gridDim.x * blockIdx.y
    + blockIdx.x;

  int id_thread =
    blockDim.x * blockDim.y * threadIdx.z
    + blockDim.x * threadIdx.y
    + threadIdx.x;

  int id = id_block * size_block + id_thread;

  return id;
}

__global__ void saturates_red_component(unsigned char *d_img, const int size)
{
  int id = compute_index();

  if (id < size)
    {
      d_img[id * N_COMPONENT + 0] = FULL;
    }
}

__global__ void saturates_green_component(unsigned char *d_img, const int size)
{
  int id = compute_index();

  if (id < size)
    {
      d_img[id * N_COMPONENT + 1] = FULL;
    }
}

__global__ void saturates_blue_component(unsigned char *d_img, const int size)
{
  int id = compute_index();

  if (id < size)
    {
      d_img[id * N_COMPONENT + 2] = FULL;
    }
}

__global__ void horizontal_symetry(unsigned char *d_img,
                                   const unsigned char *d_tmp,
                                   const int size)
{
  int id = compute_index();

  if (id < size)
    {
      d_img[id * N_COMPONENT + 0] = d_tmp[(size - id) * N_COMPONENT + 0];
      d_img[id * N_COMPONENT + 1] = d_tmp[(size - id) * N_COMPONENT + 1];
      d_img[id * N_COMPONENT + 2] = d_tmp[(size - id) * N_COMPONENT + 2];
    }
}

__global__ void blur(unsigned char *d_img, const unsigned char *d_tmp,
                     const int height, const int width)
{
  int id = compute_index();

  int size = height * width;

  if (id < size)
    {
      if (id % width != 0 && (id + 1) % width != 0
          && (id / width) % height != 0 && ((id + 1) / width) % height != 0)
        {
          unsigned char mean_x = d_tmp[id * N_COMPONENT + 0];
          unsigned char mean_y = d_tmp[id * N_COMPONENT + 1];
          unsigned char mean_z = d_tmp[id * N_COMPONENT + 2];

          mean_x += d_tmp[(id - 1) * N_COMPONENT + 0];
          mean_y += d_tmp[(id - 1) * N_COMPONENT + 1];
          mean_z += d_tmp[(id - 1) * N_COMPONENT + 2];

          mean_x += d_tmp[(id + 1) * N_COMPONENT + 0];
          mean_y += d_tmp[(id + 1) * N_COMPONENT + 1];
          mean_z += d_tmp[(id + 1) * N_COMPONENT + 2];

          mean_x += d_tmp[(id - width) * N_COMPONENT + 0];
          mean_y += d_tmp[(id - width) * N_COMPONENT + 1];
          mean_z += d_tmp[(id - width) * N_COMPONENT + 2];

          mean_x += d_tmp[(id + width) * N_COMPONENT + 0];
          mean_y += d_tmp[(id + width) * N_COMPONENT + 1];
          mean_z += d_tmp[(id + width) * N_COMPONENT + 2];

          d_img[id * N_COMPONENT + 0] = mean_x / (unsigned char)5;
          d_img[id * N_COMPONENT + 1] = mean_y / (unsigned char)5;
          d_img[id * N_COMPONENT + 2] = mean_z / (unsigned char)5;
      }
    }
}

__global__ void rgb_to_gray(unsigned char *d_img, const int size)
{
  int id = compute_index();

  if (id < size)
    {
      unsigned char rgb =
        (unsigned char) (  (float)d_img[id * N_COMPONENT + 0] * 0.299f
                         + (float)d_img[id * N_COMPONENT + 1] * 0.587f
                         + (float)d_img[id * N_COMPONENT + 2] * 0.114f
                         );

      d_img[id * N_COMPONENT + 0] = rgb;
      d_img[id * N_COMPONENT + 1] = rgb;
      d_img[id * N_COMPONENT + 2] = rgb;
    }
}

__global__ void sobel(unsigned char *d_img, const unsigned char *d_tmp,
                      const int height, const int width)
{
  int id = compute_index();

  int size = height * width;

  if (id < size)
    {
      if (id % width != 0 && (id + 1) % width != 0
          && (id / width) % height != 0 && ((id + 1) / width) % height != 0)
        {
          // Init Gx and Gy
          unsigned char gx_x = 0;
          unsigned char gx_y = 0;
          unsigned char gx_z = 0;

          unsigned char gy_x = 0;
          unsigned char gy_y = 0;
          unsigned char gy_z = 0;

          /*********************/
          /* First convolution */
          /*********************/

          // Center column
          gx_x -= 2 * d_tmp[(id - 1) * N_COMPONENT + 0];
          gx_y -= 2 * d_tmp[(id - 1) * N_COMPONENT + 1];
          gx_z -= 2 * d_tmp[(id - 1) * N_COMPONENT + 2];

          gx_x += 2 * d_tmp[(id + 1) * N_COMPONENT + 0];
          gx_y += 2 * d_tmp[(id + 1) * N_COMPONENT + 1];
          gx_z += 2 * d_tmp[(id + 1) * N_COMPONENT + 2];

          // Left column
          gx_x -= d_tmp[(id - width - 1) * N_COMPONENT + 0];
          gx_y -= d_tmp[(id - width - 1) * N_COMPONENT + 1];
          gx_z -= d_tmp[(id - width - 1) * N_COMPONENT + 2];

          gx_x += d_tmp[(id - width + 1) * N_COMPONENT + 0];
          gx_y += d_tmp[(id - width + 1) * N_COMPONENT + 1];
          gx_z += d_tmp[(id - width + 1) * N_COMPONENT + 2];

          // Right column
          gx_x -= d_tmp[(id + width - 1) * N_COMPONENT + 0];
          gx_y -= d_tmp[(id + width - 1) * N_COMPONENT + 1];
          gx_z -= d_tmp[(id + width - 1) * N_COMPONENT + 2];

          gx_x += d_tmp[(id + width + 1) * N_COMPONENT + 0];
          gx_y += d_tmp[(id + width + 1) * N_COMPONENT + 1];
          gx_z += d_tmp[(id + width + 1) * N_COMPONENT + 2];

          /**********************/
          /* Second convolution */
          /**********************/

          // Line below
          gy_x += d_tmp[(id - width - 1) * N_COMPONENT + 0];
          gy_y += d_tmp[(id - width - 1) * N_COMPONENT + 1];
          gy_z += d_tmp[(id - width - 1) * N_COMPONENT + 2];

          gy_x += 2 * d_tmp[(id - width) * N_COMPONENT + 0];
          gy_y += 2 * d_tmp[(id - width) * N_COMPONENT + 1];
          gy_z += 2 * d_tmp[(id - width) * N_COMPONENT + 2];

          gy_x += d_tmp[(id - width + 1) * N_COMPONENT + 0];
          gy_y += d_tmp[(id - width + 1) * N_COMPONENT + 1];
          gy_z += d_tmp[(id - width + 1) * N_COMPONENT + 2];

          // Line above
          gy_x -= d_tmp[(id + width - 1) * N_COMPONENT + 0];
          gy_y -= d_tmp[(id + width - 1) * N_COMPONENT + 1];
          gy_z -= d_tmp[(id + width - 1) * N_COMPONENT + 2];

          gy_x -= 2 * d_tmp[(id + width) * N_COMPONENT + 0];
          gy_y -= 2 * d_tmp[(id + width) * N_COMPONENT + 1];
          gy_z -= 2 * d_tmp[(id + width) * N_COMPONENT + 2];

          gy_x -= d_tmp[(id + width + 1) * N_COMPONENT + 0];
          gy_y -= d_tmp[(id + width + 1) * N_COMPONENT + 1];
          gy_z -= d_tmp[(id + width + 1) * N_COMPONENT + 2];

          // Update
          d_img[id * N_COMPONENT + 0] = (unsigned char)sqrt((float)(gx_x * gx_x + gy_x * gy_x));
          d_img[id * N_COMPONENT + 1] = (unsigned char)sqrt((float)(gx_y * gx_y + gy_y * gy_y));
          d_img[id * N_COMPONENT + 2] = (unsigned char)sqrt((float)(gx_z * gx_z + gy_z * gy_z));
      }
    }
}

__global__ void slide_effect(unsigned char *d_img,
                             const int size, const int c)
{
  int id = compute_index();

  if (id < size)
    {
      d_img[id * N_COMPONENT + 0] -= c;
      d_img[id * N_COMPONENT + 1] -= c;
      d_img[id * N_COMPONENT + 2] -= c;
    }
}

__global__ void keep_red_component(unsigned char *d_img, const int size)
{
  int id = compute_index();

  if (id < size)
    {
      d_img[id * N_COMPONENT + 1] = 0;
      d_img[id * N_COMPONENT + 2] = 0;
    }
}

__global__ void keep_green_component(unsigned char *d_img, const int size)
{
  int id = compute_index();

  if (id < size)
    {
      d_img[id * N_COMPONENT + 0] = 0;
      d_img[id * N_COMPONENT + 2] = 0;
    }
}

__global__ void keep_blue_component(unsigned char *d_img, const int size)
{
  int id = compute_index();

  if (id < size)
    {
      d_img[id * N_COMPONENT + 0] = 0;
      d_img[id * N_COMPONENT + 1] = 0;
    }
}

__global__ void opposite_components(unsigned char *d_img, const int size)
{
  int id = compute_index();

  if (id < size)
    {
      d_img[id * N_COMPONENT + 0] = (unsigned char)255 - d_img[id * N_COMPONENT + 0];
      d_img[id * N_COMPONENT + 1] = (unsigned char)255 - d_img[id * N_COMPONENT + 1];
      d_img[id * N_COMPONENT + 2] = (unsigned char)255 - d_img[id * N_COMPONENT + 2];
    }
}
