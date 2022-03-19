#include <stdio.h>

#include "kernel.h"
#include "launcher.h"

struct timespec ts;

void launch_saturation(const dim3 dim_grid,
                       const dim3 dim_block,
                       unsigned char *d_img,
                       const int height,
                       const int width,
                       const char c)
{
  clock_gettime(CLOCK_MONOTONIC, &ts);
  double before_saturate = ts.tv_sec + ts.tv_nsec * 1.0e-9;

  // Apply saturation
  if (c == 'r')
    {
      saturates_red_component<<<dim_grid, dim_block>>>(d_img, height * width);
    }
  else if (c == 'g')
    {
      saturates_green_component<<<dim_grid, dim_block>>>(d_img, height * width);
    }
  else if (c == 'b')
    {
      saturates_blue_component<<<dim_grid, dim_block>>>(d_img, height * width);
    }
  else
    {
      fprintf(stderr, "WTF?! You can't even read the fucking manual !\n");
      exit(3);
    }

  cudaDeviceSynchronize();

  clock_gettime(CLOCK_MONOTONIC, &ts);
  double after_saturate = ts.tv_sec + ts.tv_nsec * 1.0e-9;

  char color[256] = "";

  if (c == 'r')
    strcpy(color, "red");
  else if (c == 'g')
    strcpy(color, "green");
  else if (c == 'b')
    strcpy(color, "blue");

  fprintf(stderr, "Saturate %s component takes %e seconds\n",
          color, after_saturate - before_saturate);
}

void launch_horizontal_symetry(const dim3 dim_grid,
                               const dim3 dim_block,
                               unsigned char *d_img,
                               unsigned char *d_tmp,
                               const int height,
                               const int width,
                               const unsigned int size_in_bytes)
{
  // Copy d_img in d_tmp
  cudaMemcpy(d_tmp, d_img, size_in_bytes, cudaMemcpyDeviceToDevice);

  clock_gettime(CLOCK_MONOTONIC, &ts);
  double before_symetry = ts.tv_sec + ts.tv_nsec * 1.0e-9;

  // Apply horizontal symetry
  horizontal_symetry<<<dim_grid, dim_block>>>(d_img, d_tmp, height * width);

  cudaDeviceSynchronize();

  clock_gettime(CLOCK_MONOTONIC, &ts);
  double after_symetry = ts.tv_sec + ts.tv_nsec * 1.0e-9;

  fprintf(stderr, "Horizontal symetry takes %e seconds\n",
          after_symetry - before_symetry);
}

void launch_blur(const dim3 dim_grid, const dim3 dim_block,
                 unsigned char *d_img, unsigned char *d_tmp,
                 const int height, const int width,
                 const unsigned int size_in_bytes, const int n)
{
  // Copy d_img in d_tmp
  cudaMemcpy(d_tmp, d_img, size_in_bytes, cudaMemcpyDeviceToDevice);

  clock_gettime(CLOCK_MONOTONIC, &ts);
  double before_blur = ts.tv_sec + ts.tv_nsec * 1.0e-9;

  // Apply blur
  for (int i = 0; i < n; i++)
    {
      blur<<<dim_grid, dim_block>>>(d_img, d_tmp, height, width);
      cudaMemcpy(d_tmp, d_img, size_in_bytes, cudaMemcpyDeviceToDevice);
      cudaDeviceSynchronize();
    }

  clock_gettime(CLOCK_MONOTONIC, &ts);
  double after_blur = ts.tv_sec + ts.tv_nsec * 1.0e-9;

  fprintf(stderr, "Blur level %d takes %e seconds\n",
          n, after_blur - before_blur);
}

void launch_rgb_to_gray(const dim3 dim_grid, const dim3 dim_block,
                        unsigned char *d_img, unsigned char *d_tmp,
                        const int height, const int width,
                        const unsigned int size_in_bytes)
{
  clock_gettime(CLOCK_MONOTONIC, &ts);
  double before_gray = ts.tv_sec + ts.tv_nsec * 1.0e-9;

  // Transform rgb to gray
  rgb_to_gray<<<dim_grid, dim_block>>>(d_img, height * width);
  cudaDeviceSynchronize();

  clock_gettime(CLOCK_MONOTONIC, &ts);
  double after_gray = ts.tv_sec + ts.tv_nsec * 1.0e-9;

  fprintf(stderr, "RGB to gray takes %e seconds\n",
          after_gray - before_gray);
}

void launch_sobel(const dim3 dim_grid, const dim3 dim_block,
                  unsigned char *d_img, unsigned char *d_tmp,
                  const int height, const int width,
                  const unsigned int size_in_bytes)
{
  clock_gettime(CLOCK_MONOTONIC, &ts);
  double before_sobel = ts.tv_sec + ts.tv_nsec * 1.0e-9;

  // Transform colorfull image to gray
  rgb_to_gray<<<dim_grid, dim_block>>>(d_img, height * width);

  // Copy d_img in d_tmp
  cudaMemcpy(d_tmp, d_img, size_in_bytes, cudaMemcpyDeviceToDevice);

  // Apply sobel filter
  sobel<<<dim_grid, dim_block>>>(d_img, d_tmp, height, width);
  cudaDeviceSynchronize();

  clock_gettime(CLOCK_MONOTONIC, &ts);
  double after_sobel = ts.tv_sec + ts.tv_nsec * 1.0e-9;

  fprintf(stderr, "Sobel filter takes %e seconds\n",
          after_sobel - before_sobel);
}

void launch_slide_effect(const dim3 dim_grid, const dim3 dim_block,
                         unsigned char *d_img, unsigned char *d_tmp,
                         const int height, const int width,
                         const unsigned int c)
{
  clock_gettime(CLOCK_MONOTONIC, &ts);
  double before_slide_effect = ts.tv_sec + ts.tv_nsec * 1.0e-9;

  // Apply slide effect
  slide_effect<<<dim_grid, dim_block>>>(d_img, height * width, c);
  cudaDeviceSynchronize();

  clock_gettime(CLOCK_MONOTONIC, &ts);
  double after_slide_effect = ts.tv_sec + ts.tv_nsec * 1.0e-9;

  fprintf(stderr, "Slide effect takes %e seconds\n",
          after_slide_effect - before_slide_effect);
}

void launch_streams(unsigned char *h_img, unsigned char *d_img,
                    const int height, const int width,
                    unsigned int size_in_bytes)
{
  // Grid and Block
  unsigned int nthread = 32;
  unsigned int vertical_limit = height / 2;
  unsigned int horizontal_limit = width / 2;
  unsigned int grid_x = horizontal_limit / nthread + 1;
  unsigned int grid_y = vertical_limit / nthread + 1;

  dim3 dim_grid(grid_x, grid_y, 1);
  dim3 dim_block(nthread, nthread, 1);

  unsigned int old_size_in_bytes = size_in_bytes;
  size_in_bytes = vertical_limit * horizontal_limit * N_COMPONENT * sizeof(unsigned char);

  // Allocation
  unsigned char *h_img_top_left = (unsigned char *)malloc(size_in_bytes);;
  unsigned char *h_img_top_right = (unsigned char *)malloc(size_in_bytes);;
  unsigned char *h_img_bottom_left = (unsigned char *)malloc(size_in_bytes);;
  unsigned char *h_img_bottom_right = (unsigned char *)malloc(size_in_bytes);;

  unsigned char *d_img_top_left = NULL;
  unsigned char *d_img_top_right = NULL;
  unsigned char *d_img_bottom_left = NULL;
  unsigned char *d_img_bottom_right = NULL;

  cudaMalloc(&d_img_top_left, size_in_bytes);
  cudaMalloc(&d_img_top_right, size_in_bytes);
  cudaMalloc(&d_img_bottom_left, size_in_bytes);
  cudaMalloc(&d_img_bottom_right, size_in_bytes);

  // Check allocation
  if (!h_img_top_left || !h_img_top_right
      || !h_img_bottom_left || !h_img_bottom_right
      || !d_img_top_left || !d_img_top_right
      || !d_img_bottom_left || !d_img_bottom_right)
    {
      fprintf(stderr, "WTF?! We can't even allocate memories ? Die !\n");
      exit(2);
    }

  // Stream
  const unsigned int NSTREAM = 4;
  cudaStream_t stream[NSTREAM];

  for (unsigned int i = 0; i < NSTREAM; i++)
    cudaStreamCreate(&stream[i]);

  // Cut
  for (unsigned int y = 0; y < vertical_limit; y++)
    {
      unsigned int line_img = y * width * N_COMPONENT;
      unsigned int line_cpy = y * horizontal_limit * N_COMPONENT;

      for (unsigned int x = 0; x < horizontal_limit; x++)
        {
          h_img_bottom_left[line_cpy + x * N_COMPONENT + 0] = h_img[line_img + x * N_COMPONENT + 0];
          h_img_bottom_left[line_cpy + x * N_COMPONENT + 1] = h_img[line_img + x * N_COMPONENT + 1];
          h_img_bottom_left[line_cpy + x * N_COMPONENT + 2] = h_img[line_img + x * N_COMPONENT + 2];
        }
    }

  for (unsigned int y = 0; y < vertical_limit; y++)
    {
      unsigned int line_img = y * width * N_COMPONENT + horizontal_limit * N_COMPONENT;
      unsigned int line_cpy = y * horizontal_limit * N_COMPONENT;

      for (unsigned int x = 0; x < horizontal_limit; x++)
        {
          h_img_bottom_right[line_cpy + x * N_COMPONENT + 0] = h_img[line_img + x * N_COMPONENT + 0];
          h_img_bottom_right[line_cpy + x * N_COMPONENT + 1] = h_img[line_img + x * N_COMPONENT + 1];
          h_img_bottom_right[line_cpy + x * N_COMPONENT + 2] = h_img[line_img + x * N_COMPONENT + 2];
        }
    }

  for (unsigned int y = 0; y < vertical_limit; y++)
    {
      unsigned int line_img = y * width * N_COMPONENT + vertical_limit * width * N_COMPONENT;
      unsigned int line_cpy = y * horizontal_limit * N_COMPONENT;

      for (unsigned int x = 0; x < horizontal_limit; x++)
        {
          h_img_top_left[line_cpy + x * N_COMPONENT + 0] = h_img[line_img + x * N_COMPONENT + 0];
          h_img_top_left[line_cpy + x * N_COMPONENT + 1] = h_img[line_img + x * N_COMPONENT + 1];
          h_img_top_left[line_cpy + x * N_COMPONENT + 2] = h_img[line_img + x * N_COMPONENT + 2];
        }
    }

  for (unsigned int y = 0; y < vertical_limit; y++)
    {
      unsigned int line_img = y * width * N_COMPONENT + vertical_limit * width * N_COMPONENT + horizontal_limit * N_COMPONENT;;
      unsigned int line_cpy = y * horizontal_limit * N_COMPONENT;

      for (unsigned int x = 0; x < horizontal_limit; x++)
        {
          h_img_top_right[line_cpy + x * N_COMPONENT + 0] = h_img[line_img + x * N_COMPONENT + 0];
          h_img_top_right[line_cpy + x * N_COMPONENT + 1] = h_img[line_img + x * N_COMPONENT + 1];
          h_img_top_right[line_cpy + x * N_COMPONENT + 2] = h_img[line_img + x * N_COMPONENT + 2];
        }
    }

  //
  clock_gettime(CLOCK_MONOTONIC, &ts);
  double before_streams = ts.tv_sec + ts.tv_nsec * 1.0e-9;

  // Send to Device
  cudaMemcpyAsync(d_img_bottom_left, h_img_bottom_left, size_in_bytes, cudaMemcpyHostToDevice, stream[0]);
  cudaMemcpyAsync(d_img_bottom_right, h_img_bottom_right, size_in_bytes, cudaMemcpyHostToDevice, stream[1]);
  cudaMemcpyAsync(d_img_top_left, h_img_top_left, size_in_bytes, cudaMemcpyHostToDevice, stream[2]);
  cudaMemcpyAsync(d_img_top_right, h_img_top_right, size_in_bytes, cudaMemcpyHostToDevice, stream[3]);

  /*****************/
  /* Apply streams */
  /*****************/

  // Saturate green on bottom left
  saturates_green_component<<<dim_grid, dim_block, 0, stream[0]>>>(d_img_bottom_left, horizontal_limit * vertical_limit);

  // Saturate red on bottom right
  saturates_red_component<<<dim_grid, dim_block, 0, stream[1]>>>(d_img_bottom_right, horizontal_limit * vertical_limit);

  // Apply gray filter on top left
  rgb_to_gray<<<dim_grid, dim_block, 0, stream[2]>>>(d_img_top_left, horizontal_limit * vertical_limit);

  // Saturate blue on top right
  saturates_blue_component<<<dim_grid, dim_block, 0, stream[3]>>>(d_img_top_right, horizontal_limit * vertical_limit);

  // Get from Device
  cudaMemcpyAsync(h_img_bottom_left, d_img_bottom_left, size_in_bytes, cudaMemcpyDeviceToHost, stream[0]);
  cudaMemcpyAsync(h_img_bottom_right, d_img_bottom_right, size_in_bytes, cudaMemcpyDeviceToHost, stream[1]);
  cudaMemcpyAsync(h_img_top_left, d_img_top_left, size_in_bytes, cudaMemcpyDeviceToHost, stream[2]);
  cudaMemcpyAsync(h_img_top_right, d_img_top_right, size_in_bytes, cudaMemcpyDeviceToHost, stream[3]);
  cudaDeviceSynchronize();

  //
  clock_gettime(CLOCK_MONOTONIC, &ts);
  double after_streams = ts.tv_sec + ts.tv_nsec * 1.0e-9;

  fprintf(stderr, "Streams takes %e seconds\n",
          after_streams - before_streams);

  // Uncut
  for (unsigned int y = 0; y < vertical_limit; y++)
    {
      unsigned int line_img = y * width * N_COMPONENT;
      unsigned int line_cpy = y * horizontal_limit * N_COMPONENT;

      for (unsigned int x = 0; x < horizontal_limit; x++)
        {
          h_img[line_img + x * N_COMPONENT + 0] = h_img_bottom_left[line_cpy + x * N_COMPONENT + 0];
          h_img[line_img + x * N_COMPONENT + 1] = h_img_bottom_left[line_cpy + x * N_COMPONENT + 1];
          h_img[line_img + x * N_COMPONENT + 2] = h_img_bottom_left[line_cpy + x * N_COMPONENT + 2];
        }
    }

  for (unsigned int y = 0; y < vertical_limit; y++)
    {
      unsigned int line_img = y * width * N_COMPONENT + horizontal_limit * N_COMPONENT;
      unsigned int line_cpy = y * horizontal_limit * N_COMPONENT;

      for (unsigned int x = 0; x < horizontal_limit; x++)
        {
          h_img[line_img + x * N_COMPONENT + 0] = h_img_bottom_right[line_cpy + x * N_COMPONENT + 0] ;
          h_img[line_img + x * N_COMPONENT + 1] = h_img_bottom_right[line_cpy + x * N_COMPONENT + 1];
          h_img[line_img + x * N_COMPONENT + 2] = h_img_bottom_right[line_cpy + x * N_COMPONENT + 2];
        }
    }

  for (unsigned int y = 0; y < vertical_limit; y++)
    {
      unsigned int line_img = y * width * N_COMPONENT + vertical_limit * width * N_COMPONENT;
      unsigned int line_cpy = y * horizontal_limit * N_COMPONENT;

      for (unsigned int x = 0; x < horizontal_limit; x++)
        {
          h_img[line_img + x * N_COMPONENT + 0] = h_img_top_left[line_cpy + x * N_COMPONENT + 0];
          h_img[line_img + x * N_COMPONENT + 1] = h_img_top_left[line_cpy + x * N_COMPONENT + 1];
          h_img[line_img + x * N_COMPONENT + 2] = h_img_top_left[line_cpy + x * N_COMPONENT + 2];
        }
    }

  for (unsigned int y = 0; y < vertical_limit; y++)
    {
      unsigned int line_img = y * width * N_COMPONENT + vertical_limit * width * N_COMPONENT + horizontal_limit * N_COMPONENT;;
      unsigned int line_cpy = y * horizontal_limit * N_COMPONENT;

      for (unsigned int x = 0; x < horizontal_limit; x++)
        {
          h_img[line_img + x * N_COMPONENT + 0] = h_img_top_right[line_cpy + x * N_COMPONENT + 0];
          h_img[line_img + x * N_COMPONENT + 1] = h_img_top_right[line_cpy + x * N_COMPONENT + 1];
          h_img[line_img + x * N_COMPONENT + 2] = h_img_top_right[line_cpy + x * N_COMPONENT + 2];
        }
    }

  // Because after the function I make a memcpy form device to host of d_img and h_img
  cudaMemcpy(d_img, h_img, old_size_in_bytes, cudaMemcpyHostToDevice);

  // Release
  for (unsigned int i = 0; i < NSTREAM; ++i)
    cudaStreamDestroy(stream[i]);

  cudaFree(d_img_top_left);
  cudaFree(d_img_top_right);
  cudaFree(d_img_bottom_left);
  cudaFree(d_img_bottom_right);

  free(h_img_top_left);
  free(h_img_top_right);
  free(h_img_bottom_left);
  free(h_img_bottom_right);
}

void launch_keeper(const dim3 dim_grid,
                   const dim3 dim_block,
                   unsigned char *d_img,
                   const int height,
                   const int width,
                   const char c)
{
  clock_gettime(CLOCK_MONOTONIC, &ts);
  double before_keep = ts.tv_sec + ts.tv_nsec * 1.0e-9;

  // Apply saturation
  if (c == 'r')
    {
      keep_red_component<<<dim_grid, dim_block>>>(d_img, height * width);
    }
  else if (c == 'g')
    {
      keep_green_component<<<dim_grid, dim_block>>>(d_img, height * width);
    }
  else if (c == 'b')
    {
      keep_blue_component<<<dim_grid, dim_block>>>(d_img, height * width);
    }
  else
    {
      fprintf(stderr, "WTF?! You can't even read the fucking manual !\n");
      exit(3);
    }

  cudaDeviceSynchronize();

  clock_gettime(CLOCK_MONOTONIC, &ts);
  double after_keep = ts.tv_sec + ts.tv_nsec * 1.0e-9;

  char color[256] = "";

  if (c == 'r')
    strcpy(color, "red");
  else if (c == 'g')
    strcpy(color, "green");
  else if (c == 'b')
    strcpy(color, "blue");

  fprintf(stderr, "Keep %s component takes %e seconds\n",
          color, after_keep - before_keep);
}

void launch_opposite(const dim3 dim_grid,
                   const dim3 dim_block,
                   unsigned char *d_img,
                   const int height,
                   const int width)
{
  clock_gettime(CLOCK_MONOTONIC, &ts);
  double before_opposite = ts.tv_sec + ts.tv_nsec * 1.0e-9;

  // Apply inversion
  opposite_components<<<dim_grid, dim_block>>>(d_img, height * width);
  cudaDeviceSynchronize();

  clock_gettime(CLOCK_MONOTONIC, &ts);
  double after_opposite = ts.tv_sec + ts.tv_nsec * 1.0e-9;

  fprintf(stderr, "Opposite all components takes %e seconds\n",
          after_opposite - before_opposite);
}
