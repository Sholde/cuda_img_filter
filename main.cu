#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "FreeImage.h"
#include "kernel.h"
#include "launcher.h"

//#define WIDTH       1920
//#define HEIGHT      1024
#define BPP         24   // Since we're outputting three 8 bit RGB values

__host__ static void get_pixels(FIBITMAP *bitmap, unsigned char *img,
                                int height, int width, int pitch)
{
  //
  BYTE *bits = (BYTE*)FreeImage_GetBits(bitmap);

  //
  for (int y = 0; y < height; y++)
    {
      BYTE *pixel = (BYTE *)bits;
      for (int x = 0; x < width; x++)
        {
          int idx = ((y * width) + x) * N_COMPONENT;
          img[idx + 0] = pixel[FI_RGBA_RED];
          img[idx + 1] = pixel[FI_RGBA_GREEN];
          img[idx + 2] = pixel[FI_RGBA_BLUE];

          // next pixel
          pixel += N_COMPONENT;
        }

      // next line
      bits += pitch;
    }
}

__host__ static void store_pixels(FIBITMAP *bitmap,
                                  const unsigned char *img,
                                  const int height, const int width,
                                  const int pitch)
{
  BYTE *bits = (BYTE*)FreeImage_GetBits(bitmap);

  for (int y = 0; y < height; y++)
    {
      BYTE *pixel = (BYTE *)bits;
      for (int x = 0; x < width; x++)
        {
          // Compute new pixel
          RGBQUAD newcolor;

          int idx = ((y * width) + x) * N_COMPONENT;
          newcolor.rgbRed = img[idx + 0];
          newcolor.rgbGreen = img[idx + 1];
          newcolor.rgbBlue = img[idx + 2];

          // Update pixel
          if(!FreeImage_SetPixelColor(bitmap, x, y, &newcolor))
            {
              fprintf(stderr, "(%d, %d) Fail...\n", x, y);
            }

          // next pixel
          pixel += N_COMPONENT;
        }

      // next line
      bits += pitch;
    }
}

int main (int argc, char **argv)
{
  // Check argument
  if (argc < 2)
    {
      fprintf(stderr, "WTF?! You can't even read the fucking manual !\n");
      exit(1);
    }

  if (strcmp(argv[1], "--help") == 0)
    {
      // Print usage
      fprintf(stderr, "Usage: %s [INPUT] [OUTPUT] [OPTION] [PARAMETER]\n", argv[0]);
      fprintf(stderr, "         --saturate C          C = r, g, b\n");
      fprintf(stderr, "         --horizontal-symetry  -\n");
      fprintf(stderr, "         --blur INTENSITY      INTENSITY = INTEGER\n");
      fprintf(stderr, "         --rgb-to-gray         -\n");
      fprintf(stderr, "         --sobel               -\n");
      fprintf(stderr, "         --slide-effect C      C = INTEGER\n");
      fprintf(stderr, "         --streams             -\n");
      fprintf(stderr, "         --keep C              C = r, g, b\n");
      fprintf(stderr, "         --opposite            -\n");

      // Exit success
      exit(0);
    }

  if (argc < 4)
    {
      fprintf(stderr, "WTF?! You can't even read the fucking manual !\n");
      exit(1);
    }

  FreeImage_Initialise();

  char PathName[256] = "img.jpg";
  char PathDest[256] = "new_img.png";

  strcpy(PathName, argv[1]);
  strcpy(PathDest, argv[2]);

  // Load and decode a regular file
  FREE_IMAGE_FORMAT fif = FreeImage_GetFileType(PathName);

  FREE_IMAGE_FORMAT fif_pathname = FreeImage_GetFIFFromFilename(PathName);

  FIBITMAP *bitmap = FreeImage_Load(fif_pathname, PathName, 0);

  if(!bitmap)
    {
      fprintf(stderr, "WTF?! We can't even allocate images ? Die !\n");
      exit(1); //WTF?! We can't even allocate images ? Die !
    }

  unsigned width  = FreeImage_GetWidth(bitmap);
  unsigned height = FreeImage_GetHeight(bitmap);
  unsigned pitch  = FreeImage_GetPitch(bitmap);

  fprintf(stderr, "Processing Image of size %d x %d\n", width, height);

  // Allocate memories
  unsigned int size_in_bytes =
    sizeof(unsigned char) * N_COMPONENT * width * height;

  unsigned char *h_img = (unsigned char *)malloc(size_in_bytes);
  unsigned char *d_img = NULL;
  unsigned char *d_tmp = NULL;
  cudaMalloc(&d_img, size_in_bytes);
  cudaMalloc(&d_tmp, size_in_bytes);

  if (!h_img || !d_img || !d_tmp)
    {
      fprintf(stderr, "WTF?! We can't even allocate memories ? Die !\n");
      exit(2);
    }

  // Get pixels
  get_pixels(bitmap, h_img, height, width, pitch);

  // Copy host array to device array
  cudaMemcpy(d_img, h_img, size_in_bytes, cudaMemcpyHostToDevice);

  // Define grid and blocks
  unsigned int nthread = 32;
  unsigned int grid_x = width / nthread + 1;;
  unsigned int grid_y = height / nthread + 1;
  dim3 dim_grid(grid_x, grid_y, 1);
  dim3 dim_block(nthread, nthread, 1);

  fprintf(stderr, "Using a grid (%d, %d, %d) of blocks (%d, %d, %d)\n",
          dim_grid.x, dim_grid.y, dim_grid.z,
          dim_block.x, dim_block.y, dim_block.z);

  /**********/
  /* Kernel */
  /**********/

  if (strcmp(argv[3], "--saturate") == 0)
    {
      if (argc < 5)
        {
          fprintf(stderr, "WTF?! You can't even read the fucking manual !\n");
          exit(4);
        }

      // Saturate component
      if (strcmp(argv[4], "r") == 0)
        {
          launch_saturation(dim_grid, dim_block, d_img, height, width, 'r');
        }
      else if (strcmp(argv[4], "g") == 0)
        {
          launch_saturation(dim_grid, dim_block, d_img, height, width, 'g');
        }
      else if (strcmp(argv[4], "b") == 0)
        {
          launch_saturation(dim_grid, dim_block, d_img, height, width, 'b');
        }
      else
        {
          fprintf(stderr, "WTF?! You can't even read the fucking manual !\n");
          exit(4);
        }
    }
  else if (strcmp(argv[3], "--horizontal-symetry") == 0)
    {
      // Apply horizontal symetry
      launch_horizontal_symetry(dim_grid, dim_block,
                                d_img, d_tmp, height, width, size_in_bytes);
    }
  else if (strcmp(argv[3], "--blur") == 0)
    {
      if (argc < 5)
        {
          fprintf(stderr, "WTF?! You can't even read the fucking manual !\n");
          exit(4);
        }

      unsigned int intensity = strtol(argv[4], NULL, 10);

      // Apply blur
      launch_blur(dim_grid, dim_block,
                  d_img, d_tmp, height, width, size_in_bytes, intensity);
    }
  else if (strcmp(argv[3], "--rgb-to-gray") == 0)
    {
      // Transform rgb to gray
      launch_rgb_to_gray(dim_grid, dim_block, d_img, d_tmp, height, width,
                         size_in_bytes);
    }
  else if (strcmp(argv[3], "--sobel") == 0)
    {
      // Apply sobel filter
      launch_sobel(dim_grid, dim_block, d_img, d_tmp, height, width,
                   size_in_bytes);
    }
  else if (strcmp(argv[3], "--slide-effect") == 0)
    {
      if (argc < 5)
        {
          fprintf(stderr, "WTF?! You can't even read the fucking manual !\n");
          exit(4);
        }

      unsigned int c = strtol(argv[4], NULL, 10);

      // Apply slide effect
      launch_slide_effect(dim_grid, dim_block,
                          d_img, d_tmp, height, width, c);
    }
  else if (strcmp(argv[3], "--streams") == 0)
    {
      // Apply streams
      launch_streams(h_img, d_img, height, width, size_in_bytes);
    }
  else if (strcmp(argv[3], "--keep") == 0)
    {
      if (argc < 5)
        {
          fprintf(stderr, "WTF?! You can't even read the fucking manual !\n");
          exit(4);
        }

      // Saturate component
      if (strcmp(argv[4], "r") == 0)
        {
          launch_keeper(dim_grid, dim_block, d_img, height, width, 'r');
        }
      else if (strcmp(argv[4], "g") == 0)
        {
          launch_keeper(dim_grid, dim_block, d_img, height, width, 'g');
        }
      else if (strcmp(argv[4], "b") == 0)
        {
          launch_keeper(dim_grid, dim_block, d_img, height, width, 'b');
        }
      else
        {
          fprintf(stderr, "WTF?! You can't even read the fucking manual !\n");
          exit(4);
        }
    }
  else if (strcmp(argv[3], "--opposite") == 0)
    {
      // Apply opposite
      launch_opposite(dim_grid, dim_block, d_img, height, width);
    }
  else
    {
      fprintf(stderr, "You don't specify a known kernel, try --help\n");
    }

  // Copy back
  cudaMemcpy(h_img, d_img, size_in_bytes, cudaMemcpyDeviceToHost);

  /*******/
  /* End */
  /*******/

  // Store pixels
  store_pixels(bitmap, h_img, height, width, pitch);

  FREE_IMAGE_FORMAT fif_dest = FreeImage_GetFIFFromFilename(PathDest);

  if(FreeImage_Save(fif_dest, bitmap, PathDest, 0))
    fprintf(stderr, "Image successfully saved !\n");
  else
    fprintf(stderr, "WTF?! We can't even saved image!\n");

  // Cleanup !
  FreeImage_DeInitialise();

  free(h_img);
  cudaFree(d_img);
  cudaFree(d_tmp);
}
