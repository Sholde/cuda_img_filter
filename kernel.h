#ifndef _KERNEL_H_
#define _KERNEL_H_

#define N_COMPONENT 3    // we have 3 component (RGB)
#define FULL        255

/**
 * Compute index of the thread in 3 dimension of grid and block
 */
__device__ int compute_index(void);

/**
 * Saturate one color of the pixel (red, green or blue)
 */
__global__ void saturates_red_component(unsigned char *d_img, const int size);
__global__ void saturates_green_component(unsigned char *d_img, const int size);
__global__ void saturates_blue_component(unsigned char *d_img, const int size);

/**
 * Apply a horizontal symetry
 */
__global__ void horizontal_symetry(unsigned char *d_img,
                                   const unsigned char *d_tmp,
                                   const int size);

/**
 * Blur the image
 */
__global__ void blur(unsigned char *d_img, const unsigned char *d_tmp,
                     const int height, const int width);

/**
 * Convert colorfull image into a gray image
 */
__global__ void rgb_to_gray(unsigned char *d_img, const int size);

/**
 * Sobel filter
 */
__global__ void sobel(unsigned char *d_img, const unsigned char *d_tmp,
                      const int height, const int width);

/**
 * Slide effect - reduce each component by c
 */
__global__ void slide_effect(unsigned char *d_img,
                             const int size, const int c);

/**
 * Keep only one composant (others are set to 0)
 */
__global__ void keep_red_component(unsigned char *d_img, const int size);
__global__ void keep_green_component(unsigned char *d_img, const int size);
__global__ void keep_blue_component(unsigned char *d_img, const int size);

/**
 * Opposite all components
 */
__global__ void opposite_components(unsigned char *d_img, const int size);

#endif // _KERNEL_H_
