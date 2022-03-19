#ifndef _LAUNCHER_H_
#define _LAUNCHER_H_

//
void launch_saturation(const dim3 dim_grid,
                       const dim3 dim_block,
                       unsigned char *d_img,
                       const int height,
                       const int width,
                       const char c);

//
void launch_horizontal_symetry(const dim3 dim_grid,
                               const dim3 dim_block,
                               unsigned char *d_img,
                               unsigned char *d_tmp,
                               const int height,
                               const int width,
                               const unsigned int size_in_bytes);

//
void launch_blur(const dim3 dim_grid, const dim3 dim_block,
                 unsigned char *d_img, unsigned char *d_tmp,
                 const int height, const int width,
                 const unsigned int size_in_bytes, const int n);

//
void launch_rgb_to_gray(const dim3 dim_grid, const dim3 dim_block,
                        unsigned char *d_img, unsigned char *d_tmp,
                        const int height, const int width,
                        const unsigned int size_in_bytes);

//
void launch_sobel(const dim3 dim_grid, const dim3 dim_block,
                  unsigned char *d_img, unsigned char *d_tmp,
                  const int height, const int width,
                  const unsigned int size_in_bytes);

//
void launch_slide_effect(const dim3 dim_grid, const dim3 dim_block,
                         unsigned char *d_img, unsigned char *d_tmp,
                         const int height, const int width,
                         const unsigned int c);

//
void launch_streams(unsigned char *img, unsigned char *d_img,
                    const int height, const int width,
                    unsigned int size_in_bytes);

//
void launch_keeper(const dim3 dim_grid,
                   const dim3 dim_block,
                   unsigned char *d_img,
                   const int height,
                   const int width,
                   const char c);

//
void launch_opposite(const dim3 dim_grid,
                     const dim3 dim_block,
                     unsigned char *d_img,
                     const int height,
                     const int width);

#endif // _LAUNCHER_H_
