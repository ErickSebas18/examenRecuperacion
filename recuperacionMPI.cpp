#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <mpi.h>
 
int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
   
    int width, height, channels;
    uint8_t *rgb_pixels;
    if (rank == 0) {
        rgb_pixels = stbi_load("./image01.jpg", &width, &height, &channels, STBI_rgb);
        if (!rgb_pixels) {
            printf("Error al cargar la imagen.\n");
            MPI_Finalize();
            return 1;
        }
    }
    MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int blocksize = height / nprocs;
    int padding = height % nprocs;
    int block = padding;
    if (rank < padding) {
        block++;
    }

    uint8_t *my_gray_pixels = (uint8_t *)malloc(width * padding);

    MPI_Scatter(rgb_pixels, width * blocksize * 3, MPI_UNSIGNED_CHAR, 
                rgb_pixels + rank * blocksize * width * 3, 
                block * width * 3, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    // Convertir el segmento de imagen a escala de grises
    for (int i = 0; i < block * width; i++) {
        int r = rgb_pixels[i * 3];
        int g = rgb_pixels[i * 3 + 1];
        int b = rgb_pixels[i * 3 + 2];
        gray_pixels[i] = 0.21 * r + 0.72 * g + 0.07 * b;
    }

    MPI_Gather(gray_pixels, block * width, MPI_UNSIGNED_CHAR, 
               rgb_pixels, block * width, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        stbi_write_png("./imagen-gris.jpg", width, height, 1, rgb_pixels, width);
        stbi_image_free(rgb_pixels);
    }

    free(gray_pixels);
    MPI_Finalize();
    return 0;
}