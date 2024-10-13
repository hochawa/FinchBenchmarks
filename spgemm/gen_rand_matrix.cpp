#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

typedef struct {
    int row;
    int col;
    double value;
} Element;

int compare(const void *a, const void *b) {
    Element *elemA = (Element *)a;
    Element *elemB = (Element *)b;
    if (elemA->row != elemB->row) {
        return elemA->row - elemB->row;
    }
    return elemA->col - elemB->col;
}

void generate_sparse_matrix(int dimension, double sparsity, const char *filename) {
    int num_elements = dimension * dimension;
    int num_nonzeros = (int)(num_elements * sparsity);
    
    Element *elements = (Element *)malloc(num_nonzeros * sizeof(Element));

    if (!elements) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }

    srand(time(NULL));
    int count = 0;

    while (count < num_nonzeros) {
        int row = rand() % dimension;
        int col = rand() % dimension;

        int is_unique = 1;
        for (int i = 0; i < count; i++) {
            if (elements[i].row == row && elements[i].col == col) {
                is_unique = 0;
                break;
            }
        }

        if (is_unique) {
            elements[count].row = row;
            elements[count].col = col;
            elements[count].value = (double)rand() / RAND_MAX;
            count++;
        }
    }

    qsort(elements, num_nonzeros, sizeof(Element), compare);

    FILE *fp = fopen(filename, "w");
    if (fp == NULL) {
        fprintf(stderr, "Failed to open file\n");
        exit(EXIT_FAILURE);
    }

    fprintf(fp, "%%%%MatrixMarket matrix coordinate real general\n");
    fprintf(fp, "%d %d %d\n", dimension, dimension, num_nonzeros);

    for (int i = 0; i < num_nonzeros; i++) {
        fprintf(fp, "%d %d %lf\n", elements[i].row + 1, elements[i].col + 1, elements[i].value);
    }

    fclose(fp);
    
    free(elements);
}

int main() {
    double sparsity = 0.001;
    
    for (int power = 7; power <= 14; power++) {
        int dimension = 1 << power;
        char filename[30];
        snprintf(filename, sizeof(filename), "rand_%d.mtx", dimension);
        generate_sparse_matrix(dimension, sparsity, filename);
        printf("Generated %dx%d sparse matrix in %s\n", dimension, dimension, filename);
    }

    return 0;
}

