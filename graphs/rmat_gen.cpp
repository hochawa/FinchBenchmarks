#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef struct {
    int src;
    int dst;
    double weight;
} Edge;

void generate_rmat_edge(int *src, int *dst, int scale, double a, double b, double c, double d) {
    *src = 1;
    *dst = 1;
    int stride = 1 << (scale - 1);

    for (int i = 0; i < scale; i++) {
        double r = (double)rand() / RAND_MAX;
        if (r < a) {
        } else if (r < a + b) {
            *dst += stride;
        } else if (r < a + b + c) {
            *src += stride;
        } else {
            *src += stride;
            *dst += stride;
        }
        stride /= 2;
    }
}

int edge_compare(const void *a, const void *b) {
    Edge *edge1 = (Edge *)a;
    Edge *edge2 = (Edge *)b;

    if (edge1->src != edge2->src) {
        return edge1->src - edge2->src;
    }
    return edge1->dst - edge2->dst;
}

void generate_rmat_graph(int num_edges, int scale, double a, double b, double c, double d, const char *filename) {
    int num_vertices = 1 << scale;
    Edge *edges = (Edge *)malloc(num_edges * sizeof(Edge));
    int edge_count = 0;

    for (int i = 0; i < num_edges; i++) {
        int src, dst;
        generate_rmat_edge(&src, &dst, scale, a, b, c, d);
        if (src != dst) {
            double weight = (double)rand() / RAND_MAX;
            edges[edge_count++] = (Edge){src, dst, weight};
        }
    }

    qsort(edges, edge_count, sizeof(Edge), edge_compare);

    FILE *file = fopen(filename, "w");
    if (!file) {
        fprintf(stderr, "Error opening file for writing\n");
        exit(1);
    }

    fprintf(file, "%%MatrixMarket matrix coordinate real general\n");
    fprintf(file, "%d %d %d\n", num_vertices, num_vertices, edge_count);

    for (int i = 0; i < edge_count; i++) {
        fprintf(file, "%d %d %f\n", edges[i].src, edges[i].dst, edges[i].weight);
    }

    fclose(file);
    free(edges);
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <scale> <edge_factor>\n", argv[0]);
        return 1;
    }

    int scale = atoi(argv[1]);
    int edge_factor = atoi(argv[2]);
    int num_edges = edge_factor * (1 << scale);
    double a = 0.57;
    double b = 0.19;
    double c = 0.19;
    double d = 0.05;

    srand(time(NULL));

    char filename[50];
    snprintf(filename, sizeof(filename), "rmat_s%d_e%d.mtx", scale, edge_factor);

    generate_rmat_graph(num_edges, scale, a, b, c, d, filename);

    printf("RMAT graph generation complete. Output saved to %s\n", filename);
    return 0;
}

