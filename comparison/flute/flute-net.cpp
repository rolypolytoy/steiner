#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "flute.h"

int main(int argc, char *argv[])
{
    int d = 0;
    int x[100], y[100];
    int a = FLUTE_ACCURACY;

    if (argc > 1)
        a = atoi(argv[1]);

    while (!feof(stdin)) {
        scanf("%d %d\n", &x[d], &y[d]);
        d++;
    }

    Flute::FluteState *flute1 = Flute::flute_init(FLUTE_POWVFILE, FLUTE_PORTFILE);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    Flute::Tree flutetree = Flute::flute(flute1, d, x, y, a);
    clock_gettime(CLOCK_MONOTONIC, &t1);

    double ns = (t1.tv_sec - t0.tv_sec) * 1e9 + (t1.tv_nsec - t0.tv_nsec);
    printf("%d %.3f\n", flutetree.length, ns / 1000.0);

    Flute::flute_free(flute1);
    return 0;
}
