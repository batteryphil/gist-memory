/**
 * test_gist_c.c — Standalone Verification Suite for C Gist Engine
 */

#include "gist_memory.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

int main(void) {
    printf("[*] Starting Gist C99 Recurrence Engine Verification...\n");

    const int d_map = 32;
    const int d_model = 512;
    const float decay = 0.9995f;
    const float eps = 1e-4f;

    GistStateC state;
    int err = gist_init_state(&state, d_map, d_model, decay, eps);
    assert(err == 0);
    assert(state.M != NULL);
    assert(state.Z != NULL);
    printf("  [+] State initialized: M [%d x %d], Z [%d], footprint: %.2f KB\n",
           d_map, d_model, d_map,
           ((d_map * d_model + d_map) * sizeof(float)) / 1024.0f);

    float *m_k = (float *)malloc(d_map * sizeof(float));
    float *q_k = (float *)malloc(d_map * sizeof(float));
    float *v = (float *)malloc(d_model * sizeof(float));
    float *recall = (float *)malloc(d_model * sizeof(float));

    /* Initialize dummy vectors */
    for (int i = 0; i < d_map; ++i) {
        m_k[i] = (float)(i + 1) * 0.05f;
        q_k[i] = (float)(i + 1) * 0.05f;
    }
    for (int j = 0; j < d_model; ++j) {
        v[j] = (float)(j % 17) * 0.1f;
    }

    /* Run 1000 streaming steps */
    printf("  [*] Executing 1,000 streaming autoregressive steps...\n");
    for (int step = 0; step < 1000; ++step) {
        gist_step(&state, m_k, q_k, v, recall);
        /* Verify no NaN or Inf */
        if (step % 200 == 0) {
            for (int j = 0; j < 5; ++j) {
                assert(!isnan(recall[j]) && !isinf(recall[j]));
            }
        }
    }

    printf("  [+] Step count reached: %zu\n", state.step_count);
    assert(state.step_count == 1000);

    /* Verify non-zero recall output */
    float norm = 0.0f;
    for (int j = 0; j < d_model; ++j) {
        norm += recall[j] * recall[j];
    }
    norm = sqrtf(norm);
    printf("  [+] Final recall L2 norm: %.6f\n", norm);
    assert(norm > 0.01f);

    /* Test reset */
    gist_reset_state(&state);
    assert(state.step_count == 0);
    assert(state.Z[0] == 0.0f);
    assert(state.M[0] == 0.0f);
    printf("  [+] Reset verified.\n");

    /* Cleanup */
    gist_free_state(&state);
    free(m_k);
    free(q_k);
    free(v);
    free(recall);

    printf("[SUCCESS] All C99 Gist Engine tests passed flawlessly!\n");
    return 0;
}
