/**
 * gist_memory.h — Pure C99 Recurrent Engine for Gist Associative Memory
 * ====================================================================
 * Header-only, zero-dependency C implementation of O(1) state space updates
 * and associative recall. Suitable for edge inference, Cosmopolitan libc,
 * and embedded deployment.
 */

#ifndef GIST_MEMORY_H
#define GIST_MEMORY_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float *M;          /* Associative memory matrix: [d_map x d_model] */
    float *Z;          /* Normalization vector: [d_map] */
    int d_map;         /* Compressed blueprint dimension (e.g. 32) */
    int d_model;       /* Full hidden dimension (e.g. 512, 2048) */
    float decay;       /* Recency decay factor lambda in (0, 1) */
    float eps;         /* Stabilization epsilon (default: 1e-4) */
    size_t step_count; /* Number of tokens processed */
} GistStateC;

/**
 * Allocates and initializes Gist associative memory state.
 */
static inline int gist_init_state(GistStateC *state, int d_map, int d_model, float decay, float eps) {
    if (!state || d_map <= 0 || d_model <= 0) return -1;
    state->d_map = d_map;
    state->d_model = d_model;
    state->decay = decay;
    state->eps = (eps <= 0.0f) ? 1e-4f : eps;
    state->step_count = 0;

    state->M = (float *)calloc((size_t)d_map * (size_t)d_model, sizeof(float));
    state->Z = (float *)calloc((size_t)d_map, sizeof(float));

    if (!state->M || !state->Z) {
        free(state->M);
        free(state->Z);
        state->M = NULL;
        state->Z = NULL;
        return -2;
    }
    return 0;
}

/**
 * Frees internal state memory.
 */
static inline void gist_free_state(GistStateC *state) {
    if (state) {
        if (state->M) { free(state->M); state->M = NULL; }
        if (state->Z) { free(state->Z); state->Z = NULL; }
        state->step_count = 0;
    }
}

/**
 * Resets state to zeros.
 */
static inline void gist_reset_state(GistStateC *state) {
    if (state && state->M && state->Z) {
        memset(state->M, 0, (size_t)state->d_map * (size_t)state->d_model * sizeof(float));
        memset(state->Z, 0, (size_t)state->d_map * sizeof(float));
        state->step_count = 0;
    }
}

/**
 * Performs a single O(1) step update and associative query recall:
 *   1. Normalizer recurrence: Z = lambda * Z + m_k
 *   2. Associative recurrence: M = lambda * M + m_k (x) v
 *   3. Associative recall: recall = (q_k^T * M) / (q_k^T * Z + eps)
 *
 * Parameters:
 *   state: Pointer to initialized GistStateC
 *   m_k: Kernelized and salience-gated blueprint vector [d_map]
 *   q_k: Kernelized query vector [d_map]
 *   v: Value payload vector [d_model]
 *   recall_out: Output recalled vector [d_model] (allocated by caller)
 */
static inline void gist_step(
    GistStateC *state,
    const float *m_k,
    const float *q_k,
    const float *v,
    float *recall_out
) {
    const int d_map = state->d_map;
    const int d_model = state->d_model;
    const float lambda = state->decay;
    float *M = state->M;
    float *Z = state->Z;

    /* Step 1: Update Normalizer Z = lambda * Z + m_k */
    for (int i = 0; i < d_map; ++i) {
        Z[i] = lambda * Z[i] + m_k[i];
    }

    /* Step 2: Update Memory M = lambda * M + m_k (x) v */
    for (int i = 0; i < d_map; ++i) {
        const float mi = m_k[i];
        float *row = M + ((size_t)i * (size_t)d_model);
        for (int j = 0; j < d_model; ++j) {
            row[j] = lambda * row[j] + mi * v[j];
        }
    }

    /* Step 3: Compute Denominator den = q_k^T * Z + eps */
    float den = 0.0f;
    for (int i = 0; i < d_map; ++i) {
        den += q_k[i] * Z[i];
    }
    den += state->eps;
    if (den < state->eps) den = state->eps;
    const float inv_den = 1.0f / den;

    /* Step 4: Compute Numerator num = q_k^T * M and multiply by inv_den */
    for (int j = 0; j < d_model; ++j) {
        float sum = 0.0f;
        for (int i = 0; i < d_map; ++i) {
            sum += q_k[i] * M[(size_t)i * (size_t)d_model + (size_t)j];
        }
        recall_out[j] = sum * inv_den;
    }

    state->step_count++;
}

#ifdef __cplusplus
}
#endif

#endif /* GIST_MEMORY_H */
