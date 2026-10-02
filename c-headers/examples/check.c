/* Smoke test of the C API: cc check.c -I.. -L../../rust/target/release -llombokvector */
#include <stdio.h>
#include "lombokvector.h"

int main(void) {
    const float a[] = {1.0f, 2.0f, 3.0f};
    const float b[] = {4.0f, 5.0f, 6.0f};
    float out = 0.0f;
    if (lombokvector_dot_f32(a, b, 3, &out) != LOMBOKVECTOR_OK || out != 32.0f) return 1;
    if (lombokvector_cosine_f32(a, b, 0, &out) != LOMBOKVECTOR_ERR_EMPTY_VECTOR) return 2;
    float v[] = {3.0f, 4.0f};
    if (lombokvector_normalize_inplace_f32(v, 2) != LOMBOKVECTOR_OK || v[0] != 0.6f) return 3;
    printf("ok, backend %s\n", lombokvector_active_backend());
    return 0;
}
