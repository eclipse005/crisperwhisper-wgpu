#include <immintrin.h>

__m512 _ZGVeN16v_expf(__m512);
__m512 _ZGVeN16v_erff(__m512);

void cw_libmvec_expf16(const float *in, float *out) {
    _mm512_storeu_ps(out, _ZGVeN16v_expf(_mm512_loadu_ps(in)));
}

void cw_libmvec_erff16(const float *in, float *out) {
    _mm512_storeu_ps(out, _ZGVeN16v_erff(_mm512_loadu_ps(in)));
}
