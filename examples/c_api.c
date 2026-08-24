/* The C interface: double precision keeps the plain name, single prefixes admf_. */
#include <admiral/admiral.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    adm_plan plan;
    adm_complex data[1024];
    if (adm_plan_1d(&plan, 1024, NULL) != ADM_SUCCESS) {           /* NULL = defaults */
        fprintf(stderr, "plan failed: %s\n", adm_plan_error_message(plan));
        adm_plan_destroy(plan);                                    /* failure handle is still owned */
        return 1;
    }

    for (size_t i = 0; i < 1024; ++i) {
        data[i].real = sin(0.01 * (double)i);
        data[i].imag = 0.0;
    }
    if (adm_plan_execute_forward(plan, data) != ADM_SUCCESS) return 1;
    if (adm_plan_execute_inverse(plan, data) != ADM_SUCCESS) return 1;  /* divides by 1024 */
    adm_plan_destroy(plan);

    double err = 0;
    for (size_t i = 0; i < 1024; ++i) {
        const double d = fabs(data[i].real - sin(0.01 * (double)i)) + fabs(data[i].imag);
        if (d > err) err = d;
    }
    if (err > 1e-10) {
        fprintf(stderr, "C round trip failed: err=%g\n", err);
        return 1;
    }
    return 0;
}
