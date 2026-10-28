# Positive control for check (e): one instruction per ISA tier boundary. scripts/check_dispatch.sh
# assembles it with `as` and runs scripts/check_xed.py over it: under the floor allow-list the
# v2..v4 instructions must fire, under the v3 list the AVX-512 instructions must fire.
.intel_syntax noprefix
.text
f:
movsxd rax, ecx
addsubpd xmm0, xmm1
cmpxchg16b [rdi]
lahf
pshufb xmm0, xmm1
popcnt rax, rcx
lzcnt rax, rcx
tzcnt rax, rcx
xgetbv
vpaddd ymm0, ymm1, ymm2
vpaddq zmm0, zmm1, zmm2
ptest xmm0, xmm1
pcmpistri xmm0, xmm1, 0
movbe rax, qword ptr [rdi]
andn rax, rbx, rcx
bzhi rax, rbx, rcx
vcvtph2ps ymm0, xmm1
vfmadd231pd ymm0, ymm1, ymm2
vpgatherdd ymm0, [rdi+ymm1*4], ymm2
vpaddq xmm0{k1}, xmm1, xmm2
vpmullq zmm0, zmm1, zmm2
vplzcntd zmm0, zmm1
kmovw k1, eax
.att_syntax noprefix
ret
