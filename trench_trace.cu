// 粒子追跡。RF 全周期では粒子時計に応じて単位電圧応答を重ねる。
// CPU 版と同じ double 精度、真空セルによる補間、CFL、壁面障壁を使う。
__device__ int periodic_index(int i, int n) {
    int r = i % n;
    return r < 0 ? r + n : r;
}

__device__ int clip_index(int i, int n) {
    return i < 0 ? 0 : (i >= n ? n - 1 : i);
}

__device__ double periodic_position(double x, double length) {
    double r = fmod(x, length);
    return r < 0.0 ? r + length : r;
}

__device__ void interpolate(double x, double y, const double *ex, const double *ey,
                            const unsigned char *vac, int nx, int ny, double dx,
                            double &out_x, double &out_y) {
    double fx = x / dx - 0.5, fy = y / dx - 0.5;
    int i = (int)floor(fx), j = (int)floor(fy);
    double tx = fx - i, ty = fy - j;
    int cells[4] = {periodic_index(i, nx) * ny + clip_index(j, ny),
                    periodic_index(i + 1, nx) * ny + clip_index(j, ny),
                    periodic_index(i, nx) * ny + clip_index(j + 1, ny),
                    periodic_index(i + 1, nx) * ny + clip_index(j + 1, ny)};
    double weights[4] = {(1.0-tx)*(1.0-ty), tx*(1.0-ty), (1.0-tx)*ty, tx*ty};
    double sx = 0.0, sy = 0.0, sw = 0.0;
    for (int k = 0; k < 4; ++k) {
        int c = cells[k];
        // NumPy の F = (Ex*vac, Ey*vac, vac) と同じ演算順。
        sx += weights[k] * (vac[c] ? ex[c] : 0.0);
        sy += weights[k] * (vac[c] ? ey[c] : 0.0);
        sw += weights[k] * (double)vac[c];
    }
    sw = fmax(sw, 1e-12);
    out_x = sx / sw;
    out_y = sy / sw;
}

__device__ void wall_entry(double x0, double y0, double sx, double sy,
                           const unsigned char *vac, int nx, int ny, double dx,
                           int &cx, int &cy, int &vcx, int &vcy, bool &face_x) {
    int ix0 = min((int)(x0 / dx), nx - 1), iy0 = clip_index((int)(y0 / dx), ny);
    int dix = (int)floor((x0 + sx) / dx) - ix0;
    int diy = clip_index((int)floor((y0 + sy) / dx), ny) - iy0;
    double infinity = __longlong_as_double(0x7ff0000000000000LL);
    double fx = dix != 0 ? ((ix0 + (dix > 0)) * dx - x0) / sx : infinity;
    double fy = diy != 0 ? ((iy0 + (diy > 0)) * dx - y0) / sy : infinity;
    bool first_x = fx < fy;
    int ax = first_x ? periodic_index(ix0 + dix, nx) : ix0;
    int ay = first_x ? iy0 : iy0 + diy;
    bool use_last = dix != 0 && diy != 0 && vac[ax * ny + ay];
    cx = use_last ? periodic_index(ix0 + dix, nx) : ax;
    cy = use_last ? iy0 + diy : ay;
    vcx = use_last ? ax : ix0;
    vcy = use_last ? ay : iy0;
    face_x = use_last ? !first_x : first_x;
}

__device__ void rf_coefficients(double at, const double *wave, int n, double period,
                                double &vs, double &vm) {
    double f = periodic_position(at, period) / period * n;
    int i = ((int)floor(f)) % n, j = (i + 1) % n;
    double w = f - floor(f);
    vs = wave[2*i] * (1.0-w) + wave[2*j] * w;
    vm = wave[2*i+1] * (1.0-w) + wave[2*j+1] * w;
}

extern "C" __global__ void trace_particles(
    double *particles, const double *ex, const double *ey, const unsigned char *vac,
    const double *phi, const double *wface, signed char *status, int *hit_cells,
    double *endpoints, const int *slots, double *paths, int *path_lengths,
    const double *rf_ex, const double *rf_ey, const double *rf_phi, const double *wave, double *clocks,
    int n, int nx, int ny, int steps, int first_chunk, int barrier, int record, int tapered,
    int path_stride, double dx, double cfl, int n_phase, double period, double dt_max,
    double bound_vs, double bound_vm) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= n || status[id] != 3) return;
    double x = particles[id*5], y = particles[id*5+1];
    double vx = particles[id*5+2], vy = particles[id*5+3], qm = particles[id*5+4];
    double lx = nx * dx, ly = ny * dx;
    double clock = n_phase ? clocks[2*id] : 0.0, elapsed = n_phase ? clocks[2*id+1] : 0.0;
    int cells = nx * ny;
    int slot = record ? slots[id] : -1;
    int path_len = slot >= 0 ? path_lengths[slot] : 0;
    if (slot >= 0 && first_chunk) {
        paths[2 * (slot * path_stride)] = x;
        paths[2 * (slot * path_stride) + 1] = y;
    }
    double xp = x, yp = y;
    for (int k = 0; k < steps; ++k) {
        double ex_p, ey_p;
        interpolate(x, y, ex, ey, vac, nx, ny, dx, ex_p, ey_p);
        double ax = qm * ex_p, ay = qm * ey_p;
        double dt = cfl * dx / (hypot(vx, vy) + sqrt(2.0 * dx * hypot(ax, ay)) + 1.0);
        if (n_phase) {
            double ex_s, ey_s, ex_m, ey_m, vs, vm;
            interpolate(x, y, rf_ex, rf_ey, vac, nx, ny, dx, ex_s, ey_s);
            interpolate(x, y, rf_ex+cells, rf_ey+cells, vac, nx, ny, dx, ex_m, ey_m);
            double bx = fabs(ex_p) + bound_vs * fabs(ex_s) + bound_vm * fabs(ex_m);
            double by = fabs(ey_p) + bound_vs * fabs(ey_s) + bound_vm * fabs(ey_m);
            dt = fmin(dt_max, cfl * dx / (hypot(vx, vy) + sqrt(2.0 * dx * fabs(qm) * hypot(bx, by)) + 1.0));
            rf_coefficients(clock + dt/2, wave, n_phase, period, vs, vm);
            ex_p += vs * ex_s + vm * ex_m;
            ey_p += vs * ey_s + vm * ey_m;
            ax = qm * ex_p; ay = qm * ey_p;
            clock = periodic_position(clock + dt, period);
            elapsed += dt;
        }
        double vxn = vx + ax * dt, vyn = vy + ay * dt;
        double x0 = x, y0 = y;
        double sx = 0.5 * (vx + vxn) * dt, sy = 0.5 * (vy + vyn) * dt;
        x = periodic_position(x + sx, lx);
        y += sy;
        vx = vxn; vy = vyn;
        bool escaped = y >= ly;
        int ix = min((int)(x / dx), nx - 1), iy = clip_index((int)(y / dx), ny);
        bool hit = !escaped && !vac[ix * ny + iy];
        int cx = ix, cy = iy, vcx = 0, vcy = 0;
        bool face_x = false;
        // 階段の角は、終点が真空でも途中で固体面を横切る場合がある。
        bool corner_crossing = tapered && !hit && !escaped
            && ix != min((int)(x0 / dx), nx-1) && iy != clip_index((int)(y0 / dx), ny);
        if ((hit && (barrier || record || tapered)) || corner_crossing) {
            wall_entry(x0, y0, sx, sy, vac, nx, ny, dx, cx, cy, vcx, vcy, face_x);
            hit = !vac[cx*ny+cy];
            if (hit && barrier) {
                double vn = face_x ? vx : vy;
                double dphi = wface[cx*ny+cy] * (phi[cx*ny+cy] - phi[vcx*ny+vcy]);
                if (n_phase) {
                    double vs, vm;
                    rf_coefficients(clock, wave, n_phase, period, vs, vm);
                    int c = cx*ny+cy, v = vcx*ny+vcy;
                    dphi += wface[c] * (vs * (rf_phi[c]-rf_phi[v]) + vm * (rf_phi[cells+c]-rf_phi[cells+v]));
                }
                if (0.5 * vn * vn < qm * dphi) {
                    if (face_x) vx = -vx; else vy = -vy;
                    x = x0; y = y0;
                    hit = false;
                }
            }
            ix = cx; iy = cy;
        }
        xp = x; yp = y;
        if (hit) {
            status[id] = 1;
            hit_cells[id] = ix * ny + iy;
            if (record) {
                double fraction;
                if (face_x) {
                    double xb = (cx + (sx < 0.0)) * dx;
                    xb += nearbyint((x0 - xb) / lx) * lx;
                    fraction = (xb - x0) / sx;
                } else {
                    fraction = ((cy + (sy < 0.0)) * dx - y0) / sy;
                }
                fraction = fmin(fmax(fraction, 0.0), 1.0);
                if (n_phase) {
                    clock = periodic_position(clock - (1.0-fraction)*dt, period);
                    elapsed -= (1.0-fraction)*dt;
                }
                xp = periodic_position(x0 + fraction * sx, lx);
                yp = y0 + fraction * sy;
            }
        }
        if (escaped) {
            status[id] = 2;
            if (record) {
                double fraction = fmin(fmax((ly - y0) / sy, 0.0), 1.0);
                if (n_phase) {
                    clock = periodic_position(clock - (1.0-fraction)*dt, period);
                    elapsed -= (1.0-fraction)*dt;
                }
                xp = periodic_position(x0 + fraction * sx, lx);
                yp = ly;
            }
        }
        if (slot >= 0) {
            paths[2 * (slot * path_stride + path_len)] = xp;
            paths[2 * (slot * path_stride + path_len) + 1] = yp;
            ++path_len;
        }
        if (hit || escaped) break;
    }
    particles[id*5] = x; particles[id*5+1] = y;
    particles[id*5+2] = vx; particles[id*5+3] = vy;
    if (n_phase) { clocks[2*id] = clock; clocks[2*id+1] = elapsed; }
    if (record) {
        endpoints[2*id] = xp; endpoints[2*id+1] = yp;
        if (slot >= 0) path_lengths[slot] = path_len;
    }
}
