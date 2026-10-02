"""1D シースの CPU JIT カーネル。倍精度、CIC、Newton、leapfrog、衝突モデルを維持する。"""
import math

import numpy as np
from numba import njit


@njit(cache=True, nogil=True)
def solve_phi(phi, ni, Te, n_s, k_e, dy, phi_left, diagonal, rhs, delta):
    """厳密対角優位の Newton 行列を Thomas 法で解く。許容値・反復上限は参照版と同じ。"""
    phi[0] = phi_left
    m = len(phi)-2
    off = 1/dy**2
    for iteration in range(20):
        for i in range(m):
            j = i+1
            ex = math.exp(min(phi[j]/Te, 50.0))
            lap = -2*phi[j]
            if i > 0:
                lap += phi[j-1]
            if i < m-1:
                lap += phi[j+1]
            if i == 0:
                lap += phi[0]
            lap /= dy**2
            rhs[i] = -(lap+k_e*(ni[j]-n_s*ex))
            diagonal[i] = -2/dy**2-k_e*n_s*ex/Te
        for i in range(1, m):
            ratio = off/diagonal[i-1]
            diagonal[i] -= ratio*off
            rhs[i] -= ratio*rhs[i-1]
        delta[m-1] = rhs[m-1]/diagonal[m-1]
        for i in range(m-2, -1, -1):
            delta[i] = (rhs[i]-off*delta[i+1])/diagonal[i]
        largest = 0.0
        for i in range(m):
            phi[i+1] += delta[i]
            largest = max(largest, abs(delta[i]))
        if largest < 1e-5:
            return
    raise RuntimeError("1 次元シースの Poisson 方程式 (Newton 法) が収束しませんでした")


@njit(cache=True)
def elliptic_k(parameter):
    """完全楕円積分 K(m) を算術幾何平均で評価する (散乱に使う 0 <= m < 1)。"""
    a, b = 1.0, math.sqrt(1.0-parameter)
    for _ in range(30):
        an = .5*(a+b)
        b = math.sqrt(a*b)
        a = an
        if abs(a-b) <= 2e-16*a:
            break
    return math.pi/(2*a)


@njit(cache=True, nogil=True)
def scatter(vx, v, vz, ids, nc, rng, M, kT, beta_m, charge, beta0, nk_a):
    """参照版と同じ順序で Generator を消費し、対象粒子だけを散乱する。"""
    neutral = rng.normal(0.0, math.sqrt(kT/M), (nc, 3))
    beta = beta_m*np.sqrt(rng.random(nc))
    cx_draw = rng.random(nc)
    iso = np.empty(nc, np.int64)
    direct = np.empty(nc, np.int64)
    ni = nd = over = cx_count = 0
    for k in range(nc):
        if beta[k] < beta0:
            iso[ni] = k
            ni += 1
        else:
            direct[nd] = k
            nd += 1
    iso_c = 1-2*rng.random(ni)
    iso_ph = 2*math.pi*rng.random(ni)
    direct_ph = 2*math.pi*rng.random(nd)
    for subset in range(2):
        size = ni if subset == 0 else nd
        for h in range(size):
            k = iso[h] if subset == 0 else direct[h]
            i = ids[k]
            oldx, oldy, oldz = vx[i], v[i], vz[i]
            gx, gy, gz = neutral[k, 0]-oldx, neutral[k, 1]-oldy, neutral[k, 2]-oldz
            gm = math.sqrt(gx*gx+gy*gy+gz*gz)
            beta_ex = max(nk_a*(.25*M*gm**2/charge)**.25, 1.0)
            over += int(beta_ex > beta_m)
            cx = beta[k] < beta_ex and cx_draw[k] < .5
            if subset == 0:
                c = iso_c[h]
                s = math.sqrt(1-c*c)
                ph = iso_ph[h]
                vx[i] = .5*(oldx+neutral[k, 0])-.5*gm*s*math.cos(ph)
                v[i] = .5*(oldy+neutral[k, 1])-.5*gm*s*math.sin(ph)
                vz[i] = .5*(oldz+neutral[k, 2])-.5*gm*c
            else:
                e1 = math.sqrt(beta[k]**2+math.sqrt(beta[k]**4-1))
                chi = math.pi-2*math.sqrt(2)*beta[k]*elliptic_k(1/e1**4)/e1
                cosx, sinx = math.cos(chi), abs(math.sin(chi))
                cph, sph = math.cos(direct_ph[h]), math.sin(direct_ph[h])
                gp = max(math.sqrt(gy*gy+gz*gz), 1e-300)
                hx = gp*cph
                hy = -(gx*gy*cph+gm*gz*sph)/gp
                hz = -(gx*gz*cph-gm*gy*sph)/gp
                dx = .5*(gx*(1-cosx)+hx*sinx)
                dy = .5*(gy*(1-cosx)+hy*sinx)
                dz = .5*(gz*(1-cosx)+hz*sinx)
                if cx:
                    vx[i], v[i], vz[i] = neutral[k, 0]-dx, neutral[k, 1]-dy, neutral[k, 2]-dz
                    cx_count += 1
                else:
                    vx[i], v[i], vz[i] = oldx+dx, oldy+dy, oldz+dz
    return cx_count, over


@njit(cache=True, nogil=True)
def simulate(tg, Vg, phi, y0, v0, vx0, vz0, rng, Te, M, n_s, dy, dt, L, T,
             w, qm, k_e, warmup, n_steps, inj_rate, u_B, v_th, P_col, kT_gas,
             beta_m, charge, beta0, nk_a):
    """配列を再利用して全時間ステップを実行する。fastmath / 並列 RNG は使用しない。"""
    N = len(phi)-1
    capacity = len(y0)+int(math.ceil(inj_rate*n_steps))+4
    y, v, vx, vz = np.empty(capacity), np.empty(capacity), np.empty(capacity), np.empty(capacity)
    n = len(y0)
    y[:n], v[:n], vx[:n], vz[:n] = y0, v0, vx0, vz0
    old_v = np.empty(capacity)
    cells = np.empty(capacity, np.int64)
    weights = np.empty(capacity)
    collision_ids = np.empty(capacity, np.int64)
    recorded = np.empty((capacity, 3))
    lower, upper, ni, field = np.empty(N+1), np.empty(N+1), np.empty(N+1), np.empty(N+1)
    diagonal, rhs, delta = np.empty(N-1), np.empty(N-1), np.empty(N-1)
    n_hit = n_col = n_cx = n_over = 0
    inj_acc = 0.0
    for k in range(n_steps):
        tk = -warmup+k*dt
        lower[:] = 0.0
        upper[:] = 0.0
        for i in range(n):
            g = y[i]/dy
            j = min(int(g), N-1)
            f = g-j
            cells[i], weights[i] = j, f
            lower[j] += 1-f
            upper[j+1] += f
        for j in range(N+1):
            ni[j] = (lower[j]+upper[j])*(w/dy)
        ni[0] *= 2
        ni[N] *= 2
        solve_phi(phi, ni, Te, n_s, k_e, dy, -np.interp(tk % T, tg, Vg), diagonal, rhs, delta)
        for j in range(1, N):
            field[j] = -(phi[j+1]-phi[j-1])/(2*dy)
        field[0] = -(phi[1]-phi[0])/dy
        field[N] = -(phi[N]-phi[N-1])/dy
        for i in range(n):
            j, f = cells[i], weights[i]
            Ep = field[j]*(1-f)+field[j+1]*f
            old_v[i] = v[i]
            v[i] = v[i]+qm*Ep*dt
        if P_col > 0:
            nc = 0
            for i in range(n):
                if rng.random() < P_col:
                    collision_ids[nc] = i
                    nc += 1
            if nc:
                cx, over = scatter(vx, v, vz, collision_ids, nc, rng, M, kT_gas, beta_m, charge, beta0, nk_a)
                n_over += over
                if tk >= 0:
                    n_col += nc
                    n_cx += cx
        kept = 0
        for i in range(n):
            new_y = y[i]+v[i]*dt
            if new_y <= 0:
                if tk >= 0:
                    j, f = cells[i], weights[i]
                    vn = .5*(old_v[i]+v[i])
                    phi_y = phi[j]*(1-f)+phi[j+1]*f
                    vn2 = max(vn**2+2*qm*(phi_y-phi[0]), 0.0)
                    recorded[n_hit, 0] = math.sqrt(vn2)
                    recorded[n_hit, 1], recorded[n_hit, 2] = vx[i], vz[i]
                    n_hit += 1
            elif new_y < L:
                y[kept], v[kept], vx[kept], vz[kept] = new_y, v[i], vx[i], vz[i]
                kept += 1
        n = kept
        inj_acc += inj_rate
        n_new = int(inj_acc)
        inj_acc -= n_new
        if n_new:
            new_y = rng.uniform(0.0, 1.0, n_new)
            new_x = rng.normal(0.0, v_th, n_new)
            new_z = rng.normal(0.0, v_th, n_new)
            for i in range(n_new):
                y[n+i] = L-u_B*dt*new_y[i]
                v[n+i] = -u_B
                vx[n+i], vz[n+i] = new_x[i], new_z[i]
            n += n_new
    return recorded[:n_hit].copy(), n_col, n_cx, n_over
