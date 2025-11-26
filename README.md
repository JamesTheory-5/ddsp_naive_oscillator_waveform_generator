## ✅ DDSP MODULE SPEC — `ddsp_naive_oscillator_waveforms_generator`

**MODULE NAME:**

`ddsp_naive_oscillator_waveforms_generator`

---

**DESCRIPTION:**

A fully differentiable, phase-driven **naive oscillator waveform generator** in GDSP core style.

This module:

* Takes an **external normalized phase** (typically from a phasor core).

* Generates, in parallel, a **bundle of naive analytic waveforms**:

  * sine
  * saw (ramp up)
  * ramp down
  * square
  * pulse (same as square but PW-smoothed)
  * rectangle (two-level with pulse-width)
  * triangle
  * parabolic
  * trapezoid (fixed shape)

* Shares a single **amplitude smoothing** state and a **pulse-width smoothing** state.

* Does **not** own frequency or perform phase accumulation.

* Is pure functional JAX, jit/grad/vmap/scan-safe, with tuple-only state.

* Returns all waveforms as a single array `y[..., num_waveforms]`.

---

**INPUTS (per tick):**

* `phase` : normalized phase (any real; internally wrapped to `[0,1)`).
* `dt` : phase increment per sample (cycles/sample). Included for API symmetry; unused by this naive generator.
* `amp_target` : target amplitude for all waveforms (in `params`).
* `amp_smooth_coef` : amplitude smoothing coefficient α ∈ `[0,1)` (in `params`).
* `pw_target` : target pulse width ∈ `(0,1)` used by square/pulse/rectangle (in `params`).
* `pw_smooth_coef` : pulse-width smoothing coefficient α_pw ∈ `[0,1)` (in `params`).
* `bias` : DC bias added after amplitude (in `params`).
* `dist_amt` : distortion blend amount ∈ `[0,1]`, 0 = pure waveform, 1 = `tanh(waveform)` (in `params`).
* `clip_flag` : ∈ `[0,1]`, 0 = off, 1 = soft clip final output with `tanh`.

`phase` and `dt` are passed as arguments to `*_tick` / `*_process`; everything else lives in the `params` tuple.

---

**OUTPUTS:**

* `y` : multi-waveform output array with shape `(..., 9)`, in order:

  0. sine
  1. saw (ramp up)
  2. ramp down
  3. square
  4. pulse (same as square)
  5. rectangle
  6. triangle
  7. parabolic
  8. trapezoid

* `new_state` : updated state tuple.

---

**STATE VARIABLES (tuple):**

```python
(
    amp_smooth_vec,   # shape (1,), smoothed amplitude
    pw_smooth_vec,    # shape (1,), smoothed pulse width
)
```

* `amp_smooth_vec[0]` is the smoothed amplitude.
* `pw_smooth_vec[0]` is the smoothed pulse width.
* Length-1 vectors are used so we can update state via `lax.dynamic_update_slice`.

---

**EQUATIONS / MATH:**

Let:

* `a_target = amp_target`
* `α = amp_smooth_coef`
* `pw_target = pw_target`
* `α_pw = pw_smooth_coef`
* `A[n] = amp_smooth[n]`
* `PW[n] = pw_smooth[n]`

1. **Amplitude smoothing:**

[
A[n+1] = a_{\text{target}} + \alpha \cdot \left(A[n] - a_{\text{target}}\right)
]

2. **Pulse width smoothing:**

[
PW[n+1] = pw_{\text{target}} + \alpha_{\text{pw}} \cdot \left(PW[n] - pw_{\text{target}}\right)
]

3. **Phase wrapping:**

Given arbitrary real `phase[n]`, normalize to `[0,1)`:

[
p[n] = \text{mod}(\text{phase}[n], 1.0)
]

(using `jnp.mod`)

4. **Naive waveform definitions:**

Let `x = 2p - 1`.

* **Sine:**

  [
  y_{\text{sine}}[n] = \sin(2\pi p[n])
  ]

* **Saw (ramp up):**

  [
  y_{\text{saw}}[n] = 2 p[n] - 1
  ]

* **Ramp down:**

  [
  y_{\text{ramp_down}}[n] = -y_{\text{saw}}[n]
  ]

* **Square (and Pulse):**

  [
  y_{\text{square}}[n] =
  \begin{cases}
  +1, & p[n] < PW[n+1] \
  -1, & \text{otherwise}
  \end{cases}
  ]

  implemented with `jnp.where`.

  Pulse is the same naive definition here.

* **Rectangle:**

  Two-level wave with asymmetric high/low:

  [
  y_{\text{rect}}[n] =
  \begin{cases}
  +1, & p[n] < PW[n+1] \
  -0.5, & \text{otherwise}
  \end{cases}
  ]

* **Triangle:**

  A symmetric -1→+1→-1 shape:

  [
  y_{\text{tri}}[n] = 2 \cdot \left|2p[n] - 1\right| - 1
  ]

* **Parabolic:**

  Simple centered parabola:

  [
  x[n] = 2p[n] - 1
  ]
  [
  y_{\text{parab}}[n] = 2 \cdot \left(1 - x[n]^2\right) - 1
  ]

* **Trapezoid (fixed shape):**

  Let `a = 0.25`, `b = 0.75`.
  Three segments: rise, high, fall.

  Masks:

  [
  m_1 = [p < a], \quad
  m_2 = [a \le p < b], \quad
  m_3 = [p \ge b]
  ]

  Segment outputs:

  * Rising from -1 to +1 over `[0, a]`:

    [
    y_1 = -1 + \frac{2}{a} p
    ]

  * High plateau:

    [
    y_2 = 1
    ]

  * Falling from +1 to -1 over `[b, 1]`:

    [
    y_3 = 1 - \frac{2}{1-b} (p - b)
    ]

  Blend with masks (implemented via `jnp.where` / arithmetic):

  [
  y_{\text{trap}} = m_1 y_1 + m_2 y_2 + m_3 y_3
  ]

5. **Distortion (shared for each waveform):**

For each naive waveform `w`:

[
w_{\text{shaped}} = (1 - d) \cdot w + d \cdot \tanh(w)
]

where `d = dist_amt`.

6. **Amplitude and bias:**

[
w_{\text{amp}} = A[n+1] \cdot w_{\text{shaped}} + \text{bias}
]

7. **Soft clipping (optional):**

[
w_{\text{out}} = (1 - c) \cdot w_{\text{amp}} + c \cdot \tanh(w_{\text{amp}})
]

where `c = clip_flag`.

All nine waveforms are stacked into a final vector:

[
\mathbf{y}[n] =
\left[
y_{\text{sine}}, y_{\text{saw}}, y_{\text{ramp_down}},
y_{\text{square}}, y_{\text{pulse}}, y_{\text{rect}},
y_{\text{tri}}, y_{\text{parab}}, y_{\text{trap}}
\right]
]

---

**NOTES / CONSTRAINTS:**

* No phase or frequency state; phase is fully external.
* `amp_smooth_coef`, `pw_smooth_coef` ∈ `[0,1)` for stability.
* `pw_target` kept in `(0,1)` by caller.
* `dist_amt` and `clip_flag` are treated as continuous blends (`[0,1]`), but intended as knobs.
* State updates use `lax.dynamic_update_slice`.
* No Python conditionals or loops inside the jitted tick/scan; all control via `jnp.where` and arithmetic masks.
* No `jnp.arange` or `jnp.zeros` inside jitted code; only in the `__main__` section.

---

## 🐍 Full Python Module: `ddsp_naive_oscillator_waveforms_generator.py`

```python
# -----------------------------------------------------------------------------
# ddsp_naive_oscillator_waveforms_generator.py
#
# GDSP-style naive oscillator waveform generator in pure JAX.
#
# Behavior:
#   - Consumes an external phase (normalized, typically in [0,1)).
#   - Generates multiple naive analytic waveforms in parallel:
#       0: sine
#       1: saw (ramp up)
#       2: ramp down
#       3: square
#       4: pulse (same as square, PW controlled)
#       5: rectangle (asymmetric high/low)
#       6: triangle
#       7: parabolic
#       8: trapezoid (fixed rise/high/fall)
#
#   - Shares amplitude and pulse-width smoothing states across all waveforms.
#   - Applies optional distortion, bias, and soft clipping.
#
# Design:
#   - Pure functional JAX, jit/grad/vmap/scan-safe.
#   - No internal phase/frequency; phase is provided by an external phasor.
#   - State is a tuple of arrays:
#         (amp_smooth_vec, pw_smooth_vec)
#     where each has shape (1,).
#   - Uses lax.scan for buffer processing.
#   - No Python branching in jitted paths; all control is via jnp.where and masks.
# -----------------------------------------------------------------------------


from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax


# -----------------------------------------------------------------------------
# INIT
# -----------------------------------------------------------------------------


def ddsp_naive_oscillator_waveforms_generator_init(
    amp_init=1.0,
    amp_target=None,
    amp_smooth_coef=0.0,
    pw_init=0.5,
    pw_target=None,
    pw_smooth_coef=0.0,
    bias=0.0,
    dist_amt=0.0,
    clip_flag=0.0,
    dtype=jnp.float32,
):
    """
    Initialize the naive oscillator waveform generator.

    Args
    ----
    amp_init        : initial smoothed amplitude value.
    amp_target      : target amplitude; if None, defaults to amp_init.
    amp_smooth_coef : amplitude smoothing coefficient in [0,1).
    pw_init         : initial smoothed pulse width in (0,1).
    pw_target       : target pulse width; if None, defaults to pw_init.
    pw_smooth_coef  : pulse width smoothing coefficient in [0,1).
    bias            : DC bias added after amplitude.
    dist_amt        : distortion blend amount in [0,1].
                      0 = pure waveform, 1 = tanh(waveform).
    clip_flag       : 0.0 = no clipping, 1.0 = soft clip with tanh.
    dtype           : JAX dtype, e.g. jnp.float32.

    Returns
    -------
    state  : (amp_smooth_vec, pw_smooth_vec)
    params : (amp_target, amp_smooth_coef,
              pw_target, pw_smooth_coef,
              bias, dist_amt, clip_flag)
    """
    if amp_target is None:
        amp_target = amp_init
    if pw_target is None:
        pw_target = pw_init

    amp_init_arr = jnp.asarray(amp_init, dtype=dtype)
    amp_smooth_vec = jnp.reshape(amp_init_arr, (1,))

    pw_init_arr = jnp.asarray(pw_init, dtype=dtype)
    pw_smooth_vec = jnp.reshape(pw_init_arr, (1,))

    amp_target = jnp.asarray(amp_target, dtype=dtype)
    amp_smooth_coef = jnp.asarray(amp_smooth_coef, dtype=dtype)
    pw_target = jnp.asarray(pw_target, dtype=dtype)
    pw_smooth_coef = jnp.asarray(pw_smooth_coef, dtype=dtype)
    bias = jnp.asarray(bias, dtype=dtype)
    dist_amt = jnp.asarray(dist_amt, dtype=dtype)
    clip_flag = jnp.asarray(clip_flag, dtype=dtype)

    state = (amp_smooth_vec, pw_smooth_vec)
    params = (
        amp_target,
        amp_smooth_coef,
        pw_target,
        pw_smooth_coef,
        bias,
        dist_amt,
        clip_flag,
    )
    return state, params


# -----------------------------------------------------------------------------
# UPDATE STATE (optional control-rate update)
# -----------------------------------------------------------------------------


def ddsp_naive_oscillator_waveforms_generator_update_state(state, params):
    """
    Optional control-rate state update (e.g., once per block instead of per sample).

    Applies one step of amplitude and pulse-width smoothing using the same
    rules as in the sample-rate tick.
    """
    (amp_smooth_vec, pw_smooth_vec) = state
    (
        amp_target,
        amp_smooth_coef,
        pw_target,
        pw_smooth_coef,
        bias,
        dist_amt,
        clip_flag,
    ) = params

    amp_smooth = amp_smooth_vec[0]
    pw_smooth = pw_smooth_vec[0]

    amp_smooth_new = amp_target + amp_smooth_coef * (amp_smooth - amp_target)
    pw_smooth_new = pw_target + pw_smooth_coef * (pw_smooth - pw_target)

    amp_smooth_vec_new = lax.dynamic_update_slice(
        amp_smooth_vec,
        jnp.reshape(amp_smooth_new, (1,)),
        (0,),
    )
    pw_smooth_vec_new = lax.dynamic_update_slice(
        pw_smooth_vec,
        jnp.reshape(pw_smooth_new, (1,)),
        (0,),
    )

    return (amp_smooth_vec_new, pw_smooth_vec_new)


# -----------------------------------------------------------------------------
# TICK
# -----------------------------------------------------------------------------


def ddsp_naive_oscillator_waveforms_generator_tick(phase, dt, state, params):
    """
    Single-sample naive waveform generator tick.

    Args
    ----
    phase : scalar or array; normalized phase (any real, wrapped to [0,1)).
    dt    : scalar or array; phase increment (unused here).
    state : (amp_smooth_vec, pw_smooth_vec)
    params: (amp_target, amp_smooth_coef,
             pw_target, pw_smooth_coef,
             bias, dist_amt, clip_flag)

    Returns
    -------
    y          : array with shape (..., 9) containing:
                 [sine, saw, ramp_down, square, pulse, rect,
                  triangle, parabolic, trapezoid]
    new_state  : updated (amp_smooth_vec, pw_smooth_vec)
    """
    del dt  # unused, kept for API symmetry

    (amp_smooth_vec, pw_smooth_vec) = state
    (
        amp_target,
        amp_smooth_coef,
        pw_target,
        pw_smooth_coef,
        bias,
        dist_amt,
        clip_flag,
    ) = params

    amp_smooth = amp_smooth_vec[0]
    pw_smooth = pw_smooth_vec[0]

    # Update amplitude and pulse width via one-pole smoothing
    amp_smooth_new = amp_target + amp_smooth_coef * (amp_smooth - amp_target)
    pw_smooth_new = pw_target + pw_smooth_coef * (pw_smooth - pw_target)

    amp_smooth_vec_new = lax.dynamic_update_slice(
        amp_smooth_vec,
        jnp.reshape(amp_smooth_new, (1,)),
        (0,),
    )
    pw_smooth_vec_new = lax.dynamic_update_slice(
        pw_smooth_vec,
        jnp.reshape(pw_smooth_new, (1,)),
        (0,),
    )

    # Wrap phase to [0,1)
    one = jnp.asarray(1.0, dtype=phase.dtype)
    phase_wrapped = jnp.mod(phase, one)

    # Precompute helpers
    two_pi = 2.0 * jnp.pi
    x = 2.0 * phase_wrapped - 1.0

    # 0: sine
    sine = jnp.sin(two_pi * phase_wrapped)

    # 1: saw (ramp up)
    saw = x  # 2p - 1

    # 2: ramp down
    ramp_down = -saw

    # 3 & 4: square / pulse (same naive shape)
    square = jnp.where(phase_wrapped < pw_smooth_new, 1.0, -1.0)
    pulse = square

    # 5: rectangle (asymmetric high/low)
    rect = jnp.where(phase_wrapped < pw_smooth_new, 1.0, -0.5)

    # 6: triangle
    triangle = 2.0 * jnp.abs(2.0 * phase_wrapped - 1.0) - 1.0

    # 7: parabolic
    parabola = 2.0 * (1.0 - x * x) - 1.0

    # 8: trapezoid (fixed shape)
    a = jnp.asarray(0.25, dtype=phase.dtype)
    b = jnp.asarray(0.75, dtype=phase.dtype)

    # Mask definitions
    # m1: p < a
    # m2: a <= p < b
    # m3: p >= b
    m1 = (phase_wrapped < a).astype(phase_wrapped.dtype)
    m2 = ((phase_wrapped >= a) & (phase_wrapped < b)).astype(phase_wrapped.dtype)
    m3 = (phase_wrapped >= b).astype(phase_wrapped.dtype)

    # Rising edge from -1 to +1 over [0, a]
    # y1(p=0) = -1, y1(p=a) = +1
    # slope = 2 / a
    y1 = -1.0 + (2.0 / a) * phase_wrapped

    # Plateau at +1
    y2 = jnp.asarray(1.0, dtype=phase_wrapped.dtype)

    # Falling edge from +1 to -1 over [b, 1]
    # y3(p=b) = +1, y3(p=1) = -1
    # slope = -2 / (1 - b)
    denom = (1.0 - b)
    y3 = 1.0 - (2.0 / denom) * (phase_wrapped - b)

    trapezoid = m1 * y1 + m2 * y2 + m3 * y3

    # Stack raw waveforms
    waveforms = jnp.stack(
        [
            sine,
            saw,
            ramp_down,
            square,
            pulse,
            rect,
            triangle,
            parabola,
            trapezoid,
        ],
        axis=-1,
    )

    # Distortion blend
    waveforms_shaped = (1.0 - dist_amt) * waveforms + dist_amt * jnp.tanh(waveforms)

    # Amplitude and bias
    waveforms_amp = amp_smooth_new * waveforms_shaped + bias

    # Soft clipping
    y = (1.0 - clip_flag) * waveforms_amp + clip_flag * jnp.tanh(waveforms_amp)

    new_state = (amp_smooth_vec_new, pw_smooth_vec_new)
    return y, new_state


# -----------------------------------------------------------------------------
# PROCESS (buffer wrapper using lax.scan)
# -----------------------------------------------------------------------------


def ddsp_naive_oscillator_waveforms_generator_process(
    phase_buf,
    dt_buf,
    state,
    params,
):
    """
    Process a buffer of phases through the naive waveform generator.

    Args
    ----
    phase_buf : [T, ...] array of phases.
    dt_buf    : [T, ...] array of phase increments (unused here).
    state     : (amp_smooth_vec, pw_smooth_vec)
    params    : (amp_target, amp_smooth_coef,
                 pw_target, pw_smooth_coef,
                 bias, dist_amt, clip_flag)

    Returns
    -------
    y_buf      : [T, ..., 9] array of waveforms.
    final_state: final (amp_smooth_vec, pw_smooth_vec)
    """

    def _scan_fn(carry, inputs):
        st = carry
        ph_t, dt_t = inputs
        y_t, st_new = ddsp_naive_oscillator_waveforms_generator_tick(
            ph_t,
            dt_t,
            st,
            params,
        )
        return st_new, y_t

    final_state, y_buf = lax.scan(
        _scan_fn,
        state,
        (phase_buf, dt_buf),
    )
    return y_buf, final_state


# -----------------------------------------------------------------------------
# Smoke test / plotting / listening example
# -----------------------------------------------------------------------------


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    try:
        import sounddevice as sd
        HAVE_SD = True
    except Exception:
        HAVE_SD = False

    # Parameters
    sample_rate = 48000
    duration_s = 1.0
    freq_hz = 220.0

    num_samples = int(sample_rate * duration_s)
    n = np.arange(num_samples, dtype=np.float32)

    # Normalized frequency (cycles/sample)
    freq_norm = freq_hz / float(sample_rate)

    # Build phase buffer: phase[n] = (freq_norm * n) mod 1
    phase_np = np.mod(freq_norm * n, 1.0).astype(np.float32)
    dt_np = np.full_like(phase_np, freq_norm, dtype=np.float32)

    phase_buf = jnp.asarray(phase_np)
    dt_buf = jnp.asarray(dt_np)

    # Initialize module
    state, params = ddsp_naive_oscillator_waveforms_generator_init(
        amp_init=0.0,
        amp_target=0.8,
        amp_smooth_coef=0.99,
        pw_init=0.5,
        pw_target=0.25,
        pw_smooth_coef=0.98,
        bias=0.0,
        dist_amt=0.0,
        clip_flag=0.0,
        dtype=jnp.float32,
    )

    # JIT the process
    process_jit = jax.jit(ddsp_naive_oscillator_waveforms_generator_process)

    # Run
    y_buf, final_state = process_jit(phase_buf, dt_buf, state, params)

    y = np.array(y_buf)  # [T, 9]

    print("ddsp_naive_oscillator_waveforms_generator smoke test:")
    print("  Output shape:", y.shape)
    print("  Wave 0 (sine) mean/std:", float(y[:, 0].mean()), float(y[:, 0].std()))

    # Plot first 1000 samples of all waveforms
    num_plot = min(1000, num_samples)
    t_ms = (n[:num_plot] / sample_rate) * 1000.0

    names = [
        "sine",
        "saw",
        "ramp_down",
        "square",
        "pulse",
        "rect",
        "triangle",
        "parabolic",
        "trapezoid",
    ]

    fig, axes = plt.subplots(9, 1, figsize=(10, 12), sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(t_ms, y[:num_plot, i])
        ax.set_ylabel(names[i])
        ax.grid(True)
    axes[-1].set_xlabel("Time (ms)")
    fig.suptitle("ddsp_naive_oscillator_waveforms_generator output (first ~1000 samples)")
    plt.tight_layout()
    plt.show()

    # Optional: listen to one waveform (e.g. sine, index 0)
    if HAVE_SD:
        print("Playing sine waveform (wave 0)...")
        sd.play(y[:, 0], samplerate=sample_rate)
        sd.wait()
        print("Done.")
    else:
        print("sounddevice not available; skipping audio playback.")
```

---

If you’d like, next I can:

* Split this generator into individual modules like `ddsp_naive_sine_from_phase.py`, `ddsp_naive_triangle_from_phase.py`, etc., or
* Build a **phasor-driven version** wired directly to your `phasor_core`, or
* Add a **polygonal / breakpoint-based** oscillator module with N control points.

```
