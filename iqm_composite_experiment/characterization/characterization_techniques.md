# Mathematical description of the three gate-characterization techniques

## 0. Gate being characterized

All three experiments characterize the same **composite single-qubit gate** selected from `sequences.py`.
A sequence is specified by

$$
S=(\theta_{\mathrm{target}},\theta_c,\{\phi_k\}_{k=1}^{N}),
$$

where:

- $\theta_{\mathrm{target}}$ is the ideal net rotation angle,
- $\theta_c$ is the rotation angle of every constituent pulse,
- $\phi_k$ are the phases of the constituent pulses,
- $N$ is the pulse count.

The implemented composite gate is the serial product of those constituent PRX pulses,

$$
G_S = U(\theta_c,-\phi_N)\cdots U(\theta_c,-\phi_2)U(\theta_c,-\phi_1),
$$

where the minus sign comes from converting the phase convention used in `sequences.py` to the IQM PRX convention. In the experiments, amplitude and detuning errors can be injected into each constituent physical pulse before the full sequence is executed.

In the current `main.py`, the selected sequence is **H11a**. It has

$$
\theta_{\mathrm{target}}=\frac{\pi}{2},\qquad
\theta_c=\frac{\pi}{2},\qquad N=11,
$$

so the intended logical operation is an $R_x(\pi/2)$ gate implemented by 11 phased $\pi/2$ pulses.

The three characterization methods ask three different questions about this same $G_S$:

| Technique | Main question | Primary gate property returned |
|---|---|---|
| Process tomography | What affine map does the gate apply to the Bloch sphere? | Average gate fidelity, effective rotation angle/axis, non-unitary distortion |
| Transition-phase Ramsey | Does the gate give the transition the correct phase? | Transition-phase error from a Ramsey-fringe displacement |
| Interleaved randomized benchmarking | How much error does repeated use of this gate add on average? | RB gate decay and estimated gate infidelity |

---

# 1. Process tomography

## Circuit that is set up

The code prepares four input states

$$
|0\rangle,\quad |1\rangle,\quad |+x\rangle,\quad |+y\rangle,
$$

applies the composite gate $G_S$, and then measures each output state in the three Pauli bases

$$
X,\quad Y,\quad Z.
$$

Therefore, for every sequence/error setting, the experiment runs

$$
4\ \text{input states}\times 3\ \text{measurement bases}=12
$$

circuits per qubit.

Conceptually each circuit is

`prepare s` → $G_S$ → `basis rotation for b` → computational-basis measurement,

with $s\in\{0,1,+x,+y\}$ and $b\in\{x,y,z\}$.

The preparation pulses used by the implementation are:

- $|0\rangle$: no pulse,
- $|1\rangle$: calibrated $\pi$ pulse,
- $|+x\rangle$: calibrated $\pi/2$ PRX with the phase used in the code,
- $|+y\rangle$: calibrated $\pi/2$ PRX with the phase used in the code.

For the analysis basis:

- $Z$: direct readout,
- $X$: a calibrated $\pi/2$ analysis pulse before readout,
- $Y$: a calibrated $\pi/2$ analysis pulse before readout.

## What is measured

The hardware always returns the excited-state probability

$$
P_1(s,b)=P(\text{measure }1\mid s,b).
$$

After optional readout correction, it is converted to the corresponding Pauli expectation value through

$$
\langle \sigma_b\rangle_s = 1-2P_1(s,b).
$$

Thus, for every prepared input state $s$, the three measurements give an output Bloch vector

$$
\mathbf r_s=
\begin{pmatrix}
\langle X\rangle_s\\
\langle Y\rangle_s\\
\langle Z\rangle_s
\end{pmatrix}.
$$

## Mathematical reconstruction

The single-qubit channel is represented as an affine Bloch-sphere map

$$
\mathbf r_{\mathrm{out}}=M\mathbf r_{\mathrm{in}}+\mathbf t,
$$

where $M$ is a $3\times3$ real matrix and $\mathbf t$ is a translation vector.

From the measured output vectors, the implementation reconstructs

$$
\mathbf t=\frac{\mathbf r_0+\mathbf r_1}{2},
$$

and

$$
M=
\begin{bmatrix}
\mathbf r_{+x}-\mathbf t &
\mathbf r_{+y}-\mathbf t &
(\mathbf r_0-\mathbf r_1)/2
\end{bmatrix}.
$$

The corresponding Pauli-transfer matrix is

$$
R=
\begin{pmatrix}
1 & 0 & 0 & 0\\
t_x & & &\\
t_y & & M &\\
t_z & & &
\end{pmatrix}.
$$

The target PTM is the ideal rotation about $x$ by the sequence target angle,

$$
R_{\mathrm{target}}=
1\oplus R_x(\theta_{\mathrm{target}}).
$$

The code reports the average gate fidelity as

$$
F_{\mathrm{avg}}
=
\frac{\operatorname{Tr}(R_{\mathrm{target}}^T R)+2}{6},
$$

and the corresponding infidelity

$$
r_{\mathrm{tomo}}=1-F_{\mathrm{avg}}.
$$

It also performs an SVD of the $3\times3$ Bloch block,

$$
M=U\Sigma V^T,
$$

and takes the nearest proper rotation

$$
R_{\mathrm{near}}=U D V^T,
\qquad
D=\operatorname{diag}(1,1,\det(UV^T)).
$$

From $R_{\mathrm{near}}$, it extracts an effective rotation vector, and therefore:

- effective rotation angle $\theta_{\mathrm{eff}}$,
- angle error $\theta_{\mathrm{eff}}-\theta_{\mathrm{target}}$ wrapped to $[-\pi,\pi)$,
- rotation-axis components $(n_x,n_y,n_z)$,
- axis azimuth/tilt/distance from $+x$.

The singular values of $M$ and the residual

$$
\|M-R_{\mathrm{near}}\|_F
$$

quantify how far the reconstructed action is from a pure rotation.

## What property of the sequence does tomography give us?

Tomography answers: **what transformation does this composite sequence actually implement?**

For a sequence such as H11a, it tells us whether the intended $R_x(\pi/2)$ is realized with:

1. the correct overall angle,
2. the correct rotation axis,
3. high average gate fidelity,
4. little translation or contraction/distortion of the Bloch sphere.

So tomography is the most complete of the three methods: it separates coherent geometric errors (wrong angle/axis) from non-unitary distortion visible in the reconstructed Bloch map.

---

# 2. Transition-phase Ramsey characterization

## Circuit that is set up

The Ramsey experiment scans the phase $\varphi$ of a calibrated $\pi/2$ analysis pulse.

For an $R_x(\pi/2)$ target sequence such as H11a, the circuit is

$$
|0\rangle
\;\xrightarrow{\;G_S\;}\;
\xrightarrow{\;R_{\varphi}(\pi/2)\;}\;
\text{measure }Z.
$$

For a target $R_x(\pi)$ sequence, the code first adds a calibrated $\pi/2$ pulse:

$$
|0\rangle
\;\xrightarrow{\;R_x(\pi/2)\;}\;
\xrightarrow{\;G_S\;}\;
\xrightarrow{\;R_{\varphi}(\pi/2)\;}\;
\text{measure }Z.
$$

The analysis phase $\varphi$ is swept over the configured Ramsey phases.

## What is measured

For every $\varphi$, the experiment measures

$$
P_1(\varphi).
$$

It fits the fringe to the linear sinusoidal model

$$
P_1(\varphi)
=
c_0+c_c\cos\varphi+c_s\sin\varphi.
$$

Equivalently,

$$
P_1(\varphi)=c_0+A\cos(\varphi-\delta),
$$

with

$$
A=\sqrt{c_c^2+c_s^2},
\qquad
\delta=\operatorname{atan2}(c_s,c_c).
$$

The fit therefore provides the fringe offset, amplitude, visibility, phase, and fit residual.

## Mathematical gate quantity extracted

The same ideal circuit is evaluated for the ideal target gate $R_x(\theta_{\mathrm{target}})$, giving an ideal fringe phase $\delta_{\mathrm{ideal}}$.

The measured fringe displacement is

$$
\Delta\delta
=
\operatorname{wrap}(\delta_{\mathrm{meas}}-\delta_{\mathrm{ideal}}).
$$

The implementation reports the transition-phase error as

$$
\epsilon_{\phi}
=
-\frac{\Delta\delta}{d},
$$

where

$$
d=
\begin{cases}
2, & \theta_{\mathrm{target}}=\pi,\\
1, & \text{otherwise}.
\end{cases}
$$

The minus sign is explicitly included because the IQM analysis-pulse phase convention is opposite to the paper convention used for the sequence phases.

For **H11a**, $\theta_{\mathrm{target}}=\pi/2$, so $d=1$ and

$$
\epsilon_{\phi}=-\Delta\delta.
$$

The code only reports this phase error when the measured Ramsey fringe is sufficiently larger than its estimated shot-noise floor.

## What property of the sequence does Ramsey give us?

This experiment isolates the **transition-phase error** of the composite gate.

In practical terms, it tells us whether the sequence performs the correct logical rotation but leaves the qubit with an incorrect azimuthal phase. A sequence can therefore have a reasonably correct population transfer while still showing a displaced Ramsey fringe; this method is designed to make that coherent phase error directly visible.

For H11a, the quantity of interest is specifically the phase error associated with the composite realization of the intended $R_x(\pi/2)$.

---

# 3. Interleaved randomized benchmarking (IRB)

## Circuit that is set up

The code generates random single-qubit Clifford sequences of length $m$. For each random sequence it runs two experiments.

### Reference RB

$$
|0\rangle
\xrightarrow{C_1}
\xrightarrow{C_2}
\cdots
\xrightarrow{C_m}
\xrightarrow{C_{\mathrm{rec}}}
\text{measure }Z.
$$

The recovery Clifford $C_{\mathrm{rec}}$ is chosen so that the ideal net operation is the identity.

### Interleaved RB

The composite gate under test is inserted after every random Clifford:

$$
|0\rangle
\xrightarrow{C_1}
\xrightarrow{G_S}
\xrightarrow{C_2}
\xrightarrow{G_S}
\cdots
\xrightarrow{C_m}
\xrightarrow{G_S}
\xrightarrow{C_{\mathrm{rec}}^{(G)}}
\text{measure }Z.
$$

The interleaved recovery is computed using the **ideal target unitary** of the selected sequence. Consequently, the selected target must be a single-qubit Clifford; the $R_x(\pi/2)$ target of H11a satisfies this condition.

## What is measured

The hardware measures $P_1$, but RB converts this to the ground-state survival probability

$$
S(m)=P_0(m)=1-P_1(m).
$$

For every sequence length $m$, the survival probabilities are averaged over random samples separately for the reference and interleaved experiments.

Each averaged decay curve is fitted to

$$
S(m)=A p^m+B.
$$

This gives two decay parameters:

$$
p_{\mathrm{ref}}
\quad\text{and}\quad
p_{\mathrm{int}}.
$$

## Mathematical gate quantity extracted

The interleaved-gate decay is estimated by taking the ratio

$$
p_G=\frac{p_{\mathrm{int}}}{p_{\mathrm{ref}}}.
$$

For a single qubit, the implementation reports

$$
r_{\mathrm{RB}}
=\frac{1-p_G}{2}.
$$

This is stored as `rb_infidelity`.

The reference division is important: the reference curve estimates the background error of the random Clifford sequence, whereas the faster interleaved decay contains that background plus the additional error from repeatedly applying $G_S$.

## What property of the sequence does IRB give us?

IRB answers: **how much average error does one use of this composite gate add when embedded in long randomized circuits?**

For H11a, it gives an operational estimate of the average infidelity of the 11-pulse composite $R_x(\pi/2)$ gate. Unlike tomography, RB does not reconstruct the error direction or distinguish angle error from axis error. Instead, it compresses the accumulated effect into a decay/inﬁdelity number and is comparatively insensitive to fixed state-preparation and measurement offsets because those largely enter the fitted constants $A$ and $B$.

---

# 4. How the three results fit together

The same composite sequence can fail in different ways, and the three experiments expose different projections of that failure.

### Tomography: *What gate did we actually implement?*

Reconstructs the affine Bloch/PTM map and reports

$$
F_{\mathrm{avg}},\quad
\theta_{\mathrm{eff}},\quad
\Delta\theta,\quad
\mathbf n,\quad
\Sigma(M),\quad
\mathbf t.
$$

This is the best method here for diagnosing **angle error, axis error, and non-unitary deformation**.

### Ramsey: *Did the gate get the phase right?*

Measures a phase-scanned interference fringe and reports

$$
\epsilon_{\phi}.
$$

This is a targeted measurement of the gate's **transition/azimuthal phase error**.

### Interleaved RB: *How damaging is the gate in repeated computation?*

Measures survival decay versus random-sequence length and reports

$$
p_G,\qquad r_{\mathrm{RB}}=\frac{1-p_G}{2}.
$$

This is an operational estimate of the gate's **average error per application in randomized circuits**.

A useful interpretation is therefore:

| Result | Interpretation |
|---|---|
| Tomography angle/axis error is large | The composite sequence implements the wrong coherent rotation |
| Ramsey transition-phase error is large | The gate introduces a coherent phase/azimuth error |
| Tomography singular values/residual are poor | The output is not well described by a pure unitary rotation |
| RB infidelity is large | Repeated use of the gate causes substantial average computational error |
| Tomography looks good but RB is worse | Small errors may accumulate under repetition, or noise may vary between applications |
| RB is good but a small tomography bias remains | The coherent deviation may be small enough that the randomized average is still favorable |

---

# 5. Relation to the amplitude-error and detuning sweeps

The gate under test is not only characterized at its nominal calibration. The implementation can alter each constituent PRX pulse before constructing the composite sequence:

- **amplitude error** rescales the physical I/Q pulse amplitudes by $1+\epsilon_A$,
- **detuning** offsets the pulse modulation frequency by the requested detuning.

The three techniques can therefore be repeated as functions of

$$
\epsilon_A\quad\text{and}\quad\Delta f.
$$

This turns the characterization into a robustness test of the sequence itself:

$$
G_S(\epsilon_A,\Delta f)
\longrightarrow
\begin{cases}
\text{tomography metrics},\\
\text{transition-phase error},\\
\text{RB infidelity}.
\end{cases}
$$

For a composite sequence such as H11a, the central experimental question is not only whether it realizes $R_x(\pi/2)$ at the nominal point, but **how slowly these three classes of error grow when the constituent pulses are systematically miscalibrated or detuned**.

---

# 6. One-line summary

For each sequence in `sequences.py`, the code characterizes the same composite gate in three complementary ways: **tomography reconstructs its full effective transformation, Ramsey measures its coherent transition-phase error, and interleaved RB measures its average accumulated gate infidelity.**

## Source mapping

This note follows the implementation in:

- `sequences(1).py`: `SequenceSpec` and the built-in composite sequences, including H11a.
- `common(2).py`: physical construction of the composite gate and error injection.
- `tomography.py`: 4-state × 3-basis acquisition, PTM reconstruction, fidelity, effective rotation, and singular-value diagnostics.
- `ramsey.py`: phase-scanned analysis circuit, sinusoidal fringe fit, and transition-phase error.
- `rb.py`: reference/interleaved Clifford circuits, exponential decay fits, and RB infidelity.
- `main.py`: selects H11a and executes all three characterization blocks.
