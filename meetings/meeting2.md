# Intermediate Meeting November 2024

## Implementing Augmented Lagrangian method

* solve discretised equation

```math
\begin{bmatrix}
T & -G \\
-G^T & O
\end{bmatrix}
\begin{bmatrix}
v \\
p
\end{bmatrix}
= 
\begin{bmatrix}
f \\ 
0
\end{bmatrix}
```

* via iteration

  > **while** not converged **do**
  >
  >> solve $(T + \gamma G G^T) v = f + G p$
  >>
  >> set $p = p - \gamma G^T v$

* using CG for the inner solve

implemented on standard staggered grid and Arakawa E-grid


## Evaluation of Current state

**Weak inclusion on standard grid**

![](../figures/2_output_miniapp_10.png)

**Weak inclusion on Arakawa grid**

* Both match the result with PT
* Both converge faster than PT

In "easy" setting, can verify that method works

**Strong inclusion with $\eta$-ratio $10^3$**

* still works, but takes longer to converge


**Strong inclusion with $\eta$-ratio $10^6$**

CG doesn't converge, but overall result accepted after 3 outer steps

If compare to result if $\rho g = 1$ instead of $10^{-6}$:

* CG doesn't converge in early steps, towards the end it behaves better
* The result is identical to the previous, up to scaling of pressure & velocity

## Next Steps

* Strong inclusion should work, and be independent on choice of $\rho g$
  - convergence criteria should be non-dimensionalised
  - $\gamma$ can be chosen smaller than $0.01$

* Test coupled CG (applied directly to discretised Stokes)
  - Ivan provides further explanation on implementation & preconditioning