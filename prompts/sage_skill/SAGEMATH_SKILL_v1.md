SageMath usage notes for tool-assisted reasoning.

Execution model:
- Code runs under `sage -python`.
- Use Python syntax, not notebook shorthand.
- Use `**` for exponentiation, never `^`.
- Always assign the final structured result to `RESULT`.

Core polynomial patterns:
- Create rings with `R = PolynomialRing(QQ, ['x', 'y'])` then `x, y = R.gens()`.
- Prefer exact rings like `QQ`, finite fields, or explicit algebraic extensions for verification.
- For a coefficient of `x`, prefer `p.monomial_coefficient(x)`.
- For substitution, prefer calls like `p(x=value)` or `p.subs(...)` depending on the parent object.

Factorization and inspection:
- Factorizations are iterable. Prefer `for f, e in list(fac):`.
- When a method fails, inspect the object before guessing:
  - `type(obj)`
  - `obj.parent()`
  - `dir(obj)`
- Use small smoke-test snippets to confirm an API pattern before embedding it into a larger verification script.

Verification discipline:
- Separate direct proof from suggestive evidence.
- Numerical tests, random samples, and specializations can refute or suggest; they do not certify a global claim by themselves.
- If a required condition is not directly proved, mark it unresolved instead of passing it.

Required verification envelope:
- Put verification data in `RESULT["verification"]`.
- Use `summary` as one of `pass`, `fail`, or `unresolved`.
- Use `checks` as an ordered list of objects with keys:
  - `id`
  - `status`
  - `evidence`
- Use `outputs` as a mapping from required output ids to exact computed values.
