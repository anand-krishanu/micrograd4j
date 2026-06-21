# How autograd works — a worked example

This is the whole idea behind micrograd4j (and behind PyTorch, TensorFlow, JAX…) in one page. We'll
build a tiny expression, run `backward()`, and follow the gradient by hand through every node.

## 1. A `Value` records how it was computed

When you write `a.multiply(b)`, the result is not just a number — it's a new `Value` that remembers:

- its **data** (the forward result),
- its **parents** (`prev`) — the `Value`s it was built from,
- the **op** that made it (`"*"`, `"+"`, …),
- a slot for its **grad**, initially `0`.

So a chain of arithmetic quietly builds a **graph**.

```java
Value a = new Value(2.0);
Value b = new Value(-3.0);
Value c = new Value(10.0);

Value e = a.multiply(b);  // e = a*b = -6
Value d = e.add(c);       // d = e+c =  4
Value f = new Value(-2.0);
Value L = d.multiply(f);  // L = d*f = -8
```

The graph (forward values shown):

```
  a=2 ─┐
       (×)─► e=-6 ─┐
  b=-3 ┘           (+)─► d=4 ─┐
           c=10 ──┘           (×)─► L=-8
                       f=-2 ──┘
```

## 2. The chain rule, locally

We want `dL/dx` for every input `x`. The chain rule says: the gradient of `L` w.r.t. some input is
the gradient arriving **from above** multiplied by the op's **local derivative**.

Each op knows only its own local rule:

| op            | forward        | local rule (what `_backward()` does)                          |
|---------------|----------------|---------------------------------------------------------------|
| `add`         | `out = x + y`  | `x.grad += out.grad`; `y.grad += out.grad`                    |
| `multiply`    | `out = x * y`  | `x.grad += y.data * out.grad`; `y.grad += x.data * out.grad`  |
| `power(n)`    | `out = x^n`    | `x.grad += n * x^(n-1) * out.grad`                            |
| `relu`        | `out = max(0,x)` | `x.grad += (x > 0 ? 1 : 0) * out.grad`                      |
| `tanh`        | `out = tanh(x)`  | `x.grad += (1 - out^2) * out.grad`                          |

Addition just **passes the gradient straight through**. Multiplication **swaps in the other operand**.
That's it — no symbolic differentiation, no math library. Each op contributes one line.

## 3. `backward()` runs the rules in the right order

`backward()` does three things:

1. **Topological sort** — orders nodes so every node comes *after* its parents
   (`buildTopo` is a depth-first walk).
2. **Seed the output**: `L.grad = 1.0` — the derivative of `L` with respect to itself is `1`.
3. **Walk in reverse** and call each node's `_backward()`, pushing gradient toward the inputs.

Tracing our example, from the output back:

```
start:  L.grad = 1

L = d*f :  d.grad += f.data * L.grad = (-2)(1) = -2
           f.grad += d.data * L.grad = ( 4)(1) =  4

d = e+c :  e.grad += d.grad = -2          (addition passes through)
           c.grad += d.grad = -2

e = a*b :  a.grad += b.data * e.grad = (-3)(-2) =  6
           b.grad += a.data * e.grad = ( 2)(-2) = -4
```

Result:

```java
L.backward();
a.grad;  //  6.0   = dL/da
b.grad;  // -4.0   = dL/db
c.grad;  // -2.0   = dL/dc
```

Run it yourself: [`examples/Quickstart.java`](../examples/Quickstart.java), or interactively with
`jbang examples/Playground.java` → **Autograd playground** → type `(a*b) + c`, then `:step` to watch
exactly this flow animate.

## 4. Two subtleties worth internalizing

**Gradients accumulate (`+=`, never `=`).** If a node feeds into the output along more than one path —
`x.multiply(x)`, or a weight reused across data points — each path contributes, and the *total*
gradient is their **sum**. That's why every `_backward()` uses `+=`, and why you must call
`zeroGrad()` before re-running `backward()` on a graph you've already used.

```java
Value x = new Value(3.0);
Value y = x.add(x);  // = 2x
y.backward();
x.grad;  // 2.0, not 1.0 — both edges into x contribute
```

**Why reverse topological order?** A node's gradient is only correct once *every* node downstream of
it has already added its contribution. Reverse-topo guarantees that: by the time we call a node's
`_backward()`, its own `grad` is final, so it can safely push that value to its parents.

## 5. From scalars to a neural network

A neuron is just `w·x + b` followed by an activation — i.e. a few `multiply`/`add`/`tanh` `Value`s.
A `Layer` is several neurons; an `MLP` is several layers. Training is the same loop you'd write in any
framework, except you can read every line:

```java
loss.backward();                 // fill in .grad on every parameter
for (Value p : model.parameters())
    p.data -= learningRate * p.grad;   // step downhill
```

No magic — the same chain rule from §2, applied to a bigger graph.

→ For how the code is organized, see [ARCHITECTURE.md](ARCHITECTURE.md).
