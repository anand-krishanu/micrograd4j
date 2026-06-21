# Contributing to micrograd4j

Thanks for your interest! micrograd4j is an **educational** project — its job is to make reverse-mode
autodiff easy to *read and understand*, not to be a fast or feature-complete framework. The best
contributions keep the code small, clear, and faithful to that goal.

> New to open source? This is a friendly place to start. Look for issues labelled
> [`good first issue`](https://github.com/anand-krishanu/micrograd4j/labels/good%20first%20issue).

## Ways to help

- **Add a new op** (e.g. `sigmoid`, `log`, `sin`, `cos`) — see the recipe below.
- **Improve the docs** — clearer explanations, diagrams, or fixes to
  [`docs/HOW_AUTOGRAD_WORKS.md`](docs/HOW_AUTOGRAD_WORKS.md).
- **Improve the playground** — a new dataset, preset, or visual.
- **Report a bug** or a confusing explanation via [an issue](https://github.com/anand-krishanu/micrograd4j/issues/new/choose).

## Recipe: add a new op in 3 steps

Every operation on `Value` follows the same pattern — a forward value plus a local backward rule.
Adding `sigmoid` as an example:

1. **Add the method to `Value.java`.** Return a new anonymous subclass that overrides `_backward()`
   with the op's *local derivative*. Always **accumulate** (`+=`) into parents' `grad`:

   ```java
   /** Sigmoid activation: 1 / (1 + e^-x). */
   public Value sigmoid() {
       double s = 1.0 / (1.0 + Math.exp(-data));
       return new Value(s, children(this), "sigmoid") {
           @Override
           public void _backward() {
               // d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
               Value.this.grad += s * (1 - s) * this.grad;
           }
       };
   }
   ```

2. **Add a test in `ValueTest.java`** that checks both the forward value and the gradient (a
   finite-difference or known-derivative check is perfect):

   ```java
   @Test
   void sigmoidBackward() {
       Value x = new Value(0.5);
       Value y = x.sigmoid();
       y.backward();
       assertEquals(0.6224, y.data, EPS);
       assertEquals(0.2350, x.grad, EPS);  // s*(1-s)
   }
   ```

3. **Wire it up where relevant** — e.g. add it to the `Activation` enum and the playground's
   `ExprParser` if it should be usable from the TUI. Update the op list in `README.md`.

## Development setup

You only need a JDK (17+); the Maven Wrapper is committed.

```bash
./mvnw test                                  # run all tests (mvnw.cmd on Windows)
./mvnw test -Dtest=ValueTest                 # one class
./mvnw test -Dtest=ValueTest#powerBackward   # one method
```

For the playground (needs [JBang](https://www.jbang.dev)):

```bash
jbang examples/Playground.java          # interactive
jbang examples/Playground.java --demo   # non-interactive smoke run
```

> The playground **must** stay readable when output is piped or `--demo` is passed: verify
> `jbang examples/Playground.java --demo` emits **zero ANSI escape sequences**.

## Pull request guidelines

- Keep PRs focused and small; one logical change per PR.
- Match the surrounding code style, comment density, and naming.
- Add or update tests for any behaviour change — CI runs on JDK 17 and 21.
- Update relevant docs (README op list, `docs/`) when you change the public API.
- Don't add heavyweight dependencies to the core library (it is zero-dependency by design).

## Scope: what *not* to add

micrograd4j is intentionally tiny. Some features would turn a readable teaching tool into a slow,
complicated framework and will be declined:

- Tensors / n-d arrays / broadcasting — the point is *scalar* autograd.
- GPU / vectorization / multithreading — performance is not the goal.
- An optimizer zoo (Adam, schedulers…) — `p.data -= lr * p.grad` *is* the lesson.
- A layer zoo (Conv, BatchNorm, attention…) and serialization / ONNX.
- Heavy dependencies in the core (it's zero-dependency; JLine lives only in the playground).

A good test: *does this make backprop easier to **understand**, or just easier to **scale**?* If it's
the second one, it belongs in PyTorch, not here.

## License

By contributing, you agree that your contributions are licensed under the
[MIT License](LICENSE).
