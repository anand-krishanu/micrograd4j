# Architecture

A map of the codebase for contributors and curious readers. For the *concepts*, read
[HOW_AUTOGRAD_WORKS.md](HOW_AUTOGRAD_WORKS.md) first.

## The core engine

`src/main/java/io/github/anandkrishanu/micrograd/`

```
Value ───────────── the whole autograd engine
Activation ──────── enum: LINEAR, RELU, TANH
Module (abstract) ─ parameters() + zeroGrad()
  ├─ Neuron ─────── w·x + b, then an Activation
  ├─ Layer  ─────── neurons over a shared input
  └─ MLP    ─────── a stack of Layers
```

### `Value` — the one class that matters

Each arithmetic op (`add`, `multiply`, `power`, `relu`, `tanh`, `exp`, …) returns a **new anonymous
subclass of `Value`** that overrides `_backward()` with that op's local derivative rule. This is the
key trick: the backward logic for an op lives right next to its forward computation, closing over the
operands it needs.

Invariants to preserve when touching this code:

- **Gradients accumulate** (`+=`, never `=`) so a node reused in multiple places gets the *sum* of all
  incoming gradients. The private `children(...)` helper uses a `HashSet` (not `Set.of`) so duplicate
  parents — e.g. `x.multiply(x)` — don't throw.
- **`backward()`** runs a DFS **topological sort**, seeds the root's `grad` to `1.0`, then calls
  `_backward()` on nodes in reverse-topo order. Call **`zeroGrad()` before `backward()`** on any reused
  graph.
- **Derived ops are built from primitives:** `subtract = add(negate)`, `divide = multiply(power(-1))`,
  `negate = multiply(-1)`. Adding a genuinely new primitive means a new anonymous subclass with its own
  `_backward()` — see the [CONTRIBUTING recipe](../CONTRIBUTING.md#recipe-add-a-new-op-in-3-steps).

### The NN layer

`Module` is the abstract base (`parameters()` + `zeroGrad()`). The hierarchy is
`Neuron → Layer → MLP`. Convention: an `MLP`'s hidden layers use the chosen `Activation`; the **output
layer is always `LINEAR`** so it can produce any real value. Pass a seeded `Random` through the
constructors for reproducible weight init (the tests rely on this).

## Examples — two separate worlds

There are two example mechanisms, and they are intentionally decoupled:

| Location | Build tool | Depends on |
|----------|-----------|------------|
| `src/main/.../examples/` (`GradCheck`, `MoonsDemo`) | Maven (`exec:java`) | the **local** `src/main` |
| `examples/*.java` (`Quickstart`, `Playground`) | **JBang** | the **published JitPack artifact** |

The JBang scripts pin a released tag via `//DEPS com.github.anand-krishanu:micrograd4j:vX.Y.Z`, so
**editing `src/main` does not change what `jbang examples/...` runs** until a new tag is published. The
`examples/playground/*.java` files are flat (default package) and wired into `Playground.java` via
`//SOURCES` directives — they are JBang sources, never compiled by Maven.

## The playground

A keyboard-driven terminal UI built on JLine — no web/browser code despite the name. Module roles:

| Module | Role |
|--------|------|
| `Tui` | JLine wrapper (all-static): colours, cursor, raw-key input, dumb-terminal detection |
| `Menu` | arrow-key selection |
| `ExprParser` | recursive-descent parser → `Value` graph |
| `GraphView` / `GraphInteractive` | computation-graph rendering and the backprop scrubber |
| `Charts` / `Braille` | braille loss curve, sparkline, 256-colour decision-boundary heatmap |
| `Datasets` | moons / xor / circles / custom |
| `Trainer` | SVM max-margin training loop + `Config` |
| `Explain` | plain-English chain-rule text |

**Graceful degradation is a hard requirement.** Every visual must fall back to plain ASCII — no ANSI
codes, cursor moves, or animation — when JLine reports a "dumb" terminal (piped input or `--demo`).
When changing the playground, verify `jbang examples/Playground.java --demo` emits **zero ANSI escape
sequences** so it stays readable in logs and CI.

## Versioning / release coupling

The version string lives in several places that must move together on a release: `pom.xml`
`<version>`, the `//DEPS` lines in `examples/Quickstart.java` and `examples/Playground.java`, and the
version references in `README.md`. JitPack builds from a pushed git tag, so examples only work against
a tag that has actually been published. Note the coordinate mismatch: Maven coordinates are
`io.github.anandkrishanu`, but JitPack consumes the repo as `com.github.anand-krishanu`.

## CI

`.github/workflows/ci.yml` runs `./mvnw -B test` on JDK 17 and 21 for every push to `main` and every
PR. It does not exercise the JBang examples.
