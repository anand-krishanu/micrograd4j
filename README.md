# micrograd4j

[![CI](https://github.com/anand-krishanu/micrograd4j/actions/workflows/ci.yml/badge.svg)](https://github.com/anand-krishanu/micrograd4j/actions/workflows/ci.yml)
[![JitPack](https://jitpack.io/v/anand-krishanu/micrograd4j.svg)](https://jitpack.io/#anand-krishanu/micrograd4j)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A tiny scalar-valued **autograd engine** and a small **neural-network library** on top of it, in plain Java.
A faithful port of [Andrej Karpathy's micrograd](https://github.com/karpathy/micrograd) — small enough to read in one sitting, correct enough to train a real classifier.

```java
Value a = new Value(2.0);
Value b = new Value(-3.0);
Value L = a.multiply(b).add(10.0).multiply(-2.0);
L.backward();
System.out.println(a.grad);  // 6.0  ->  dL/da
```

## Try it in 10 seconds (no install)

With [JBang](https://www.jbang.dev):

```bash
jbang https://raw.githubusercontent.com/anand-krishanu/micrograd4j/main/examples/Quickstart.java
```

JBang pulls the library from JitPack and runs it — no clone, no build.

## Interactive playground (no install)

Want to actually *drive* the engine instead of reading a script? Launch the interactive
terminal playground with one command:

```bash
jbang https://raw.githubusercontent.com/anand-krishanu/micrograd4j/main/examples/Playground.java
```

It's a menu-driven TUI (built on [JLine](https://github.com/jline/jline3)) with four modes:

- **Autograd playground** — type your own expressions (`(a*b) + c.tanh()`, `relu(a) / 2`),
  assign variables, and see the value plus every input's gradient after `backward()`.
- **Train a network** — pick a dataset (`moons`, `xor`, `circles`, or enter your own points)
  and watch a **live ASCII loss curve** train, then a **decision-boundary** plot of the result.
- **Step through backprop** — walk the backward pass node-by-node and watch gradients flow
  through the computation graph.
- **Settings** — tune the dataset, hidden layers, activation, epochs, learning rate, and seed,
  then re-run.

Run it from a clone with `jbang examples/Playground.java`, or do a quick non-interactive
smoke run with `jbang examples/Playground.java --demo`.

## Use it as a dependency (Maven via JitPack)

Add the JitPack repository and the dependency to your `pom.xml`:

```xml
<repositories>
  <repository>
    <id>jitpack.io</id>
    <url>https://jitpack.io</url>
  </repository>
</repositories>

<dependency>
  <groupId>com.github.anand-krishanu</groupId>
  <artifactId>micrograd4j</artifactId>
  <version>v1.1.0</version>
</dependency>
```

> Replace `v1.1.0` with any released tag, a branch as `main-SNAPSHOT`, or a commit hash.

## What's inside

| Class        | Role                                                                 |
|--------------|----------------------------------------------------------------------|
| `Value`      | A scalar that records how it was computed; supports `backward()`     |
| `Activation` | `LINEAR`, `RELU`, `TANH`                                             |
| `Neuron`     | `w · x + b` followed by an activation                                |
| `Layer`      | A row of neurons over the same input                                 |
| `MLP`        | A stack of layers (hidden activation configurable, linear output)    |

### Autograd

```java
Value x = new Value(-4.0);
Value y = x.power(2).add(x.multiply(3)).add(1);  // y = x^2 + 3x + 1
y.backward();
System.out.println(y.data);  // 5.0
System.out.println(x.grad);  // 2x + 3 = -5.0
```

Supported ops: `add`, `subtract`, `multiply`, `divide`, `power`, `negate`, `relu`, `tanh`, `exp`
(each with `double` overloads where it makes sense).

### Training a network

```java
MLP model = new MLP(2, new int[]{16, 16, 1}, Activation.TANH);

for (int step = 0; step < 100; step++) {
    Value loss = computeLoss(model, xs, ys);  // your loss as a Value graph
    model.zeroGrad();
    loss.backward();
    for (Value p : model.parameters()) {
        p.data -= 0.05 * p.grad;               // SGD step
    }
}
```

## Build & test

This repo ships a Maven Wrapper, so you don't need Maven installed — just a JDK (17+).

```bash
./mvnw test          # Linux / macOS
mvnw.cmd test        # Windows
```

## Run the examples

```bash
# Gradient check against PyTorch reference values
./mvnw -q exec:java -Dexec.mainClass=io.github.anandkrishanu.micrograd.examples.GradCheck

# Two-moons binary classification with a tanh MLP (reaches 100% accuracy)
./mvnw -q exec:java -Dexec.mainClass=io.github.anandkrishanu.micrograd.examples.MoonsDemo
```

## Project layout

```
src/main/java/io/github/anandkrishanu/micrograd/   core: Value, Module, Neuron, Layer, MLP
                                         .../examples/   runnable demos
src/test/java/io/github/anandkrishanu/micrograd/   JUnit 5 tests
examples/Quickstart.java                           JBang zero-install demo
examples/Playground.java                           JBang interactive TUI playground
                       .../playground/             playground modules (parser, charts, trainer, ...)
```

## Credits

Port of [karpathy/micrograd](https://github.com/karpathy/micrograd). Licensed under the [MIT License](LICENSE).
