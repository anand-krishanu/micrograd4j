package io.github.anandkrishanu.micrograd;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * A single scalar value that records how it was computed, so we can autograd through it.
 */
public class Value {
    public double data;
    public double grad;
    public Set<Value> prev;
    public String op;

    /**
     * Leaf value created directly from a number (inputs, weights, constants).
     */
    public Value(double data) {
        this.data = data;
        this.grad = 0.0;
        this.prev = new HashSet<>();
        this.op = "";
    }

    /**
     * Intermediate value created by an operation, with its parents and op label.
     */
    public Value(double data, Set<Value> prev, String op) {
        this.data = data;
        this.grad = 0.0;
        this.prev = prev;
        this.op = op;
    }

    /**
     * Local backward step
     * Does nothing by his own lol
     * No-op for leaves
     * Overridden by each operation below.
     */
    public void _backward () {}

    /**
     * HashSet instead of Set.of so duplicate parents (e.g. x.multiply(x)) don't throw.
     */
    private static Set<Value> children(Value... vs) {
        return new HashSet<>(Arrays.asList(vs));
    }

    public Value add(Value other) {
        return new Value(data + other.data, children(this, other), "+") {
            @Override
            public void _backward () {
                // d(out)/d(each input) = 1 in addition, so the gradient passes straight through.
                Value.this.grad += this.grad;
                other.grad += this.grad;
            }
        };
    }

    /**
     * This is made to add another number and not just the Value obj
     */
    public Value add(double other) {
        return add(new Value(other));
    }

    public Value multiply(Value other) {
        return new Value(data * other.data, children(this, other), "*") {
            @Override
            public void _backward () {
                /* product rule: each input's gradient is scaled by the other's value. */
                Value.this.grad += other.data * this.grad;
                other.grad += Value.this.data * this.grad;
            }
        };
    }

    /**
     * Same here, to multiply number and not just Value obj
     */
    public Value multiply(double other) {
        return multiply(new Value(other));
    }

    /**
     * Raise to a constant power
     * The exponent is not part of the graph.
     */
    public Value power (double exponent) {
        return new Value(Math.pow(data, exponent), children(this), "^" + exponent) {
            @Override
            public void _backward () {
                /* power rule: d(out)/d(in) = exponent * in^(exponent - 1). */
                Value.this.grad += exponent * Math.pow(Value.this.data, exponent - 1) * this.grad;
            }
        };
    }

    /**
     * -this, implemented as multiply by -1.
     */
    public Value negate() {
        return multiply(-1.0);
    }

    /**
     * this - other, implemented as this + (-other).
     */
    public Value subtract(Value other) {
        return add(other.negate());
    }

    public Value subtract(double other) {
        return add(-other);
    }

    /**
     * this / other, implemented as this * other^(-1).
     */
    public Value divide(Value other) {
        return multiply(other.power(-1.0));
    }

    public Value divide(double other) {
        return multiply(1.0 / other);
    }

    /**
     * ReLU activation.
     * forward:  relu(x) = max(0, x)
     * backward: d/dx relu(x) = 1 if x > 0, else 0
     */
    public Value relu() {
        return new Value(Math.max(0.0, data), children(this), "ReLU") {
            @Override
            public void _backward () {
                /* gradient flows only where the input was positive. */
                Value.this.grad += (Value.this.data > 0 ? 1.0 : 0.0) * this.grad;
            }
        };
    }

    /**
     * Tanh activation.
     * forward:  tanh(x) = (e^x - e^-x) / (e^x + e^-x)
     * backward: d/dx tanh(x) = 1 - tanh(x)^2
     */
    public Value tanh() {
        double t = Math.tanh(data);
        return new Value(t, children(this), "tanh") {
            @Override
            public void _backward () {
                /* derivative of tanh(x) is 1 - tanh(x)^2. */
                Value.this.grad += (1 - t * t) * this.grad;
            }
        };
    }

    /**
     * Exponential.
     * forward:  exp(x) = e^x
     * backward: d/dx e^x = e^x
     */
    public Value exp() {
        double e = Math.exp(data);
        return new Value(e, children(this), "exp") {
            @Override
            public void _backward () {
                /* e^x is its own derivative. */
                Value.this.grad += e * this.grad;
            }
        };
    }

    /**
     *  Full backward pass: call on the final value to fill in grad on every node.
     */
    public void backward() {
        List<Value> topo = new ArrayList<>();
        Set<Value> visited = new HashSet<>();
        buildTopo(this, visited, topo);

        this.grad = 1.0;     // derivative of the output with respect to itself is 1.
        Collections.reverse(topo);
        for (Value v : topo) {
            v._backward();
        }
    }

    /**
     * Reset grad to 0 on this value and every value reachable from it.
     * Call before backward() so gradients from a previous pass don't accumulate.
     */
    public void zeroGrad() {
        zeroGrad(this, new HashSet<>());
    }

    private void zeroGrad(Value v, Set<Value> visited) {
        if (visited.add(v)) {
            v.grad = 0.0;
            for (Value child : v.prev) {
                zeroGrad(child, visited);
            }
        }
    }

    /**
     * DFS topological sort: adds children before parents so backward() can reverse it.
     */
    private void buildTopo (Value v, Set<Value> visited, List<Value> topo) {
        if (!visited.contains(v)) {
            visited.add(v);
            for (Value child : v.prev) {
                buildTopo(child, visited, topo);
            }

            topo.add(v);
        }
    }

    @Override
    public String toString() {
        return String.format("Value(data=%.4f, grad=%.4f)", data, grad);
    }
}