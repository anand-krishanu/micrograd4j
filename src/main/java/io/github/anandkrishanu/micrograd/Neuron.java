package io.github.anandkrishanu.micrograd;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * A single neuron: computes w . x + b, then applies an activation.
 */
public class Neuron extends Module {
    public List<Value> w;            // one weight per input
    public Value b;                  // bias
    public Activation activation;    // activation applied to w . x + b

    private final Random rng;

    public Neuron(int nin, Activation activation) {
        this(nin, activation, new Random());
    }

    /** Seeded constructor for reproducible weights (handy in tests). */
    public Neuron(int nin, Activation activation, Random rng) {
        this.rng = rng;
        w = new ArrayList<>();
        for (int i = 0; i < nin; i++) {
            w.add(new Value(rng.nextDouble() * 2 - 1));  // uniform in [-1, 1)
        }
        b = new Value(0.0);
        this.activation = activation;
    }

    /** Forward pass: act = b + sum(w_i * x_i), then the activation. */
    public Value forward(List<Value> x) {
        Value act = b;
        for (int i = 0; i < w.size(); i++) {
            act = act.add(w.get(i).multiply(x.get(i)));
        }
        return activation.apply(act);
    }

    @Override
    public List<Value> parameters() {
        List<Value> params = new ArrayList<>(w);
        params.add(b);
        return params;
    }

    @Override
    public String toString() {
        return activation + "Neuron(" + w.size() + ")";
    }
}
