package io.github.anandkrishanu.micrograd;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * A multi-layer perceptron: a stack of Layers. Every hidden layer uses the chosen
 * activation; the final layer is LINEAR so the output can take any value.
 */
public class MLP extends Module {
    public List<Layer> layers;

    /** Defaults to ReLU hidden activations and a random seed. */
    public MLP(int nin, int[] nouts) {
        this(nin, nouts, Activation.RELU, new Random());
    }

    /** Choose the hidden activation (e.g. TANH) with a random seed. */
    public MLP(int nin, int[] nouts, Activation hidden) {
        this(nin, nouts, hidden, new Random());
    }

    /**
     * @param nin    number of inputs
     * @param nouts  size of each layer, e.g. {16, 16, 1} for two hidden layers then one output
     * @param hidden activation for hidden layers (output layer is always LINEAR)
     * @param rng    source of randomness for weight init (pass a seeded Random for reproducibility)
     */
    public MLP(int nin, int[] nouts, Activation hidden, Random rng) {
        int[] sz = new int[nouts.length + 1];
        sz[0] = nin;
        for (int i = 0; i < nouts.length; i++) {
            sz[i + 1] = nouts[i];
        }

        layers = new ArrayList<>();
        for (int i = 0; i < nouts.length; i++) {
            boolean last = i == nouts.length - 1;
            Activation act = last ? Activation.LINEAR : hidden;
            layers.add(new Layer(sz[i], sz[i + 1], act, rng));
        }
    }

    /** Forward pass: feed the input through every layer in turn. */
    public List<Value> forward(List<Value> x) {
        for (Layer layer : layers) {
            x = layer.forward(x);
        }
        return x;
    }

    /** Convenience: forward a plain double[] input and return the first output. */
    public Value forward(double[] x) {
        List<Value> input = new ArrayList<>();
        for (double v : x) {
            input.add(new Value(v));
        }
        return forward(input).get(0);
    }

    @Override
    public List<Value> parameters() {
        List<Value> params = new ArrayList<>();
        for (Layer layer : layers) {
            params.addAll(layer.parameters());
        }
        return params;
    }

    @Override
    public String toString() {
        return "MLP" + layers;
    }
}
