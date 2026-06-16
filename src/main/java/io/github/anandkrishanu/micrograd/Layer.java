package io.github.anandkrishanu.micrograd;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * A layer of independent neurons that all see the same input.
 */
public class Layer extends Module {
    public List<Neuron> neurons;

    public Layer(int nin, int nout, Activation activation, Random rng) {
        neurons = new ArrayList<>();
        for (int i = 0; i < nout; i++) {
            neurons.add(new Neuron(nin, activation, rng));
        }
    }

    /** Forward pass: one output Value per neuron. */
    public List<Value> forward(List<Value> x) {
        List<Value> out = new ArrayList<>();
        for (Neuron n : neurons) {
            out.add(n.forward(x));
        }
        return out;
    }

    @Override
    public List<Value> parameters() {
        List<Value> params = new ArrayList<>();
        for (Neuron n : neurons) {
            params.addAll(n.parameters());
        }
        return params;
    }

    @Override
    public String toString() {
        return "Layer[" + neurons.size() + " neurons]";
    }
}
