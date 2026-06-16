package io.github.anandkrishanu.micrograd;

/**
 * Activation function applied to a neuron's pre-activation value.
 */
public enum Activation {
    LINEAR,
    RELU,
    TANH;

    /** Apply this activation to a Value, returning the activated Value. */
    public Value apply(Value v) {
        switch (this) {
            case RELU: return v.relu();
            case TANH: return v.tanh();
            default:   return v;   // LINEAR: pass through unchanged
        }
    }
}
