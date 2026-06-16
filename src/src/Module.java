import java.util.List;

/**
 * Base class for anything with trainable parameters (Neuron, Layer, MLP).
 * Gives everyone a shared zeroGrad().
 */
public abstract class Module {

    /** All trainable Values owned by this module. */
    public abstract List<Value> parameters();

    /** Reset every parameter's gradient to 0 before the next backward pass. */
    public void zeroGrad() {
        for (Value p : parameters()) {
            p.grad = 0.0;
        }
    }
}
