import java.util.ArrayList;
import java.util.List;

/**
 * A multi-layer perceptron: a stack of Layers. Every hidden layer is ReLU,
 * the final layer is linear so the output can take any value.
 */
public class MLP extends Module {
    public List<Layer> layers;

    /**
     * @param nin   number of inputs
     * @param nouts size of each layer, e.g. {4, 4, 1} for two hidden layers then one output
     */
    public MLP(int nin, int[] nouts) {
        int[] sz = new int[nouts.length + 1];
        sz[0] = nin;
        for (int i = 0; i < nouts.length; i++) {
            sz[i + 1] = nouts[i];
        }

        layers = new ArrayList<>();
        for (int i = 0; i < nouts.length; i++) {
            boolean nonlin = i != nouts.length - 1;  /* last layer is linear */
            layers.add(new Layer(sz[i], sz[i + 1], nonlin));
        }
    }

    /** Forward pass: feed the input through every layer in turn. */
    public List<Value> forward(List<Value> x) {
        for (Layer layer : layers) {
            x = layer.forward(x);
        }
        return x;
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
