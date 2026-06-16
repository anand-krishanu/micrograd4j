import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * A single neuron: computes w . x + b, then optionally a ReLU.
 */
public class Neuron extends Module {
    public List<Value> w;
    public Value b;
    public boolean nonlin;    // apply ReLU if true, otherwise stay linear

    private static final Random RNG = new Random();

    public Neuron(int nin, boolean nonlin) {
        w = new ArrayList<>();
        for (int i = 0; i < nin; i++) {
            w.add(new Value(RNG.nextDouble() * 2 - 1));  /* uniform in [-1, 1) */
        }
        b = new Value(0.0);
        this.nonlin = nonlin;
    }

    /** Forward pass: act = b + sum(w_i * x_i), then ReLU if nonlinear. */
    public Value forward(List<Value> x) {
        Value act = b;
        for (int i = 0; i < w.size(); i++) {
            act = act.add(w.get(i).multiply(x.get(i)));
        }
        return nonlin ? act.relu() : act;
    }

    @Override
    public List<Value> parameters() {
        List<Value> params = new ArrayList<>(w);
        params.add(b);
        return params;
    }

    @Override
    public String toString() {
        return (nonlin ? "ReLU" : "Linear") + "Neuron(" + w.size() + ")";
    }
}
