import io.github.anandkrishanu.micrograd.Activation;
import io.github.anandkrishanu.micrograd.MLP;
import io.github.anandkrishanu.micrograd.Value;

import java.util.Random;

/** Mutable bag of everything the user can tune from the Settings screen. */
final class Config {
    String dataset = "moons";   // moons | xor | circles | custom
    double noise = 0.10;
    int samples = 100;
    int[] hidden = {16, 16};    // hidden layer sizes; output layer (size 1) is appended by the trainer
    Activation activation = Activation.TANH;
    int epochs = 100;
    double baseLr = 1.0;        // decays to 0.1*baseLr by the last epoch
    double regAlpha = 1e-4;
    long seed = 1337L;

    String hiddenString() {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < hidden.length; i++) {
            if (i > 0) sb.append(',');
            sb.append(hidden[i]);
        }
        return sb.toString();
    }

    /** Parse "16,16" into the hidden-layer sizes; ignores malformed input. */
    void setHidden(String csv) {
        String[] parts = csv.split(",");
        java.util.List<Integer> sizes = new java.util.ArrayList<>();
        for (String p : parts) {
            p = p.trim();
            if (p.isEmpty()) continue;
            try {
                int v = Integer.parseInt(p);
                if (v > 0) sizes.add(v);
            } catch (NumberFormatException ignored) {
            }
        }
        if (!sizes.isEmpty()) {
            hidden = sizes.stream().mapToInt(Integer::intValue).toArray();
        }
    }
}

/**
 * The SVM max-margin training loop from MoonsDemo, parameterized by {@link Config}
 * and emitting a per-step callback so the UI can animate the loss curve.
 */
final class Trainer {
    /** Called after every SGD step with live metrics. */
    interface StepCallback {
        void onStep(int step, int epochs, double loss, double acc, MLP model);
    }

    private Trainer() {}

    /** Build a fresh MLP for this config: nin=2, the hidden sizes, then a single linear output. */
    static MLP newModel(Config cfg) {
        int[] nouts = new int[cfg.hidden.length + 1];
        System.arraycopy(cfg.hidden, 0, nouts, 0, cfg.hidden.length);
        nouts[nouts.length - 1] = 1;
        return new MLP(2, nouts, cfg.activation, new Random(cfg.seed));
    }

    /** Train {@code model} on the dataset, invoking {@code cb} each step. Returns the same model. */
    static MLP train(Config cfg, MLP model, double[][] xs, double[] ys, StepCallback cb) {
        int n = xs.length;
        for (int step = 0; step <= cfg.epochs; step++) {
            // ----- forward: hinge loss relu(1 - y*score), averaged -----
            Value dataLoss = new Value(0.0);
            for (int i = 0; i < n; i++) {
                Value score = model.forward(xs[i]);
                Value margin = score.multiply(ys[i]).negate().add(1.0).relu();
                dataLoss = dataLoss.add(margin);
            }
            dataLoss = dataLoss.divide(n);

            // ----- L2 regularization -----
            Value reg = new Value(0.0);
            for (Value p : model.parameters()) {
                reg = reg.add(p.multiply(p));
            }
            Value loss = dataLoss.add(reg.multiply(cfg.regAlpha));

            // ----- backward -----
            model.zeroGrad();
            loss.backward();

            // ----- SGD step with learning-rate decay -----
            double lr = cfg.baseLr * (1.0 - 0.9 * step / cfg.epochs);
            for (Value p : model.parameters()) {
                p.data -= lr * p.grad;
            }

            if (cb != null) {
                cb.onStep(step, cfg.epochs, loss.data, accuracy(model, xs, ys), model);
            }
        }
        return model;
    }

    /** Fraction of points whose predicted sign matches the label. */
    static double accuracy(MLP model, double[][] xs, double[] ys) {
        int correct = 0;
        for (int i = 0; i < xs.length; i++) {
            double pred = model.forward(xs[i]).data;
            if ((pred > 0) == (ys[i] > 0)) correct++;
        }
        return (double) correct / xs.length;
    }
}
