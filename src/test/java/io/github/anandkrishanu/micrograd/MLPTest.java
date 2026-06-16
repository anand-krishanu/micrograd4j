package io.github.anandkrishanu.micrograd;

import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class MLPTest {

    @Test
    void parameterCountMatchesArchitecture() {
        // 2 -> 3 -> 1: layer1 = 3*(2+1)=9, layer2 = 1*(3+1)=4 => 13
        MLP model = new MLP(2, new int[]{3, 1}, Activation.TANH, new Random(0));
        assertEquals(13, model.parameters().size());
    }

    @Test
    void outputLayerIsLinear() {
        MLP model = new MLP(2, new int[]{4, 1}, Activation.RELU, new Random(0));
        Activation last = model.layers.get(model.layers.size() - 1)
                                      .neurons.get(0).activation;
        assertEquals(Activation.LINEAR, last);
    }

    @Test
    void trainingReducesLoss() {
        Random rng = new Random(42);
        double[][] xs = {{2.0, 3.0}, {3.0, -1.0}, {0.5, 1.0}, {-1.0, -1.0}};
        double[] ys = {1.0, -1.0, -1.0, 1.0};

        MLP model = new MLP(2, new int[]{8, 8, 1}, Activation.TANH, rng);

        double firstLoss = Double.NaN;
        double lastLoss = Double.NaN;
        for (int step = 0; step < 50; step++) {
            Value loss = new Value(0.0);
            for (int i = 0; i < xs.length; i++) {
                Value pred = model.forward(xs[i]);
                loss = loss.add(pred.subtract(ys[i]).power(2));
            }
            model.zeroGrad();
            loss.backward();
            for (Value p : model.parameters()) {
                p.data -= 0.05 * p.grad;
            }
            if (step == 0) firstLoss = loss.data;
            lastLoss = loss.data;
        }

        assertTrue(lastLoss < firstLoss * 0.1,
                "expected loss to drop substantially, was " + firstLoss + " -> " + lastLoss);
    }
}
