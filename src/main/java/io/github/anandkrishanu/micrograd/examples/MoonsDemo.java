package io.github.anandkrishanu.micrograd.examples;

import io.github.anandkrishanu.micrograd.Activation;
import io.github.anandkrishanu.micrograd.MLP;
import io.github.anandkrishanu.micrograd.Value;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * The classic micrograd demo: a two-moons binary classification problem solved by
 * a small tanh MLP trained with a max-margin (SVM) loss plus L2 regularization.
 *
 * Run: java io.github.anandkrishanu.micrograd.examples.MoonsDemo
 */
public class MoonsDemo {
    public static void main(String[] args) {
        Random rng = new Random(1337);

        // Dataset: 100 points, two interleaving half-moons, labels in {-1, +1}.
        int n = 100;
        double[][] xs = new double[n][2];
        double[] ys = new double[n];
        makeMoons(n, 0.1, rng, xs, ys);

        // 2 inputs -> 16 -> 16 -> 1 output, tanh hidden activations.
        MLP model = new MLP(2, new int[]{16, 16, 1}, Activation.TANH, rng);
        System.out.println(model + "  (" + model.parameters().size() + " params)");

        int steps = 100;
        for (int step = 0; step <= steps; step++) {
            // ----- forward: SVM "max-margin" loss -----
            Value dataLoss = new Value(0.0);
            for (int i = 0; i < n; i++) {
                Value score = model.forward(xs[i]);
                // hinge loss: relu(1 - y * score)
                Value margin = score.multiply(ys[i]).negate().add(1.0).relu();
                dataLoss = dataLoss.add(margin);
            }
            dataLoss = dataLoss.divide(n);

            // L2 regularization to keep weights small
            double alpha = 1e-4;
            Value reg = new Value(0.0);
            for (Value p : model.parameters()) {
                reg = reg.add(p.multiply(p));
            }
            Value loss = dataLoss.add(reg.multiply(alpha));

            // ----- backward -----
            model.zeroGrad();
            loss.backward();

            // ----- SGD update with a simple learning-rate decay -----
            double lr = 1.0 - 0.9 * step / steps;
            for (Value p : model.parameters()) {
                p.data -= lr * p.grad;
            }

            if (step % 10 == 0) {
                double acc = accuracy(model, xs, ys);
                System.out.printf("step %3d  loss %.4f  acc %.0f%%%n", step, loss.data, acc * 100);
            }
        }
    }

    /** Fraction of points whose predicted sign matches the label. */
    private static double accuracy(MLP model, double[][] xs, double[] ys) {
        int correct = 0;
        for (int i = 0; i < xs.length; i++) {
            double pred = model.forward(xs[i]).data;
            if ((pred > 0) == (ys[i] > 0)) {
                correct++;
            }
        }
        return (double) correct / xs.length;
    }

    /**
     * Generate a two-moons dataset (mirrors sklearn.datasets.make_moons).
     * Fills xs (n x 2 features) and ys (labels in {-1, +1}).
     */
    private static void makeMoons(int n, double noise, Random rng, double[][] xs, double[] ys) {
        int nOut = n / 2;
        int nIn = n - nOut;
        List<double[]> pts = new ArrayList<>();
        List<Double> labels = new ArrayList<>();

        for (int i = 0; i < nOut; i++) {
            double t = Math.PI * i / (nOut - 1);
            pts.add(new double[]{Math.cos(t), Math.sin(t)});
            labels.add(-1.0);
        }
        for (int i = 0; i < nIn; i++) {
            double t = Math.PI * i / (nIn - 1);
            pts.add(new double[]{1 - Math.cos(t), 0.5 - Math.sin(t)});
            labels.add(1.0);
        }

        for (int i = 0; i < n; i++) {
            xs[i][0] = pts.get(i)[0] + rng.nextGaussian() * noise;
            xs[i][1] = pts.get(i)[1] + rng.nextGaussian() * noise;
            ys[i] = labels.get(i);
        }
    }
}
