import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/** A 2D binary-classification dataset: features {@code xs} and labels {@code ys} in {-1,+1}. */
final class Dataset {
    final double[][] xs;
    final double[] ys;
    final String name;

    Dataset(double[][] xs, double[] ys, String name) {
        this.xs = xs;
        this.ys = ys;
        this.name = name;
    }
}

/** Built-in toy datasets plus an interactive custom-point entry mode. */
final class Datasets {
    private Datasets() {}

    /** Build the dataset named by {@code cfg.dataset}. "custom" prompts the user for points. */
    static Dataset build(Config cfg) {
        Random rng = new Random(cfg.seed);
        switch (cfg.dataset) {
            case "xor":
                return makeXor(cfg.samples, cfg.noise, rng);
            case "circles":
                return makeCircles(cfg.samples, cfg.noise, rng);
            case "custom":
                return readCustom();
            case "moons":
            default:
                return makeMoons(cfg.samples, cfg.noise, rng);
        }
    }

    /** Two interleaving half-moons (mirrors sklearn.datasets.make_moons / MoonsDemo). */
    static Dataset makeMoons(int n, double noise, Random rng) {
        int nOut = n / 2;
        int nIn = n - nOut;
        double[][] xs = new double[n][2];
        double[] ys = new double[n];
        List<double[]> pts = new ArrayList<>();
        List<Double> labels = new ArrayList<>();

        for (int i = 0; i < nOut; i++) {
            double t = Math.PI * i / Math.max(1, nOut - 1);
            pts.add(new double[]{Math.cos(t), Math.sin(t)});
            labels.add(-1.0);
        }
        for (int i = 0; i < nIn; i++) {
            double t = Math.PI * i / Math.max(1, nIn - 1);
            pts.add(new double[]{1 - Math.cos(t), 0.5 - Math.sin(t)});
            labels.add(1.0);
        }
        for (int i = 0; i < n; i++) {
            xs[i][0] = pts.get(i)[0] + rng.nextGaussian() * noise;
            xs[i][1] = pts.get(i)[1] + rng.nextGaussian() * noise;
            ys[i] = labels.get(i);
        }
        return new Dataset(xs, ys, "moons");
    }

    /** Classic XOR: opposite quadrants share a label — not linearly separable. */
    static Dataset makeXor(int n, double noise, Random rng) {
        double[][] centers = {{-1, -1}, {1, 1}, {-1, 1}, {1, -1}};
        double[] labels = {-1, -1, 1, 1};
        double[][] xs = new double[n][2];
        double[] ys = new double[n];
        for (int i = 0; i < n; i++) {
            int c = i % 4;
            xs[i][0] = centers[c][0] + rng.nextGaussian() * (noise + 0.15);
            xs[i][1] = centers[c][1] + rng.nextGaussian() * (noise + 0.15);
            ys[i] = labels[c];
        }
        return new Dataset(xs, ys, "xor");
    }

    /** Concentric circles: inner disk is +1, outer ring is -1. */
    static Dataset makeCircles(int n, double noise, Random rng) {
        int half = n / 2;
        double[][] xs = new double[n][2];
        double[] ys = new double[n];
        for (int i = 0; i < n; i++) {
            double t = 2 * Math.PI * rng.nextDouble();
            double r = i < half ? 0.4 : 1.0;
            double label = i < half ? 1.0 : -1.0;
            xs[i][0] = r * Math.cos(t) + rng.nextGaussian() * noise;
            xs[i][1] = r * Math.sin(t) + rng.nextGaussian() * noise;
            ys[i] = label;
        }
        return new Dataset(xs, ys, "circles");
    }

    /** Let the user type points as "x1,x2 -> label" (label +1 / -1). Blank line ends entry. */
    static Dataset readCustom() {
        Tui.println(Tui.color("Enter points as  x1,x2 -> label   (label = 1 or -1). Blank line to finish.", Tui.DIM));
        List<double[]> xs = new ArrayList<>();
        List<Double> ys = new ArrayList<>();
        while (true) {
            String line = Tui.readLine("point> ");
            if (line == null) break;
            line = line.trim();
            if (line.isEmpty()) break;
            try {
                String[] halves = line.split("->");
                String[] xy = halves[0].split(",");
                double x1 = Double.parseDouble(xy[0].trim());
                double x2 = Double.parseDouble(xy[1].trim());
                double y = Double.parseDouble(halves[1].trim());
                xs.add(new double[]{x1, x2});
                ys.add(y >= 0 ? 1.0 : -1.0);
            } catch (Exception e) {
                Tui.println(Tui.color("  couldn't parse that — expected e.g.  0.5,-0.2 -> 1", Tui.RED));
            }
        }
        if (xs.size() < 2) {
            Tui.println(Tui.color("Too few points; falling back to moons.", Tui.YELLOW));
            return makeMoons(100, 0.1, new Random(1337));
        }
        double[][] xa = xs.toArray(new double[0][]);
        double[] ya = new double[ys.size()];
        for (int i = 0; i < ya.length; i++) ya[i] = ys.get(i);
        return new Dataset(xa, ya, "custom");
    }
}
