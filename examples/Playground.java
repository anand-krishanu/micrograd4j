///usr/bin/env jbang "$0" "$@" ; exit $?
//REPOS mavencentral,jitpack=https://jitpack.io
//DEPS com.github.anand-krishanu:micrograd4j:v1.1.0
//DEPS org.jline:jline:3.26.3
//JAVA_OPTIONS --enable-native-access=ALL-UNNAMED
//SOURCES playground/Tui.java playground/ExprParser.java playground/Charts.java
//SOURCES playground/Datasets.java playground/Trainer.java playground/GraphView.java
//
// Interactive micrograd4j playground. Zero-install:
//   jbang https://raw.githubusercontent.com/anand-krishanu/micrograd4j/main/examples/Playground.java
// Local:
//   jbang examples/Playground.java
// Non-interactive smoke run:
//   jbang examples/Playground.java --demo

import io.github.anandkrishanu.micrograd.Activation;
import io.github.anandkrishanu.micrograd.MLP;
import io.github.anandkrishanu.micrograd.Neuron;
import io.github.anandkrishanu.micrograd.Value;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Random;

public class Playground {
    static final Config cfg = new Config();
    static final ExprParser parser = new ExprParser();
    static Value lastExpr = null;

    public static void main(String[] args) {
        boolean demo = args.length > 0 && args[0].equals("--demo");
        Tui.init();
        try {
            if (demo) {
                runDemo();
            } else {
                mainLoop();
            }
        } finally {
            Tui.close();
        }
    }

    // ---------------------------------------------------------------- main menu
    private static void mainLoop() {
        while (true) {
            Tui.clear();
            banner();
            Tui.header("main menu");
            Tui.println("  " + Tui.color("[1]", Tui.YELLOW) + " Autograd playground   – type expressions, see gradients & graph");
            Tui.println("  " + Tui.color("[2]", Tui.YELLOW) + " Train a network       – live loss curve + decision boundary");
            Tui.println("  " + Tui.color("[3]", Tui.YELLOW) + " Step through backprop – watch gradients flow node by node");
            Tui.println("  " + Tui.color("[4]", Tui.YELLOW) + " Settings              – dataset & hyperparameters");
            Tui.println("  " + Tui.color("[q]", Tui.YELLOW) + " Quit");
            Tui.flush();
            String c = Tui.readLine("\n» ");
            if (c == null) return;
            switch (c.trim().toLowerCase()) {
                case "1": autogradPlayground(); break;
                case "2": trainScreen(); break;
                case "3": backpropScreen(); break;
                case "4": settingsScreen(); break;
                case "q": case "quit": case "exit": return;
                default: break;
            }
        }
    }

    private static void banner() {
        Tui.println(Tui.color("  micrograd4j", Tui.BOLD + Tui.CYAN)
                + Tui.color("  ·  a tiny autograd engine you can poke at", Tui.DIM));
        Tui.println();
    }

    // ----------------------------------------------------- 1. autograd playground
    private static void autogradPlayground() {
        Tui.clear();
        Tui.header("Autograd playground");
        playgroundHelp();
        while (true) {
            Tui.flush();
            String line = Tui.readLine(Tui.color("expr> ", Tui.CYAN));
            if (line == null) return;
            line = line.trim();
            if (line.isEmpty()) continue;
            if (line.equals(":back") || line.equalsIgnoreCase("q")) return;
            if (line.equals(":help")) { playgroundHelp(); continue; }
            if (line.equals(":vars")) { printVars(); continue; }
            if (line.equals(":graph")) {
                if (lastExpr == null) Tui.println(Tui.color("  evaluate an expression first", Tui.YELLOW));
                else Tui.print(GraphView.renderGraph(lastExpr, parser.names, null));
                continue;
            }
            if (line.equals(":step")) {
                if (lastExpr == null) Tui.println(Tui.color("  evaluate an expression first", Tui.YELLOW));
                else stepBackprop(lastExpr, parser.names, true);
                continue;
            }
            // assignment?  name = <number or expression>
            int eq = line.indexOf('=');
            if (eq > 0 && isIdentifier(line.substring(0, eq).trim())) {
                handleAssign(line.substring(0, eq).trim(), line.substring(eq + 1).trim());
                continue;
            }
            // otherwise evaluate
            try {
                Value root = parser.eval(line);
                root.zeroGrad();
                root.backward();
                lastExpr = root;
                printResult(root);
            } catch (RuntimeException e) {
                Tui.println(Tui.color("  error: " + e.getMessage(), Tui.RED));
            }
        }
    }

    private static void playgroundHelp() {
        Tui.println(Tui.color("  assign a variable:", Tui.DIM) + "  a = 2.0");
        Tui.println(Tui.color("  type an expression:", Tui.DIM) + " (a*b) + c.tanh()   or   relu(a) / 2");
        Tui.println(Tui.color("  functions:", Tui.DIM) + " tanh, relu, exp     operators: + - * / ^");
        Tui.println(Tui.color("  commands:", Tui.DIM) + "  :graph  :step  :vars  :help  :back");
        Tui.println();
    }

    private static void handleAssign(String name, String rhs) {
        if (!isIdentifier(name)) {
            Tui.println(Tui.color("  '" + name + "' is not a valid name", Tui.RED));
            return;
        }
        double val;
        try {
            val = Double.parseDouble(rhs);
        } catch (NumberFormatException nfe) {
            try {
                val = parser.eval(rhs).data;
            } catch (RuntimeException e) {
                Tui.println(Tui.color("  error: " + e.getMessage(), Tui.RED));
                return;
            }
        }
        parser.assign(name, val);
        Tui.println("  " + Tui.color(name, Tui.BOLD) + " = " + ExprParser.trimNum(val));
    }

    private static void printResult(Value root) {
        Tui.println("  = " + Tui.color(String.format("%.4f", root.data), Tui.BOLD + Tui.GREEN));
        // only report gradients for variables that actually appear in this expression
        java.util.Set<Value> inGraph = Collections.newSetFromMap(new java.util.IdentityHashMap<>());
        inGraph.addAll(GraphView.buildTopo(root));
        StringBuilder sb = new StringBuilder("  gradients: ");
        boolean any = false;
        for (Map.Entry<String, Value> e : parser.vars.entrySet()) {
            if (!inGraph.contains(e.getValue())) continue;
            double g = e.getValue().grad;
            sb.append(e.getKey()).append("=")
              .append(g >= 0 ? Tui.color(String.format("%+.3f", g), Tui.GREEN)
                             : Tui.color(String.format("%+.3f", g), Tui.RED))
              .append("  ");
            any = true;
        }
        if (any) Tui.println(sb.toString());
        Tui.println(Tui.color("  (:graph to see the computation graph, :step to walk backprop)", Tui.DIM));
    }

    private static void printVars() {
        if (parser.vars.isEmpty()) { Tui.println(Tui.color("  (no variables yet)", Tui.DIM)); return; }
        for (Map.Entry<String, Value> e : parser.vars.entrySet()) {
            Tui.println("  " + Tui.color(e.getKey(), Tui.BOLD) + " = " + ExprParser.trimNum(e.getValue().data));
        }
    }

    // ----------------------------------------------------------- 2. train screen
    private static void trainScreen() {
        Tui.clear();
        Tui.header("Train a network");
        printConfig();
        Tui.println();
        Dataset ds = Datasets.build(cfg);
        MLP model = Trainer.newModel(cfg);
        Tui.println("  " + Tui.color("dataset: " + ds.name + " (" + ds.xs.length + " pts)   model: "
                + chainString(model), Tui.DIM));
        Tui.pressEnter();

        final List<Double> lossHist = new ArrayList<>();
        final List<Double> accHist = new ArrayList<>();
        final int drawEvery = Math.max(1, cfg.epochs / 40);

        Trainer.train(cfg, model, ds.xs, ds.ys, (step, epochs, loss, acc, m) -> {
            lossHist.add(loss);
            accHist.add(acc);
            boolean lastStep = step == epochs;
            if (!Tui.dumb && (step % drawEvery == 0 || lastStep)) {
                Tui.clear();
                Tui.header("Training  (step " + step + "/" + epochs + ")");
                Tui.print(Charts.lossCurve(lossHist, chartWidth(), 12));
                Tui.println("  acc " + Tui.color(Charts.sparkline(accHist, chartWidth()), Tui.GREEN)
                        + "  " + Tui.color(String.format("%.0f%%", acc * 100), Tui.GREEN));
                Tui.println(String.format("  loss %s   acc %s",
                        Tui.color(String.format("%.4f", loss), Tui.CYAN),
                        Tui.color(String.format("%.0f%%", acc * 100), Tui.GREEN)));
                Tui.flush();
                sleep(20);
            }
        });

        double finalAcc = Trainer.accuracy(model, ds.xs, ds.ys);
        Tui.clear();
        Tui.header("Training complete");
        Tui.println(String.format("  final loss %s   accuracy %s   (%d epochs)",
                Tui.color(String.format("%.4f", lossHist.get(lossHist.size() - 1)), Tui.CYAN),
                Tui.color(String.format("%.0f%%", finalAcc * 100), Tui.BOLD + Tui.GREEN),
                cfg.epochs));
        Tui.println();
        Tui.println(Tui.color("  loss curve", Tui.BOLD));
        Tui.print(Charts.lossCurve(lossHist, chartWidth(), 10));
        Tui.println();
        Tui.println(Tui.color("  decision boundary", Tui.BOLD));
        Tui.print(Charts.decisionBoundary(model, ds.xs, ds.ys, boundaryCols(), boundaryRows()));
        Tui.println();
        Tui.println(Tui.color("  network", Tui.BOLD));
        Tui.print(GraphView.renderNetwork(model));
        Tui.pressEnter();
    }

    // --------------------------------------------------- 3. step-through backprop
    private static void backpropScreen() {
        Tui.clear();
        Tui.header("Step through backprop");
        Tui.println("  " + Tui.color("[1]", Tui.YELLOW) + " your last expression"
                + (lastExpr == null ? Tui.color("  (none yet — build one in the playground)", Tui.DIM) : ""));
        Tui.println("  " + Tui.color("[2]", Tui.YELLOW) + " a single neuron forward pass");
        Tui.println("  " + Tui.color("[b]", Tui.YELLOW) + " back");
        Tui.flush();
        String c = Tui.readLine("\n» ");
        if (c == null) return;
        switch (c.trim().toLowerCase()) {
            case "1":
                if (lastExpr == null) {
                    Tui.println(Tui.color("  Build an expression in the Autograd playground first.", Tui.YELLOW));
                    Tui.pressEnter();
                } else {
                    stepBackprop(lastExpr, parser.names, true);
                }
                break;
            case "2":
                singleNeuronBackprop();
                break;
            default:
                break;
        }
    }

    private static void singleNeuronBackprop() {
        Value x1 = new Value(0.5);
        Value x2 = new Value(-1.0);
        Neuron neuron = new Neuron(2, cfg.activation, new Random(cfg.seed));
        Value out = neuron.forward(List.of(x1, x2));

        java.util.IdentityHashMap<Value, String> names = new java.util.IdentityHashMap<>();
        names.put(x1, "x1");
        names.put(x2, "x2");
        names.put(neuron.w.get(0), "w1");
        names.put(neuron.w.get(1), "w2");
        names.put(neuron.b, "b");
        stepBackprop(out, names, true);
    }

    /**
     * Walk the backward pass parents-first, calling {@code _backward()} on each node and
     * re-rendering the graph so the user sees gradients appear. {@code interactive} waits for Enter.
     */
    private static void stepBackprop(Value root, Map<Value, String> names, boolean interactive) {
        List<Value> topo = GraphView.buildTopo(root);
        Collections.reverse(topo);          // parents before children, like Value.backward()
        root.zeroGrad();
        root.grad = 1.0;

        Tui.clear();
        Tui.header("Backprop — " + topo.size() + " nodes");
        Tui.println(Tui.color("  output grad seeded to 1.0; each step applies one node's local backward.", Tui.DIM));
        Tui.print(GraphView.renderGraph(root, names, null));

        for (Value v : topo) {
            if (interactive) {
                String k = Tui.readLine(Tui.color("  [Enter] step   [q] stop  ", Tui.DIM));
                if (k != null && k.trim().equalsIgnoreCase("q")) break;
                v._backward();
                Tui.clear();
                Tui.header("Backprop — applied " + opName(v) + " (data=" + String.format("%.3f", v.data) + ")");
                Tui.print(GraphView.renderGraph(root, names, v));
            } else {
                v._backward();
                Tui.println("  applied " + Tui.color(opName(v), Tui.BOLD)
                        + "  data=" + String.format("%.3f", v.data)
                        + "  grad=" + String.format("%+.3f", v.grad));
            }
        }
        if (!interactive) Tui.print(GraphView.renderGraph(root, names, null));

        Tui.println(Tui.color("  done. input gradients:", Tui.BOLD));
        for (Map.Entry<Value, String> e : names.entrySet()) {
            if (e.getValue() != null) {
                double g = e.getKey().grad;
                Tui.println("    " + e.getValue() + " : "
                        + (g >= 0 ? Tui.color(String.format("%+.4f", g), Tui.GREEN)
                                  : Tui.color(String.format("%+.4f", g), Tui.RED)));
            }
        }
        if (interactive) Tui.pressEnter();
    }

    private static String opName(Value v) {
        if (v.op == null || v.op.isEmpty()) return "leaf";
        return v.op;
    }

    // -------------------------------------------------------------- 4. settings
    private static void settingsScreen() {
        Tui.clear();
        Tui.header("Settings");
        printConfig();
        Tui.println();
        Tui.println(Tui.color("  Press Enter to keep the current value.", Tui.DIM));

        String ds = Tui.readLine("  dataset (moons/xor/circles/custom) [" + cfg.dataset + "]: ");
        if (ds != null && !ds.trim().isEmpty()) {
            ds = ds.trim().toLowerCase();
            if (ds.equals("moons") || ds.equals("xor") || ds.equals("circles") || ds.equals("custom")) cfg.dataset = ds;
            else Tui.println(Tui.color("  unknown dataset, keeping " + cfg.dataset, Tui.YELLOW));
        }
        cfg.samples = Tui.readInt("  samples", cfg.samples);
        cfg.noise = Tui.readDouble("  noise", cfg.noise);

        String h = Tui.readLine("  hidden layers (e.g. 16,16) [" + cfg.hiddenString() + "]: ");
        if (h != null && !h.trim().isEmpty()) cfg.setHidden(h.trim());

        String act = Tui.readLine("  activation (tanh/relu) [" + cfg.activation.name().toLowerCase() + "]: ");
        if (act != null && !act.trim().isEmpty()) {
            cfg.activation = parseActivation(act.trim(), cfg.activation);
        }
        cfg.epochs = Tui.readInt("  epochs", cfg.epochs);
        cfg.baseLr = Tui.readDouble("  learning rate", cfg.baseLr);
        cfg.seed = Tui.readInt("  seed", (int) cfg.seed);

        Tui.println(Tui.color("  settings saved.", Tui.GREEN));
        Tui.pressEnter();
    }

    private static Activation parseActivation(String s, Activation def) {
        switch (s.toLowerCase()) {
            case "tanh": return Activation.TANH;
            case "relu": return Activation.RELU;
            default:
                Tui.println(Tui.color("  unknown activation, keeping " + def.name().toLowerCase(), Tui.YELLOW));
                return def;
        }
    }

    private static void printConfig() {
        Tui.println("  dataset=" + cfg.dataset + "  samples=" + cfg.samples + "  noise=" + cfg.noise);
        Tui.println("  hidden=[" + cfg.hiddenString() + "]  activation=" + cfg.activation.name().toLowerCase()
                + "  epochs=" + cfg.epochs + "  lr=" + cfg.baseLr + "  seed=" + cfg.seed);
    }

    private static String chainString(MLP model) {
        StringBuilder sb = new StringBuilder("2");
        for (int i = 0; i < cfg.hidden.length; i++) sb.append("→").append(cfg.hidden[i]);
        sb.append("→1");
        return sb.toString();
    }

    // ----- layout helpers -----
    private static int chartWidth() { return Math.max(20, Math.min(Tui.width() - 12, 60)); }
    private static int boundaryCols() { return Math.max(20, Math.min(Tui.width() - 6, 56)); }
    private static int boundaryRows() { return Math.max(10, Math.min(Tui.height() - 10, 22)); }

    private static boolean isIdentifier(String s) {
        if (s.isEmpty() || !Character.isLetter(s.charAt(0))) return false;
        for (int i = 1; i < s.length(); i++) {
            char ch = s.charAt(i);
            if (!Character.isLetterOrDigit(ch) && ch != '_') return false;
        }
        return !s.equals("tanh") && !s.equals("relu") && !s.equals("exp");
    }

    private static void sleep(long ms) {
        try { Thread.sleep(ms); } catch (InterruptedException ignored) { Thread.currentThread().interrupt(); }
    }

    // ---------------------------------------------------------------- demo mode
    private static void runDemo() {
        Tui.println(Tui.color("=== micrograd4j playground — demo ===", Tui.BOLD + Tui.CYAN));

        // 1) autograd
        Tui.println(Tui.color("\n[1] Autograd playground", Tui.BOLD));
        parser.assign("a", 2.0);
        parser.assign("b", -3.0);
        parser.assign("c", 10.0);
        Value root = parser.eval("(a*b) + c");
        root.zeroGrad();
        root.backward();
        lastExpr = root;
        Tui.println("  (a*b) + c  with a=2, b=-3, c=10");
        printResult(root);
        Tui.print(GraphView.renderGraph(root, parser.names, null));

        Value t = parser.eval("a.tanh() + relu(b)");
        t.zeroGrad();
        t.backward();
        Tui.println("  a.tanh() + relu(b)");
        printResult(t);

        // 2) training
        Tui.println(Tui.color("\n[2] Train a network", Tui.BOLD));
        cfg.epochs = 50;
        cfg.samples = 80;
        Dataset dsData = Datasets.build(cfg);
        MLP model = Trainer.newModel(cfg);
        List<Double> lossHist = new ArrayList<>();
        Trainer.train(cfg, model, dsData.xs, dsData.ys, (step, epochs, loss, acc, m) -> {
            lossHist.add(loss);
            if (step % 10 == 0 || step == epochs) {
                Tui.println(String.format("  step %3d  loss %.4f  acc %.0f%%", step, loss, acc * 100));
            }
        });
        Tui.println("\n  loss curve");
        Tui.print(Charts.lossCurve(lossHist, 50, 8));
        Tui.println("  decision boundary");
        Tui.print(Charts.decisionBoundary(model, dsData.xs, dsData.ys, 48, 16));
        Tui.print(GraphView.renderNetwork(model));

        // 3) backprop stepping (auto)
        Tui.println(Tui.color("\n[3] Step through backprop (single neuron)", Tui.BOLD));
        Value x1 = new Value(0.5), x2 = new Value(-1.0);
        Neuron neuron = new Neuron(2, cfg.activation, new Random(cfg.seed));
        Value out = neuron.forward(List.of(x1, x2));
        java.util.IdentityHashMap<Value, String> names = new java.util.IdentityHashMap<>();
        names.put(x1, "x1"); names.put(x2, "x2");
        names.put(neuron.w.get(0), "w1"); names.put(neuron.w.get(1), "w2");
        names.put(neuron.b, "b");
        stepBackprop(out, names, false);

        Tui.println(Tui.color("\n=== demo complete ===", Tui.BOLD + Tui.CYAN));
        Tui.flush();
    }
}
