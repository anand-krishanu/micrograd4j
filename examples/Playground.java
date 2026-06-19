///usr/bin/env jbang "$0" "$@" ; exit $?
//REPOS mavencentral,jitpack=https://jitpack.io
//DEPS com.github.anand-krishanu:micrograd4j:v1.2.0
//DEPS org.jline:jline:3.26.3
//JAVA_OPTIONS --enable-native-access=ALL-UNNAMED
//SOURCES playground/Tui.java playground/ExprParser.java playground/Charts.java
//SOURCES playground/Datasets.java playground/Trainer.java playground/GraphView.java
//SOURCES playground/Menu.java playground/Braille.java playground/Explain.java
//SOURCES playground/GraphInteractive.java
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
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

public class Playground {
    static final Config cfg = new Config();
    static final ExprParser parser = new ExprParser();
    static Value lastExpr = null;
    static boolean explainMode = false;

    static final String BANNER = "▲ micrograd4j\na tiny autograd engine you can poke at";

    public static void main(String[] args) {
        boolean demo = args.length > 0 && args[0].equals("--demo");
        Tui.init();
        Tui.dynamicWords = parser.vars::keySet;   // feed live variable names to tab-completion
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
        List<Menu.Item> items = List.of(
                Menu.Item.of("Autograd playground", "type expressions, see gradients & graph"),
                Menu.Item.of("Train a network", "live loss curve + decision boundary"),
                Menu.Item.of("Step through backprop", "watch gradients flow node by node"),
                Menu.Item.of("Learn how autograd works", "a one-minute guided tour"),
                Menu.Item.of("Settings", "dataset & hyperparameters"),
                Menu.Item.of("Quit", "")
        );
        while (true) {
            int c = Menu.select(BANNER, "arrow keys move · Enter selects · q quits", items, null);
            switch (c) {
                case 0: autogradPlayground(); break;
                case 1: trainScreen(); break;
                case 2: backpropScreen(); break;
                case 3: learnScreen(); break;
                case 4: settingsScreen(); break;
                case 5: case -1: return;     // Quit, or q / Esc
                default: break;              // -2 (dumb: unrecognised): loop and re-render
            }
        }
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
            if (line.equals(":examples")) { examplesScreen(); continue; }
            if (line.equals(":explain")) {
                explainMode = !explainMode;
                Tui.println(Tui.color("  explain mode " + (explainMode ? "on" : "off"), Tui.YELLOW)
                        + Tui.color("  (per-op chain rule printed after each result)", Tui.DIM));
                continue;
            }
            if (line.equals(":graph")) {
                if (lastExpr == null) Tui.println(Tui.color("  evaluate an expression first", Tui.YELLOW));
                else { GraphInteractive.view(lastExpr, parser.names); redrawPlaygroundHeader(); }
                continue;
            }
            if (line.equals(":step")) {
                if (lastExpr == null) Tui.println(Tui.color("  evaluate an expression first", Tui.YELLOW));
                else { GraphInteractive.backprop(lastExpr, parser.names); redrawPlaygroundHeader(); }
                continue;
            }
            // assignment?  name = <number or expression>
            int eq = line.indexOf('=');
            if (eq > 0 && isIdentifier(line.substring(0, eq).trim())) {
                handleAssign(line.substring(0, eq).trim(), line.substring(eq + 1).trim());
                continue;
            }
            evaluate(line);
        }
    }

    private static void redrawPlaygroundHeader() {
        Tui.clear();
        Tui.header("Autograd playground");
        Tui.println(Tui.color("  back in the playground — type an expression, or :help / :examples / :back", Tui.DIM));
    }

    private static void playgroundHelp() {
        Tui.println(Tui.color("  assign a variable:", Tui.DIM) + "  a = 2.0");
        Tui.println(Tui.color("  type an expression:", Tui.DIM) + " (a*b) + c.tanh()   or   relu(a) / 2");
        Tui.println(Tui.color("  functions:", Tui.DIM) + " tanh, relu, exp     operators: + - * / ^");
        Tui.println(Tui.color("  commands:", Tui.DIM) + "  :examples  :graph  :step  :explain  :vars  :help  :back");
        Tui.println();
    }

    /** Parse + backward + show; the shared path for typed expressions and presets. */
    private static void evaluate(String line) {
        try {
            Value root = parser.eval(line);
            root.zeroGrad();
            root.backward();
            lastExpr = root;
            printResult(root);
            if (explainMode) printExplain(root);
        } catch (RuntimeException e) {
            showParseError(line, e.getMessage());
        }
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
                showParseError(rhs, e.getMessage());
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

    /** Print the per-op chain rule for every internal node (root first). */
    private static void printExplain(Value root) {
        List<Value> topo = GraphView.buildTopo(root);
        Collections.reverse(topo);
        Tui.println(Tui.color("  how each op passes gradient back:", Tui.DIM));
        for (Value v : topo) {
            if (v.prev.isEmpty()) continue;
            Tui.println(Tui.color("    " + Explain.localRule(v), Tui.DIM));
        }
    }

    private static void showParseError(String line, String msg) {
        Tui.println(Tui.color("  error: " + msg, Tui.RED));
        int col = extractColumn(msg);
        if (col >= 1 && col <= line.length() + 1) {
            Tui.println("  " + line);
            Tui.println("  " + " ".repeat(col - 1) + Tui.color("^", Tui.BOLD + Tui.RED));
        }
    }

    private static int extractColumn(String msg) {
        java.util.regex.Matcher m = java.util.regex.Pattern.compile("column (\\d+)").matcher(msg);
        return m.find() ? Integer.parseInt(m.group(1)) : -1;
    }

    private static void printVars() {
        if (parser.vars.isEmpty()) { Tui.println(Tui.color("  (no variables yet)", Tui.DIM)); return; }
        for (Map.Entry<String, Value> e : parser.vars.entrySet()) {
            Tui.println("  " + Tui.color(e.getKey(), Tui.BOLD) + " = " + ExprParser.trimNum(e.getValue().data));
        }
    }

    // ----- expression presets -----
    private record Preset(String desc, String expr, LinkedHashMap<String, Double> vars) {}

    private static Preset preset(String desc, String expr, Object... kv) {
        LinkedHashMap<String, Double> m = new LinkedHashMap<>();
        for (int i = 0; i + 1 < kv.length; i += 2) m.put((String) kv[i], ((Number) kv[i + 1]).doubleValue());
        return new Preset(desc, expr, m);
    }

    private static final List<Preset> PRESETS = List.of(
            preset("two ops: (a*b) + c", "(a*b) + c", "a", 2.0, "b", -3.0, "c", 10.0),
            preset("a polynomial: x² + 3x + 1   (dy/dx = 2x+3)", "x^2 + 3*x + 1", "x", -4.0),
            preset("a tiny neuron: tanh(w1·x1 + w2·x2 + b)", "(w1*x1 + w2*x2 + b).tanh()",
                    "w1", 0.5, "x1", 1.0, "w2", -0.3, "x2", 2.0, "b", 0.1),
            preset("mixing activations: a.tanh() + relu(b)", "a.tanh() + relu(b)", "a", 0.8, "b", -1.5),
            preset("a sigmoid: exp(z) / (exp(z) + 1)", "exp(z) / (exp(z) + 1)", "z", 0.5)
    );

    private static void examplesScreen() {
        List<Menu.Item> items = new ArrayList<>();
        for (Preset p : PRESETS) items.add(Menu.Item.of(p.expr(), p.desc()));
        int idx = Menu.select("Example expressions", "pick one to load its variables and evaluate", items, null);
        if (idx < 0) { Tui.clear(); Tui.header("Autograd playground"); playgroundHelp(); return; }
        Preset p = PRESETS.get(idx);
        for (Map.Entry<String, Double> e : p.vars().entrySet()) parser.assign(e.getKey(), e.getValue());

        Tui.clear();
        Tui.header("Autograd playground");
        Tui.println(Tui.color("  loaded: ", Tui.DIM) + p.desc());
        StringBuilder a = new StringBuilder("  ");
        for (Map.Entry<String, Double> e : p.vars().entrySet()) {
            a.append(Tui.color(e.getKey(), Tui.BOLD)).append("=").append(ExprParser.trimNum(e.getValue())).append("  ");
        }
        Tui.println(a.toString());
        Tui.println("  " + Tui.color("expr> ", Tui.CYAN) + p.expr());
        evaluate(p.expr());
    }

    // ------------------------------------------------------------- Learn screen
    private static void learnScreen() {
        List<String> pages = Explain.intro();
        for (int i = 0; i < pages.size(); i++) {
            Tui.clear();
            Tui.panel("How autograd works   (" + (i + 1) + "/" + pages.size() + ")", pages.get(i));
            Tui.println();
            if (Tui.dumb) continue;
            String k = Tui.readLine(Tui.color("  [Enter] next   [q] back  ", Tui.DIM));
            if (k != null && k.trim().equalsIgnoreCase("q")) return;
        }
        if (!Tui.dumb) Tui.pressEnter();
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
        final int[] tick = {0};
        if (!Tui.dumb) Tui.clear();

        Trainer.train(cfg, model, ds.xs, ds.ys, (step, epochs, loss, acc, m) -> {
            lossHist.add(loss);
            accHist.add(acc);
            boolean lastStep = step == epochs;
            if (!Tui.dumb && (step % drawEvery == 0 || lastStep)) {
                tick[0]++;
                double frac = epochs == 0 ? 1.0 : (double) step / epochs;
                double lr = cfg.baseLr * (1.0 - 0.9 * step / Math.max(1, epochs));
                StringBuilder f = new StringBuilder();
                f.append(frameLine(Tui.color("  Training a network", Tui.BOLD + Tui.CYAN)));
                f.append(frameLine("  " + Tui.color(Tui.spinner(tick[0]), Tui.CYAN)
                        + String.format("  step %d/%d   ", step, epochs)
                        + Tui.progressBar(frac, 22) + String.format(" %3.0f%%", frac * 100)));
                f.append(frameLine(""));
                f.append(frameBlock(Charts.lossCurve(lossHist, chartWidth(), 10)));
                f.append(frameLine("  acc " + Tui.color(Charts.sparkline(accHist, chartWidth()), Tui.GREEN)
                        + "  " + Tui.color(String.format("%.0f%%", acc * 100), Tui.GREEN)));
                f.append(frameLine(String.format("  loss %s    lr %s",
                        Tui.color(String.format("%.4f", loss), Tui.CYAN),
                        Tui.color(String.format("%.3f", lr), Tui.DIM))));
                Tui.print(Tui.HOME + f + Tui.CLR_DOWN);
                Tui.flush();
                sleep(20);
            }
        });

        double finalAcc = Trainer.accuracy(model, ds.xs, ds.ys);
        Tui.clear();
        Tui.panel("Training complete", String.format("final loss %s    accuracy %s    (%d epochs)",
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

    private static String frameLine(String s) {
        return s + Tui.CLR_EOL + "\n";
    }

    private static String frameBlock(String multi) {
        if (multi.endsWith("\n")) multi = multi.substring(0, multi.length() - 1);
        StringBuilder sb = new StringBuilder();
        for (String l : multi.split("\n", -1)) sb.append(l).append(Tui.CLR_EOL).append('\n');
        return sb.toString();
    }

    // --------------------------------------------------- 3. step-through backprop
    private static void backpropScreen() {
        List<Menu.Item> items = List.of(
                Menu.Item.of("Your last expression",
                        lastExpr == null ? "(none yet — build one in the playground)" : "walk its backward pass"),
                Menu.Item.of("A single neuron forward pass", "tanh(w1·x1 + w2·x2 + b)")
        );
        int c = Menu.select("Step through backprop", "apply one node's local backward at a time", items, null);
        switch (c) {
            case 0:
                if (lastExpr == null) {
                    Tui.clear();
                    Tui.header("Step through backprop");
                    Tui.println(Tui.color("  Build an expression in the Autograd playground first.", Tui.YELLOW));
                    Tui.pressEnter();
                } else {
                    GraphInteractive.backprop(lastExpr, parser.names);
                }
                break;
            case 1:
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
        GraphInteractive.backprop(out, names);
    }

    // -------------------------------------------------------------- 4. settings
    private static void settingsScreen() {
        List<Menu.Item> items = List.of(
                Menu.Item.of("Edit fields one by one", "dataset, layers, activation, epochs, lr, seed"),
                Menu.Item.of("Preset · Quick", "60 pts · [8] · 60 epochs — fast"),
                Menu.Item.of("Preset · Balanced", "100 pts · [16,16] · 100 epochs — default"),
                Menu.Item.of("Preset · Thorough", "200 pts · [16,16,8] · 250 epochs — slow"),
                Menu.Item.of("Back", "")
        );
        int c = Menu.select("Settings", "current:  " + oneLineConfig(), items, null);
        switch (c) {
            case 0: editSettings(); break;
            case 1: applyPreset(60, new int[]{8}, 60); break;
            case 2: applyPreset(100, new int[]{16, 16}, 100); break;
            case 3: applyPreset(200, new int[]{16, 16, 8}, 250); break;
            default: break;
        }
    }

    private static void applyPreset(int samples, int[] hidden, int epochs) {
        cfg.samples = samples;
        cfg.hidden = hidden;
        cfg.epochs = epochs;
        cfg.activation = Activation.TANH;
        Tui.clear();
        Tui.header("Settings");
        Tui.println(Tui.color("  preset applied.", Tui.GREEN));
        Tui.println();
        printConfig();
        Tui.pressEnter();
    }

    private static void editSettings() {
        Tui.clear();
        Tui.header("Settings");
        printConfig();
        Tui.println();
        Tui.println(Tui.color("  Press Enter to keep the current value.", Tui.DIM));

        String ds = Tui.readLine("  dataset (moons/xor/circles/custom) ["
                + Tui.color(cfg.dataset, Tui.BOLD) + "]: ");
        if (ds != null && !ds.trim().isEmpty()) {
            ds = ds.trim().toLowerCase();
            if (ds.equals("moons") || ds.equals("xor") || ds.equals("circles") || ds.equals("custom")) cfg.dataset = ds;
            else Tui.println(Tui.color("  unknown dataset, keeping " + cfg.dataset, Tui.YELLOW));
        }
        cfg.samples = Tui.readInt("  samples", cfg.samples);
        cfg.noise = Tui.readDouble("  noise", cfg.noise);

        String h = Tui.readLine("  hidden layers (e.g. 16,16) ["
                + Tui.color(cfg.hiddenString(), Tui.BOLD) + "]: ");
        if (h != null && !h.trim().isEmpty()) cfg.setHidden(h.trim());

        String act = Tui.readLine("  activation (tanh/relu) ["
                + Tui.color(cfg.activation.name().toLowerCase(), Tui.BOLD) + "]: ");
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

    private static String oneLineConfig() {
        return cfg.dataset + " · [" + cfg.hiddenString() + "] · " + cfg.activation.name().toLowerCase()
                + " · " + cfg.epochs + " ep";
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
        GraphInteractive.backpropStatic(out, names);

        Tui.println(Tui.color("\n=== demo complete ===", Tui.BOLD + Tui.CYAN));
        Tui.flush();
    }
}
