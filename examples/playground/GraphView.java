import io.github.anandkrishanu.micrograd.Activation;
import io.github.anandkrishanu.micrograd.Layer;
import io.github.anandkrishanu.micrograd.MLP;
import io.github.anandkrishanu.micrograd.Neuron;
import io.github.anandkrishanu.micrograd.Value;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/** ASCII rendering of the autograd DAG and the MLP architecture. Reads only public fields. */
final class GraphView {
    private GraphView() {}

    /** Children-before-parents topological order, built from the public {@code prev} sets. */
    static List<Value> buildTopo(Value root) {
        List<Value> topo = new ArrayList<>();
        Set<Value> visited = Collections.newSetFromMap(new IdentityHashMap<>());
        build(root, visited, topo);
        return topo;
    }

    private static void build(Value v, Set<Value> visited, List<Value> topo) {
        if (visited.add(v)) {
            for (Value c : v.prev) build(c, visited, topo);
            topo.add(v);
        }
    }

    /**
     * Nodes in the same order {@link #renderGraph} prints them (first-occurrence pre-order, children
     * sorted by data then op). Index {@code i} corresponds to the rendered id {@code #(i+1)}.
     */
    static List<Value> displayOrder(Value root) {
        List<Value> order = new ArrayList<>();
        Set<Value> shown = Collections.newSetFromMap(new IdentityHashMap<>());
        visitDisplay(root, order, shown);
        return order;
    }

    private static void visitDisplay(Value v, List<Value> order, Set<Value> shown) {
        if (!shown.add(v)) return;
        order.add(v);
        List<Value> children = new ArrayList<>(v.prev);
        children.sort(Comparator.comparingDouble((Value c) -> c.data).thenComparing(c -> c.op));
        for (Value c : children) visitDisplay(c, order, shown);
    }

    // ----- computation graph -----
    // gradient heat ramps (dim → vivid), matching the decision-boundary heatmap's visual language
    private static final int[] POS_RAMP = {22, 28, 34, 40, 46};    // greens for +grad
    private static final int[] NEG_RAMP = {52, 88, 124, 160, 196};  // reds for −grad

    /** Render the graph rooted at {@code root} as an ASCII tree. {@code highlight} (nullable) is marked. */
    static String renderGraph(Value root, Map<Value, String> names, Value highlight) {
        return renderGraph(root, names, highlight, false);
    }

    /**
     * Rich tree render. Each node shows a coloured op chip, its data, and a gradient heat-gauge
     * (bar length ∝ |grad|, colour on a green/red ramp). When {@code backpropMode} is set, nodes that
     * gradient has not reached yet (grad ≈ 0) are dimmed, so {@code :step} shows a visible flow front.
     */
    static String renderGraph(Value root, Map<Value, String> names, Value highlight, boolean backpropMode) {
        StringBuilder sb = new StringBuilder();
        Map<Value, Integer> ids = new IdentityHashMap<>();
        Set<Value> shown = Collections.newSetFromMap(new IdentityHashMap<>());
        int[] counter = {0};
        double maxAbs = 0;
        for (Value v : buildTopo(root)) maxAbs = Math.max(maxAbs, Math.abs(v.grad));
        renderNode(root, "", true, true, sb, names, ids, shown, counter, highlight, backpropMode, maxAbs);
        return sb.toString();
    }

    private static void renderNode(Value v, String prefix, boolean isLast, boolean isRoot,
                                   StringBuilder sb, Map<Value, String> names,
                                   Map<Value, Integer> ids, Set<Value> shown,
                                   int[] counter, Value highlight, boolean backpropMode, double maxAbs) {
        String rawConn = isRoot ? "" : (isLast ? "└─ " : "├─ ");
        String connector = Tui.dumb ? rawConn : Tui.color(rawConn, Tui.GRAY);
        String marker = v == highlight
                ? (Tui.dumb ? "▶ " : Tui.color("▸ ", Tui.BOLD + Tui.YELLOW))
                : "  ";

        if (shown.contains(v)) {
            int seenId = ids.get(v);
            if (Tui.dumb) {
                sb.append(prefix).append(rawConn).append("↑ #").append(seenId).append(' ')
                  .append(label(v, names)).append('\n');
            } else {
                sb.append(prefix).append(connector)
                  .append(Tui.color("↑ #" + seenId + " " + label(v, names), Tui.GRAY)).append('\n');
            }
            return;
        }
        int id = ++counter[0];
        ids.put(v, id);
        shown.add(v);

        if (Tui.dumb) {
            sb.append(prefix).append(rawConn).append(marker)
              .append("#").append(id).append(' ').append(label(v, names))
              .append("  data=").append(fmt(v.data))
              .append("  grad=").append(String.format("%+.3f", v.grad)).append('\n');
        } else {
            boolean lit = !backpropMode || Math.abs(v.grad) > 1e-12;
            String data = Tui.color(fmt(v.data), lit ? Tui.DIM : Tui.GRAY);
            String grad = lit ? gradColored(v.grad)
                              : Tui.color(String.format("%+.3f", v.grad), Tui.GRAY);
            sb.append(prefix).append(connector).append(marker)
              .append(Tui.color("#" + id, Tui.GRAY)).append(' ')
              .append(opChip(v, names, lit)).append("  ")
              .append(data).append("   ")
              .append(gradGauge(v.grad, maxAbs, 6, lit)).append(' ').append(grad).append('\n');
        }

        List<Value> children = new ArrayList<>(v.prev);
        children.sort(Comparator.comparingDouble((Value c) -> c.data).thenComparing(c -> c.op));
        String bar = Tui.dumb ? "│  " : Tui.color("│", Tui.GRAY) + "  ";
        String childPrefix = prefix + (isRoot ? "" : (isLast ? "   " : bar));
        for (int i = 0; i < children.size(); i++) {
            renderNode(children.get(i), childPrefix, i == children.size() - 1, false,
                    sb, names, ids, shown, counter, highlight, backpropMode, maxAbs);
        }
    }

    /** A coloured chip identifying a node: the variable name, a constant value, or the op symbol. */
    private static String opChip(Value v, Map<Value, String> names, boolean lit) {
        String padded = " " + chipText(v, names) + " ";
        if (!lit) return Tui.color(padded, Tui.DIM);
        int[] c = chipColor(v, names);
        return Tui.chip(padded, c[0], c[1]);
    }

    /** The bare text shown inside a node's chip (no padding, no colour). */
    static String chipText(Value v, Map<Value, String> names) {
        String name = names == null ? null : names.get(v);
        if (name != null) return name;
        if (v.prev.isEmpty()) return ExprParser.trimNum(v.data);   // constant — show its value
        String op = v.op == null ? "" : v.op;
        if (op.startsWith("^")) return "^" + op.substring(1);
        switch (op) {
            case "+":            return "+";
            case "-":            return "−";
            case "*":            return "×";
            case "/":            return "÷";
            case "tanh":         return "tanh";
            case "relu": case "ReLU": case "RELU": return "relu";
            case "exp":          return "exp";
            default:             return op.isEmpty() ? "·" : op;
        }
    }

    /** {bg, fg} 256-colour pair for a node's chip, keyed by what the node is. */
    private static int[] chipColor(Value v, Map<Value, String> names) {
        if (names != null && names.get(v) != null) return new int[]{24, 231};   // input variable
        if (v.prev.isEmpty()) return new int[]{240, 233};                        // constant
        String op = v.op == null ? "" : v.op;
        if (op.startsWith("^")) return new int[]{130, 231};
        switch (op) {
            case "+": case "-":                    return new int[]{26, 231};
            case "*":                              return new int[]{91, 231};
            case "/":                              return new int[]{55, 231};
            case "tanh":                           return new int[]{30, 231};
            case "relu": case "ReLU": case "RELU": return new int[]{22, 231};
            case "exp":                            return new int[]{94, 231};
            default:                               return new int[]{238, 231};
        }
    }

    /** A small heat-bar: length ∝ |grad|/maxAbs, coloured green (+) / red (−), brighter = larger. */
    private static String gradGauge(double g, double maxAbs, int w, boolean lit) {
        double f = maxAbs < 1e-12 ? 0.0 : Math.min(1.0, Math.abs(g) / maxAbs);
        int full = Math.max(0, Math.min(w, (int) Math.round(f * w)));
        String empty = Tui.color("░".repeat(w - full), Tui.GRAY);
        if (full == 0) return empty;
        if (!lit) return Tui.color("▒".repeat(full), Tui.GRAY) + empty;
        int lvl = Math.max(0, Math.min(POS_RAMP.length - 1, (int) Math.round(f * (POS_RAMP.length - 1))));
        int code = g >= 0 ? POS_RAMP[lvl] : NEG_RAMP[lvl];
        return Tui.color256("█".repeat(full), code, g >= 0 ? Tui.GREEN : Tui.RED) + empty;
    }

    /** The lit chip for a node, exposed so the interactive views can echo the selected/applied node. */
    static String badge(Value v, Map<Value, String> names) {
        return opChip(v, names, true);
    }

    /** A one-line key explaining the heat-gauge, for the bottom of the interactive views. */
    static String gradLegend() {
        if (Tui.dumb) return "gradient: longer bar = larger |grad|; green +, red −";
        return Tui.color("bar ∝ |grad|   ", Tui.DIM)
             + Tui.color256("███", NEG_RAMP[4], Tui.RED) + Tui.color(" −", Tui.DIM)
             + "   " + Tui.color256("███", POS_RAMP[4], Tui.GREEN) + Tui.color(" +", Tui.DIM)
             + Tui.color("    ▸ ", Tui.BOLD + Tui.YELLOW) + Tui.color("current node", Tui.DIM);
    }

    // ===================================================== horizontal (left→right) layout
    // A node-link diagram: inputs on the left, output on the right, chips joined by box-drawing
    // connectors. Edges are routed on a direction-bitmask canvas (so junctions pick the right glyph)
    // with one vertical "bus" lane per node, which keeps converging edges readable.
    private static final int DN = 1, DE = 2, DS = 4, DW = 8;   // connector direction bits
    private static final int ROW_STEP = 3;
    private static final int GAUGE_W = 4;                       // per-node grad heat-bar width

    /**
     * Render the graph as a horizontal node-link diagram. Returns {@code null} when it can't be drawn
     * nicely (dumb terminal, or it wouldn't fit the window) so callers fall back to {@link #renderGraph}.
     * {@code highlight} is drawn in bright yellow; in {@code backpropMode} un-reached nodes are dimmed.
     */
    static String renderLayered(Value root, Map<Value, String> names, Value highlight, boolean backpropMode) {
        if (Tui.dumb) return null;
        List<Value> topo = buildTopo(root);                      // children before parents

        Map<Value, Integer> col = new IdentityHashMap<>();        // column = longest path from a leaf
        int maxCol = 0;
        for (Value v : topo) {
            int d = 0;
            for (Value c : v.prev) d = Math.max(d, col.get(c) + 1);
            col.put(v, d);
            maxCol = Math.max(maxCol, d);
        }
        List<List<Value>> cols = new ArrayList<>();
        for (int i = 0; i <= maxCol; i++) cols.add(new ArrayList<>());
        for (Value v : topo) cols.get(col.get(v)).add(v);

        Map<Value, List<Value>> parents = new IdentityHashMap<>();
        for (Value v : topo) parents.put(v, new ArrayList<>());
        for (Value v : topo) for (Value c : v.prev) parents.get(c).add(v);

        // rows: leaves stack; internal nodes sit at the average of their children, then de-overlap
        Map<Value, Integer> y = new IdentityHashMap<>();
        for (int c = 0; c <= maxCol; c++) {
            List<Value> g = cols.get(c);
            if (c == 0) {
                for (int i = 0; i < g.size(); i++) y.put(g.get(i), i * ROW_STEP);
            } else {
                for (Value v : g) {
                    int sum = 0, k = 0;
                    for (Value ch : v.prev) { sum += y.get(ch); k++; }
                    y.put(v, k == 0 ? 0 : Math.round((float) sum / k));
                }
                g.sort(Comparator.comparingInt(y::get));
                int prev = Integer.MIN_VALUE;
                for (Value v : g) { int yy = Math.max(y.get(v), prev + ROW_STEP); y.put(v, yy); prev = yy; }
            }
        }

        double maxAbs = 0;                                        // for the per-node grad heat-bars
        for (Value v : topo) maxAbs = Math.max(maxAbs, Math.abs(v.grad));

        int[] cw = new int[maxCol + 1];                           // column width = widest chip + gauge
        for (int c = 0; c <= maxCol; c++) {
            int w = 1;
            for (Value v : cols.get(c)) w = Math.max(w, chipText(v, names).length() + 2);
            cw[c] = w + 1 + GAUGE_W;
        }
        int[] colX = new int[maxCol + 1];
        for (int c = 1; c <= maxCol; c++) {
            int lanes = 0;
            for (Value v : cols.get(c - 1)) if (!parents.get(v).isEmpty()) lanes++;
            colX[c] = colX[c - 1] + cw[c - 1] + Math.max(4, lanes + 3);
        }
        int width = colX[maxCol] + cw[maxCol] + 1;
        int height = 1;
        for (Value v : topo) height = Math.max(height, y.get(v) + 1);
        // the output row also carries a "◂ output = <value>" tag; make sure that fits too
        int outRow = colX[maxCol] + chipText(root, names).length() + 3 + GAUGE_W + outTagPlain(root).length();
        if (Math.max(width, outRow) > Tui.width() - 2 || height > Tui.height() - 10) return null;

        int[][] mask = new int[height][width];
        for (int c = 0; c <= maxCol; c++) {
            int lane = 0;
            for (Value ch : cols.get(c)) {
                List<Value> ps = parents.get(ch);
                if (ps.isEmpty()) continue;
                int sx = colX[c] + chipText(ch, names).length() + 3 + GAUGE_W;   // right edge: chip + gauge
                int sy = y.get(ch);
                int busX = sx + 1 + lane++;                            // detached vertical lane
                int top = sy, bot = sy;
                for (Value p : ps) { top = Math.min(top, y.get(p)); bot = Math.max(bot, y.get(p)); }
                hseg(mask, sy, sx, busX);
                vseg(mask, busX, top, bot);
                for (Value p : ps) hseg(mask, y.get(p), busX, colX[col.get(p)] - 1);
            }
        }

        Map<Integer, java.util.TreeMap<Integer, Value>> rowNodes = new java.util.HashMap<>();
        for (Value v : topo) rowNodes.computeIfAbsent(y.get(v), r -> new java.util.TreeMap<>())
                                     .put(colX[col.get(v)], v);

        StringBuilder sb = new StringBuilder();
        for (int r = 0; r < height; r++) {
            java.util.TreeMap<Integer, Value> nodes = rowNodes.get(r);
            StringBuilder line = new StringBuilder("  ");        // match the 2-space UI margin
            int x = 0;
            boolean done = false;
            while (x < width && !done) {
                Value v = nodes == null ? null : nodes.get(x);
                if (v != null) {
                    boolean lit = !backpropMode || Math.abs(v.grad) > 1e-12;
                    line.append(nodeChip(v, names, highlight, backpropMode))
                        .append(' ').append(gradGauge(v.grad, maxAbs, GAUGE_W, lit));
                    x += chipText(v, names).length() + 3 + GAUGE_W;
                    if (v == root) { line.append(outTag(root)); done = true; }   // nothing is right of the output
                } else {
                    int b = mask[r][x];
                    line.append(b == 0 ? " " : Tui.color(String.valueOf(glyph(b)), Tui.GRAY));
                    x++;
                }
            }
            sb.append(rtrim(line.toString())).append('\n');
        }
        return sb.toString();
    }

    private static String nodeChip(Value v, Map<Value, String> names, Value highlight, boolean backpropMode) {
        if (v == highlight) return Tui.chip(" " + chipText(v, names) + " ", 226, 16);   // bright = current
        boolean lit = !backpropMode || Math.abs(v.grad) > 1e-12;
        return opChip(v, names, lit);
    }

    /** A short "◂ output" marker after the root node; the value is shown in the summary line below. */
    private static String outTag(Value root) {
        return Tui.color(" ◂ ", Tui.BOLD + Tui.YELLOW) + Tui.color("output", Tui.BOLD + Tui.YELLOW);
    }

    /** Plain (un-coloured) form of {@link #outTag}, used to measure whether the row fits. */
    private static String outTagPlain(Value root) {
        return " ◂ output";
    }

    private static void hseg(int[][] m, int yy, int xa, int xb) {
        if (xa > xb) { int t = xa; xa = xb; xb = t; }
        for (int x = xa; x <= xb; x++) bit(m, x, yy, (x > xa ? DW : 0) | (x < xb ? DE : 0));
    }
    private static void vseg(int[][] m, int xx, int ya, int yb) {
        if (ya > yb) { int t = ya; ya = yb; yb = t; }
        for (int y = ya; y <= yb; y++) bit(m, xx, y, (y > ya ? DN : 0) | (y < yb ? DS : 0));
    }
    private static void bit(int[][] m, int x, int y, int bits) {
        if (y >= 0 && y < m.length && x >= 0 && x < m[0].length) m[y][x] |= bits;
    }
    private static char glyph(int b) {
        switch (b) {
            case 0:                      return ' ';
            case DN: case DS: case DN | DS:           return '│';
            case DE: case DW: case DE | DW:           return '─';
            case DE | DS:                return '┌';
            case DS | DW:                return '┐';
            case DN | DE:                return '└';
            case DN | DW:                return '┘';
            case DN | DE | DS:           return '├';
            case DN | DS | DW:           return '┤';
            case DE | DS | DW:           return '┬';
            case DN | DE | DW:           return '┴';
            default:                     return '┼';
        }
    }
    private static String rtrim(String s) {
        int e = s.length();
        while (e > 0 && s.charAt(e - 1) == ' ') e--;
        return s.substring(0, e);
    }

    static String label(Value v, Map<Value, String> names) {
        String name = names == null ? null : names.get(v);
        if (name != null) return name;
        if (v.prev.isEmpty()) return "leaf";
        return prettyOp(v.op);
    }

    private static String prettyOp(String op) {
        if (op == null || op.isEmpty()) return "leaf";
        if (op.startsWith("^")) return "pow(" + op.substring(1) + ")";
        return op;
    }

    private static String fmt(double d) {
        return String.format("%.3f", d);
    }

    private static String gradColored(double g) {
        String s = String.format("%+.3f", g);
        if (g > 1e-12) return Tui.color(s, Tui.GREEN);
        if (g < -1e-12) return Tui.color(s, Tui.RED);
        return Tui.color(s, Tui.GRAY);
    }

    // ----- network diagram -----
    static String renderNetwork(MLP model) {
        StringBuilder sb = new StringBuilder();
        List<Layer> layers = model.layers;
        int nin = layers.get(0).neurons.get(0).w.size();

        StringBuilder chain = new StringBuilder("  Input(").append(nin).append(")");
        for (int li = 0; li < layers.size(); li++) {
            Layer layer = layers.get(li);
            int size = layer.neurons.size();
            Activation act = layer.neurons.get(0).activation;
            String kind = li == layers.size() - 1 ? "out" : "hidden";
            chain.append(Tui.color(" → ", Tui.GRAY))
                 .append("[").append(size).append(" ").append(act.name().toLowerCase())
                 .append(" ").append(kind).append("]");
        }
        sb.append(Tui.color(chain.toString(), Tui.BOLD)).append("\n\n");

        // a small vertical sketch of each layer's neurons (capped)
        for (int li = 0; li < layers.size(); li++) {
            Layer layer = layers.get(li);
            int size = layer.neurons.size();
            int dots = Math.min(size, 12);
            StringBuilder row = new StringBuilder();
            for (int i = 0; i < dots; i++) row.append("o ");
            if (size > dots) row.append("…");
            Activation act = layer.neurons.get(0).activation;
            sb.append(String.format("  layer %d  %-10s %s%n",
                    li + 1, "(" + act.name().toLowerCase() + ")",
                    Tui.color(row.toString(), Tui.CYAN)));
        }
        sb.append("\n  ").append(Tui.color(model.parameters().size() + " parameters", Tui.YELLOW))
          .append("\n");
        return sb.toString();
    }
}
