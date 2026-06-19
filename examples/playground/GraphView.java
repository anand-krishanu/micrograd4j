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
    /** Render the graph rooted at {@code root} as an ASCII tree. {@code highlight} (nullable) is marked. */
    static String renderGraph(Value root, Map<Value, String> names, Value highlight) {
        StringBuilder sb = new StringBuilder();
        Map<Value, Integer> ids = new IdentityHashMap<>();
        Set<Value> shown = Collections.newSetFromMap(new IdentityHashMap<>());
        int[] counter = {0};
        renderNode(root, "", true, true, sb, names, ids, shown, counter, highlight);
        return sb.toString();
    }

    private static void renderNode(Value v, String prefix, boolean isLast, boolean isRoot,
                                   StringBuilder sb, Map<Value, String> names,
                                   Map<Value, Integer> ids, Set<Value> shown,
                                   int[] counter, Value highlight) {
        String connector = isRoot ? "" : (isLast ? "└─ " : "├─ ");

        if (shown.contains(v)) {
            int id = ids.get(v);
            sb.append(prefix).append(connector)
              .append(Tui.color("↑ #" + id + " " + label(v, names), Tui.GRAY))
              .append('\n');
            return;
        }
        int id = ++counter[0];
        ids.put(v, id);
        shown.add(v);

        String marker = v == highlight ? Tui.color("▶ ", Tui.YELLOW) : "  ";
        sb.append(prefix).append(connector).append(marker)
          .append(Tui.color("#" + id + " ", Tui.GRAY))
          .append(Tui.color(label(v, names), Tui.BOLD))
          .append("  data=").append(fmt(v.data))
          .append("  grad=").append(gradColored(v.grad))
          .append('\n');

        List<Value> children = new ArrayList<>(v.prev);
        children.sort(Comparator.comparingDouble((Value c) -> c.data).thenComparing(c -> c.op));
        String childPrefix = prefix + (isRoot ? "" : (isLast ? "   " : "│  "));
        for (int i = 0; i < children.size(); i++) {
            renderNode(children.get(i), childPrefix, i == children.size() - 1, false,
                    sb, names, ids, shown, counter, highlight);
        }
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
