import io.github.anandkrishanu.micrograd.Value;

import org.jline.terminal.Attributes;
import org.jline.utils.NonBlockingReader;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Map;

/**
 * Live, key-driven views of the computation graph that mirror the loss-curve's "it moves" feel:
 *
 *  - {@link #view}     browse the graph node by node, inspecting each one's data / grad / rule.
 *  - {@link #backprop} scrub the backward pass: step it forward and back, or auto-play the
 *                      animation and watch gradients flow.
 *
 * Both redraw in place (cursor home + erase-down, no flicker) and read single keypresses in raw
 * mode (no Enter needed). On a dumb terminal they fall back to the plain static rendering.
 */
final class GraphInteractive {
    private GraphInteractive() {}

    private enum Key { UP, DOWN, LEFT, RIGHT, ENTER, AUTO, RESTART, STEPMODE, QUIT, NONE }

    private static final long STEP_DELAY_MS = 180;

    // ============================================================= backprop scrubber
    static void backprop(Value root, Map<Value, String> names) {
        if (Tui.dumb) { backpropStatic(root, names); return; }

        List<Value> topo = GraphView.buildTopo(root);
        Collections.reverse(topo);                 // parents-first, like Value.backward()
        int n = topo.size();
        int k = 0;                                  // nodes whose _backward has been applied
        boolean auto = false;

        NonBlockingReader in = Tui.terminal.reader();
        Attributes saved = Tui.terminal.enterRawMode();
        try {
            Tui.clear();
            while (true) {
                renderBackprop(root, names, topo, k, n, auto);
                Key key;
                if (auto) {
                    int c = in.read(STEP_DELAY_MS);
                    if (c == NonBlockingReader.READ_EXPIRED) {   // no key: advance a frame
                        if (k < n) k++; else auto = false;
                        continue;
                    }
                    auto = false;                                 // any key pauses, then is handled
                    key = parseKey(in, c);
                } else {
                    key = parseKey(in, in.read());
                }
                switch (key) {
                    case RIGHT: case ENTER: case DOWN: if (k < n) k++; break;
                    case LEFT: case UP: if (k > 0) k--; break;
                    case AUTO: auto = k < n; break;
                    case RESTART: k = 0; break;
                    case QUIT: return;
                    default: break;
                }
            }
        } catch (IOException e) {
            // terminal closed mid-read: just leave the view
        } finally {
            Tui.terminal.setAttributes(saved);
            Tui.print("\r\n");
            Tui.flush();
        }
    }

    private static void renderBackprop(Value root, Map<Value, String> names, List<Value> topo,
                                       int k, int n, boolean auto) {
        // replay the first k local-backwards from a clean slate so stepping back also works
        root.zeroGrad();
        root.grad = 1.0;
        for (int i = 0; i < k; i++) topo.get(i)._backward();
        Value applied = k > 0 ? topo.get(k - 1) : null;

        StringBuilder f = new StringBuilder(Tui.HOME);
        String title = k == 0 ? "Backprop — output gradient seeded to 1.0"
                : k == n ? "Backprop — complete"
                : "Backprop — applied " + opName(applied);
        f.append(crlf(Tui.color("  " + title, Tui.BOLD + Tui.CYAN)));
        double frac = n == 0 ? 1.0 : (double) k / n;
        f.append(crlf("  step " + k + "/" + n + "   " + Tui.progressBar(frac, 22)
                + (auto ? Tui.color("  ▶ playing", Tui.GREEN) : "")));
        f.append(crlf(""));
        f.append(block(GraphView.renderGraph(root, names, applied)));
        if (applied != null && !applied.prev.isEmpty()) {
            f.append(crlf(Tui.color("  " + Explain.localRule(applied), Tui.DIM)));
        } else if (k == n) {
            f.append(crlf(Tui.color("  ✓ gradients have reached every input", Tui.GREEN)));
        } else {
            f.append(crlf(Tui.color("  press → to apply the first local backward", Tui.DIM)));
        }
        f.append(crlf(""));
        f.append(crlf(Tui.color("  →/Enter step · ← back · a/space auto-play · r restart · q back", Tui.DIM)));
        f.append(Tui.CLR_DOWN);
        Tui.print(f.toString());
        Tui.flush();
    }

    /** Non-interactive walk used by --demo and on dumb terminals. */
    static void backpropStatic(Value root, Map<Value, String> names) {
        List<Value> topo = GraphView.buildTopo(root);
        Collections.reverse(topo);
        root.zeroGrad();
        root.grad = 1.0;

        Tui.header("Backprop — " + topo.size() + " nodes");
        Tui.println(Tui.color("  output grad seeded to 1.0; each step applies one node's local backward.", Tui.DIM));
        Tui.print(GraphView.renderGraph(root, names, null));
        for (Value v : topo) {
            v._backward();
            Tui.println("  applied " + Tui.color(opName(v), Tui.BOLD)
                    + "  data=" + String.format("%.3f", v.data)
                    + "  grad=" + String.format("%+.3f", v.grad));
        }
        Tui.print(GraphView.renderGraph(root, names, null));
        Tui.println(Tui.color("  done. input gradients:", Tui.BOLD));
        for (Map.Entry<Value, String> e : names.entrySet()) {
            if (e.getValue() != null) Tui.println("    " + e.getValue() + " : " + gradStr(e.getKey().grad));
        }
    }

    // ============================================================= graph inspector
    static void view(Value root, Map<Value, String> names) {
        if (Tui.dumb) { Tui.print(GraphView.renderGraph(root, names, null)); return; }

        List<Value> order = GraphView.displayOrder(root);
        int sel = 0;
        NonBlockingReader in = Tui.terminal.reader();
        Attributes saved = Tui.terminal.enterRawMode();
        try {
            Tui.clear();
            while (true) {
                renderView(root, names, order, sel);
                Key key = parseKey(in, in.read());
                switch (key) {
                    case UP: case LEFT: sel = (sel - 1 + order.size()) % order.size(); break;
                    case DOWN: case RIGHT: sel = (sel + 1) % order.size(); break;
                    case STEPMODE: case ENTER:
                        Tui.terminal.setAttributes(saved);
                        Tui.print("\r\n");
                        backprop(root, names);       // hands off; returns to caller after
                        return;
                    case QUIT: return;
                    default: break;
                }
            }
        } catch (IOException e) {
            // terminal closed mid-read: leave the view
        } finally {
            Tui.terminal.setAttributes(saved);
            Tui.print("\r\n");
            Tui.flush();
        }
    }

    private static void renderView(Value root, Map<Value, String> names, List<Value> order, int sel) {
        Value v = order.get(sel);
        StringBuilder f = new StringBuilder(Tui.HOME);
        f.append(crlf(Tui.color("  Computation graph", Tui.BOLD + Tui.CYAN)
                + Tui.color("   " + order.size() + " nodes · ↑/↓ to inspect", Tui.DIM)));
        f.append(crlf(""));
        f.append(block(GraphView.renderGraph(root, names, v)));
        f.append(crlf(""));
        f.append(crlf(Tui.color("  selected ", Tui.DIM)
                + Tui.color("#" + (sel + 1) + " " + label(v, names), Tui.BOLD)
                + "   data=" + String.format("%.3f", v.data) + "   grad=" + gradStr(v.grad)));
        if (!v.prev.isEmpty()) {
            StringBuilder ins = new StringBuilder("  inputs:  ");
            for (Value c : sortedChildren(v)) {
                ins.append(label(c, names)).append("=").append(String.format("%.3f", c.data)).append("   ");
            }
            f.append(crlf(Tui.color(ins.toString(), Tui.DIM)));
            f.append(crlf(Tui.color("  rule:  " + Explain.localRule(v), Tui.DIM)));
        } else {
            f.append(crlf(Tui.color("  this is a leaf (an input); its grad is dL/d" + label(v, names), Tui.DIM)));
            f.append(crlf(""));
        }
        f.append(crlf(""));
        f.append(crlf(Tui.color("  ↑/↓ inspect · s/Enter step through backprop · q back", Tui.DIM)));
        f.append(Tui.CLR_DOWN);
        Tui.print(f.toString());
        Tui.flush();
    }

    // ============================================================= helpers
    /** Parse one keypress (handling ESC [ A.. arrow sequences) into an action. */
    private static Key parseKey(NonBlockingReader in, int c) throws IOException {
        if (c < 0) return Key.QUIT;                       // EOF
        if (c == 27) {                                     // ESC — maybe an arrow
            int c2 = in.read(30L);
            if (c2 == '[' || c2 == 'O') {
                switch (in.read(30L)) {
                    case 'A': return Key.UP;
                    case 'B': return Key.DOWN;
                    case 'C': return Key.RIGHT;
                    case 'D': return Key.LEFT;
                    default: return Key.NONE;
                }
            }
            return Key.QUIT;                               // lone Esc
        }
        switch (c) {
            case '\r': case '\n': return Key.ENTER;
            case ' ': case 'a': case 'A': return Key.AUTO;
            case 'r': case 'R': return Key.RESTART;
            case 's': case 'S': return Key.STEPMODE;
            case 'q': case 'Q': return Key.QUIT;
            case 'h': return Key.LEFT;
            case 'l': return Key.RIGHT;
            case 'j': return Key.DOWN;
            case 'k': return Key.UP;
            default: return Key.NONE;
        }
    }

    private static List<Value> sortedChildren(Value v) {
        List<Value> children = new ArrayList<>(v.prev);
        children.sort(Comparator.comparingDouble((Value c) -> c.data).thenComparing(c -> c.op));
        return children;
    }

    private static String label(Value v, Map<Value, String> names) {
        String n = names == null ? null : names.get(v);
        if (n != null) return n;
        if (v.prev.isEmpty()) return "leaf";
        if (v.op == null || v.op.isEmpty()) return "leaf";
        return v.op.startsWith("^") ? "pow(" + v.op.substring(1) + ")" : v.op;
    }

    private static String opName(Value v) {
        return v.op == null || v.op.isEmpty() ? "leaf" : v.op;
    }

    private static String gradStr(double g) {
        String s = String.format("%+.3f", g);
        if (g > 1e-12) return Tui.color(s, Tui.GREEN);
        if (g < -1e-12) return Tui.color(s, Tui.RED);
        return Tui.color(s, Tui.GRAY);
    }

    /** One in-place line: clear to EOL then CRLF (raw mode does no \n translation). */
    private static String crlf(String s) {
        return s + Tui.CLR_EOL + "\r\n";
    }

    /** A multi-line block (e.g. the rendered graph) with each line cleared to EOL. */
    private static String block(String multi) {
        if (multi.endsWith("\n")) multi = multi.substring(0, multi.length() - 1);
        StringBuilder sb = new StringBuilder();
        for (String l : multi.split("\n", -1)) sb.append(l).append(Tui.CLR_EOL).append("\r\n");
        return sb.toString();
    }
}
